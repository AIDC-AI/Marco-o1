import os
import json
import math
import random
from tree_search.mcts_nodes import *
from tree_search.evaluator.evaluator import *

from tree_search.utils import snake_to_pascal


class MCTS_Engine():
    def __init__(self, max_rollout_depth, generate_func='local', exploration_value=1.414, max_new_tokens=1024,
                 mini_step=False, args={}):
        assert "action_tree" in args, "args must contain action_tree"
        self.max_rollout_depth = max_rollout_depth
        self.exploration_value = exploration_value
        self.max_new_tokens = max_new_tokens
        self.mini_step = mini_step
        self.action_tree = args.action_tree
        self.mode = args.mode  # for debug
        self.output_tree = args.output_tree
        self.args = args
        evaluator_name = '%s_evaluator' % (args.evaluate_func)
        if evaluator_name in globals():
            self.evaluate_func = globals()[evaluator_name]
        else:
            raise ValueError(f"Invalid evaluate_func: {args.evaluate_func}")

        if generate_func == 'api':
            from tree_search.utils.model_IO.openAI_API import generate_with_api_model, load_api_model
            self.model = load_api_model()
            self.generate_func = generate_with_api_model
        elif generate_func == 'hf':
            from tree_search.utils.model_IO.hf_API import generate_with_hf_model, load_hf_model
            self.tokenizer, self.model = load_hf_model()
            self.generate_func = generate_with_hf_model
        elif generate_func == 'vLLM':
            from tree_search.utils.model_IO.vLLM_API import generate_with_vLLM_model, load_vLLM_model
            self.tokenizer, self.model = load_vLLM_model()
            self.generate_func = generate_with_vLLM_model
        elif generate_func == 'local':
            from tree_search.utils.model_IO.local_http_API import get_response
            self.generate_func = get_response
        else:
            raise ValueError(f"Invalid generate_func: {generate_func}")

    def get_rollout_reward(self, node, ground_truth):
        # special node function_call is not terminal
        assert node.is_terminal, "Rollout node must be terminal"

        # for function_call node, return the reward of all function_call nodes
        if self.args.use_function_call:

            if self.args.mask_asking is False and node.find_parent_by_action('asking') is None:
                return 0

            function_call_node_list = []
            cur_node = node
            while cur_node is not None:
                cur_node = cur_node.find_parent_by_action('function_call')
                if cur_node is not None:
                    function_call_node_list.append(cur_node)
            return self.evaluate_func(function_call_node_list, ground_truth)

        return self.evaluate_func(node, ground_truth)

    def do_rollout(self, question_dict):
        final_wrong_answer = ''
        final_correct_answer = ''
        total_reward = 0

        if self.args.use_multi_turn:
            original_question = question_dict['conversation'][0]['content']
            question = "%s%s" % (self.args.base_prompt, original_question)
            if self.args.use_function_call:
                function_call_response = question_dict['function_call_response']
                function_list = question_dict['function']
                ground_truth = question_dict['solution']
                ask_response = question_dict['ask_response']

                if len(ask_response) > 0:
                    self.args.mask_asking = False
                else:
                    self.args.mask_asking = True
        else:
            original_question = question_dict['problem']
            question = "%s%s" % (self.args.base_prompt, original_question)
            ground_truth = question_dict['solution']
        root = BaseNode(parent=None)
        root.user_question = question
        root.node_value = original_question
        root.other_info['ground_truth'] = ground_truth

        if self.args.use_function_call:
            root.other_info['function_call_response'] = function_call_response
            root.other_info['function_list'] = function_list
            root.other_info['ask_response'] = ask_response

        cur_rollout_id = 0
        search_success = False

        if self.mode == 'debug':
            print('mask_asking:', self.args.mask_asking, flush=True)

        while True:
            print(f'rollout {cur_rollout_id} start: ', flush=True)
            node = self.do_selection(root)
            reward = self.get_rollout_reward(node, ground_truth)
            self.do_backpropagation(node, reward)

            # log wrong answer and add special node for wrong answer
            if reward == 0:
                final_wrong_answer = node.all_path_value
                if len(self.args.use_for_wrong_answer) != 0:
                    self.add_wrong_answer_node(node)
            else:
                final_correct_answer = node.all_path_value
            total_reward += reward

            print(f'currect total node num: {NodeCounter._id_counter}', flush=True)

            # compute out condition
            if cur_rollout_id > self.args.max_rollout_time * 4 and total_reward < self.args.search_reward_threshold[0]:
                break
            if cur_rollout_id > self.args.max_rollout_time and total_reward >= self.args.search_reward_threshold[0]:
                search_success = True
                break
            if total_reward >= self.args.search_reward_threshold[1]:
                search_success = True
                break

            cur_rollout_id += 1

        best_child = self.get_best_chain(root)

        # save search info
        if self.output_tree:
            self.print_tree(root, question_dict['id'], final_correct_answer, final_wrong_answer,
                            question_dict['solution'], search_success)
        return best_child[-1], search_success

    def do_backpropagation(self, node, reward):
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

    def do_selection(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_value)
            else:
                self.do_expansion(node)
        return node

    def do_expansion(self, node, default_actions={}):
        if len(default_actions)==0:
            default_actions = self.get_actions(node)

        for action, n in default_actions.items():
            if 'mask_asking' in self.args and self.args.mask_asking and action == 'asking':
                continue
            # init node info
            action_class_name = snake_to_pascal(action)
            new_node = globals()[f'{action_class_name}Node'](parent=node)
            new_node.user_question = node.user_question
            new_node.all_path_value = node.all_path_value
            new_node.other_info = node.other_info

            # when use step, use no tag, set tag to \n
            if self.args.use_tag is False:
                new_node.node_action_description = ["", "\n"]

            output_texts = self.do_generate(new_node, n=n)

            if self.mode == 'debug':
                print(action, ':', new_node.node_id, '->', output_texts[0][:200].replace('\n', '<\\n>'), flush=True)

            # create new node
            for i in range(n):
                if i != 0:
                    new_node = globals()[f'{action_class_name}Node'](parent=node)
                new_node.node_value = output_texts[i]
                new_node.user_question = node.user_question
                new_node.other_info = node.other_info
                new_node.parent = node
                if not new_node.show_in_history:
                    new_node.all_path_value = node.all_path_value
                else:
                    new_node.all_path_value = node.all_path_value + output_texts[i]

                if action == 'answer':
                    new_node.is_terminal = True

                # 使用step的时候，不使用tag引导，将结束符置为\n
                if self.args.use_tag is False:
                    new_node.node_action_description = ["", "\n"]

                # for function_call node, post init
                if new_node.node_action_name == 'function_call':
                    if len(node.other_info['function_call_response']) > 0:
                        new_node.post_init(node.other_info['function_call_response'], node.other_info['ground_truth'])

                if new_node.node_action_name == 'asking':
                    if 'ask_response' in node.other_info:
                        new_node.post_init(node.other_info['ask_response'])

                node.addChild(action, new_node)

        node.is_fully_expanded = True
        return node

    def add_wrong_answer_node(self, node):
        for action, back_info in self.args.use_for_wrong_answer.items():
            back_to_node = node.find_parent_by_action(action)
            if back_to_node is None:
                continue
            # 对于首次错误的路径，引导回归正确答案
            # if '<thinking_>' not in node.all_path_value:
            if self.mode == 'debug':
                print(f'**** -> add wrong answer node {action},{node.node_value}', flush=True)
            self.do_expansion(back_to_node, back_info)


    def do_generate(self, node, n):
        prefill_text = node.prefill_text
        user_question = node.user_question
        if len(prefill_text) > 0:
            # 如果prefill有结束标签，说明不需要继续输出
            prefill = random.choice(prefill_text)
            # if node.node_action_name == 'reflection':
            if prefill.endswith(node.node_action_description[-1]):
                return ['%s%s\n' % (node.node_action_description[0], prefill)] * n
            history_text = '%s%s%s' % (node.all_path_value, node.node_action_description[0], prefill)
        else:
            history_text = '%s%s' % (node.all_path_value, node.node_action_description[0])

        if node.use_special_model:
            special_model = True
        else:
            special_model = False

        if 'function_list' in node.other_info:
            tools = node.other_info['function_list']
        else:
            tools = None

        end_tokens = node.node_action_description[-1] if self.args.use_tag else '<|im_end|>'
        output_text = self.generate_func(user_question=user_question, history_text=history_text,
                                         max_tokens=self.max_new_tokens, n=n, end_tokens=end_tokens,
                                         special_model=special_model, tools=tools)

        # add to node_post_init

        if len(prefill_text) > 0:
            output_text = ['%s%s%s%s\n' % (
            node.node_action_description[0], prefill, output_text[i], node.node_action_description[-1]) for i in
                           range(n)]
        else:
            output_text = [
                '%s%s%s\n' % (node.node_action_description[0], output_text[i], node.node_action_description[-1]) for i
                in range(n)]
        return output_text

    def get_actions(self, node):
        if node.node_action_name == 'evaluate' and node.node_value.count('True') > 1:
            return {'answer': 1}
        if node.node_action_name in self.action_tree:
            return self.action_tree[node.node_action_name]
        else:
            raise ValueError(f"Invalid node action name: {node.node_action_name}")

    def get_best_child(self, node, explorationValue):
        """
        根据UCT值获取最佳子节点  如果有多个最佳  随机返回一个
        """
        bestValue = float("-inf")
        bestNodes = []
        total_children = node.getChildren()
        for child in total_children:
            # 如果没探索，则返回当前节点
            if child.num_visits == 0:
                return child
            # 计算UCT值
            nodeValue = child.total_reward / child.num_visits + explorationValue * math.sqrt(
                2 * math.log(node.num_visits) / child.num_visits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def get_best_chain(self, node):

        def choose_max_node(nodes):
            max_node = nodes[0]
            for node in nodes:
                if node.num_visits > max_node.num_visits:
                    max_node = node
            return max_node

        best_chain = []
        while not node.is_terminal:
            best_chain.append(node)
            node = choose_max_node(node.getChildren())
        best_chain.append(node)
        return best_chain

    def print_tree(self, root, question_id, final_correct_answer, final_wrong_answer, question_solution,
                   search_success):
        os.makedirs('./output/%s/tree_output' % (self.args.output_folder), exist_ok=True)
        save_file = os.path.basename(self.args.config).replace('_config.json', '')
        save_file_path = './output/%s/tree_output/%s_%s.txt' % (self.args.output_folder, save_file, question_id)
        tree_str = ''
        total_children = []

        def dfs_search(node):
            nonlocal tree_str
            node_value = node.node_value.replace("\n", "<\\n>")
            cur_node_str = '    ' * (node.getDepth() - 1) + f'{node}{node_value}'
            tree_str += cur_node_str + '\n'
            total_children.append(node.get_json_dict())
            for child in node.getChildren():
                dfs_search(child)

        # print_tree
        dfs_search(root)

        with open(save_file_path, 'w+') as f:
            f.write(tree_str)

        save_file_path = './output/%s/tree_output/%s_%s.json' % (self.args.output_folder, save_file, question_id)
        with open(save_file_path, 'w+') as f:
            json.dump(total_children, f, indent=4, ensure_ascii=False)

        os.makedirs('./output/%s/search_res' % (self.args.output_folder), exist_ok=True)
        search_res_path = './output/%s/search_res/%s_%s.txt' % (self.args.output_folder, save_file, question_id)

        best_chain_str = 'Question: %s\n' % (root.node_value)
        best_chain_str += '\tAnswer: %s\n\n*******************\n\n' % question_solution
        if search_success:
            # best_chain = self.get_best_chain(root)
            # best_chain_str += best_chain[-1].all_path_value
            best_chain_str += final_correct_answer
        if len(final_wrong_answer) > 0:
            best_chain_str += '\n\n*******************\n\nWrong Answer:\n%s\n' % (final_wrong_answer)

        with open(search_res_path, 'w+') as f:
            f.write(best_chain_str)
        print(f'Search result has been saved to {save_file_path}')
