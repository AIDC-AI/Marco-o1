import re
import json


def planning_evaluator(answer_node_value, ground_truth):
    # Todo your code here
    return True


def math_evaluator(answer_node_value, ground_truth):
    # Todo your code here
    return True


def count_latter_evaluator(answer_node, ground_truth):
    answer_node_value = answer_node.node_value
    answer = re.findall(r'boxed{(.*)}', answer_node_value)
    # 防止标签连续乱出
    if answer_node_value.count('<') > 2:
        return 0
    if len(answer) == 0:
        return 0
    if str(answer[0]).strip() == str(ground_truth).strip():
        return 1
    else:
        return 0


def function_call_evaluator(function_call_node_list, ground_truth):
    if len(ground_truth) == 0 and len(function_call_node_list) == 0:
        return 1
    # Todo your code here
    return True


def instruct_follow_evaluator(answer_node_value, ground_truth):
    import copy
    from tree_search.evaluator import ifeval

    instruction_id_list = ground_truth['instruction_id_list']
    kwargs = ground_truth['kwargs']
    response = answer_node_value.node_value
    language = ground_truth['language']

    input_dict = {
        'instruction_id_list': instruction_id_list,
        'kwargs': [json.loads(each) for each in kwargs],
        'response': response
    }

    def parse_result(outputs):

        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        for example in outputs:
            follow_instruction_list = example["follow_instruction_list"]
            instruction_id_list = example["instruction_id_list"]

            prompt_total += 1
            if all(follow_instruction_list):
                prompt_correct += 1

            instruction_total += len(instruction_id_list)
            instruction_correct += sum(follow_instruction_list)

        return prompt_correct / prompt_total, instruction_correct / instruction_total

    def gen_acc_loose(x):
        response = str(x["response"])
        r = response.split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        all_responses = [
            response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]
        instruction_list = x["instruction_id_list"]
        is_following_list = []
        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = ifeval.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            instruction.build_description(**x["kwargs"][index])

            is_following = False
            for r in all_responses:  # type: ignore
                if r.strip() and instruction.check_following(r):  # type: ignore
                    is_following = True
                    break

            is_following_list.append(is_following)
        return {
            "follow_instruction_list": is_following_list,
            "instruction_id_list": instruction_list,
        }

    outputs_loose = [gen_acc_loose(input_dict)]
    cur_other_info = copy.copy(answer_node_value.other_info)
    cur_other_info['outputs_loose'] = outputs_loose[0]['follow_instruction_list']
    answer_node_value.other_info = copy.copy(cur_other_info)
    if False in outputs_loose[0]['follow_instruction_list']:
        print(answer_node_value.node_id, outputs_loose[0]['follow_instruction_list'])
    res_loose = parse_result(outputs_loose)
    return int(res_loose[0])


if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from tools.tree_search.mcts_nodes import *

    answer = AnswerNode(None)
    answer.node_value = "the social network, directed by david fincher and written by aaron sorkin, is a masterful exploration of ambition, betrayal, and the birth of a global phenomenon. released in 2010, the film chronicles the creation of facebook, focusing on its co-founder mark zuckerberg, played with enigmatic intensity by jesse eisenberg."
    ground_true = {
        "instruction_id_list": [
            "change_case:english_lowercase"
        ],
        "kwargs": [
            "{}"
        ],
        "language": "English"
    }
    print(instruct_follow_evaluator(answer, ground_true))
