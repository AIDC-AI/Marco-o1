import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from tree_search.mcts_nodes import *
from tree_search.utils import snake_to_pascal
from tree_search.utils.mcts_engine import MCTS_Engine
from tree_search.utils.model_IO.local_http_API import set_urls

args = argparse.ArgumentParser()
args.add_argument('--config', type=str, default='./tree_search/configs/demo_config.json')
args = args.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_config_to_node_class(config):
    action_tree = {}

    for action_name, action_info in config['action_tree'].items():
        action_class_name = snake_to_pascal(action_name)
        action_class = globals()[f'{action_class_name}Node']
        action_class.prefill_text = action_info['prefill_text']
        action_class.description = action_info['description']
        action_class.show_in_history = action_info['show_in_history']
        action_class.use_special_model = action_info['special_model']
        action_tree[action_name] = action_info['next_step']

    args.action_tree = action_tree
    args.max_rollout_time = config['max_rollout_time']
    args.max_tokens = config['max_tokens']
    args.base_prompt = config['base_prompt']
    args.output_tree = config['output_tree']
    args.input_path = config['input_path']
    args.evaluate_func = config['evaluate_func']
    args.search_reward_threshold = config['search_reward_threshold']
    args.use_for_wrong_answer = config['use_for_wrong_answer']
    args.output_folder = config['output_folder']
    args.generate_func = config['generate_func']
    args.mask_asking = False
    if args.generate_func == 'local':
        set_urls(config['url1'], config['url2'])
    args.mode = config['mode']
    args.use_step = config['use_step']
    args.use_mini_step = config['use_mini_step']
    args.use_function_call = config['use_function_call']

    if args.use_function_call:
        args.use_multi_turn = True
    else:
        args.use_multi_turn = False

    if args.use_step or args.use_mini_step:
        args.use_tag = False
    else:
        args.use_tag = True


def read_data(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data


def main():
    config = load_config(args.config)
    load_config_to_node_class(config)
    data = read_data(args.input_path)
    if config['mode'] == 'debug':
        data = data[:2]
    mcts = MCTS_Engine(max_rollout_depth=args.max_rollout_time, generate_func=args.generate_func,
                       max_new_tokens=args.max_tokens, args=args)
    for d in tqdm(data):
        best_child = mcts.do_rollout(d)


if __name__ == "__main__":
    main()