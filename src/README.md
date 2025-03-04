# mcts search

We are now open-sourcing our tree search framework.It can be used to search the best chain for a given question.
We use this framework to generate the Long CoT and train our model.

## description

We are now open-sourcing our tree search framework.Its main features are as follows:
1. It can generate serval chain for a given question.
2. When find wrong answer, we will rollback to latest node, add a new node to hint model to generate a new answer. We call it reflection time.
3. Multi-model double check. Inspired by pair programming, we use two models to generate data, and another model to generate some node, such as double check, to prevent thinking dependency.
4. We define some node as MCTS action.
5. We opensource a data annotation tool to filter the data.

<img src="/assets/cot_in_code.jpg"/>


### 1. reflection time

When we find a wrong answer, we will rollback to the latest node（if set `use_for_wrong_answer`), add a new node to hint model to generate a new answer.
˚
As shown in the figure, we rollback to latest node(`thinking` in the figure), and add a new node `Reflection` to hint model to generate a new answer.


### 2. Multi-model double check˚

Inspired by pair programming, we use two models to generate data, one model for normal node generate and another model used to generate some node, such as double check, to prevent thinking dependency.

Specifically, we use `Qwen-2.5-72B-Instruct` and `Llama-3.1-70B-Instruct`, we add a `double-check` node between `thinking` and `answer` node. When generate `double-check`, 
we use `Llama-3.1-70B-Instruct` to generate `double-check` node, we change the model generate server to another model to prevent thinking dependency.

Also, for reflection, we also use `Llama-3.1-70B-Instruct` to generate `reflection` node.

You can configure it in config file by set `special_model` to True.

### 3. MCTS action

We define some node as MCTS action, each node have different action. You need to implement it in `tree_search/mcts_nodes`, new node needs to extend Class `BaseNode` 
and register it in `tree_search/mcts_nodes/__init__.py`.


For more detail, you can check in the code.

### 4. data annotation tool

We also release a data annotation tool to filter the data.You can check it in `tree_search/http_server.py`. 
We use this tool to filter the data which have bad process but have the correct answer and it support latex.

## how to use

1. create your python environment
2. start your inference server
3. add your config file in `tree_search/configs/`
4. run `python main.py`
5. check the result in `output/<your_output_folder>`


### 1. create your python environment

We strongly recommend using VLLM/Huggingface for inference.

you can create your python environment by

`pip install -r requirements.txt`

### 2. start your inference server

Our framework supports VLLM、Huggingface、API call as a inference server. For simple, we only implemented and debug vllm version, 
but reserved the interface for other two versions.

For example, We use `Qwen-2.5-72B-Instruct` and `Llama-3.1-70B-Instruct` to generate the Long CoT.
One for normal generation and one for double check.

```commandline
python ./tree_search/utils/start_vllm_server.py --path /mnt/workspace/checkpoints/public_models/Qwen2.5-72B-Instruct --port 40000 --usage 0.45
python ./tree_search/utils/start_vllm_server.py --path /mnt/workspace/checkpoints/public_models/Llama-3.1-70B-Instruct --port 40001 --usage 0.9
```

### 3. add your config file in `tree_search/configs/`

We show a demo config file in `tree_search/configs/demo_config.json`.It can be run directly.


For you own config, you can add your config file in `tree_search/configs/`.Detail description are as follows:

```json
{
    "desc":"数字母",  // for description, no use in our framework
    "mode":"debug",   // run mode, if debug, it will output more detail log
    "max_rollout_time":32,  // max rollout time
    "max_tokens":512,   // max token for each node
    "output_tree":true, // save tree or not
    "input_path":"./tree_search/init_datasets/latter_count_hard.json",  // input path
    "output_folder":"demo", // output folder
    "generate_func":"local",  // generate function, in ['api','hf, 'local']
    "search_reward_threshold":[4,8],  // min and max correct time. It will make sure it have at least 4 correct answer in 32 times rollout and at most 8 correct answer in 32 times rollout.
    "evaluate_func":"count_latter", // evaluator function, implement in `tree_search/utils/evaluator.py`
    "use_for_wrong_answer":{"double_check":{"reflection":1}}, // when find a wrong answer, use double check. if don't need, set it to {}
    "use_mini_step":false,  // use mini step or not
    "use_function_call":false, // use function call or not
    "use_step":false, // use step or not
    "url1":"https://localhost:40000/generate", // url1, for normal generation
    "url2":"https://localhost:40001/generate", // url2, for double check
    "base_prompt":"Now you need to complete a task related to letters.\nFirst, you need to break down the task in <sub-task>, and your breakdown should be as detailed as possible, considering verification and correction.\nIn <thinking>, you need to complete the tasks broken down in subtask, with one sub-task per <thinking>. For example, if you breakdown into 3 sub-tasks, you need to output 3 <thinking> to complete each of these steps. You need to thinking carefully, you need to do split the string and compute *letter by letter*.\nIn <double-check>, ensure that the steps in <thinking> are correct, including but not limited to word spelling and calculations. If you find errors, you need to clearly point them out.\nIn <reflection>, correct your mistakes and provide the correct answer.\nIn <answer>, output your answer, and the answer should be clearly stated in boxed{xx}.\nNote that these tags cannot be nested, but they can be sequential, so try to keep actions within tags atomic.\n\nNow the question is: ",
    "action_tree":{   // main action tree
        "base":{    // node name, it need to implement in `./tree_search/mcts_nodes`.Same to file name. eg. base -> base_node.py
            "prefill_text":[],    // prefix text for this node.It will add to the end of node input.If multiple, it will be randomly selected
            "description":"User Question",  // node description, no use in our framework
            "show_in_history":true,   // node output show in history or not
            "special_model":false,    // use special model or not.If True, the model will be set to args.url2,default is args.url1
            "next_step":{       // next step for this node, eg. sub_task=2 means will generate 2 sub_task nodes
                "sub_task":2
            }
        }
        ...
    }
}

```


### 4. run `python main.py`

run `python main.py` to start the search.Specify your config file in `--config`.

```commandline
python main.py --config ./tree_search/configs/demo_config.json
```
if output_tree is True, it will output the tree in `output/<your_output_folder>/tree.json`.


### 5. check the result in `output/<your_output_folder>`

We show the lastest correct and wrong answer in folder `./output/<your_output_folder>/search_res/***.txt`.

Also, you can check the detail of holistic tree result in `./output/<your_output_folder>/tree_output`.
