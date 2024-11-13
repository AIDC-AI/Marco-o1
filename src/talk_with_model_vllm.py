"""
Copyright (C) 2024 AIDC-AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from vllm import LLM, SamplingParams
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(path):
    llm = LLM(model=path, tensor_parallel_size=4)
    return llm

def build_prompt(history,tokenizer):
    text = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
    # only use for output Chinese CoT
    text +=  '\n<Thought>\n好的，'
    print(text,flush=True)
    print('<Thought>\n好的，',end='')
    return text

def chat(llm, tokenizer):
    history = []
    print("输入 'q' 退出，输入 'c' 清空对话历史。")
    while True:
        user_input = input("User: ")
        
        if user_input.lower() == 'q':
            print("退出对话。")
            break
        elif user_input.lower() == 'c':
            print("清空对话历史。")
            history = []
            continue
        
        # if not any(msg['role'] == 'system' for msg in history):
        #     history.append({
        #         "role": "system",
        #         "content": "You are a helpful assistant."
        #     })
        
        history.append({"role": "user", "content": user_input})
        
        prompt = build_prompt(history, tokenizer)
        prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n约翰点了一些披萨和他的朋友分享。一共有 20 位朋友，约翰想确保每个人都能吃到 4 块。披萨只能切成 8 块出售。约翰需要点多少披萨？<|im_end|>\n<|im_start|>assistant\n\n<Thought>\n好的，'
        max_new_tokens = 4096

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,
            top_p=0.9
        )
        
        outputs = llm.generate(
            [prompt],
            sampling_params=sampling_params,
        )
        
        response = outputs[0].outputs[0].text.strip()
        
        print('Assistant:', response)
        history.append({"role": "assistant", "content": response})

def main():
    path = "/mnt/nas/mangguo/o1_test/OpenRLHF/checkpoint/Qwen2-7B-Instruct-SFT-1024v2/global_step450"
    tokenizer = AutoTokenizer.from_pretrained(path)
    if not os.path.exists(path):
        print(f"路径不存在：{path}")
        return
    
    llm = load_model(path)
    print('开始对话。')
    chat(llm, tokenizer)
    
if __name__ == "__main__":
    main()
