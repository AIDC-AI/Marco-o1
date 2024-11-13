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
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to("cuda:0")
    model.eval()  
    return tokenizer, model

def chat(model, tokenizer):
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
        
        history.append({"role":"system","content":'You are a helpful assistant.对于<though>内的文本要求：1.尽可能使用中文。2.应该尽可能多的使用‘\n’以确保每一行文本不会太长。'})
        history.append({"role":"user","content":user_input})
        
        text = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
        
        model_inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        
        generated_ids = input_ids
        
        print('Assistant:', end=' ', flush=True) 
        
        max_new_tokens = 4096
        stopping_criteria = tokenizer.eos_token_id
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
                
                new_token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)
                print(new_token, end='', flush=True)
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        print()  
        response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        history.append({"role":"assistant","content":response})

def main():
    path = "/mnt/nas/mangguo/o1_test/OpenRLHF/checkpoint/Qwen2-7B-Instruct-SFT-1024v2/global_step450" 
    if not os.path.exists(path):
        print(f"路径不存在：{path}")
        return
    
    tokenizer, model = load_model_and_tokenizer(path)
    print('开始对话。')
    chat(model, tokenizer)

if __name__ == "__main__":
    main()
