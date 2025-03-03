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
import torch
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to('cuda:0')
    model.eval()
    return tokenizer, model


def generate_response(model, tokenizer,
                      input_ids, attention_mask,
                      max_new_tokens=4096):
    generated_ids = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
            new_token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)
            print(new_token, end='', flush=True)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)


def chat(model, tokenizer):
    history: List[Dict[str, str]] = []
    print("Enter 'q' to quit, 'c' to clear chat history.")
    while True:
        user_input = input("User: ").strip().lower()
        if user_input == 'q':
            print("Exiting chat.")
            break
        if user_input == 'c':
            print("Clearing chat history.")
            history.clear()
            continue
        if not user_input:
            print("Input cannot be empty.")
            continue

        history.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to('cuda:0')

        print('Assistant:', end=' ', flush=True)
        response = generate_response(model, tokenizer, model_inputs.input_ids, model_inputs.attention_mask)
        print()
        history.append({"role": "assistant", "content": response})


def main():
    path = "AIDC-AI/Marco-o1"
    tokenizer, model = load_model_and_tokenizer(path)
    print('Starting chat.')
    chat(model, tokenizer)


if __name__ == "__main__":
    main()
