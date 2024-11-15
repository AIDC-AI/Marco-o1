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
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = LLM(model=path, tensor_parallel_size=4)
    return tokenizer, model


def generate_response(model,
                      text,
                      max_new_tokens=4096):
    new_output = ''
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        top_p=0.9
    )
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model.generate(
                [f'{text}{new_output}'],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            new_output += outputs[0].outputs[0].text
            print(outputs[0].outputs[0].text, end='', flush=True)
            if new_output.endswith('</Output>'):
                break
    return new_output


def chat(model, tokenizer):
    history = []
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
        print('Assistant:', end=' ', flush=True)
        response = generate_response(model, text)
        print()
        history.append({"role": "assistant", "content": response})


def main():
    path = "AIDC-AI/Marco-o1"
    #path = 'Your local path here'

    tokenizer, model = load_model_and_tokenizer(path)
    print('Starting chat.')
    chat(model, tokenizer)


if __name__ == "__main__":
    main()
