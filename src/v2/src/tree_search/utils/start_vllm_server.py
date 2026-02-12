
import os
import json
import argparse
from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,
                    default='your_model_path')
parser.add_argument('--port', type=int, default=40000)
parser.add_argument('--usage', type=float, default=0.9)
args = parser.parse_args()

app = Flask(__name__)


def load_model(path):
    llm = LLM(model=path, tensor_parallel_size=8, gpu_memory_utilization=args.usage)
    return llm


def build_prompt(history, tokenizer, tools={}):
    if len(tools) > 0:
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            tools=tools,
            add_generation_prompt=True,
            use_tools=True,
        )
    else:
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
    print(text)
    return text


def generate_response(llm, tokenizer, user_question, model_output, max_tokens, end_tokens, temperature=0, n=1,
                      use_tools=False):
    history = [{"role": "user", "content": user_question}]
    prompt = build_prompt(history, tokenizer, use_tools)
    if len(model_output) > 0:
        prompt += model_output

    if ',' in end_tokens:
        end_tokens = end_tokens.split(",")
    else:
        end_tokens = [end_tokens]
    print('end_token: ', end_tokens)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        stop=end_tokens,
        n=n
    )

    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params,
    )
    responses = [each.text for each in outputs[0].outputs]
    return responses


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        user_question = data.get('user_question', '')
        model_output = data.get('model_output', '')
        max_tokens = int(data.get('max_tokens', 4096))
        end_tokens = data.get('end_tokens', '')
        use_tools = data.get('use_tools', '{}')
        try:
            use_tools = json.loads(use_tools)
        except:
            use_tools = {}
        if '<|im_end|>' in end_tokens and 'Llama' in args.path:
            end_tokens = '<|end_of_text|>'
            print('Llama')
        temperature = float(data.get('temperature', 0.7))
        n = int(data.get('n', 1))
        response = generate_response(llm, tokenizer, user_question, model_output, max_tokens, end_tokens, temperature,
                                     n, use_tools=use_tools)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    global llm, tokenizer
    path = args.path
    tokenizer = AutoTokenizer.from_pretrained(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在：{path}")

    llm = load_model(path)
    print('模型加载完成。')


if __name__ == "__main__":
    main()
    app.run(host='0.0.0.0', port=args.port)
