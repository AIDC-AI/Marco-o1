from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()


class ChatRequest(BaseModel):
    user_input: str
    history: list


tokenizer = None
model = None


@app.on_event("startup")
def load_model_and_tokenizer():
    global tokenizer, model
    path = "AIDC-AI/Marco-o1"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = LLM(model=path, tensor_parallel_size=4)


def generate_response(model, text, max_new_tokens=4096):
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
            if new_output.endswith('</Output>'):
                break
    return new_output


@app.post("/chat/")
async def chat(request: ChatRequest):
    if not request.user_input:
        raise HTTPException(status_code=400, detail="Input cannot be empty.")

    if request.user_input.lower() in ['q', 'quit']:
        return {"response": "Exiting chat."}

    if request.user_input.lower() == 'c':
        request.history.clear()
        return {"response": "Clearing chat history."}

    request.history.append({"role": "user", "content": request.user_input})
    text = tokenizer.apply_chat_template(request.history, tokenize=False, add_generation_prompt=True)
    response = generate_response(model, text)
    request.history.append({"role": "assistant", "content": response})

    return {"response": response, "history": request.history}