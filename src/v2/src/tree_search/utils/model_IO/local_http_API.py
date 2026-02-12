import json
import requests

url1 = ''
url2 = ''


def set_urls(new_url1, new_url2):
    global url1, url2
    url1 = new_url1
    url2 = new_url2


headers = {
}


def get_response(user_question, history_text, end_tokens='<|im_end|>', n=1, special_model=False, max_tokens=1024,
                 temperature=0.7, tools=None):
    if special_model:
        url = url2
    else:
        url = url1
        end_tokens += ',</tool_call>'

    json_data = {
        'user_question': user_question,
        'model_output': history_text,
        'end_tokens': end_tokens,
        'n': n,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'use_tools': json.dumps(tools) if tools else '{}',
    }

    response = requests.post(url, json=json_data, headers=headers)
    # breakpoint()
    return response.json()['response']

