from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

load_file_path = '/Users/sniper/codes/mainTest/cache/aime_res_23.json'
target_save_path = './output_data/aime_01_save1.json'
current_index = 0

if os.path.exists(target_save_path):
    with open(target_save_path, 'r') as file:
        saved_problems = json.load(file)
        current_index = len(saved_problems)
else:
    saved_problems = []

with open(load_file_path, 'r') as file:
    problems = json.load(file)


@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET' and request.headers.get('Content-Type') == 'application/json':
        global current_index
        print(current_index)
        data = problems[current_index].copy()
        data['id'] = current_index
        data['solution'] = data.get('source_text', '')
        data['file_idx'] = data.get('ground_truth', '')
        return jsonify(data)

    print('init at ', current_index)
    initial_data = problems[current_index].copy()
    initial_data['id'] = current_index
    initial_data['solution'] = initial_data.get('source_text', '')
    initial_data['file_idx'] = initial_data.get('ground_truth', '')
    # print(problems[current_index]['id'])
    return render_template('index.html', data=initial_data, current_index=current_index)


@app.route('/chosen', methods=['POST'])
def chosen():
    if request.method == 'POST':
        global current_index
        data = request.get_json()
        choice = data.get('choice')
        solution = data.get('solution', '')
        file_idx = data.get('file_idx', '')
        problem_id = data.get('id')

        new_data = {
            'id': problem_id,
            'problem': problems[current_index]['problem'],
            'ground_truth': file_idx,
            'solution': solution,
            'choice': choice,
            "file_idx": problems[current_index]['id']
        }

        saved_problems.append(new_data)
        with open(target_save_path, 'w', encoding='utf-8') as file:
            json.dump(saved_problems, file, indent=4, ensure_ascii=False)

        current_index = (current_index + 1) % len(problems)

        return jsonify({"status": "success"})


@app.route('/prev', methods=['POST'])
def prev():
    if request.method == 'POST':
        global current_index
        current_index = (current_index - 1) if current_index > 0 else len(problems) - 1
        return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(host='localhost', port=40010, debug=True)