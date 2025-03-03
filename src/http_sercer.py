from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

load_file_path = '/Users/sniper/codes/mainTest/cache/aime_data.json'
load_file_path = '/Users/sniper/codes/mainTest/cache/aime_res_23.json'
load_file_path = '/Users/sniper/codes/macro-o1-local/init_datasets/qwq_aime_data.json'
target_save_path = './output_data/aime_01_save1.json'
target_save_path = './output_data/aime_23_save.json'
target_save_path = './output_data/qwq_res.json'
current_index = 0

if os.path.exists(target_save_path):
    with open(target_save_path, 'r') as file:
        saved_problems = json.load(file)
        current_index = len(saved_problems)
else:
    saved_problems = []
"""
    {
        "id": "train-00001-of-00005.json_601",
        "problem": "An inverted cone with base radius $12  \\mathrm{cm}$ and height $18  \\mathrm{cm}$ is full of water. The water is poured into a tall cylinder whose horizontal base has radius of $24  \\mathrm{cm}$. What is the height in centimeters of the water in the cylinder?\n$\\textbf{(A)} ~1.5 \\qquad\\textbf{(B)} ~3 \\qquad\\textbf{(C)} ~4 \\qquad\\textbf{(D)} ~4.5 \\qquad\\textbf{(E)} ~6$",
        "ground_truth": "1. **Calculate the volume of the water in the cone**:  \n   The formula for the volume $V$ of a cone is given by:\n   \\[\n   V = \\frac{1}{3} \\pi r^2 h\n   \\]\n   where $r$ is the radius of the base and $h$ is the height of the cone. For the given cone, $r = 12 \\text{ cm}$ and $h = 18 \\text{ cm}$. Plugging in these values, we get:\n   \\[\n   V = \\frac{1}{3} \\pi (12)^2 (18) = \\frac{1}{3} \\pi \\cdot 144 \\cdot 18 = 864 \\pi \\text{ cm}^3\n   \\]\n\n2. **Calculate the volume of the water in the cylinder**:  \n   The formula for the volume $V$ of a cylinder is:\n   \\[\n   V = \\pi r^2 h\n   \\]\n   where $r$ is the radius of the base and $h$ is the height of the water in the cylinder. For the given cylinder, $r = 24 \\text{ cm}$. Let $h$ be the height of the water in the cylinder. The volume of the water remains the same, so:\n   \\[\n   \\pi (24)^2 h = 864 \\pi\n   \\]\n   Simplifying, we find:\n   \\[\n   576 \\pi h = 864 \\pi\n   \\]\n   Dividing both sides by $576 \\pi$, we get:\n   \\[\n   h = \\frac{864 \\pi}{576 \\pi} = \\frac{864}{576} = \\frac{3}{2} = 1.5 \\text{ cm}\n   \\]\n\n3. **Conclusion**:  \n   The height of the water in the cylinder is $1.5$ cm. Therefore, the correct answer is $\\boxed{\\textbf{(A)} ~1.5}$.",
        "solution": "AIME####Z####\\textbf{(A)} ~1.5",
        "source_text": "\n\n让我们一步一步解决这个问题。我们有一个倒置的圆锥装满水，并将这些水倒入一个圆柱体，我们需要找到圆柱中水的高度。圆锥的底半径为12厘米，高度为18厘米，而圆柱的底半径为24厘米。\n\n首先，我需要回忆一下圆锥和圆柱的体积公式，因为水的体积在这两种形状中是相同的，因为我们只是将水从一个形状转移到另一个形状。\n\n圆锥的体积 \\( v_{\\text{cone}} \\) 由以下公式给出：\n\\[\nv_{\\text{cone}} = \\frac{1}{3} \\pi r_{\\text{cone}}^2 h_{\\text{cone}}\n\\]\n其中 \\( r_{\\text{cone}} \\) 是底面的半径，\\( h_{\\text{cone}} \\) 是高度。\n\n对于圆柱体，体积 \\( v_{\\text{cylinder}} \\) 是：\n\\[\nv_{\\text{cylinder}} = \\pi r_{\\text{cylinder}}^2 h_{\\text{cylinder}}\n\\]\n其中 \\( r_{\\text{cylinder}} \\) 是底面的半径，\\( h_{\\text{cylinder}} \\) 是我们试图找到的圆柱中水的高度。\n\n因为水从圆锥倒入圆柱，所以圆锥中的水量等于圆柱中的水量：\n\\[\nv_{\\text{cone}} = v_{\\text{cylinder}}\n\\]\n\n因此：\n\\[\n\\frac{1}{3} \\pi (12)^2 (18) = \\pi (24)^2 h_{\\text{cylinder}}\n\\]\n\n我们可以代入已知值：\n\\[\n\\frac{1}{3} \\pi (12^2) (18) = \\pi (24^2) h_{\\text{cylinder}}\n\\]\n即：\n\\[\n\\frac{1}{3} \\pi (144)(18) = \\pi (576) h_{\\text{cylinder}}\n\\]\n\n接下来，我们计算：\n\\[\n\\frac{1}{3} \\times 144 \\times 18 = 576 h_{\\text{cylinder}}\n\\]\n\\[\n864 = 576 h_{\\text{cylinder}}\n\\]\n\n接下来解决这个方程以找到 \\( h_{\\text{cylinder}} \\)：\n\\[\nh_{\\text{cylinder}} = \\frac{864}{576} = 1.5 \\text{ cm}\n\\]\n\n所以圆柱中水的高度应该是1.5 cm。",
        "target_text

"""
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