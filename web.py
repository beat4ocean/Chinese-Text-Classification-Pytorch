import argparse
import os

from flask import Flask, request, render_template, jsonify

from predict import Predictor

app = Flask(__name__)

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--model', type=str, default='TextRCNN', help='the model to be used')
parser.add_argument('--dataset', type=str, default='data/Comments', help='the dataset path')
# 解析参数
args = parser.parse_args()

model = args.model
dataset = args.dataset

# 加载标签列表
key_map = {}

with open(os.path.join(dataset, 'data/class.txt'), 'r') as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        class_name = line.strip()
        key_map[i] = class_name


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/text_predict', methods=['POST'])
def text_predict():
    # 获取 POST 请求的数据
    data = request.get_json()  # 获取 POST 请求的 JSON 数据
    text = data.get('text')  # 获取 content 字段的值

    print("text:", text)

    # 输入验证
    if text is None or text.strip() == "":
        return jsonify({"error": "Invalid input"})

    result = pred.predict_text(text)

    print("result:", result)

    return jsonify({'result': result})


# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--model', type=str, default='TextRCNN', help='the model to be used')
parser.add_argument('--dataset', type=str, default='data/Comments', help='the dataset path')
# 解析参数
args = parser.parse_args()

if __name__ == "__main__":
    model = args.model
    dataset = args.dataset
    embedding = 'vocab.embedding.sougou.npz'
    use_word = False

    pred = Predictor(model, dataset, embedding, use_word)

    app.run(host='0.0.0.0', port=5001)
