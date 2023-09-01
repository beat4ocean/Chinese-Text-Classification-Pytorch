import argparse
import jieba
import pickle as pkl
import os
import torch
import numpy as np
from importlib import import_module

from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

key_map = {}


class Predictor:
    def __init__(self, model_name, dataset, embedding, use_word=False):
        self.use_word = use_word
        self.tokenizer = self.get_tokenizer()
        self.x = import_module('models.' + model_name)
        self.config = self.x.Config(dataset, embedding)
        self.vocab = self.load_vocab()
        self.pad_size = self.config.pad_size
        self.model = self.load_model()

        self.key_map = self.get_key_map(dataset)

    def get_tokenizer(self):
        if self.use_word:
            return lambda x: list(jieba.cut(x))
        else:
            return lambda x: [y for y in x]

    def load_vocab(self):
        with open(self.config.vocab_path, 'rb') as f:
            vocab = pkl.load(f)
        return vocab

    def load_model(self):
        model = self.x.Model(self.config).to('cpu')
        model.load_state_dict(torch.load(self.config.save_path, map_location='cpu'))
        return model

    def get_key_map(self, dataset):
        key_map = {}
        with open(os.path.join(dataset, 'data/class.txt'), 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                class_name = line.strip()
                key_map[i] = class_name
        return key_map

    def preprocess_texts(self, texts):
        words_lines = []
        seq_lens = []
        for text in texts:
            words_line = []
            tokens = self.tokenizer(text)
            seq_len = len(tokens)
            if self.pad_size:
                if len(tokens) < self.pad_size:
                    tokens.extend(['<PAD>'] * (self.pad_size - len(tokens)))
                else:
                    tokens = tokens[:self.pad_size]
                    seq_len = self.pad_size
            # Convert words to ids
            for token in tokens:
                words_line.append(self.vocab.get(token, self.vocab.get('<UNK>')))
            words_lines.append(words_line)
            seq_lens.append(seq_len)
        return torch.LongTensor(words_lines), torch.LongTensor(seq_lens)

    def predict_text_with_all_labels(self, query):
        query = [query]
        data = self.preprocess_texts(query)
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=0)
            labels = self.key_map.values()
            probabilities = [prob.item() for prob in probabilities]
        return list(zip(labels, probabilities))


# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--model', type=str, default='TextRCNN', help='the model to be used')
parser.add_argument('--dataset', type=str, default='data/Comments', help='the dataset path')
# 解析参数
args = parser.parse_args()

model = args.model
dataset = args.dataset

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

    # 输入验证
    if text is None or text.strip() == "":
        return jsonify({"error": "Invalid input"})

    result_list = pred.predict_text_with_all_labels(text)
    print("text:", text, "\tpredict", result_list)

    # 构建返回的 JSON 数据
    return jsonify(result_list)


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
