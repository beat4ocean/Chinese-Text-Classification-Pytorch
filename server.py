import argparse
import jieba
import pickle as pkl
import os
import torch
import torch.nn.functional as F
from importlib import import_module

from flask import Flask, request, render_template, jsonify

key_map = {}

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--model', type=str, default='TextRCNN', help='the model to be used')
parser.add_argument('--dataset', type=str, default='data/Comments', help='the dataset path')
parser.add_argument('--use_word', default=0, type=int, help='1 for word, 0 for char')
parser.add_argument('--port', type=int, default=5000, help='the server port')
# 解析参数
args = parser.parse_args()

model = args.model
dataset = args.dataset
use_word = bool(args.use_word)
port = args.port
embedding = 'vocab.embedding.npz'


class Predictor:
    def __init__(self, model_name, dataset, embedding, use_word):
        self.use_word = use_word
        self.tokenizer = self._tokenizer_word() if self.use_word else self._tokenizer_char()
        self.x = import_module('models.' + model_name)
        self.config = self.x.Config(dataset, embedding)
        self.vocab = self.load_vocab()
        self.pad_size = self.config.pad_size
        self.model = self.load_model()
        self.key_map = self._get_key_map()

    def _get_key_map(self):
        with open(os.path.join(dataset, 'data/class.txt'), 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                class_name = line.strip()
                key_map[i] = class_name
        return key_map

    # 使用字符级别的tokenizer
    def _tokenizer_char(self):
        def tokenizer(text):
            return [char for char in text]

        return tokenizer

    # 使用单词级别的tokenizer
    def _tokenizer_word(self):
        def tokenizer(text):
            # return text.strip().split()
            return list(jieba.cut(text))

        return tokenizer

    def load_vocab(self):
        with open(self.config.vocab_path, 'rb') as f:
            vocab = pkl.load(f)
        return vocab

    # def load_model(self):
    #     model = self.x.Model(self.config).to('cpu')
    #     model.load_state_dict(torch.load(self.config.save_path, map_location='cpu'))
    #     return model

    def load_model(self):
        model = self.x.Model(self.config)
        model.load_state_dict(torch.load(self.config.save_path))
        return model

    def get_key_map(self, dataset):
        key_map = {}
        with open(os.path.join(dataset, 'data/class.txt'), 'r', encoding='utf-8') as file:
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
            # probabilities = torch.softmax(outputs, dim=0) # 在指定维度上计算 softmax，而 F.softmax(outputs) 则是在默认维度上计算 softmax。
            # probabilities = F.softmax(outputs)  # 算输出张量的 softmax 函数，将输出的每个元素转换为表示概率的值，确保所有概率相加等于1。适用于多分类问题。
            probabilities = F.sigmoid(outputs)  # 获取张量中最大值的索引，返回张量中最大值元素的索引。常用于多分类问题中确定最可能的类别。
            labels = self.key_map.values()

            probabilities = probabilities.tolist()

        return list(zip(labels, probabilities))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/text_predict', methods=['POST'])
def text_predict():
    # 获取 POST 请求的数据
    data = request.get_json()  # 获取 POST 请求的 JSON 数据
    text = data.get('text')  # 获取 content 字段的值

    # 输入验证
    if not text or text.strip() == "":
        return jsonify({"error": "Invalid input"})

    result_list = pred.predict_text_with_all_labels(text)
    print("text:", text, "\tpredict:", result_list)

    # 构建返回的 JSON 数据
    return jsonify(result_list)


if __name__ == "__main__":
    pred = Predictor(model, dataset, embedding, use_word)

    app.run(host='0.0.0.0', port=port)
