import argparse
import jieba
import pickle as pkl
import os
import torch
import numpy as np
from importlib import import_module

key_map = {}

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--model', type=str, default='TextRCNN', help='the model to be used')
parser.add_argument('--dataset', type=str, default='data/Comments', help='the dataset path')
parser.add_argument('--use_word', default=0, type=int, help='1 for word, 0 for char')
# 解析参数
args = parser.parse_args()

model = args.model
dataset = args.dataset
use_word = bool(args.use_word)
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
        with open(os.path.join(dataset, 'data/class.txt'), 'r') as file:
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

    def predict_text(self, query):
        query = [query]
        data = self.preprocess_texts(query)
        with torch.no_grad():
            outputs = self.model(data)
            num = torch.argmax(outputs)
        return self.key_map[int(num)]

    def predict_text_with_all_labels(self, query):
        query = [query]
        data = self.preprocess_texts(query)
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=0)
            labels = self.key_map.values()
            probabilities = [prob.item() for prob in probabilities]
        return list(zip(labels, probabilities))

    def predict_list(self, queries):
        data = self.preprocess_texts(queries)
        with torch.no_grad():
            outputs = self.model(data)
            nums = torch.argmax(outputs, dim=1)
            preds = [self.key_map[int(num)] for num in list(np.array(nums))]
        return preds


if __name__ == "__main__":
    pred = Predictor(model, dataset, embedding, use_word)

    # 预测一条
    query = "火凤凰租房"
    print(pred.predict_text(query))
    print(pred.predict_text_with_all_labels(query))

    # 预测一个列表
    querys = ["比亚迪路过", "240511480车友+"]
    print(pred.predict_list(querys))
