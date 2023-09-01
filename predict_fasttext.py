import jieba
import pickle as pkl
import os
import torch
import numpy as np
from importlib import import_module

key_map = {}


class Predictor:

    def __init__(self, model_name, dataset, embedding, use_word=False):
        if use_word:
            # self.tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
            self.tokenizer = lambda x: list(jieba.cut(x))
        else:
            self.tokenizer = lambda x: [y for y in x]
        self.x = import_module('models.' + model_name)
        self.config = self.x.Config(dataset, embedding)
        self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        self.pad_size = self.config.pad_size
        self.model = self.x.Model(self.config).to('cpu')
        self.model.load_state_dict(torch.load(self.config.save_path, map_location='cpu'))

        with open(os.path.join(dataset, 'data/class.txt'), 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                class_name = line.strip()
                key_map[i] = class_name

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
        return key_map[int(num)]

    def predict_list(self, queries):
        data = self.preprocess_texts(queries)
        with torch.no_grad():
            outputs = self.model(data)
            nums = torch.argmax(outputs, dim=1)
            preds = [key_map[int(num)] for num in list(np.array(nums))]
        return preds


if __name__ == "__main__":
    model = 'FastText'
    # model = 'TextRCNN'
    # dataset = 'data/Comments'
    dataset = 'data/THUCNews'
    embedding = 'embedding_SougouNews.npz'
    use_word = False
    pred = Predictor(model, dataset, embedding, use_word)

    # 预测一条
    query = "学费太贵怎么办？"
    print(pred.predict_text(query))
    # 预测一个列表
    querys = ["学费太贵怎么办？", "金融怎么样"]
    print(pred.predict_list(querys))
