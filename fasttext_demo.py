# import fasttext
#
# model = fasttext.load_model('data/Comments/saved_dict/model.bin')
#
# text = '加我微信2342ewf'
# predictions = model.predict(text)
#
# print(predictions)


import torch
from models import FastText
import pickle as pkl
from importlib import import_module
from utils_fasttext import build_vocab, build_iterator
from train_eval import predict, text_to_tensor
import json

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

model_path = 'FastText'
dataset = 'data/Comments'
embedding = 'random'

# 加载配置参数
config = FastText.Config(dataset=dataset, embedding=embedding)
# print(config.class_list) # ['advertisement', 'attract_traffic', 'competitor_guidance']

# 加载词表
vocab = pkl.load(open(config.vocab_path, 'rb'))
config.n_vocab = len(vocab)

# 构建模型
x = import_module('models.' + model_path)
model = x.Model(config).to(config.device)

# 加载模型参数
model.load_state_dict(torch.load(config.save_path))

# 设置待预测的文本
text = "一八九二四二二五七六八 深蓝交流"
tensor = text_to_tensor(text, vocab).to(config.device)  # 将tensor移至GPU

result = predict(config, model, tensor, test=False)
print(result)
