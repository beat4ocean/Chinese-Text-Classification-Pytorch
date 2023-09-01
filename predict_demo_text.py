import torch
from models import FastText
import pickle as pkl
from importlib import import_module
from utils_fasttext import build_vocab, build_iterator
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

# 文本预处理（根据需要进行padding等操作）
pad_size = config.pad_size

tokenizer = lambda x: x.split(' ')
words = [vocab[word] if word in vocab else vocab[UNK] for word in tokenizer(text)]
input_ids = words[:pad_size] + [vocab[PAD]] * max(0, pad_size - len(words))
input_ids = torch.tensor([input_ids]).unsqueeze(0).to(config.device)

# 执行预测
with torch.no_grad():
    model.eval()
    output = model(input_ids)

# 获取预测概率
probabilities = torch.softmax(output, dim=1)[0].tolist()
label_probabilities = {config.class_list[i]: probability for i, probability in enumerate(probabilities)}

# 打印预测结果
print("预测结果：", json.dumps(label_probabilities))
