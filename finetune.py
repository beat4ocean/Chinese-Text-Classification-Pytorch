# coding: UTF-8
import os.path
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--dataset', required=True, type=str, help='choose a dataset')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--use_word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = args.dataset

    # 设置词向量文件名
    embedding = 'vocab.embedding.npz'

    # 根据参数选择词向量类型
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model

    # 导入模型和工具函数
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif

        embedding = 'random'

    else:  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # 加载数据集并生成迭代器
    vocab, train_data, dev_data, test_data = build_dataset(config, args.use_word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 训练模型
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
