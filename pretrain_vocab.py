# coding: UTF-8
import argparse
import os
import numpy as np
import pickle as pkl
import jieba

from utils import build_vocab, MAX_VOCAB_SIZE

# 创建解析器
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data/THUCNews', help='the dataset path')
args = parser.parse_args()

if __name__ == "__main__":
    '''提取预训练词向量'''
    use_word = False
    # 下面的目录、文件名按需更改。
    dataset = args.dataset
    train_dir = os.path.join(dataset, "data/train.txt")
    vocab_dir = os.path.join(dataset, "data/vocab.pkl")
    pretrain_dir = os.path.join(dataset, "data/sgns.sogou.char")
    emb_dim = 300
    filename_trimmed_dir = os.path.join(dataset, "data/vocab.embedding.sougou")
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        if use_word:
            tokenizer = lambda x: list(jieba.cut(x))  # 以词为单位构建词表
        else:
            tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
