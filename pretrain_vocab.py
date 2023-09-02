# coding: UTF-8
import argparse
import os
import numpy as np
import pickle as pkl
import jieba

from utils import build_vocab, MAX_VOCAB_SIZE


def load_pretrained_embeddings(dataset, emb_dim, use_word):
    # 加载数据集和预训练词向量的路径
    train_dir = os.path.join(dataset, "data/train.txt")
    vocab_dir = os.path.join(dataset, "data/vocab.pkl")
    comp_vocab_dir = os.path.join(dataset, "data/vocab.embedding")

    # 加载词汇表
    if os.path.exists(vocab_dir):
        # 如果词表已存在，则直接加载
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        if use_word:
            # 以词为单位构建词表
            tokenizer = lambda x: list(jieba.cut(x))
        else:
            # 以字为单位构建词表
            tokenizer = lambda x: [y for y in x]
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 保存词表
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    # 初始化词向量矩阵
    embeddings = np.random.rand(len(word_to_id), emb_dim)

    with open(word_vector, "r", encoding='UTF-8') as f:
        for i, line in enumerate(f.readlines()):
            # if i == 0:  # 若第一行是标题，则跳过
            #     continue
            lin = line.strip().split(" ")
            if lin[0] in word_to_id:
                idx = word_to_id[lin[0]]
                emb = [float(x) for x in lin[1:301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')

    # 保存压缩后的词向量矩阵
    np.savez_compressed(comp_vocab_dir, embeddings=embeddings)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/Comments', help='the dataset path')
    parser.add_argument('--word_vector', type=str, default='source/sgns.sogou.char', help='the dataset path')
    parser.add_argument('--use_word', default=0, type=int, help='1 for word, 0 for char')
    args = parser.parse_args()

    dataset = args.dataset
    use_word = bool(args.use_word)
    word_vector = args.word_vector

    emb_dim = 300

    load_pretrained_embeddings(dataset, emb_dim, use_word)
