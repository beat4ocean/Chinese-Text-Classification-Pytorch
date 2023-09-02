# coding: UTF-8
import argparse
import os
import numpy as np
import pickle as pkl


def load_pretrained_embeddings(dataset, emb_dim):
    # 加载数据集和预训练词向量的路径
    vocab_dir = os.path.join(dataset, "data/vocab.pkl")
    comp_vocab_dir = os.path.join(dataset, "data/vocab.embedding")

    # 加载词汇表
    with open(vocab_dir, 'rb') as f:
        word_to_id = pkl.load(f)

    # 初始化嵌入矩阵
    embeddings = np.random.rand(len(word_to_id), emb_dim)

    # 加载预训练词向量
    with open(word_vector, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            lin = line.strip().split(" ")
            if lin[0] in word_to_id:
                idx = word_to_id[lin[0]]
                emb = [float(x) for x in lin[1:emb_dim + 1]]
                embeddings[idx] = np.asarray(emb, dtype='float32')

    # 保存压缩后的词向量
    np.savez_compressed(comp_vocab_dir, embeddings=embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/Comments', help='the dataset path')
    parser.add_argument('--word_vector', type=str, default='source/sgns.sogou.char', help='the dataset path')
    args = parser.parse_args()

    dataset = args.dataset
    word_vector = args.word_vector
    emb_dim = 300

    load_pretrained_embeddings(dataset, emb_dim)
