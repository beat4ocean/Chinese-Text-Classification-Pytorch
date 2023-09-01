# coding: UTF-8
import argparse
import os
import numpy as np
import pickle as pkl

# 创建解析器
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data/THUCNews', help='the dataset path')
args = parser.parse_args()

if __name__ == "__main__":
    '''提取预训练词向量'''
    dataset = args.dataset
    vocab_dir = os.path.join(dataset, "data/vocab.pkl")
    pretrain_dir = os.path.join(dataset, "data/sgns.sogou.char")
    emb_dim = 300
    filename_trimmed_dir = os.path.join(dataset, "data/vocab.embedding.sougou")
    word_to_id = pkl.load(open(vocab_dir, 'rb'))
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
