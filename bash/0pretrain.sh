#!/bin/bash

#python pretrain_vocab.py \
#       --dataset data/Comments \
#       --word_vector source/sgns.sogou.char \
#       --use_word 0

python pretrain_vocab.py \
       --dataset data/Comments \
       --word_vector source/sgns.merge.char \
       --use_word 0