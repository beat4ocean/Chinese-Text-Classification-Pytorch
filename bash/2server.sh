#!/bin/bash

#python server.py \
#       --model TextRCNN \
#       --dataset data/Comments \
#       --use_word 0 \
#       --port 5432

python server_fasttext.py \
       --model FastText \
       --dataset data/Comments \
       --use_word 0 \
       --port 5432