#!/bin/bash

#python finetune.py \
#       --model TextRCNN \
#       --dataset data/Comments \
#       --embedding pre_trained \
#       --use_word 0

python finetune.py \
       --model TextRCNN \
       --dataset data/Comments \
       --embedding random \
       --use_word 0