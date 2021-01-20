# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash
bert_vocab_file=cffex_uda/pretrained_models/albert_base/vocab_chinese.txt

# Changeable variables
FILE=CFFEX
SUB_FILE=theme
SUP_TRAIN_FILE="train_0.8.csv"
SUP_DEV_FILE="eval_0.2.csv"
UNSUP_FILE="unsup.csv"
# 注意标签顺序在训练与验证，预测时保持顺序一致
LABELS="A","B"
SUP_SIZE=-1
EPOCHS=50
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=8
MAX_SEQ_LENGTH=256

# Preprocess supervised training set
echo "运行LSTM"
python lstm_uda/main.py \
  --do_train=True \
  --do_eval=True \
  --epochs=${EPOCHS} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --eval_batch_size=${EVAL_BATCH_SIZE} \
  --labels=${LABELS} \
  --lstm_model_dir=cffex_uda/ckpt/lstm_uda/${SUB_FILE}_${SUP_SIZE}_${MAX_SEQ_LENGTH} \
  --raw_data_dir=cffex_uda/data/$FILE/$SUB_FILE \
  --sup_train_file=${SUP_TRAIN_FILE} \
  --sup_dev_file=${SUP_DEV_FILE} \
  --unsup_file=${UNSUP_FILE} \
