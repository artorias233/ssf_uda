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
LABELS="A","B"
MAX_SEQ_LENGTH=256
SUP_SIZE=-1
SEGMENT_LEN=200
SEGMENT_OVERLAP=50


# Preprocess supervised training set
echo "csv长文本分段"
python utils/csv_process.py \
  --process_for_split=True \
  --raw_data_dir=cffex_uda/data/$FILE/$SUB_FILE \
  --sup_train_file=${SUP_TRAIN_FILE} \
  --sup_dev_file=${SUP_DEV_FILE} \
  --unsup_file=${UNSUP_FILE} \
  --segment_len=${SEGMENT_LEN}\
  --segment_overlap=${SEGMENT_OVERLAP}

# Preprocess supervised training set
echo "csv长文本词向量编码"
python utils/csv_process.py \
  --process_for_embedding=True \
  --raw_data_dir=cffex_uda/data/$FILE/$SUB_FILE \
  --sup_train_file=${SUP_TRAIN_FILE} \
  --sup_dev_file=${SUP_DEV_FILE} \
  --unsup_file=${UNSUP_FILE} \
  --labels=${LABELS} \
  --vocab_file=cffex_uda/pretrained_models/albert_base/vocab_chinese.txt \
  --model_dir=cffex_uda/ckpt/base_uda/${SUB_FILE}_${SUP_SIZE}_${MAX_SEQ_LENGTH} \
  --max_seq_len=${MAX_SEQ_LENGTH} \
  --segment_len=${SEGMENT_LEN} \
  --segment_overlap=${SEGMENT_OVERLAP}