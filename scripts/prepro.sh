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
bert_vocab_file=ssf_uda/pretrained_models/albert_base/vocab_chinese.txt

FILE=CFFEX
SUB_FILE=illegal
SUP_TRAIN_FILE="train_0.8.csv"
SUP_DEV_FILE="eval_0.2.csv"
UNSUP_FILE="unsup.csv"
LABELS=0,1,2,3,4
MAX_SEQ_LENGTH=256
MAX_SEG_NUM=20
SUP_SIZE=60

#FILE=CFFEX
#SUB_FILE=multitags/data_tag_4
#SUP_TRAIN_FILE="train_0.8.csv"
#SUP_DEV_FILE="eval_0.2.csv"
#LABELS="时政要闻","农产品","工信部","none"
#MAX_SEQ_LENGTH=256
#MAX_SEG_NUM=20
#SUP_SIZE=-1

# Preprocess supervised training set
echo "处理有监督数据"
python ssf_uda/preprocess.py \
  --raw_data_dir=ssf_uda/data/$FILE/$SUB_FILE \
  --labels=${LABELS} \
  --trunc_keep_right=False\
  --sup_train_file=${SUP_TRAIN_FILE} \
  --sup_dev_file=${SUP_DEV_FILE} \
  --unsup_file=${UNSUP_FILE} \
  --output_base_dir=ssf_uda/data/proc_data/$FILE/$SUB_FILE/train_${SUP_SIZE}_${MAX_SEQ_LENGTH} \
  --data_type=sup \
  --sub_set=train \
  --sup_size=$SUP_SIZE \
  --vocab_file=$bert_vocab_file \
  --max_seq_length=${MAX_SEQ_LENGTH} \
#  --task_name=$TASK_NAME \
#  $@
#
# Preprocess test set
echo "处理验证集"
python ssf_uda/preprocess.py \
  --raw_data_dir=ssf_uda/data/$FILE/$SUB_FILE \
  --labels=${LABELS} \
  --trunc_keep_right=False\
  --sup_train_file=${SUP_TRAIN_FILE} \
  --sup_dev_file=${SUP_DEV_FILE} \
  --unsup_file=${UNSUP_FILE} \
  --output_base_dir=ssf_uda/data/proc_data/$FILE/$SUB_FILE/dev_${MAX_SEQ_LENGTH} \
  --data_type=sup \
  --sub_set=dev \
  --vocab_file=$bert_vocab_file \
  --max_seq_length=${MAX_SEQ_LENGTH} \
#  --task_name=$TASK_NAME \
#  $@


## Preprocess unlabeled set
#echo "处理无监督数据"
#python ssf_uda/preprocess.py \
#  --raw_data_dir=ssf_uda/data/$FILE/$SUB_FILE \
#  --labels=${LABELS} \
#  --trunc_keep_right=False\
#  --sup_train_file=${SUP_TRAIN_FILE} \
#  --sup_dev_file=${SUP_DEV_FILE} \
#  --unsup_file=${UNSUP_FILE} \
#  --output_base_dir=ssf_uda/data/proc_data/$FILE/$SUB_FILE/unsup_${MAX_SEQ_LENGTH} \
#  --back_translation_dir=ssf_uda/data/back_translation/$FILE_back_trans \
#  --data_type=unsup \
#  --sub_set=unsup_in \
#  --aug_ops=tf_idf-0.2 \
#  --aug_copy_num=0 \
#  --vocab_file=$bert_vocab_file \
#  --max_seq_length=${MAX_SEQ_LENGTH} \
##  --task_name=$TASK_NAME \
#  $@
