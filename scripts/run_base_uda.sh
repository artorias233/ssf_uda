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
FILE=CFFEX
MAX_SEQ_LENGTH=256
SUP_SIZE=60
SUB_FILE=illegal
TAG=supsize60
SUP_TRAIN_FILE="train_0.8.csv"
SUP_DEV_FILE="eval_0.2.csv"
UNSUP_FILE="unsup.csv"
LABELS=0,1,2,3,4
#TASK_NAME=cffextheme

python ssf_uda/main.py \
  --use_tpu=False \
  --do_train=True \
  --do_eval=True \
  --labels=${LABELS} \
  --raw_data_dir=ssf_uda/data/$FILE/$SUB_FILE \
  --sup_train_file=${SUP_TRAIN_FILE} \
  --sup_dev_file=${SUP_DEV_FILE} \
  --unsup_file=${UNSUP_FILE} \
  --sup_train_data_dir=ssf_uda/data/proc_data/$FILE/$SUB_FILE/train_${SUP_SIZE}_${MAX_SEQ_LENGTH} \
  --unsup_data_dir=ssf_uda/data/proc_data/$FILE/$SUB_FILE/unsup_${MAX_SEQ_LENGTH} \
  --eval_data_dir=ssf_uda/data/proc_data/$FILE/$SUB_FILE/dev_${MAX_SEQ_LENGTH} \
  --bert_config_file=ssf_uda/pretrained_models/albert_base/albert_config.json \
  --vocab_file=ssf_uda/pretrained_models/albert_base/vocab_chinese.txt \
  --init_checkpoint=ssf_uda/pretrained_models/albert_base/model.ckpt-best \
  --model_dir=ssf_uda/ckpt/base_uda/${SUB_FILE}_${SUP_SIZE}_${MAX_SEQ_LENGTH}_${TAG} \
  --num_train_steps=5000 \
  --learning_rate=2e-05 \
  --num_warmup_steps=1000 \
  --train_batch_size=8 \
  --unsup_ratio=3 \
  --tsa=linear_schedule \
  --aug_ops=tf_idf-0.2 \
  --aug_copy=1 \
  --uda_coeff=1 \
  --train_batch_size=8 \
  --max_seq_length=${MAX_SEQ_LENGTH} \
  $@
