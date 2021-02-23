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
SUP_SIZE=-1
SUB_FILE=multitags/data_tag_0
#SUP_TRAIN_FILE="train_0.8.csv"
#SUP_DEV_FILE="eval_0.2.csv"
#LABELS="物流","公告速递","财经早报","非银金融","建筑材料","宏观政策","A股IPO动态","医药","机械设备","纺织服装","A股入摩进展","比特币","none"
#
#SUB_FILE=multitags/data_tag_1
#SUP_TRAIN_FILE="train_0.8.csv"
#SUP_DEV_FILE="eval_0.2.csv"
#LABELS="财经早餐","A股策略","中国央行动态","电力","贸易","化工","快递物流","智能制造","原油","none"
#
#SUB_FILE=multitags/data_tag_2
#SUP_TRAIN_FILE="train_0.8.csv"
#SUP_DEV_FILE="eval_0.2.csv"
#LABELS="研判优选","物联网","轻工制造","严重事故","民企","午盘点评","口罩","投融资","none"
#
#SUB_FILE=multitags/data_tag_3
#SUP_TRAIN_FILE="train_0.8.csv"
#SUP_DEV_FILE="eval_0.2.csv"
#LABELS="电子","工业互联网","焦点评论","商务部","none"
#
SUB_FILE=multitags/data_tag_4
SUP_TRAIN_FILE="train_0.8.csv"
SUP_DEV_FILE="eval_0.2.csv"
LABELS="时政要闻","农产品","工信部","none"

python cffex_uda/main.py \
  --use_tpu=False \
  --do_train=True  \
  --do_eval=True \
  --labels=${LABELS}\
  --raw_data_dir=cffex_uda/data/$FILE/$SUB_FILE \
  --sup_train_file=${SUP_TRAIN_FILE} \
  --sup_dev_file=${SUP_DEV_FILE} \
  --sup_train_data_dir=cffex_uda/data/proc_data/$FILE/$SUB_FILE/train_${SUP_SIZE}_${MAX_SEQ_LENGTH} \
  --eval_data_dir=cffex_uda/data/proc_data/$FILE/$SUB_FILE/dev_${MAX_SEQ_LENGTH} \
  --bert_config_file=cffex_uda/pretrained_models/albert_base/albert_config.json \
  --vocab_file=cffex_uda/pretrained_models/albert_base/vocab_chinese.txt \
  --init_checkpoint=cffex_uda/pretrained_models/albert_base/model.ckpt-best \
  --model_dir=cffex_uda/ckpt/base/${SUB_FILE}_${SUP_SIZE}_${MAX_SEQ_LENGTH}/bese_model \
  --num_train_steps=5000 \
  --learning_rate=3e-05 \
  --num_warmup_steps=1000 \
  --max_seq_length=${MAX_SEQ_LENGTH} \
  #  --unsup_file=${UNSUP_FILE} \
#  --unsup_data_dir=cffex_uda/data/proc_data/$FILE/$SUB_FILE/unsup_${MAX_SEQ_LENGTH} \
  $@
