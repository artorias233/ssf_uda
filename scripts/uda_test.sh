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
echo "deleting saved test model"
rm -r ckpt/base_uda/base_uda_test
echo "model deleted"

MAX_SEQ_LENGTH=256
python main.py \
  --use_tpu=False \
  --do_train=True \
  --do_eval=True \
  --sup_train_data_dir=data/proc_data/THUCNews/train_test \
  --unsup_data_dir=data/proc_data/THUCNews/unsup_test \
  --eval_data_dir=data/proc_data/THUCNews/dev_test \
  --bert_config_file=pretrained_models/albert_base/albert_config.json \
  --vocab_file=pretrained_models/albert_base/vocab_chinese.txt \
  --init_checkpoint=pretrained_models/albert_base/model.ckpt-best \
  --task_name=thuc \
  --model_dir=ckpt/base_uda/base_uda_test \
  --num_train_steps=100 \
  --learning_rate=2e-05 \
  --num_warmup_steps=10 \
  --unsup_ratio=3 \
  --tsa=linear_schedule \
  --train_batch_size=8 \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --aug_ops=tf_idf-0.2 \
  --aug_copy=1 \
  --uda_coeff=1 \
  $@
