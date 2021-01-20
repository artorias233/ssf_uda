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

# **** download pretrained models ****
mkdir pretrained_models
# download bert base
cd pretrained_models
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip && rm uncased_L-12_H-768_A-12.zip
mv uncased_L-12_H-768_A-12 bert_base
cd ..

# download bert large ft
cd pretrained_models
wget https://storage.googleapis.com/uda_model/text/imdb_bert_ft.zip
unzip imdb_bert_ft.zip && rm imdb_bert_ft.zip
cd ..

# **** download back translated data ****
mkdir -p data/back_translation
cd data/back_translation
wget https://storage.googleapis.com/uda_model/text/imdb_back_trans.zip
unzip imdb_back_trans.zip && rm imdb_back_trans.zip
cd ../../

# **** download IMDB data and convert it to csv files ****
mkdir data/IMDB_raw
cd data/IMDB_raw
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xzvf aclImdb_v1.tar.gz && rm aclImdb_v1.tar.gz
cd ../..
python utils/imdb_format.py --raw_data_dir=data/IMDB_raw/aclImdb --train_id_path=data/IMDB_raw/train_id_list.txt --output_dir=data/IMDB_raw/csv

curl 'http://127.0.0.1:8000/predict/' -X POST -H "Content-Type:application/json"  -d '{"uuid": "4596707759","content": "2月CPI公布：同比上涨1.5%  看看你买啥贵了，买啥便宜了？】国家统计局今天公布的数据显示：2月全国居民消费价格指数CPI同比上涨1.5%，涨幅比上月回落0.2个百分点。其中，食品价格同比涨幅回落是影响CPI走势的核心因素。从单项食品看，羊肉、牛肉价格涨幅居前，鲜果和鲜菜价格小幅波动，而鸡蛋、猪肉和水产品价格却处在下行通道， 涨跌互抵，使得食品价格平稳运行 。在非食品中，医疗保健、教育文化和娱乐、居住价格依旧领涨，但涨幅平稳。国研中心宏观经济研究部研究员张立群表示：总体来看，价格还是总体平稳，或者是一个走低基本趋势。（记者 柴华 武立丰 张宏斌"}'