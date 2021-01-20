# coding=utf-8
import tensorflow as tf

labels=[0,1,2,3,4]
sub_file="illegal"
sup_size=-1
model_type = "base_uda"
tag = "supsizeall"
model_name = "model.ckpt-4250"
max_seq_len=256

vocab_file="cffex_uda/pretrained_models/albert_base/vocab_chinese.txt"

conf = {
"labels":labels,
"unsup_ratio": 0,
"aug_ops": "tf_idf-0.2",
"aug_copy": 1,
"uda_coeff": 1,
"tsa": "linear_schedule",
"uda_softmax_temp": -1,
"uda_confidence_thresh": -1,
"vocab_file": vocab_file,
"init_checkpoint": "cffex_uda/pretrained_models/albert_base/model.ckpt-best",
'bert_config_file':'cffex_uda/pretrained_models/albert_base/albert_config.json',
"model_dir": "cffex_uda/ckpt/{}/{}_{}_{}_{}/best_model/{}".format(model_type,sub_file,sup_size,max_seq_len,tag,model_name),
"use_one_hot_embeddings": True,
"max_seq_length": max_seq_len,
"model_dropout": -1,
"num_train_steps": 5000,
"learning_rate": 2e-5,
"num_warmup_steps": 1000,
"clip_norm": 1.0,
"train_batch_size": 32,
"eval_batch_size": 8,
"iterations_per_loop": 200,
}