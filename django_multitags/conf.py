# coding=utf-8


# labels=[0,1,2,3,4]
# sub_file="illegal"
# sup_size=-1
# model_type = "base_uda"
# tag = "supsizeall"
# model_name = "model.ckpt-4250"
# max_seq_len=256

# vocab_file="cffex_uda/pretrained_models/albert_base/vocab_chinese.txt"

conf_tag0 = {
"labels":["物流","公告速递","财经早报","非银金融","建筑材料","宏观政策","A股IPO动态","医药","机械设备","纺织服装","A股入摩进展","比特币","none"],
"unsup_ratio": 0,
"aug_ops": "tf_idf-0.2",
"aug_copy": 1,
"uda_coeff": 1,
"tsa": "linear_schedule",
"uda_softmax_temp": -1,
"uda_confidence_thresh": -1,
"vocab_file": "cffex_uda/pretrained_models/albert_base/vocab_chinese.txt",
"init_checkpoint": "cffex_uda/pretrained_models/albert_base/model.ckpt-best",
'bert_config_file':'cffex_uda/pretrained_models/albert_base/albert_config.json',
"model_dir": "cffex_uda/ckpt/multitags/tag_0/best_model/model.ckpt-1500",
"use_one_hot_embeddings": True,
"max_seq_length": 256,
"model_dropout": -1,
"num_train_steps": 5000,
"learning_rate": 2e-5,
"num_warmup_steps": 1000,
"clip_norm": 1.0,
"train_batch_size": 32,
"eval_batch_size": 8,
"iterations_per_loop": 200,
}

conf_tag1 = {
"labels":["财经早餐","A股策略","中国央行动态","电力","贸易","化工","快递物流","智能制造","原油","none"],
"unsup_ratio": 0,
"aug_ops": "tf_idf-0.2",
"aug_copy": 1,
"uda_coeff": 1,
"tsa": "linear_schedule",
"uda_softmax_temp": -1,
"uda_confidence_thresh": -1,
"vocab_file": "cffex_uda/pretrained_models/albert_base/vocab_chinese.txt",
"init_checkpoint": "cffex_uda/pretrained_models/albert_base/model.ckpt-best",
'bert_config_file':'cffex_uda/pretrained_models/albert_base/albert_config.json',
"model_dir": "cffex_uda/ckpt/multitags/tag_1/best_model/model.ckpt-1000",
"use_one_hot_embeddings": True,
"max_seq_length": 256,
"model_dropout": -1,
"num_train_steps": 5000,
"learning_rate": 2e-5,
"num_warmup_steps": 1000,
"clip_norm": 1.0,
"train_batch_size": 32,
"eval_batch_size": 8,
"iterations_per_loop": 200,
}

conf_tag2 = {
"labels":["研判优选","物联网","轻工制造","严重事故","民企","午盘点评","口罩","投融资","none"],
"unsup_ratio": 0,
"aug_ops": "tf_idf-0.2",
"aug_copy": 1,
"uda_coeff": 1,
"tsa": "linear_schedule",
"uda_softmax_temp": -1,
"uda_confidence_thresh": -1,
"vocab_file": "cffex_uda/pretrained_models/albert_base/vocab_chinese.txt",
"init_checkpoint": "cffex_uda/pretrained_models/albert_base/model.ckpt-best",
'bert_config_file':'cffex_uda/pretrained_models/albert_base/albert_config.json',
"model_dir": "cffex_uda/ckpt/multitags/tag_2/best_model/model.ckpt-4000",
"use_one_hot_embeddings": True,
"max_seq_length": 256,
"model_dropout": -1,
"num_train_steps": 5000,
"learning_rate": 2e-5,
"num_warmup_steps": 1000,
"clip_norm": 1.0,
"train_batch_size": 32,
"eval_batch_size": 8,
"iterations_per_loop": 200,
}

conf_tag3 = {
"labels":["电子","工业互联网","焦点评论","商务部","none"],
"unsup_ratio": 0,
"aug_ops": "tf_idf-0.2",
"aug_copy": 1,
"uda_coeff": 1,
"tsa": "linear_schedule",
"uda_softmax_temp": -1,
"uda_confidence_thresh": -1,
"vocab_file": "cffex_uda/pretrained_models/albert_base/vocab_chinese.txt",
"init_checkpoint": "cffex_uda/pretrained_models/albert_base/model.ckpt-best",
'bert_config_file':'cffex_uda/pretrained_models/albert_base/albert_config.json',
"model_dir": "cffex_uda/ckpt/multitags/tag_3/best_model/model.ckpt-750",
"use_one_hot_embeddings": True,
"max_seq_length": 256,
"model_dropout": -1,
"num_train_steps": 5000,
"learning_rate": 2e-5,
"num_warmup_steps": 1000,
"clip_norm": 1.0,
"train_batch_size": 32,
"eval_batch_size": 8,
"iterations_per_loop": 200,
}

conf_tag4 = {
"labels":["时政要闻","农产品","工信部","none"],
"unsup_ratio": 0,
"aug_ops": "tf_idf-0.2",
"aug_copy": 1,
"uda_coeff": 1,
"tsa": "linear_schedule",
"uda_softmax_temp": -1,
"uda_confidence_thresh": -1,
"vocab_file": "cffex_uda/pretrained_models/albert_base/vocab_chinese.txt",
"init_checkpoint": "cffex_uda/pretrained_models/albert_base/model.ckpt-best",
'bert_config_file':'cffex_uda/pretrained_models/albert_base/albert_config.json',
"model_dir": "cffex_uda/ckpt/multitags/tag_4/best_model/model.ckpt-1250",
"use_one_hot_embeddings": True,
"max_seq_length": 256,
"model_dropout": -1,
"num_train_steps": 5000,
"learning_rate": 2e-5,
"num_warmup_steps": 1000,
"clip_norm": 1.0,
"train_batch_size": 32,
"eval_batch_size": 8,
"iterations_per_loop": 200,
}