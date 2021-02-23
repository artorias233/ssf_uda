# coding=utf-8
from cffex_uda import  get_uda
from cffex_uda.utils import tokenization, raw_data_utils
from  conf import *
import pandas as pd
import json
import re
import os

labels_multitags = [["物流","公告速递","财经早报","非银金融","建筑材料","宏观政策","A股IPO动态","医药","机械设备","纺织服装","A股入摩进展","比特币","none"],["财经早餐","A股策略","中国央行动态","电力","贸易","化工","快递物流","智能制造","原油","none"],["研判优选","物联网","轻工制造","严重事故","民企","午盘点评","口罩","投融资","none"],["电子","工业互联网","焦点评论","商务部","none"],["时政要闻","农产品","工信部","none"]]


def to_csv(data, path_base, file_name):
    data.to_csv(os.path.join(path_base, file_name), sep=',', index=False, header=1,encoding='utf-8',line_terminator='\n')

def read_csv(path_base, file_name):
    return pd.read_csv(os.path.join(path_base, file_name), delimiter=',', error_bad_lines=False,lineterminator='\n')

def get_label(label_list, prob_list):
    max_p = -1
    label_p = -1
    for i in range(len(prob_list['probabilities'])):
        if prob_list['probabilities'][i] > max_p:
            max_p = prob_list['probabilities'][i]
            label_p = i
    return label_list[label_p],max_p

def print_list(l):
    print(str(l).decode("string_escape"))

def get_multitags_result(content, model_list, labels_map):
    label_p = []
    weight_p = []
    for i in range(len(model_list)):
        p = model_list[i].predict(content)
        label_tmp, weight_tmp = get_label(labels_map[i], p)
        if label_tmp != "none":
            label_p.append(label_tmp)
            weight_p.append(weight_tmp)
    result = {"labels": label_p, "weight": weight_p}
    return result

def get_multitags_uda(conf_list):
    uda_list = []
    labels_map = []
    for conf in conf_list:
        tokenizer = tokenization.FullTokenizer(vocab_file=conf['vocab_file'], do_lower_case=True)
        processor = raw_data_utils.CFFEXProcessor(conf["labels"])
        uda = get_uda.Get_UDA(conf, tokenizer, processor)
        uda_list.append(uda)
        labels_map.append(conf["labels"])
    return uda_list, labels_map

def multitags_eval():
    conf_list = [conf_tag0, conf_tag1,conf_tag2,conf_tag3,conf_tag4]
    model_list, labels_map = get_multitags_uda(conf_list)
    eval = read_csv('cffex_uda/data/CFFEX/multitags', 'multitags_uda.csv')
    for i in range(len(eval)):
        content = eval.iloc[i]['content']
        labels_uda = json.loads(eval.iloc[i]['predict'])
        result = get_multitags_result(content, model_list, labels_map)
        print(print_list(result))
        print(str(labels_uda).decode('unicode_escape'))
    for model in model_list:
        model.kill()

multitags_eval()