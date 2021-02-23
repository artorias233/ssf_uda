#-*- coding:utf-8 _*-
"""
@author:Yuefeng Lin
@file: view.py
@time: 2020/07/27
"""
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from django.views.decorators.csrf import csrf_exempt

sys.path.append('../cffex_uda')
sys.path.append('../cffex_uda/augmentation')
sys.path.append('../cffex_uda/albert')
sys.path.append('../cffex_uda/utils')

from django.http import JsonResponse, HttpResponse
from cffex_uda import  get_uda
from cffex_uda.utils import tokenization, raw_data_utils
from conf import *


# -------------------------------辅助函数---------------------------
# 根据模型配置列表初始化模型列表，模型列表中包含了多个模型，多标签预测一共有五个模型文件，分别是tag_0,1,2,3,4
def get_multitags_uda_model(conf_list):
    uda_list = []
    labels_map = []
    for conf in conf_list:
        tokenizer = tokenization.FullTokenizer(vocab_file=conf['vocab_file'], do_lower_case=True)
        processor = raw_data_utils.CFFEXProcessor(conf["labels"])
        uda = get_uda.Get_UDA(conf, tokenizer, processor)
        uda_list.append(uda)
        labels_map.append(conf["labels"])
    return uda_list, labels_map

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

# 预测一条文本的多标签列表与对应的权重列表
def get_uda_multitags_result(content, model_list, labels_map):
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

# -------------------------------核心代码---------------------------
# 每个conf_tag对应一个模型配置，需要哪些模型就把相应的模型配置文件放入列表中
conf_list = [conf_tag0, conf_tag1,conf_tag2,conf_tag3,conf_tag4]
model_list, labels_map = get_multitags_uda_model(conf_list)

# 心跳包
def test(request):
    request.encoding = 'utf-8'
    return JsonResponse({"status": 200, "message": "connect success"}, safe=False)

@csrf_exempt
def get_multitags(request):
    request.encoding = 'utf-8'
    if request.method != "POST":
        return JsonResponse({"status":405,"message":"Only POST supported"},safe=False)
    else:
        content = json.loads(request.body)['content']
    if content:
        # get_multitags_result函数利用uda的五个模型预测文本的多标签列表及对应的权重
        # 返回形式{"labels": [非银金融，商务部..], "weight": [0.97,0.98...]}
        result = get_uda_multitags_result(content, model_list, labels_map)
    return JsonResponse( {'status':200, 'message':'OK', 'result':'{}'.format(result)}, safe=False)



