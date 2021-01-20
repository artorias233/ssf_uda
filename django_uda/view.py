#-*- coding:utf-8 _*-
"""
@author:Yuefeng Lin
@file: view.py
@time: 2020/07/27
"""
import json
import sys

from django.views.decorators.csrf import csrf_exempt

sys.path.append('../cffex_uda')
sys.path.append('../cffex_uda/augmentation')
sys.path.append('../cffex_uda/albert')
sys.path.append('../cffex_uda/utils')

from django.http import JsonResponse
from cffex_uda import  get_uda
from cffex_uda.utils import tokenization, raw_data_utils
from  conf import *

tokenizer = tokenization.FullTokenizer(vocab_file=conf['vocab_file'], do_lower_case=True)
processor = raw_data_utils.CFFEXProcessor(conf["labels"])
uda = get_uda.Get_UDA(conf, tokenizer, processor)

# 心跳包
def test(request):
    request.encoding = 'utf-8'
    return JsonResponse({"status": 200, "message": "connect success"}, safe=False)

@csrf_exempt
def predict(request):
    request.encoding = 'utf-8'
    if request.method != "POST":
        return JsonResponse({"status":405,"message":"Only POST supported"},safe=False)
    else:
        print("content:")
        content = json.loads(request.body)['content']
        print(content)
    if content:
        # str = '股指期货就像孙悟空。人跟神，不能做亲密的小伙伴，实在不好愉快的玩耍。'
        p = uda.predict(content)
        print(p)
        # print(str)
        # print(type(str))
        # print(content)
        # print(type(content))
        # print("content: {}".format(content))
        labels = processor.get_labels()
        max = -1
        for i in range(len(p['probabilities'])):
            if(p['probabilities'][i] > max):
                max = p['probabilities'][i]
                label = labels[i]
        print(label)
        result = {"probabilities": list(p['probabilities']), "label": label}
        return JsonResponse({"status": 200, "message": "OK",  "result": "{}".format(result)}, safe=False)
