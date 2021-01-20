# coding=utf-8
from cffex_uda import  get_uda
from cffex_uda.utils import tokenization, raw_data_utils
from  conf import *
import pandas as pd
import json
import re
import os


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
    # print("max_p:{},  label_p:{}".format(max_p, label_p))
    return label_list[label_p]

def print_list(l):
    print(str(l).decode("string_escape"))

def illegal_eval_uda():
    tokenizer = tokenization.FullTokenizer(vocab_file=conf['vocab_file'], do_lower_case=True)
    processor = raw_data_utils.CFFEXProcessor(conf["labels"])
    uda = get_uda.Get_UDA(conf, tokenizer, processor)

    eval_data = read_csv('cffex_uda/data/CFFEX/illegal', 'eval_0.2.csv')

    check_data = pd.DataFrame(columns=('content', 'label', 'predict'))

    # label_dict = {7:"恶意抹黑监管机构", 3:"非法发行证券",1:"非法荐股",4:"非法配资",5:"负面言论风险言论",6:"煽动维权诈骗",
    #               2:"诱导开户",0:"正常"}
    label_dict = ["未涉嫌违规", "非法投资咨询", "非法诱导投资", "非法经营活动", "维权追损二次诈骗"]
    labels = conf['labels']

    TP = 0.0
    FN = 0.0
    FP = 0.0
    false_examples = []
    for i in range(len(eval_data)):
        content = eval_data.iloc[i]['content']
        label = label_dict[int(eval_data.iloc[i]['label'])]
        probs = uda.predict(content)
        label_p = label_dict[int(get_label(labels, probs))]
        check_data = check_data.append([{'content': content, 'label': json.dumps(label, ensure_ascii=False),
                                         'predict': json.dumps(label_p, ensure_ascii=False)}], ignore_index=True)

        if label != "未涉嫌违规" and label_p != "未涉嫌违规":
            TP += 1
        elif  label != "未涉嫌违规" and label_p == "未涉嫌违规":
            # false_examples.append([content, label, label_p])
            FN += 1
        elif label == "未涉嫌违规" and label_p != "未涉嫌违规":
            false_examples.append([content, label, label_p])
            FP += 1

    for w in false_examples:
        print(w[0])
        print("ori:{}\tpredict:{}".format(w[1], w[2]))
    pcsion = TP / (TP + FP)
    recall = TP / (TP + FN)
    to_csv(check_data, 'cffex_uda/data/CFFEX/illegal', 'illegal_uda.csv')
    print("误识别为违规言论条数：{}".format(len(false_examples)))
    print("违规言论精确率：{}, 召回率：{}".format(pcsion, recall))



    uda.kill()

def restore():
    eval = read_csv('cffex_uda/data/CFFEX/multitags', 'eval_uda_predict.csv')
    eval['uda_predict'] = "[]"
    to_csv(eval, 'cffex_uda/data/CFFEX/multitags', 'eval_uda_predict.csv')

def multitags_eval():
    # tokenizer = tokenization.FullTokenizer(vocab_file=conf['vocab_file'], do_lower_case=True)
    # processor = raw_data_utils.CFFEXProcessor(conf["labels"])
    # uda = get_uda.Get_UDA(conf, tokenizer, processor)
    eval = read_csv('cffex_uda/data/CFFEX/multitags', 'eval_uda_predict.csv')
    label_index = eval.columns.get_loc('uda_predict')
    for i in range(len(eval)):
        content = eval.iloc[i]['content']
        labels_ori = re.findall(r"'(.*?)'", eval.iloc[i]['uda_codes'])
        labels_uda = json.loads(eval.iloc[i]['uda_predict'])
        print_list(labels_ori)
        print(json.dumps(labels_uda,ensure_ascii=False))
    #     probs = uda.predict(content)
    #     # print(probs)
    #     labels_p = get_label(conf["labels"], probs)
    #     print(labels_p)
    #     if labels_p!="none":
    #         # 将utf-8格式的字符解码为unicode，这样才能正常写入csv文件
    #         labels_uda.append(labels_p.decode('utf-8'))
    #         eval.iloc[i,label_index] = json.dumps(labels_uda,ensure_ascii=False)
    #     print_list(labels_ori)
    #     print(json.dumps(labels_uda,ensure_ascii=False))
    # to_csv(eval, 'cffex_uda/data/CFFEX/multitags', 'eval_uda_predict.csv')
    # uda.kill()

def multi_tags_acc():
    eval = read_csv('cffex_uda/data/CFFEX/multitags', 'eval_uda_predict.csv')
    check_data = pd.DataFrame(columns=('content', 'label', 'predict'))

    cnt = 0
    for i in range(len(eval)):
        content = eval.iloc[i]['content']
        labels_ori = re.findall(r"'(.*?)'", eval.iloc[i]['uda_codes'])
        labels_uda = json.loads(eval.iloc[i]['uda_predict'])
        labels_uda = [label.encode("utf-8") for label in labels_uda]
        check_data = check_data.append([{'content': content, 'label': json.dumps(labels_ori, ensure_ascii=False),
                                         'predict': json.dumps(labels_uda, ensure_ascii=False)}], ignore_index=True)
        if labels_ori != labels_uda:
            print(content)
            print ("ori: {}\npredict:{}".format(str(labels_ori).decode("string_escape"), str(labels_uda).decode("string_escape")))
        else:
            cnt += 1
    to_csv(check_data, 'cffex_uda/data/CFFEX/multitags', 'multitags_uda.csv')
    print ("acc: {}".format(cnt*1.0/len(eval)))

def eval_illegal_re():
        TP = 0.0
        FN = 0.0
        FP = 0.0
        cnt = 0.0
        ep = 1e-7
        t = 2
        eval = read_csv('cffex_uda/data/CFFEX/illegal', 'eval_0.2.csv')
        for i in range(len(eval)):
            label_dict = ["未涉嫌违规", "非法投资咨询", "非法诱导投资", "非法经营活动", "维权追损二次诈骗"]
            content = eval.iloc[i]["content"]
            label = label_dict[int(eval.iloc[i]['label'])]

            label_p, key_words = re_match(content, t)
            label_p = label_dict[int(label_p)]
            if label != "未涉嫌违规" and label_p != "未涉嫌违规":
                if label != label_p:
                    # print(content)
                    print("ori:{}\tpredict:{}\t关键词：{}".format(label, label_p, str(key_words).decode("string_escape")))
                TP += 1
            elif  label != "未涉嫌违规" and label_p == "未涉嫌违规":
                # print(content)
                # print("ori:{}\tpredict:{}\t关键词：{}".format(label, label_p, str(key_words).decode("string_escape")))
                FN += 1
            elif label == "未涉嫌违规" and label_p != "未涉嫌违规":
                # print(content)
                # print("ori:{}\tpredict:{}\t关键词：{}".format(label, label_p, str(key_words).decode("string_escape")))
                FP += 1
            if label_p == label:
                cnt += 1
                # print(content)
                # print("ori:{}\tpredict:{}\t关键词：{}".format(label, label_p, str(key_words).decode("string_escape")))
        pcsion = TP / (TP + FP + ep)
        recall = TP / (TP + FN + ep)
        print("关键词阈值{}，整体准确率：{}".format(t,cnt/len(eval)))
        print("关键词阈值{}，关键词检测违规言论精确率：{}, 召回率：{}".format(t, pcsion, recall))

def re_match(text, t):
    # text:将要匹配的文本， label:真实标签，t:判断为某一违规类别的关键词阈值
    p1 = '喊单|老师|保本保收益|实盘指导|承诺盈利|稳定盈利|操盘须知|抄盘葵花宝典|盘中解盘|内幕消息|微信|qq|加群|内幕|交易公式|内参|专家|带单'
    p2 = '原始股|配发|转板|二元期权|外汇|境外|外盘|石油|邮币卡|境外指数|交易软件'
    p3 = 'T+0|最低手续费|返还手续费|期货招商|期货代理|零门槛开户|免收保证金|一对一|外盘|额外优惠|手续费+0|保证金+0|备注微博开户|纯净交易者实盘群|期货资管|无门槛|高返|交返|期货配资|配资加盟|配资利息|配资|盈利提现|虚拟盘|程序交易|免费投资|外盘|私募|急用资金|pei 姿|佩咨|增资|妻火佩子|起配|进出点位|实盘交易'
    p4 = '金融维权|期货维权|亏损维权|炒期货亏损|炒期货被骗|专业追回亏损|追损|期货法援|法律援助|追损'
    match_words = [[],[],[],[],[]]
    match_words[1] = list(set(re.findall(p1, text)))
    match_words[2] = list(set(re.findall(p2, text)))
    match_words[3] = list(set(re.findall(p3, text)))
    match_words[4] = list(set(re.findall(p4, text)))
    max, pos = 0, 0
    for i in range (1,len(match_words)):
        if len(match_words[i]) > max:
            max = len(match_words[i])
            pos = i
    if max > t:
        # print("max:{}, t:{}".format(max, t))
        label_p = pos
    else:
        label_p = 0
    # if str(label_p) != str(label):
    #     print("{}\t{}\t{}\t[{}]".format(text, label, label_p,",".join(match_words[pos])))
    #     print("")
    #     return -1
    return label_p, match_words[pos]

# restore()
# multitags_eval()
# multi_tags_acc()
illegal_eval_uda()
# illegal_eval()
# eval_illegal_re()
# illegal_eval_uda()