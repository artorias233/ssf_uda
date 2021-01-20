#coding:utf-8
from __future__ import print_function

import numpy as np
import csv
import pandas as pd
import os, json
from cffex_uda.utils import tokenization, raw_data_utils
from cffex_uda import get_uda
import tensorflow as tf
import jieba
import json
import re
from jieba import analyse
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from FastTextRank.FastTextRank4Sentence import FastTextRank4Sentence
from nlg_yongzhuo import *

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "do_train", False,
    help=("Whether to perform training. If both do_train and do_eval are True, "
          "we interleave training with evaluation. But on TPU Pods, "
          "it is not possible to interleave training and evaluation"))
flags.DEFINE_bool(
    "do_eval", False,
    help="Whether to perform evaluation.")
flags.DEFINE_bool(
    "do_predict", False,
    help="Whether to perform prediction.")
flags.DEFINE_integer(
    "train_batch_size", 32,
    help="Batch size for the supervised objective.")
flags.DEFINE_integer(
    "eval_batch_size", 8,
    help="Base batch size for evaluation.")
flags.DEFINE_integer(
    "save_checkpoints_num", 20,
    help="How many checkpoints we save in training.")
flags.DEFINE_integer(
    "epochs", None,
    help="How many checkpoints we save in training.")

flags.DEFINE_bool(
    "process_for_split", False,
    help=(""))
flags.DEFINE_bool(
    "process_for_embedding", False,
    help=(""))

flags.DEFINE_integer(
    "segment_len", 200,
    help="")

flags.DEFINE_string(
    "model_dir", None, "Data directory of the raw data")

flags.DEFINE_integer(
    "max_seq_len", 256,
    help="")

flags.DEFINE_integer(
    "segment_overlap", 50,
    help="")



FILE_PATH_BASE = 'cffex_uda/data/CFFEX/theme3'

conf = {
"labels":FLAGS.labels,
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
"model_dir": FLAGS.model_dir,
"use_one_hot_embeddings": True,
"max_seq_length": FLAGS.max_seq_len,
"model_dropout": -1,
"num_train_steps": 5000,
"learning_rate": 2e-5,
"num_warmup_steps": 1000,
"clip_norm": 1.0,
"train_batch_size": 32,
"eval_batch_size": 8,
"iterations_per_loop": 200,
}

def split_mini_test(file_name, ratio):
    data = pd.read_csv(file_name, delimiter=',', error_bad_lines=False)
    train_mini, data = split_set_by_ratio(data, ratio)
    eval_mini, data = split_set_by_ratio(data, ratio)
    test_mini, data = split_set_by_ratio(data, ratio)
    # show_info(train_mini, 'answercontent')
    # show_info(eval_mini, 'answercontent')
    # show_info(test_mini, 'answercontent')
    train_mini.to_csv(os.path.join(FILE_PATH_BASE, 'train_mini.csv'), sep=',', index=0, header=1)
    eval_mini.to_csv(os.path.join(FILE_PATH_BASE, 'eval_mini.csv'), sep=',', index=0, header=1)
    test_mini.to_csv(os.path.join(FILE_PATH_BASE, 'test_mini.csv'), sep=',', index=0, header=1)

def split_set_by_ratio(data, ratio):
    np.random.seed(43)
    shuffled_indices = np.random.permutation(len(data))
    set_size = int(len(data)*ratio)
    set_indices = shuffled_indices[:set_size]
    remain_indices = shuffled_indices[set_size:]
    return data.iloc[set_indices],data.iloc[remain_indices]


def get_eval_test_train(data, eval_ratio, test_ratio):
    eval, data = split_set_by_ratio(data, eval_ratio)
    test, data = split_set_by_ratio(data, test_ratio)
    return eval, test, data

def shuffle_set(data):
    np.random.seed(43)
    shuffled_indices = np.random.permutation(len(data))
    return data.iloc[shuffled_indices]

def split_sup_unsup(data, num, sup_dir, unsup_dir):
    sup_f = open(sup_dir, 'w')
    sup_writer = csv.writer(sup_f,delimiter="\t")
    sup_writer.writerow(data.columns.values)
    unsup_f = open(unsup_dir, 'w')
    unsup_writer = csv.writer(unsup_f,delimiter="\t")
    unsup_writer.writerow(data.columns.values)
    label_list =  data['label'].unique()
    for label in label_list:
        print('处理{}'.format(label))
        for index, item in data[data['label']==label][0:num].iterrows():
            sup_writer.writerow(list(item))
        for index, item in data[data['label'] == label][num:].iterrows():
            unsup_writer.writerow(list(item))
    sup_f.close()
    unsup_f.close()

def show_label_size(data, label_name):
    print('data columns:{}\ntotal size:{}'.format(data.columns.values, data.shape[0]))
    label_list = data[label_name].unique()
    for label in label_list:
        print("label:{}\tsize:{}\tratio:{}".format(label,
                data[data[label_name]==label].shape[0], data[data[label_name]==label].shape[0]*1.0/data.shape[0]))

def reform_unsup(data, label):
    labels = ['unsup']*len(data)
    unsup = pd.DataFrame({'content': data[label], 'label': labels})
    return unsup


def drop_sup(unsup_data, sup_data,unsup_label='content', sup_label = 'content'):
    """删除无监督数据中出现的有监督数据部分，默认提取的key均为'content"""
    drop_list = []
    sup_content_list = list(sup_data[sup_label])
    for i in range(len(unsup_data)):
        if unsup_data.iloc[i][unsup_label] in sup_content_list:
            drop_list.append(i)
            print("drop {}".format(i))
    unsup_data.drop(labels=drop_list, inplace=True)
    print("drop sup list:{}".format(drop_list))
    return unsup_data

def show_info(data,  label="content", label_y="label"):
    max_len = max([len(x) for x in data[label]])
    min_len = min([len(x) for x in data[label]])
    avg = sum([len(x) for x in data[label]]) * 1.0 / len(data)
    print("len:{}\tmax:{}\tmin:{}\tavg:{}\t".format(len(data), max_len, min_len, avg))
    show_label_size(data, label_y)

def clean_text(st):
  """clean text."""
  st=st.decode('utf-8','ignore').encode("utf-8")
  st = st.replace("<br />", " ")
  st = st.replace("&quot;", "\"")
  st = st.replace("<p>", " ")
  if "<a href=" in st:
    # print("before:\n", st)
    while "<a href=" in st:
      start_pos = st.find("<a href=")
      end_pos = st.find(">", start_pos)
      if end_pos != -1:
        st = st[:start_pos] + st[end_pos + 1:]
      else:
        print("incomplete href")
        print("before", st)
        st = st[:start_pos] + st[start_pos + len("<a href=")]
        print("after", st)

    st = st.replace("</a>", "")
    # print("after\n", st)
    # print("")
  st = st.replace("\\n", " ")
  st = st.replace("\\", " ")
  while "  " in st:
    st = st.replace("  ", " ")
  return st

def get_split(text1, chunk_size, overlap):
    text1 = text1.decode('utf-8')
    l_total = []
    valid_len = chunk_size - overlap
    if len(text1)//valid_len > 0:
        chunk_num = len(text1)//valid_len
    else:
        chunk_num = 1
    for w in range(chunk_num):
        if w == 0:
          l_parcial = text1[:chunk_size].encode('utf-8')
          l_total.append(l_parcial)
        else:
          l_parcial = text1[w*valid_len:w*valid_len + chunk_size].encode('utf-8')
          # print(l_parcial)
          # print("\n")
          l_total.append(l_parcial)
    # print(l_total)
    return json.dumps(l_total)



def encode_emb_list(uda, data):
    emb_list = []
    for text_split in data['text_split']:
        text_split = json.loads(text_split)
        emb_list_temp = []
        for text_segment in text_split:
            print(text_segment)
            emb = uda.predict(text_segment)['pooled_output'].tolist()
            emb_list_temp.append(emb)
            # print(type(emb), text_segment)
        # print(emb_list_temp)
        emb_list.append(json.dumps(emb_list_temp))
    data['emb_list'] = emb_list
    return data

def to_csv(data, path_base, file_name):
    data.to_csv(os.path.join(path_base, file_name), sep=',', index=False, header=1,encoding='utf-8',line_terminator='\n')

def read_csv(path_base, file_name):
    return pd.read_csv(os.path.join(path_base, file_name), delimiter=',', error_bad_lines=False,lineterminator='\n')


def theme3_process():
    labels = ["豆粕期权", "贵州茅台", "国泰君安", "沪深300股指期权", "汇率波动", "区块链", "中国石油", "LPR"]
    train_all = pd.DataFrame()
    for i in range(0, len(labels)):
        train = read_csv(FILE_PATH_BASE, "{}.csv".format(labels[i]))
        train['label'] = [labels[i]] * len(train)
        train = train.drop(columns=['show_tags'])
        if (len(train_all) < 1):
            train_all = train
            print("len: {}".format(len(train_all)))
        else:
            train_all = pd.merge(train_all, train, how='outer')
    print(train_all.head(5))
    show_info(train_all)
    to_csv(train_all, FILE_PATH_BASE, 'train.csv')

def textrank_extraction(text, mod):
    if len(text) <300:
        return text

    # tr4w.analyze(text=text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
    #
    # print('关键词：')
    # for item in tr4w.get_keywords(20, word_min_len=1):
    #     print(item.word, item.weight)
    #
    # print()
    # print('关键短语：')
    # for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num=2):
    #     print(phrase)
    #
    mod = FastTextRank4Sentence(use_w2v=False, tol=0.0001)
    text = mod.summarize(text, 3)
    print(text)
    # tr4s = TextRank4Sentence()
    # tr4s.analyze(text=text, lower=True, source='all_filters')
    #
    # print()
    # print('摘要：')
    # res = ""
    # for item in tr4s.get_key_sentences(num=5):
    #     if len(res) < 256:
    #         res += item.sentence
    # return res


def keyword_extraction(text):
    if len(text) <300:
        return text
    keywords = analyse.extract_tags(text, topK=50)
    # keywords2 = [喊单|老师|保本保收益|实盘指导|承诺盈利|稳定盈利|操盘须知|抄盘葵花宝典|盘中解盘|内幕消息|微信|qq|加群|内幕|交易公式|内参|专家|带单]
    word_list = jieba.cut(text)
    text_process = []
    for word in word_list:
        if word in keywords:
            text_process.append(word)
    return "".join(text_process)

def dataframe_keyword_extraction(data):
    mod = TextRank4Sentence()
    label_index = data.columns.get_loc('content')
    for i in range(len(data)):
        res = ""
        print("处理第{}条数据".format(i))
        content = data.iloc[i]['content']
        if(len(content) < 300):
            continue
        mod.analyze(text=content, lower=False, source = 'all_filters')

        for str in mod.get_key_sentences(num=3):
            if len(res) < 256:
                res += str.sentence
        print(res)
        data.iloc[i,label_index] = res
    return data

# def textrank(content):
#     sents = list(cut_sentence(content))
#     vect = TfidfVectorizer(min_df=1,tokenizer=Tokenize)
#     tfidf = vect.fit_transform(sents)
#     tfidf_graph = tfidf*tfidf.T
#     nx_graph = nx.from_scipy_sparse_matrix(tfidf_graph)
#     scores = nx.pagerank(nx_graph)
#     res = sorted(((scores[i],i) for i,s in enumerate(sents)), reverse=True)
#     summary = [sents[i] for _,i in sorted(res[:3])]
#     print 'text_rank', u'。 '.join(summary).replace('\r','').replace('\n','')+u'。'

def split_train_dev(file_path, file_name):
    print("划分训练集，验证集：")
    data = read_csv(FILE_PATH_BASE, 'train.csv')
    print("总数据")
    show_info(data)
    dev, train = split_set_by_ratio(data, 0.2)
    print("训练集")
    show_info(train)
    print("验证集")
    show_info(dev)
    to_csv(dev, FILE_PATH_BASE, "eval_0.2.csv")
    to_csv(train, FILE_PATH_BASE, "train_0.8.csv")

def csv_multi_tag_process(data):
    tags_set = [["物流","公告速递","财经早报","非银金融","建筑材料","宏观政策","A股IPO动态","医药","机械设备","纺织服装","A股入摩进展","比特币","none"],["财经早餐","A股策略","中国央行动态","电力","贸易","化工","快递物流","智能制造","原油","none"],["研判优选","物联网","轻工制造","严重事故","民企","午盘点评","口罩","投融资","none"],["电子","工业互联网","焦点评论","商务部","none"],["时政要闻","农产品","工信部","none"]]
    tags_map = {'公告速递': 0, '医药': 0, '快递物流': 1, '建筑材料': 0, '工信部': 4, 'A股IPO动态': 0, '机械设备': 0, '焦点评论': 3, '电力': 1, '纺织服装': 0, '宏观政策': 0, '智能制造': 1, '工业互联网': 3, '商务部': 3, '原油': 1, '物流': 0, '贸易': 1, 'A股入摩进展': 0, '投融资': 2, '口罩': 2, '中国央行动态': 1, '电子': 3, '严重事故': 2, '民企': 2, '化工': 1, '午盘点评': 2, '物联网': 2, '财经早报': 0, '时政要闻': 4, '农产品': 4, '非银金融': 0, '财经早餐': 1, 'A股策略': 1, '研判优选': 2, '比特币': 0, '轻工制造': 2}
    data_tag_0 = pd.DataFrame(columns=('content', 'label'))
    data_tag_1 = pd.DataFrame(columns=('content', 'label'))
    data_tag_2 = pd.DataFrame(columns=('content', 'label'))
    data_tag_3 = pd.DataFrame(columns=('content', 'label'))
    data_tag_4 = pd.DataFrame(columns=('content', 'label'))
    data_tag_all = [data_tag_0, data_tag_1, data_tag_2, data_tag_3, data_tag_4]
    for i in range(len(data)):
        tags = re.findall("'(.*?)'",data.iloc[i]['uda_codes'])
        content = data.iloc[i]['content']
        print("处理第{}条文本".format(i))
        for tag in tags:
            # print(tag)
            tag_index = tags_map[tag]
            for i in range(len(data_tag_all)):
                if i == tag_index:
                    data_tag_all[i] = data_tag_all[i].append(
                        [{'content': content,'label': tag}], ignore_index=True)
                else:
                    data_tag_all[i] = data_tag_all[i].append(
                        [{'content': content,'label': "none"}], ignore_index=True)
    for i in range(len(data_tag_all)):
        to_csv(data_tag_all[i],'cffex_uda/data/CFFEX/multitags', 'data_tag_{}.csv'.format(i))

def split_multi_tags(tag_list):
    tag_map = {}
    tags_unique = []
    result = []
    for tags in tag_list:
        for tag_key in tags:
            if not tag_map.has_key(tag_key):
                tags_unique.append(tag_key)
                tag_map[tag_key] = []
            for tag_v in tags:
                if tag_v == tag_key:
                    continue
                if tag_v not in tag_map[tag_key]:
                    tag_map[tag_key].append(tag_v)

    for tag in tags_unique:
        # print("tag: {}".format(tag))
        tag_in_result = 0
        if len(result) == 0:
            result.append([tag])
        else:
            for i in range(len(result)):
                tag_cant_in_set = 0
                for tag_key in result[i]:
                    if tag in tag_map[tag_key]:
                        tag_cant_in_set = 1
                        break
                if not tag_cant_in_set:
                    result[i].append(tag)
                    tag_in_result = 1
                    break
            if not tag_in_result:
                result.append([tag])
        # print(result)
    return result

def summary(text):
    text = re.sub(r"#.*#", "", text)
    if len(text) < 300:
        return text
    keywords = analyse.extract_tags(text, topK=3, allowPOS=('v','vd','vn','n'))
    if len(keywords)== 0:
        return text
    # print(str(keywords).decode("unicode_escape"))
    sentences = re.split('。|！|：|；|？', text)
    sentences_processed = ""
    for sentence in sentences:
        if len(sentences_processed) > 256:
            break
        for key_word in keywords:
            if key_word in sentence:
                sentences_processed += "{}。".format(sentence)
                break

    return sentences_processed


def get_data_by_size_lim(train_examples, sup_size):
    """Deterministicly get a dataset with only sup_size examples."""
    # Assuming sup_size < number of labeled data and
    # that there are same number of examples for each category
    labels = [0,1,2,3,4]
    per_label_examples = {}
    if (sup_size == -1):
        return train_examples
    else:
        for i in range(len(train_examples)):
            label = train_examples.iloc[i]['label']
            if label not in per_label_examples:
                per_label_examples[label] = []
            per_label_examples[label] += [train_examples.iloc[i]]

    new_train_examples = pd.DataFrame(columns=('content', 'label'))
    for i in range(sup_size):
        for label in labels:
            if i < len(per_label_examples[label]):
                new_train_examples =  new_train_examples.append(
                        [{'content': per_label_examples[label][i]['content'],'label': per_label_examples[label][i]['label']}], ignore_index=True)

    return new_train_examples

def eval_illegal():
    eval = read_csv('cffex_uda/data/CFFEX/illegal', 'eval_0.2.csv')
    for i in range(len(eval)):
        label_dict = ["未涉嫌违规", "非法投资咨询", "非法诱导投资", "非法经营活动", "维权追损二次诈骗"]
        content = eval.iloc[i]["content"]
        label = label_dict[int(eval.iloc[i]['label'])]

        label_p, key_words = re_match(content, label, 3)
        label_p = label_dict[int(label_p)]
        if (label_p != label):
            print(content)
            print("ori:{}\tpredict:{}\t关键词：{}".format(label, label_p, key_words))

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

def print_list(l):
    print(str(l).decode("string_escape"))

def eval_multi_tags():
    eval = read_csv('cffex_uda/data/CFFEX/multitags', 'eval.csv');
    for i in range(len(eval)):
        content = eval.iloc[i]['content']
        labels = re.findall(r"'(.*?)'",eval.iloc[i]['uda_codes'])
        print_list(labels)


def main(_):

    # for i in range(5):
    #     data  = read_csv('cffex_uda/data/CFFEX/multitags/data_tag_{}'.format(i), 'data_tag_{}.csv'.format(i))
    #     train, eval = split_set_by_ratio(data,0.8)
    #     to_csv(train,'cffex_uda/data/CFFEX/multitags/data_tag_{}'.format(i), 'train_0.8.csv')
    #     to_csv(eval, 'cffex_uda/data/CFFEX/multitags/data_tag_{}'.format(i), 'eval_0.2.csv')
    # print(len(data))
    # tag_list = []
    # # print(str(tag_list).decode("string_escape"))
    # # for tag in tag_list:
    # #     print(tag)
    # tag_set = split_multi_tags(tag_list)
    # print(str(tag_set).decode("string_escape"))
    # train = read_csv('cffex_uda/data/CFFEX/illegal', 'train_0.8.csv')
    # show_info(train)

    # for t in range(0, 30):
    #     cnt = 0
    #     for i in range(len(eval)):
    #         content = eval.iloc[i]["content"]
    #         label = eval.iloc[i]['label']
    #         res = re_match(content, label, t)
    #         if res == 1:
    #             cnt +=1
    #     print(1.0*cnt/len(eval))
    # cnt = 0

    # print(1.0 * cnt / len(eval))
    # eval = read_csv('cffex_uda/data/CFFEX/illegal', 'eval_0.2.csv')
    #     label = train.iloc[i]['label']
    #     if not label_keyword_map.has_key(label):
    #         label_keyword_map[label] = []
    #     key_words = summary(train.iloc[i]['content'])
    #     for word in key_words:
    #         if word not in label_keyword_map[label]:
    #             label_keyword_map[label].append(word)
    # for i in range(1,5):
    #     print(str(label_keyword_map[i]).decode("unicode_escape"))
    # train = dataframe_keyword_extraction(train)
    # to_csv(train, 'cffex_uda/data/CFFEX/illegal', 'train_extraction_0.8.csv')
    # unsup = read_csv('cffex_uda/data/CFFEX/illegal', 'unsup.csv')
    # unsup['label'] = 'unsup'
    # to_csv(unsup, 'cffex_uda/data/CFFEX/illegal', 'unsup.csv')
    # eval = read_csv('cffex_uda/data/CFFEX/illegal', 'eval_0.2.csv')
    # eval = dataframe_keyword_extraction(eval)
    # to_csv(eval, 'cffex_uda/data/CFFEX/illegal', 'eval_extraction_0.2.csv')
    # eval = dataframe_keyword_extraction(eval)
    # unsup = dataframe_keyword_extraction(unsup)
    # to_csv(unsup, 'cffex_uda/data/CFFEX/illegal', 'unsup_extration.csv')
    # unsup.insert(1,'label','unsup')

    # labels = ['unsup']*len(unsup)
    # unsup['label'] = labels
    # print(unsup.head())
    # to_csv(unsup,'cffex_uda/data/CFFEX/illegal', 'unsup.csv')
    # train_data = read_csv('cffex_uda/data/CFFEX/illegal', 'unsup.csv')
    # # train_data.drop(['text_split','emb_list'],axis=1,inplace=True)
    # show_info(train_data)
    # label_index = train_data.columns.get_loc('content')
    # for i in range(len(train_data)):
    #     content = train_data.iloc[i]['content']
    #     content = keyword_extraction(content)
    #
    #     train_data.iloc[i,label_index] = content
    #     # print(train_data.iloc[i]['content'])
    # show_info(train_data)
    # to_csv(train_data,'cffex_uda/data/CFFEX/illegal','unsup.csv')
    # to_csv(train_data,'cffex_uda/data/CFFEX/illegal',"train_extraction_0.8.csv")

    # eval = read_csv('cffex_uda/data/CFFEX/theme3', 'eval_0.2.csv')
    # unsup = read_csv('cffex_uda/data/CFFEX/illegal', 'unsup.csv')
    # show_info(train_data)
    # show_info(eval)
    # show_info(unsup)

    if FLAGS.process_for_split:
        train_data = read_csv(FLAGS.raw_data_dir, FLAGS.sup_train_file)
        dev_data = read_csv(FLAGS.raw_data_dir, FLAGS.sup_dev_file)
        # unsup_data = read_csv(FLAGS.raw_data_dir, FLAGS.unsup_file)
        train_data['text_split'] = train_data['content'].apply(lambda x: get_split(x, FLAGS.segment_len, FLAGS.segment_overlap))
        dev_data['text_split'] = dev_data['content'].apply(lambda x: get_split(x, FLAGS.segment_len, FLAGS.segment_overlap))
        # unsup_data['text_split'] = unsup_data['content'].apply(lambda x: get_split(x, FLAGS.segment_len, FLAGS.segment_overlap))
        to_csv(train_data, FLAGS.raw_data_dir, FLAGS.sup_train_file)
        to_csv(dev_data, FLAGS.raw_data_dir, FLAGS.sup_dev_file)
        # to_csv(unsup_data, FLAGS.raw_data_dir, FLAGS.unsup_file)
        
    if FLAGS.process_for_embedding:
        tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
        processor = raw_data_utils.CFFEXProcessor(FLAGS.labels)
        uda = get_uda.Get_UDA(conf, tokenizer, processor)
        train_data = read_csv(FLAGS.raw_data_dir, FLAGS.sup_train_file)
        dev_data = read_csv(FLAGS.raw_data_dir, FLAGS.sup_dev_file)
        train_data = encode_emb_list(uda, train_data)
        dev_data = encode_emb_list(uda, dev_data)
        to_csv(train_data, FLAGS.raw_data_dir, FLAGS.sup_train_file)
        to_csv(dev_data, FLAGS.raw_data_dir, FLAGS.sup_dev_file)
        uda.kill()


    # data = pd.read_csv(os.path.join(FILE_PATH_BASE, 'train_mini.csv'), delimiter=',', error_bad_lines=False)
    # data['text_split'] = data['weibo_content'].apply(lambda x: get_split(x, 200, 50))
    # data.to_csv(os.path.join(FILE_PATH_BASE, 'train_mini.csv'), sep=',', index=0, header=1)
    # print(data.head())
    # data.to_csv(os.path.join(FILE_PATH_BASE, 'train_0.8_split.csv'), sep=',', index=0, header=1)

    # data = pd.read_csv(os.path.join(FILE_PATH_BASE, 'train.csv'), delimiter=',', error_bad_lines=False)
    # eval, train = split_set_by_ratio(data, 0.2)
    # show_info(eval,'content', 'label')
    # show_info(train, 'content', 'label')
    # to_csv(eval,'eval_0.2.csv')
    # to_csv(train, 'train_0.8.csv')

if __name__ == "__main__":
    tf.app.run()
