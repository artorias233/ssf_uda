# coding=utf-8
from __future__ import print_function
import jieba
from jieba import analyse

text = "民众投顾被骗亏损维权免费咨询威信：562071109，不追回不收费最近有投资者向我咨询该平台有监管吗,在该平台亏钱了还能挽回损失吗？通过其提供的平tai台名称，我们查询的平taizheng据。该平台宣称受中国zheng监会监管，经查询，在中国zheng监会的相关公司名录中没有查找到该平台，显然该平台为虚假宣称。是个无监管的黑平台，目前依然在市场行骗中，希望有看到的投资者能够提高警惕，有正在了解或正在此平台做单的投资者朋友，希望能够及时悬崖勒马，不要让自己出现更大的亏损近期也有许多亏损的朋友加到我，想寻求正义维权团队的帮助。如果你的情况跟上述情况类似，接下来你就不要再去操作了你该想想自己是否可能被骗了提醒大家，对于投资者而言，切勿轻信高回报与所谓的专家策略，夯实自身交易技术，提高鉴别黑平台、诈骗公司的能力，才是保护自身财产安全最直接有效的方法当然已经投入大笔资金被骗的朋友，不要慌张，可以考虑找到维权团队寻求专业的维权帮助正义团队可以承诺维权不成功不收取一丝一毫专注外汇黄金维权，只要你zheng据齐全，我们有能力也有信心帮你拿回你的资金不论你是在现货、大宗商品、外汇等哪里亏损都可以来找我。朋友们注意了亏损被骗达到以下要求的我们可以接案。1 亏损额度超过2万人民币的；2 交易记录在一年以内的；3 有网银入金记录的；4 平台没有跑路还再运作的；满足以上条件我们有把握追回、时间5-30工作日。合作之前会签署明确的协议书，前期分文不收，已帮众多受害者挽回损失"

print(len(text))
keywords = analyse.extract_tags(text,topK=50)
word_list = jieba.cut(text)
text_process = []
for word in word_list:
    if word in keywords:
        text_process.append(word)

text = "".join(text_process)
print(text, len(text))
