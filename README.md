# 文本分类模型

作者：支龙

2020年9月

# 项目说明


## 步骤一：环境搭建

```
环境要求：

1. python2.7

2. tensorflow-gpu 1.12.2

请下载[albert的预训练语言模型]（https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz) 放入pretrained_models目录下。
```



## 步骤二：数据准备

```
训练数据类型：utf-8编码 csv 文档

该模型采用半监督训练的方式，需要同时提供无标签数据集，有标签数据集，验证集等三个csv文档。

字段说明：

  - content: 新闻内容
  - label: 标签（无标签训练文档中，也需要这个字段，并且字段内容全部设置为unsup）

放置路径：/ssf_uda/ssf_uda/data/一级目录/二级目录

说明：

一级目录，二级目录名称可以自己设置，需设置这两个目录，方便后续训练脚本识别
```



## 步骤三：预处理

```
对无监督数据进行数据增强，对文本字段进行tokenize编码，在根目录下运行：

bash scripts/prepro.sh

scripts/repro.sh  中脚本参数说明：

-  FILE= ：训练文件所在一级目录名称
-  SUB_FILE=：训练文件所在二级目录名称
-  SUP_TRAIN_FILE=：有标签训练集文档名称
-  SUP_DEV_FIL=：验证集文档名称
-  UNSUP_FILE=：无标签训练集文档名称
-  LABELS=：预测时的标签列表 （LABELS="时政要闻","农产品","工信部","none"）
-  MAX_SEQ_LENGTH=：（MAX_SEQ_LENGTH<512, 11G内存设置为256合适）
-  SUP_SIZE=：（有标签训练集中每个标签的最大样本数量，用来进行类别平衡，建议设置为-1，表示不作限制）

说明：

预处理之后，会将无标签数据集，有标签数据集，验证集等三个csv文档编码成为tf_records格式存储到/ssf_uda/ssf_uda/data/proc_data 目录中。
```



## 步骤四：训练

```
在根目录下运行

bash scripts/run_base_uda.sh

scripts/run_base_uda.sh 中脚本参数说明：

	基本参数：TAG为模型名称，可自由设置。其余参数请与scripts/prepro.sh保持一致

	运行参数：

		--dotrain=: True表示训练, False表示不训练

		--do_eval=: True表示做验证, False表示不验证

		--num_train_steps=：总训练步数

		--unsup_ratio=：一个batch\_size中，无标签样本数是有标签样本数的多少倍

		--train_batch_size=：有标签训练样本的batch_size(最终batch_size (1+unsup_ratio)*train_batch_size)

说明：

模型文件存储路径：ssf_uda/ckpt/base_uda/二级目录名称\_\${SUP_SIZE}\_\${MAX_SEQ_LENGTH}\_\${TAG}

训练时该路径下会自动生成一个best_model文件夹，并将每次训练后最好的结果保存在这个文件夹中
```



## 步骤五：运行

```
本项目在django_uda目录下提供了违规文本检测的django封装，django_multitags目录下提供了多标签分类的django封装。

以django_uda为例，django_uda/conf.py 为运行配置文件，django_uda/conf.py参数说明：

-  labels= : 与scripts/run_base_uda.sh保持一致
-  sub_file=：与scripts/run_base_uda.sh中二级目录名称保持一致
-  sup_size=：与scriptsrun_base_uda.sh保持一致
-  model_type =： "base_uda"表示采用uda训练后的模型存储路径，"base"表示基础模型训练的模型存储路径
-  tag = ：与scripts/run_base_uda.sh中TAG保持一致
-  model_name =：best_model路径下模型前缀（比如"model.ckpt-4250"）,用来识别并加载模型
-  max_seq_len=：与scripts/run_base_uda.sh中MAX_SEQ_LENGH保持一致

启动服务：

python manage.py (或manage_multitags.py) runserver

POST请求地址：

违规检测：IP/get_illegal_type

多标签识别：IP/get_multitags
```


### 请求方式 `POST`

### 请求参数：JSON格式

| 参数    | 类型   | 是否必选 | 描述             | 默认值 |
| ------- | ------ | -------- | ---------------- | ------ |
| uuid    | string | 否       | 文档的唯一标志码 | 无     |
| content | string | 是       | 文档内容         | 无     |

请求示例

```
{
"uuid": "4596707759",
"content": "2月CPI公布：同比上涨1.5%  看看你买啥贵了，买啥便宜了？】国家统计局今天公布的数据显示：2月全国居民消费价格指数CPI同比上涨1.5%，涨幅比上月回落0.2个百分点。其中，食品价格同比涨幅回落是影响CPI走势的核心因素。从单项食品看，羊肉、牛肉价格涨幅居前，鲜果和鲜菜价格小幅波动，而鸡蛋、猪肉和水产品价格却处在下行通道， 涨跌互抵，使得食品价格平稳运行 。在非食品中，医疗保健、教育文化和娱乐、居住价格依旧领涨，但涨幅平稳。国研中心宏观经济研究部研究员张立群表示：总体来看，价格还是总体平稳，或者是一个走低基本趋势。（记者 柴华 武立丰 张宏斌",
}
```



### 返回字段：JSON格式

| 字段         | 说明                          |
| ------------ | ----------------------------- |
| status       | 状态码                        |
| message      | 连接状态                      |
| result.probabilities    | 预测为各类别的概率              |
| result.label | 最终的预测标签 |

返回示例

```
{
    "status": 200,
    "message": "OK",
    "result": {
        "probabilities": array([0.01, 0.99]),
        "label": B
    }
}
```

### 状态码说明：
```
200：连接正常
405：请求方法错误，仅支持post请求
```
