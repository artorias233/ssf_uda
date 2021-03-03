uda_CFFEX
====================
半监督文本分类模型（基于UDA)  
[UDA代码](https://github.com/google-research/uda "https://github.com/google-research/uda")     
[UDA论文](arxiv.org/abs/1904.12848 "arxiv.org/abs/1904.12848")  
## 环境需求:
    python 2.7
    tensorflow-gpu 1.12.2
    显卡：使用中文albert_base模型，max_seq_len=256, batch_size=32, 大概占用11g,  确保单张显卡内存>=11g，否则请减小max_seq_len或batch_size  

## 目录说明：
    -一级目录/文件
        -二级目录/文件
## 目录结构:
*注：由于还在开发阶段，这些代码是当前主要使用的代码，没有在这上面的表述代码文件均不用考虑* 

    -albert:  albert相关模型，优化器等代码，无需修改
    -augmentation:  数据增强相关代码
        -sent_level_augment.py:  句子级别的数据增强（主要加载回译法文件），并没有没有真正实现回译法，暂时不用
        -word_level_augment.py:  词语级别的数据增强，当前采用的数据增强方式，主要方法为TF_IDF 
    -data:    存放数据的地方，包括训练数据(csv)，验证数据(csv)；预处理后的数据（TFRecord），TF_IDF词频信息等
    -pretrained_models:   预训练albert模型文件放在这里
    -script：  数据预处理，训练，验证等脚本
        -prepro.sh:  数据预处理脚本,主要执行preprocess.py
        参数说明: bert_vocab_file: bert（albert）预训练模型使用的词典文件存放目录地址
                FILE: 训练（验证）数据存放的文件夹名称，CFFEX指的是data/CFFEX这个文件夹,相关的数据均在该文件夹下
                SUB_FILE: 子文件夹，比如主题相关，违规文本相关的数据，theme表示主题相关文本数据，具体文件目录为
                           data/CFFEX/theme
                MAX_SEQ_LENGTH: bert fine-tune时采用的最大序列长度，最长为512，显卡内存不够时有限减小该数值，采用数据截断方式
                                为截断右边，只保留左边的序列长度数量的字符
                SUP_SIZE:  训练时想采用的有标签样本的数量（uda的有标签训练数据默认为均衡分布，当有3个类别，SUP_SIZE=12时，将
                            导致每个类别的有标签数据为 12/3），-1则表示采用所有有标签训练文件中的所有数据（此时不再保证类别均衡）
                TASK_NAME: 使用到的processor的名称

        -run_bash_uda.sh: 采用UDA框架的训练/验证脚本
        参数说明: --do_train=False，是否训练，这里表示不训练

                 --do_eval=True， 是否验证，这里表示需要验证

                 --sup_train_data_dir=data/proc_data/$FILE/$TASK_NAME/train_${SUP_SIZE}_${MAX_SEQ_LENGTH}
                 有标签训练数据的TFRecords存放目录（经过prepro.sh处理后的数据）

                 --unsup_data_dir=data/proc_data/$FILE/$TASK_NAME/unsup_${MAX_SEQ_LENGTH}
                 无标签训练数据的TFRecords存放目录（经过prepro.sh处理后的数据）

                 --eval_data_dir=data/proc_data/$FILE/$TASK_NAME/dev_${MAX_SEQ_LENGTH}
                 验证数据的TFRecords存放目录（经过prepro.sh处理后的数据）

                 --bert_config_file=pretrained_models/albert_base/albert_config.json
                 bert模型的配置文件地址

                 --vocab_file=pretrained_models/albert_base/vocab_chinese.txt
                 bert模型使用的词典地址

                 --init_checkpoint=pretrained_models/albert_base/model.ckpt-best
                 bert模型的初始训练check-point

                 --model_dir=ckpt/base_uda/${TASK_NAME}_${SUP_SIZE}_${MAX_SEQ_LENGTH}
                 训练之后模型的存放地址

                 --aug_ops=tf_idf-0.2，无监督数据的数据增强方式，这里表示采用tf_idf,增强因子为0.2

                 其余参数请参考论文

        -run_base.sh:  不使用UDA框架，仅进行bert fine-tune的基准分类器训练/验证脚本
        参数说明: 同run_base_UDA.sh         


    -utils:    主要是数据处理相关的代码，也用来存放辅助开发的相关代码
        -proc_data_utils.py:  bert模型fine-tune时读取数据的代码，包括训练模型，验证模型等相关文件输入函数
        -raw_data_utils.py:   bert模型fine-tune所用到的各类processor类，主要用在数据预处理（prepro.sh）上
        -split_train.py:      很杂,主要用来按比例划分csv格式的训练数据（比如划分出验证集），也有分析csv文件各种信息的相关函数
        -tokenization.py:     分词器，包含中文分词器（albert的分词器，按字划分，而非按词划分）

    -main.py:  代码主入口，uda模型训练，验证主函数
    -preprocess.py:  数据预处理代码
    -uda.py:   uda模型代码
    -conf.py:     调用Django get_uda时使用的配置文件，配置选项参考run_base_uda.sh
    -get_uda.py:  封装好的uda类，从conf加载配置项完成uda实例的创建，predict方法用来预测单条语句，将返回预测为各类别的概率。

# 代码运行流程：

1. 下载[albert的预训练语言模型]（https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz) 放入pretrained_models目录下。  
2. 将训练文件（csv）放入data/CFFEX/theme(或illegal)下。文件名称：train_0.8.csv表示训练数据，eval_0.2表示验证数据（请参考  
raw_data_utils.py中的CFFEX相关的processor类进行修改)  
3. 按照文件类型修改prepro.sh中相关参数，比如最大序列长度，albert模型的存放目录，训练数据所在目录，处理后的数据的存放目录等。  
4. 转到项目根目录下，执行  
```bash scripts/prepro.sh```  
这一步会将训练数据，验证数据添加bert相关的文本标注，数据阶段等预处理工作，并将处理后的数据按照TFRecords的形式保存在prepro.sh中定义的sup_train_data_dir等相关参数指定  
的地址下  
5. 按照需求修改run_base.sh或者run_base_uda.sh中的do_train,do_eval选项，和各项参数，执行脚本  
```bash scripts/run_base_dua.sh```  
或  
```bash scripts/run_base.sh```  
屏幕日志信息中将会显示每500个epoch下模型的准确率，每个类别的f1,精确率，召回率等指标，最终将显示准确率最高的模型名称和其对应的各项指标


             
