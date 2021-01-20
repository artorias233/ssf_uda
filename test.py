# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runner for UDA that uses BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

import tensorflow as tf
from queue import Queue

from cffex_uda import uda, preprocess
from cffex_uda.albert import modeling
from cffex_uda.utils import raw_data_utils, tokenization
from django_uda.conf import *


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example

def timeCounter(text):
    def wraper():
        print('-'*100)
        print(text, wraper.cnt, time.time()-wraper.t)
        print('-'*100)
        wraper.t = time.time()
        wraper.cnt += 1
    wraper.t = time.time()
    wraper.cnt = 0
    return wraper

def get_sup_feature_specs():
  """Get supervised feature."""
  feature_specs = collections.OrderedDict()
  feature_specs["input_ids"] = tf.FixedLenFeature(
      [conf["max_seq_length"]], tf.int64)
  feature_specs["input_mask"] = tf.FixedLenFeature(
      [conf["max_seq_length"]], tf.int64)
  feature_specs["input_type_ids"] = tf.FixedLenFeature(
      [conf["max_seq_length"]], tf.int64)
  feature_specs["label_ids"] = tf.FixedLenFeature(
      [1], tf.int64)
  return feature_specs

def prediction_input_fn_builder(G_input, tokenizer, processor):
    def gen():
        while True:
            data = G_input.get()
            feature = preprocess.proc_single_sentence(data, 256, tokenizer, processor)
            dic = feature.get_dict_features()
            yield {
                'input_ids':[[int(i) for i in dic['input_ids'].int64_list.value]],
                'input_mask':[[int(i) for i in dic['input_mask'].int64_list.value]],
                'input_type_ids':[[int(i) for i in dic['input_type_ids'].int64_list.value]],
                'label_ids':[[int(dic['label_ids'].int64_list.value[0])]],
            }
            '''
            feature = preprocess.proc_single_sentence(data, 256, tokenizer, processor)
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=feature.get_dict_features()))
            d = tf_example.SerializeToString()
            d = _decode_record(d, get_sup_feature_specs())
            print(d)
            yield d
            '''

    def input_fn(params):
        return (tf.data.Dataset.from_generator(gen,
            output_types={'input_ids':tf.int32, 'input_mask':tf.int32, 'input_type_ids':tf.int32, 'label_ids':tf.int32},
            output_shapes={'input_ids':(None, conf["max_seq_length"]), 'input_mask':(None, conf['max_seq_length']), 'input_type_ids': (None, conf['max_seq_length']), 'label_ids': (None, 1)}).
            prefetch(10))

    return input_fn

class Get_UDA(object):
    def __init__(self, conf, tokenizer, processor):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.conf = conf
        self.ckpt_path = tf.train.latest_checkpoint(self.conf['model_dir'])
        print(self.ckpt_path)
        self.processor = raw_data_utils.get_processor(conf['task_name'])
        # for example, IMDB processor returns dataset as ['pos', 'neg']
        label_list = self.processor.get_labels()

        self.bert_config = modeling.AlbertConfig.from_json_file(
            conf['bert_config_file'],)

        tf.gfile.MakeDirs(conf['model_dir'])

        tf.logging.info("warmup steps {}/{}".format(
            conf['num_warmup_steps'], conf['num_train_steps']))

        tpu_cluster_resolver = None

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=conf['model_dir'],
            save_checkpoints_steps=500,
            keep_checkpoint_max=1000,
            # train_distribute=train_distribute,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=conf['iterations_per_loop'],
                per_host_input_for_training=is_per_host))

        model_fn = uda.model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=conf['init_checkpoint'],
            learning_rate=conf['learning_rate'],
            clip_norm=conf['clip_norm'],
            num_train_steps=conf['num_train_steps'],
            num_warmup_steps=conf['num_warmup_steps'],
            use_tpu=False,
            use_one_hot_embeddings=conf['use_one_hot_embeddings'],
            label_list=label_list,
            num_labels=len(label_list),
            unsup_ratio=conf['unsup_ratio'],
            uda_coeff=conf['uda_coeff'],
            tsa=conf['tsa'],
            print_feature=False,
            print_structure=False,
        )

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            params={"model_dir": conf['model_dir']},
            train_batch_size=conf['train_batch_size'],
            eval_batch_size=conf['eval_batch_size'])

        self.q = Queue()

        self.predictor = self.estimator.predict(
            prediction_input_fn_builder(self.q, tokenizer, processor),
            checkpoint_path=self.ckpt_path,
        )

    def predict(self, str):
        try:
            counter=timeCounter('predict stage')
            counter()
            self.q.put(str)
            predict_result = self.predictor.next()
            counter()
        except Exception as e:
            print(repr(e))
        return predict_result

if __name__ == "__main__":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=conf['vocab_file'], do_lower_case=True)
    processor = raw_data_utils.get_processor(conf['task_name'])
    tokenizer = tokenization.FullTokenizer(
        vocab_file=conf['vocab_file'], do_lower_case=True)
    uda = Get_UDA(conf,tokenizer,processor)
    tf.logging.info("***** Running prediction *****")
    str = '股指期货就像孙悟空。人跟神，不能做亲密的小伙伴，实在不好愉快的玩耍。'
    predict_result = uda.predict(str)
    # for key in predict_result.keys():
    #     predict_result[key] = predict_result[key].item()
    p = predict_result
    print(type(p['probabilities']))
    print(type(list(p['probabilities'])))
    # for p in predict_result:
    #     print(p)
    #     break
    # print(predict_result)
