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
"""Load raw data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import csv
import os

from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label



class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, raw_data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, raw_data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  def get_train_size(self):
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None, delimiter=","):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = pd.read_csv(input_file, delimiter=delimiter,
                                error_bad_lines=False,lineterminator='\n')
      # lines = []
      # for index, line in reader.iterrows():
      #   # print(line)
      #   lines.append(list(line))
      # return lines
      return reader


def clean_web_text(st):
  """clean text."""
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



class CFFEXSentProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "train.csv"),
                           ), "train")

    def get_dev_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "eval.csv"),
                           quotechar='"'), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_ext":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.txt"),
                               quotechar='"'), "unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "train.csv"),
                               quotechar='"'), "unsup_in", skip_unsup=False)

    def get_labels(self):
        """See base class."""
        return ["A","B","C"]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if skip_unsup and line[1] == "unsup":
                continue
            if line[1] == "unsup" and len(line[0]) < 500:
                # tf.logging.info("skipping short samples:{:s}".format(line[0]))
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            text_a = clean_web_text(text_a)
            # tf.logging.info("text_a = {}, label = {}".format(text_a, label))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_train_size(self):
        return 25000

    def get_dev_size(self):
        return 25000

class THUCProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""
    FILE_BASE = "/home/FudanCsLab/zhilong/uda/text/data/THUCNews/csv"
    def get_train_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "train_sup_0.24.csv"),
                           ), "train")

    def get_dev_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "eval_0.02.csv"),
                           quotechar='"'), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_ext":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.txt"),
                               quotechar='"'), "unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "train_unsup_0.56.csv"),
                               quotechar='"'), "unsup_in", skip_unsup=False)

    def get_labels(self):
        """See base class."""
        return ["娱乐","家居","科技","体育","社会","时政","彩票","股票","教育","游戏","时尚","房产","财经","星座"]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if skip_unsup and line[1] == "unsup":
                continue
            if line[1] == "unsup" and len(line[0]) < 500:
                # tf.logging.info("skipping short samples:{:s}".format(line[0]))
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            text_a = clean_web_text(text_a)
            # tf.logging.info("text_a = {}, label = {}".format(text_a, label))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    # def get_train_size(self):
    #     return 25000
    #
    def get_dev_size(self):
        eval = pd.read_csv(os.path.join(self.FILE_BASE, "eval_0.02.csv"), delimiter='\t',
                                error_bad_lines=False)
        print("eval size:{}".format(eval.shape[0]))
        return eval.shape[0]




class CFFEXThemeProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""
    def get_train_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "train_0.8.csv"),
                           ), "train")

    def get_dev_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "eval_0.2.csv"),
                           quotechar='"'), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_ext":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.txt"),
                               quotechar='"'), "unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup.csv"),
                               quotechar='"'), "unsup_in", skip_unsup=False)
    def get_eval_info(self):
        ROOT_PATH = os.getcwd()
        raw_data_dir = os.path.join(ROOT_PATH, "data/CFFEX/theme")
        label_name = "answercontent"
        result = {}
        data = pd.read_csv(os.path.join(raw_data_dir, "eval_0.2.csv"), delimiter=",", error_bad_lines=False)
        label_list = data[label_name].unique()
        for label in label_list:
            result[label] = data[data[label_name] == label].shape[0]
        return result
    
    def get_labels(self):
        """See base class."""
        return ["A", "B"]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if skip_unsup and line[1] == "unsup":
            #     continue
            # if line[1] == "unsup" and len(line[0]) < 500:
            #     # tf.logging.info("skipping short samples:{:s}".format(line[0]))
            #     continue
            if set_type == "unsup_in" :
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
                text_a = clean_web_text(text_a)
                # tf.logging.info("text_a = {}, label = {}".format(text_a, label))
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            else:
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                label = line[2]
                text_a = clean_web_text(text_a)
                # tf.logging.info("text_a = {}, label = {}".format(text_a, label))
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    # def get_train_size(self):
    #     return 25000
    #
    def get_dev_size(self):
        ROOT_PATH = os.getcwd()
        raw_data_dir = os.path.join(ROOT_PATH, "data/CFFEX/theme")
        eval = pd.read_csv(os.path.join(raw_data_dir, "eval_0.2.csv"), delimiter='\t',
                                error_bad_lines=False)
        print("eval size:{}".format(eval.shape[0]))
        return eval.shape[0]

class CFFEXIllegalProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""
    FILE_BASE = os.getcwd()
    ROOT_PATH = os.path.dirname(FILE_BASE)
    FILE_BASE = os.path.join(ROOT_PATH,"data/CFFEX/illegal")
    def get_train_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "train_0.8.csv"),
                           ), "train")

    def get_dev_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "eval_0.2.csv"),
                           quotechar='"'), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_ext":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.txt"),
                               quotechar='"'), "unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup.csv"),
                               quotechar='"'), "unsup_in", skip_unsup=False)

    def get_eval_info(self):
        raw_data_dir = "/home/xxc/text/text/data/CFFEX/illegal"
        label_name = "illegal_type"
        result = {}
        data = pd.read_csv(os.path.join(raw_data_dir, "eval_0.2.csv"), delimiter=",", error_bad_lines=False)
        label_list = data[label_name].unique()
        for label in label_list:
            result[label] = data[data[label_name] == label].shape[0]
        return result

    def get_labels(self):
        """See base class."""
        return [0,1,2,3,4,5,6,7]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if skip_unsup and line[1] == "unsup":
            #     continue
            # if line[1] == "unsup" and len(line[0]) < 500:
            #     # tf.logging.info("skipping short samples:{:s}".format(line[0]))
            #     continue
            if set_type == "unsup_in" :
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
                text_a = clean_web_text(text_a)
                # tf.logging.info("text_a = {}, label = {}".format(text_a, label))
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            else:
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                label = line[3]
                text_a = clean_web_text(text_a)
                # tf.logging.info("text_a = {}, label = {}".format(text_a, label))
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    # def get_train_size(self):
    #     return 25000
    #
    def get_dev_size(self):
        eval = pd.read_csv(os.path.join(self.FILE_BASE, "eval_0.2.csv"), delimiter='\t',
                                error_bad_lines=False)
        print("eval size:{}".format(eval.shape[0]))
        return eval.shape[0]



class CFFEXProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self ,labels, raw_data_dir="", sup_train_file="train_0.8.csv", sup_dev_file="eval_0.2.csv",unsup_file = "unsup.csv"):
        self.sup_train_file = sup_train_file
        self.sup_dev_file = sup_dev_file
        self.unsup_file = unsup_file
        self.raw_data_dir = raw_data_dir
        self.labels = labels

    def get_train_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, self.sup_train_file),
                           ), "train")

    def get_dev_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, self.sup_dev_file),
                           quotechar='"'), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, self.unsup_file),
                               quotechar='"'), "unsup_in", skip_unsup=False)

    def get_eval_info(self):
        result = {}
        data = pd.read_csv(os.path.join(self.raw_data_dir, self.sup_dev_file), delimiter=",", error_bad_lines=False)
        label_list = data['label'].unique()
        for label in label_list:
            result[label] = data[data['label'] == label].shape[0]
        return result

    def get_labels(self):
        """See base class."""

        return self.labels

    def get_split(self, text, chunk_size, overlap):
        l_total = []
        valid_len = chunk_size - overlap
        if len(text)//valid_len > 0:
            chunk_num = len(text)//valid_len
        else:
            chunk_num = 1
        for w in range(chunk_num):
            if w == 0:
              l_parcial = text[:chunk_size].encode('utf-8')
              l_total.append(l_parcial)
            else:
              l_parcial = text[w*valid_len:w*valid_len + chunk_size].encode('utf-8')
              # print(l_parcial)
              # print("\n")
              l_total.append(l_parcial)
        # print(l_total)
        return l_total

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            # print(content)
            # print(lines.iloc[i])
            # if skip_unsup and line[1] == "unsup":
            #     continue
            # if line[1] == "unsup" and len(line[0]) < 500:
            #     # tf.logging.info("skipping short samples:{:s}".format(line[0]))
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = lines.iloc[i]['content']
            label = lines.iloc[i]['label']
            text_a = clean_web_text(text_a)
            # tf.logging.info("text_a = {}, label = {}".format(text_a, label))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    # def get_train_size(self):
    #     return 25000
    #
    def get_dev_size(self):
        eval = pd.read_csv(os.path.join(self.raw_data_dir, self.sup_dev_file), delimiter='\t',
                           error_bad_lines=False)
        print("eval size:{}".format(eval.shape[0]))
        return eval.shape[0]

def get_processor(task_name):
  """get processor."""
  task_name = task_name.lower()
  processors = {
      "cffexsent": CFFEXSentProcessor,
      "cffextheme": CFFEXThemeProcessor,
      "cffexillegal": CFFEXIllegalProcessor,
      "thuc": THUCProcessor,
  }
  processor = processors[task_name]()
  return processor

