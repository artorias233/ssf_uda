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


from keras import layers

import numpy as np
np.random.seed(1337)

import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from utils.csv_process import *
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

train_batch_size = FLAGS.train_batch_size
eval_batch_size = FLAGS.eval_batch_size
epochs = FLAGS.epochs
num_features= 768

def print_metric(model, eval_info, metrics):
    epsilon = 1e-7
    eval_sum = 0
    ma_p = 0
    ma_r = 0
    ma_f1 = 0
    w_p = 0
    w_r = 0
    w_f1 = 0
    print("***** Final evaluation result *****")
    print("Best model: {}".format(model))
    print("\tlabel\tprecision\trecall\tf1-score\tsize")
    label_size = len(eval_info)
    for x in eval_info.items():
        eval_sum += x[1]
    for i in range(label_size):
        label = eval_info.items()[i][0]
        size = eval_info.items()[i][1]
        p = metrics['precision_{}'.format(label)]
        r = metrics['recall_{}'.format(label)]
        f1 = 2 * p * r / (p + r + epsilon)
        # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        ma_p += p*(1.0/label_size)
        ma_r += r*(1.0/label_size)
        ma_f1 += f1*(1.0/label_size)
        w_p += p*(size*1.0/eval_sum)
        w_r += r*(size*1.0/eval_sum)
        w_f1 += f1*(size*1.0/eval_sum)
        # print(type(label),type(p),type(r),type(f1),type(size) )
        print("\t{}\t{:.3f}   \t{:.3f}\t{:.3f}   \t{}".format(label, p, r, f1, size))
    print("")
    print("Macro avg:\t{:.3f}   \t{:.3f}\t{:.3f}   \t{}".format(ma_p, ma_r, ma_f1, eval_sum))
    print("Weighted avg:\t{:.3f}   \t{:.3f}\t{:.3f}   \t{}".format(w_p, w_r, w_f1, eval_sum))
    print("Acc:   \t{:.3f}".format(metrics["eval_classify_accuracy"]))

def predict_data_generator(df, batch_size,batches_per_epoch, num_features):
    x_list= []
    for emb in df['emb_list']:
        x_list.append(json.loads(emb))
    while True:
        for b in range(batches_per_epoch):
            timesteps = len(max(x_list[:(b + 1) * batch_size][-batch_size:], key=len))
            x_train = np.full((batch_size, timesteps, num_features), -99.)
            for i in range(batch_size):
                li = b * batch_size + i
                x_train[i, 0:len(x_list[li]), :] = x_list[li]
            yield x_train

def data_generator(df, labels, batch_size,batches_per_epoch, num_features):
    x_list= []
    y_list = []
    for emb in df['emb_list']:
        x_list.append(json.loads(emb))
    for label in df['label']:
        y_list.append(label)
    label_map = {}
    for (i, label) in enumerate(labels):
        # print("label = {}, type = {}".format(label, type(label)))
        label_map[label] = i
    for i in range(len(y_list)):
        y_list[i] = label_map[str(y_list[i])]
    print(y_list)
    # Generate batches
    while True:
        for b in range(batches_per_epoch):
            timesteps = len(max(x_list[:(b + 1) * batch_size][-batch_size:], key=len))
            x_train = np.full((batch_size, timesteps, num_features), -99.)
            y_train = np.zeros((batch_size,  1))
            for i in range(batch_size):
                li = b * batch_size + i
                x_train[i, 0:len(x_list[li]), :] = x_list[li]
                y_train[i] = y_list[li]
            yield x_train, y_train


def metric_fn(y_true, y_pred):
    print("y_true: {}, y_pred:{}".format(y_true, y_pred))

    acc = K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    print("acc :{}".format(acc))
    return acc

def label_id(pred):
    max_pro = -1
    label = 0
    for i in range(0, len(pred)):
        if pred[i] > max_pro:
            max_pro = pred[i]
            label = i
    return label


if __name__ == "__main__":
    if FLAGS.do_train:
        text_input = Input(shape=(None, 768,), dtype='float32', name='text')
        l_mask = layers.Masking(mask_value=-99.)(text_input)
        # Which we encoded in a single vector via a LSTM
        encoded_text = layers.LSTM(100, )(l_mask)
        out_dense = layers.Dense(30, activation='relu')(encoded_text)
        # And we add a softmax classifier on top
        out = layers.Dense(len(FLAGS.labels), activation='softmax')(out_dense)
        # At model instantiation, we specify the input and the output:
        model = Model(text_input, out)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])
        model.summary()

        df_train = read_csv(FLAGS.raw_data_dir, FLAGS.sup_train_file)
        df_val = read_csv(FLAGS.raw_data_dir, FLAGS.sup_dev_file)
        print(df_val.head(20))
        call_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=3, verbose=2,
                                        mode='auto', min_delta=0.01, cooldown=0, min_lr=0)

        train_total_steps = len(df_train['emb_list'])
        eval_total_steps = len(df_val['emb_list'])
        train_batches_per_epoch = train_total_steps // train_batch_size
        eval_batches_per_epoch = eval_total_steps // eval_batch_size

        # gen = data_generator(df_train, FLAGS.labels, train_batch_size, train_batches_per_epoch, num_features)
        # print(next(gen))
        print("===============================")
        print("LSTM Training,:")
        print("train total steps: {}".format(train_total_steps))
        print("train batches pre epoch: {}".format(train_batches_per_epoch))
        print("eval total steps: {}".format(eval_total_steps))
        print("eval batches pre epoch: {}".format(eval_batches_per_epoch))

        tf.gfile.MakeDirs(FLAGS.lstm_model_dir)
        checkpoint = ModelCheckpoint(os.path.join(FLAGS.lstm_model_dir,"best_model.h5"), monitor='loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1)
        model.fit_generator(data_generator(df_train,FLAGS.labels,train_batch_size,train_batches_per_epoch, num_features), steps_per_epoch=train_batches_per_epoch, epochs=epochs,
                            validation_data=data_generator(df_val,FLAGS.labels,eval_batch_size,eval_batches_per_epoch, num_features), validation_steps=eval_batches_per_epoch,
                            callbacks=[call_reduce,checkpoint])


    if FLAGS.do_eval:
        best_ckpt = 'best_model.h5'
        if FLAGS.do_train == False:
            model = load_model(os.path.join(FLAGS.lstm_model_dir,best_ckpt))
            df_val = read_csv(FLAGS.raw_data_dir, FLAGS.sup_dev_file)
        print("===============================")
        print("LSTM Evaluating,:")
        predict_labels_count = []
        TP_labels_count = []
        label_map = {}
        for (i, label) in enumerate(FLAGS.labels):
            label_map[label] = i

        eval_info = {}
        for i in range(len(FLAGS.labels)):
            print("df_val['label']: {}, FLAGS.labels[i]: {}".format(type(df_val['label']),type(FLAGS.labels[i]) ))
            eval_info[FLAGS.labels[i]] = df_val[df_val['label']==FLAGS.labels[i]].shape[0]
            predict_labels_count.append(0)
            TP_labels_count.append(0)

        for i in range(len(df_val)):
            single_data = df_val.iloc[i]
            # print(single_data)
            emb = [json.loads(single_data['emb_list'])]
            true_label = label_map[str(single_data['label'])]
            x = np.array(emb)
            probabilities = model.predict_on_batch(x)[0]
            predict_label = label_id(probabilities)
            predict_labels_count[predict_label] += 1
            if predict_label == true_label:
                TP_labels_count[true_label] += 1

        acc = sum(TP_labels_count)*1.0 / len(df_val)
        epsilon = 1e-7
        eval_metrics = {}
        eval_metrics["eval_classify_accuracy"] = acc
        for i in range(len(FLAGS.labels)):
            eval_metrics['precision_{}'.format(FLAGS.labels[i])] = TP_labels_count[i]*1.0 / (predict_labels_count[i] + epsilon)
            eval_metrics['recall_{}'.format(FLAGS.labels[i])] = TP_labels_count[i]*1.0 / (eval_info[FLAGS.labels[i]] + epsilon)

        print_metric(best_ckpt,eval_info,eval_metrics)
