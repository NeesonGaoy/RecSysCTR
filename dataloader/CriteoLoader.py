'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 11:49:48
LastEditTime: 2021-09-22 11:51:28
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/dataloader/CriteoLoader.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8
import os
import json
import glob

import numpy as np
import tensorflow as tf
import torch

class CriteoData():
    def __init__(self, data_path):
        self.SAMPLES = 1
        self.FIELDS = 39
        self.data_path = data_path

    def get_data(self, data_type, batch_size, epoch):
        def read_data(raw_rec):
            feature_des = {
                "feature": tf.io.FixedLenFeature([self.SAMPLES * self.FIELDS], tf.int64),
                "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            }
            example = tf.io.parse_single_example(raw_rec, feature_des)
            inputs = {}
            inputs["ids"] = example["feature"]
            inputs["label"] = example["label"]
            return inputs
        # 
        def reshape(inputs):
            inputs["ids"] = tf.reshape(inputs["ids"], [-1, self.FIELDS])
            inputs["label"] = tf.reshape(inputs["label"], [-1, ])
            return inputs
        # 
        file_name_pattern = os.path.join(self.data_path, "{}*".format(data_type))
        print("file_name_pattern: {} ".format( file_name_pattern ))
        files = glob.glob(file_name_pattern)
        print("file: {} ".format( files ))
        dataset = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=20).batch(
                            batch_size, drop_remainder=True).repeat(epoch).prefetch(buffer_size=20)
        step = 0
        for inputs in dataset:
            step += 1
            inputs["ids"] = torch.from_numpy( inputs["ids"].numpy() ) 
            inputs["label"] = torch.from_numpy( inputs["label"].numpy() )
            yield (step, inputs)
