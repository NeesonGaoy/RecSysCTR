'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 11:31:01
LastEditTime: 2021-09-22 17:54:09
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/train/utils.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8

import os
import sys
import yaml
import json
import logging

import torch
from models import AutoInt, DeepFM, DCN, IRazorDNN

class LoggerHelper():
    def __init__(self, log_name="example"):
        self.log_name = log_name
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(logging.DEBUG)
        # self.__setFormatter()
        self.__addStreamHandler()
    def __setFormatter(self):
        fmt = "%(asctime)-15s %(levelname)s %(filename)s line:%(lineno)d pid:%(process)d %(message)s"
        datefmt = "%Y-%m-%d %H-%M-%S"
        self.formatter = logging.Formatter(fmt, datefmt) 
    def __addStreamHandler(self):
        sh = logging.StreamHandler(stream=None)
        sh.setLevel(logging.DEBUG)
        fmt = "%(asctime)-15s %(message)s"
        datefmt = "%Y-%m-%d %H-%M-%S"
        formatter = logging.Formatter(fmt, datefmt)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
    def addFileHandler(self, log_path, log_name):
        os.makedirs(log_path, exist_ok=True)
        log_file_path = os.path.join(log_path, log_name)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        fmt = "%(asctime)-15s %(message)s"
        datefmt = "%Y-%m-%d %H-%M-%S"
        formatter = logging.Formatter(fmt, datefmt)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

def load_config(config_file_path):
    with open(config_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def get_model(model_name, data_config, model_config):
    if model_name == "DeepFM":
        model = DeepFM(field_num=data_config["field_num"], 
                       id_vocab_size=data_config["id_vocab_size"], 
                       emb_dim=model_config["emb_dim"], 
                       emb_type=model_config["emb_type"], 
                       mlp_dims=model_config["mlp_dims"], 
                       dropout=model_config["dropout"], 
                       quant_config=model_config["quant_config"])
    elif model_name == "DCN":
        model = DCN(field_num=data_config["field_num"], 
                    id_vocab_size=data_config["id_vocab_size"], 
                    emb_dim=model_config["emb_dim"], 
                    emb_type=model_config["emb_type"], 
                    mlp_dims=model_config["mlp_dims"], 
                    dropout=model_config["dropout"], 
                    num_cross=model_config["num_cross"],
                    quant_config=model_config["quant_config"])
    elif model_name == "IRazorDNN":
        model = IRazorDNN(field_num=data_config["field_num"], 
                          id_feat_size=data_config["id_vocab_size"], 
                          field_feat_nums=data_config["field_feat_nums"],
                          regions_dims=model_config["regions_dims"], 
                          temperature=model_config["temperature"],

                          mlp_dims=model_config["mlp_dims"], 
                          dropout=model_config["dropout"])
    else:
        raise ValueError("Invalid model type: {} ".format(model_name))
    return model 

def get_optimizer(model, config):
    embedding_params = []
    cls_params = []
    for name,param in model.named_parameters():
        if name.startswith("embedding"):
            embedding_params.append(param)
        else:
            cls_params.append(param)
    # 
    embedding_group = {
        "params": embedding_params,
        "weight_decay": config["l2_sparse"],
    }
    cls_group = {
        "params": cls_params,
        "weight_decay": config["l2_dense"],
    }
    if config["opt_name"] == "sgd":
        opt = torch.optim.SGD([embedding_group, cls_group], lr=config["lr"])
    elif config["opt_name"] == "adam":
        opt = torch.optim.Adam([embedding_group, cls_group], lr=config["lr"])
    elif config["opt_name"] == "rmsprop":
        opt = torch.optim.RMSprop([embedding_group, cls_group], 
                lr=config["lr"], alphha=config["rmsp_alpha"], momentum=config["rmsp_momentum"])
    return opt

def get_log(log_path, log_name):
    log_helper = LoggerHelper(log_name=log_name)
    log_helper.addFileHandler(log_path=log_path, log_name=log_name + ".log")
    logger = log_helper.logger
    return logger

