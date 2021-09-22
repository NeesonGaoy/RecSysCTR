'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 11:45:23
LastEditTime: 2021-09-22 17:49:42
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/main_run.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8
import os
import yaml
import argparse

from train.utils import load_config
from dataloader.CriteoLoader import CriteoData
from train.trainer import Trainer, Trainer_Distillation

def init_args():
    parser = argparse.ArgumentParser(description="Run Torch Model on low-bit quantilization. ")
    
    parser.add_argument("--config_path", default="/Users/gaoyong/Desktop/RecSysCTR/", type=str, help="config_path for data, model, training.")
    parser.add_argument("--data_config", default="data_config/criteo18w.yaml", type=str, help="data_config_files.")
    parser.add_argument("--model_config", default="model_config/DCN.yaml", type=str, help="model_config files.")
    parser.add_argument("--train_config", default="train_config/criteo678w_dcn.yaml", type=str, help="train_config files.")
    
    parser.add_argument("--batch_size",  type=int, default=2048,  help="batch size for mini-batch training.")

    parser.add_argument("--opt_name",   type=str,    default="adam",  help="optimizer name for updating params.")
    parser.add_argument("--lr",         type=float,    default=5e-5,  help="learning rate for optimizer.")
    parser.add_argument("--rmsp_alpha", type=float,    default=0.01,  help="alpha args for rmsp optimizer.")
    parser.add_argument("--rmsp_momentum", type=float, default=0.001,  help="momentum args for rmsp optimizer.")

    parser.add_argument("--log_path", type=str, default="./logs/",  help="Saved log file path for model training. ")
    parser.add_argument("--log_name", type=str, default="base_model_info",  help="log name for base model.")
    
    parser.add_argument("--with_distillation", type=bool,  default=False,  help="with_distillation for quantilization.")
    parser.add_argument("--load_pretrain", type=bool, default=False,  help="flag for loading pretrain model.")
    parser.add_argument("--pretrain_path", type=str, default="./",  help="saved path for pretrain model.")
    parser.add_argument("--pretrain_model_name", type=str, default="pretrain_model.pt",  help="saved name for pretrain model.")
    
    args, unparsed = parser.parse_known_args()
    return args


def main():
    args = init_args()

    assert args.data_config is not None, "data_config: {} ".format(args.data_config)
    assert args.model_config is not None, "model_config: {} ".format(args.model_config)
    assert args.train_config is not None, "train_config: {} ".format(args.train_config)
    
    data_config = load_config(os.path.join(args.config_path, args.data_config))
    model_config = load_config(os.path.join(args.config_path, args.model_config))
    train_config = load_config(os.path.join(args.config_path, args.train_config))

    if data_config["data_name"] == "criteo678w":
        data_loader = CriteoData(data_config["data_path"])
    elif data_config["data_name"] == "criteo18w":
        data_loader = CriteoData(data_config["data_path"])
    # elif data_config["data_name"]  == "avazu":
    else:
        raise ValueError("Not contains the {} dataset. ".format(data_config["data_name"] ))
    
    eval_steps = int(data_config["train_size"] / args.batch_size) // 5
    early_stop_steps = eval_steps * 10
    train_base_config = {
        "with_distillation": args.with_distillation,
        
        "batch_size": args.batch_size,
        "print_steps": 100, 
        "eval_steps": eval_steps,
        "early_stop_steps": early_stop_steps,
        
        "opt_name": args.opt_name,  # ["sgd", "adam", "rmsprop"]
        "lr": args.lr,
        "rmsp_alpha": args.rmsp_alpha,
        "rmsp_momentum": args.rmsp_momentum,
        
        "log_path": args.log_path,
        "log_name": args.log_name,

        "load_pretrain": args.load_pretrain,
        "pretrain_path": args.pretrain_path,
        "pretrain_model_name": args.pretrain_model_name,
    }
    
    for key,val in train_base_config.items():
        if key not in train_config:
            train_config[ key ] = val

    if train_config["with_distillation"]:
        trainer = Trainer_Distillation(data_loader,
                                       data_config, 
                                       model_config, 
                                       train_config,
                                       use_cuda=False)
        trainer.train(max_epoch=20)
    else:
        trainer = Trainer(data_loader,
                          data_config, 
                          model_config, 
                          train_config,
                          use_cuda=False)
        trainer.train(max_epoch=20)
    


if __name__ == "__main__":

    main()
