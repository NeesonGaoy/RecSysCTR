'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 11:45:23
LastEditTime: 2021-09-22 11:47:04
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/main_run.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8
import os
import argparse

from dataloader.dataset_loader import CriteoData
from utils.trainer import Trainer, Trainer_Distillation

def init_args():
    parser = argparse.ArgumentParser(description="Run Torch Model on low-bit quantilization. ")
    parser.add_argument("--data_name", type=str, default="criteo678w", help="Dataset version.")
    parser.add_argument("--distributed", default=False, type=bool, help="distributed flag.")

    parser.add_argument("--emb_type", type=str, default="normal",  help="emb type for embedding table.")
    parser.add_argument("--emb_dim", type=int, default=64,  help="emb dim for ID embedding feature.")
    
    parser.add_argument("--quant_method", type=str,   default="uniform",  help="method name for quantilization.")
    parser.add_argument("--weight_bits",  type=int,   default=8,  help="compress bit for quantilization.")
    parser.add_argument("--layerwise",    type=bool,  default=False,  help="layerwise for params quantilization.")
    parser.add_argument("--learnable",    type=bool,  default=False,  help="learnable args for clip value in quantilization.")
    parser.add_argument("--symmetric",    type=bool,  default=False,  help="symmetric args for quantilization.")
    parser.add_argument("--clip_val",     type=float, default=2.5,  help="clip value for parmas masking.")
    # quant_method,    weight_bits,   layerwise,    learnable(clip_val),  symmetric
    #    "bwn",           1,          True/False 
    #    "twn",           2,          True/False
    #  "uniform",        >=2,         True/False,     True/False,         True/False,
    #    "lsq",          >=2,         True/False,        True,            True/False
    #    "laq",          >=1,         True/False
    
    parser.add_argument("--with_distillation", type=bool,  default=False,  help="with_distillation for quantilization.")
    parser.add_argument("--model_name", type=str,   default="dcn",  help="model name.")
    parser.add_argument("--mlp_dims",   type=list,  default=[1024, 512, 256, 128],  help="dense dim for mlp.") # nargs="+", 1个或多个;  nargs="*", 0个或多个;
    parser.add_argument("--dropout",    type=float, default=0.2,    help="dropout rate for dense layer.")
    parser.add_argument("--num_cross",  type=float, default=3,      help="num of cross layer in DCN. ")
 
    parser.add_argument("--batch_size",  type=int, default=2048,  help="batch size for mini-batch training.")

    parser.add_argument("--l2_sparse", type=float, default=5e-4,  help="l2_sparse for ID embedding feature.")
    parser.add_argument("--l2_dense",  type=float, default=5e-4,  help="l2_dense coef for dense parameters.")

    parser.add_argument("--opt_name",   type=str,    default="adam",  help="optimizer name for updating params.")
    parser.add_argument("--lr",         type=float,    default=5e-5,  help="learning rate for optimizer.")
    parser.add_argument("--rmsp_alpha", type=float,    default=0.01,  help="alpha args for rmsp optimizer.")
    parser.add_argument("--rmsp_momentum", type=float, default=0.001,  help="momentum args for rmsp optimizer.")

    parser.add_argument("--log_path", type=str, default="./logs/",  help="Saved log file path for model training. ")
    parser.add_argument("--log_name", type=str, default="base_model_info",  help="log name for base model.")

    parser.add_argument("--load_pretrain", type=bool, default=False,  help="flag for loading pretrain model.")
    parser.add_argument("--pretrain_path", type=str, default="./",  help="saved path for pretrain model.")
    parser.add_argument("--pretrain_model_name", type=str, default="pretrain_model.pt",  help="saved name for pretrain model.")
    
    args, unparsed = parser.parse_known_args()
    return args


def main():
    args = init_args()

    if args.data_name == "criteo678w":
        data_config = {
            "data_path": "/Users/gaoyong/Desktop/low_bit_quantilizer_training/process_criteo/criteo_tfreocd_threshold1/",
            "id_vocab_size": 678038,
            "field_num": 39,
            "field_feat_nums": [57, 107, 127, 49, 224, 138, 94, 80, 100, 10, 
                                31, 52, 83, 1461, 570, 1545893, 668222, 306, 25, 12399, 
                                634, 4, 74829, 5557, 1518204, 3195, 28, 13883, 1231447, 11, 
                                5310, 2151, 5, 1437678, 19, 16, 157463, 104, 99817],

            "train_size": 350000,
        }
        
        data_loader = CriteoData(data_config["data_path"])
    elif args.data_name == "criteo20w":
        data_config = {
            "data_path": "/Users/gaoyong/Desktop/low_bit_quantilizer_training/process_criteo/criteo_tfreocd_threshold100/",
            "id_vocab_size": 187135,
            "field_num": 39,
            "field_feat_nums": [40, 93, 115, 32, 213, 96, 73, 47, 85, 7, 
                                27, 32, 43, 700, 539, 21054, 23838, 181, 14, 10102, 
                                350, 3, 16432, 4495, 21497, 3102, 26, 6966, 22553, 10, 
                                3275, 1612, 4, 21967, 13, 14, 15130, 60, 12295],
            
            "train_size": 400000,
        }
        data_loader = CriteoData(data_config["data_path"])
    # elif args.data_name == "avazu":
    else:
        raise ValueError("Not contains the {} dataset. ".format(args.data_name))
    
    model_config = {
        "emb_type": args.emb_type,  # ["normal", "quant"]
        "emb_dim": args.emb_dim,
        "quant_config": {
            "quant_method": args.quant_method,
            "weight_bits": args.weight_bits,
            "layerwise": args.layerwise,
            "learnable": args.learnable,
            "symmetric": args.symmetric,
            "clip_val": args.clip_val,
        },
        
        "regions_dims": [0, 1, 2, 4, 8],
        "temperature": 0.5,
        
        "model_name": args.model_name, # ["deepfm", "dcn"]
        "mlp_dims": args.mlp_dims,
        "dropout": args.dropout,
        "num_cross": args.num_cross,
    }
    eval_steps = int(data_config["train_size"] / args.batch_size) // 5
    early_stop_steps = eval_steps * 10
    training_config = {
        "with_distillation": args.with_distillation,

        "batch_size": args.batch_size,
        "print_steps": 100, 
        "eval_steps": eval_steps,
        "early_stop_steps": early_stop_steps,

        "l2_sparse": args.l2_sparse,
        "l2_dense": args.l2_dense,

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
    if training_config["with_distillation"]:
        trainer = Trainer_Distillation(data_loader,
                                       data_config, 
                                       model_config, 
                                       training_config,
                                       use_cuda=False)
        trainer.train(max_epoch=20)
    else:
        trainer = Trainer(data_loader,
                          data_config, 
                          model_config, 
                          training_config,
                          use_cuda=False)
        trainer.train(max_epoch=20)

if __name__ == "__main__":

    main()
