'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 09:23:31
LastEditTime: 2021-09-22 09:51:31
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/models/DeepFM.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8

import torch 

from layers.core import CrossNet, MLP
from utils_quant import QuantizeEmbedding

class DeepFM(torch.nn.Module):
    def __init__(self, field_num, id_vocab_size, emb_dim, 
                        emb_type, mlp_dims, dropout, quant_config):
        super(DeepFM, self).__init__()
        self.field_num = field_num
        self.id_vocab_size = id_vocab_size
        self.emb_dim = emb_dim
        self.emb_type = emb_type
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.quant_config = quant_config
        self.embedding_output_dim = self.field_num * self.emb_dim

        if self.emb_type == "quant":
            self.embedding = QuantizeEmbedding(self.id_vocab_size, self.emb_dim,
                                    clip_val=2.5,
                                    weight_bits=self.quant_config["weight_bits"],
                                    learnable=self.quant_config["learnable"],
                                    symmetric=self.quant_config["symmetric"],
                                    embed_layerwise=self.quant_config["layerwise"],
                                    weight_quant_method=self.quant_config["quant_method"])
        else:
            self.embedding = torch.nn.Embedding(self.id_vocab_size, self.emb_dim, sparse=False)
        torch.nn.init.uniform_(self.embedding.weight, a=-0.05, b=0.05)

        self.linear = LR(self.id_vocab_size)
        self.fm = FM(reduce_sum=True)
        self.mlp = MLP(self.embedding_output_dim, self.mlp_dims, dropout=dropout)
        self.predict_dense = torch.nn.Linear(self.mlp_dims[-1], 1)
        # 
    def forward(self, x):
        id_emb = self.embedding(x)
        id_emb_flatten = id_emb.view(-1, self.embedding_output_dim)

        dnn_out = self.mlp.forward(id_emb_flatten)
        dnn_score = self.predict_dense(dnn_out)

        linear_score = self.linear.forward(x)
        fm_score = self.fm.forward(id_emb)

        logits = linear_score + fm_score + dnn_score
        logits = logits.squeeze(1)
        out_dict = {
            "id_emb": id_emb_flatten,
            "dnn_out": dnn_out
        }
        return logits, out_dict



