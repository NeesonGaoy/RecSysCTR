'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 09:21:53
LastEditTime: 2021-09-22 17:43:35
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/models/DCN.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8

import torch

from layers.core import CrossNet, MLP
from layers.utils_quant import QuantizeEmbedding


class DCN(torch.nn.Module):
    def __init__(self, field_num, id_vocab_size, emb_dim, emb_type, 
                       mlp_dims, dropout, num_cross, quant_config):
        super(DCN, self).__init__()
        self.field_num = field_num
        self.id_vocab_size = id_vocab_size
        self.emb_dim = emb_dim
        self.emb_type = emb_type
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.num_cross = num_cross
        self.quant_config = quant_config
        self.embedding_output_dim = self.field_num * self.emb_dim
        # 
        if self.emb_type == "quant":
            self.embedding = QuantizeEmbedding(self.id_vocab_size, self.emb_dim,
                                    clip_val=self.quant_config["clip_val"], # 2.5
                                    weight_bits=self.quant_config["weight_bits"],
                                    learnable=self.quant_config["learnable"],
                                    symmetric=self.quant_config["symmetric"],
                                    embed_layerwise=self.quant_config["layerwise"],
                                    weight_quant_method=self.quant_config["quant_method"])
        else:
            self.embedding = torch.nn.Embedding(self.id_vocab_size, self.emb_dim, sparse=False)
        torch.nn.init.uniform_(self.embedding.weight, a=-0.05, b=0.05)
        
        self.mlp = MLP(self.embedding_output_dim, self.mlp_dims, dropout=dropout)
        self.cross_net = CrossNet(input_dim=self.embedding_output_dim, num_layer=self.num_cross)
        
        self.predict_dense = torch.nn.Linear(self.mlp_dims[-1] + self.embedding_output_dim, 1)
        
    def forward(self, feat_ids):
        id_emb = self.embedding(feat_ids)
        id_emb_flatten = id_emb.view(-1, self.embedding_output_dim)

        dnn_out = self.mlp.forward(id_emb_flatten)
        cross_out = self.cross_net.forward(id_emb_flatten)
        
        out = torch.cat([dnn_out, cross_out], dim=1)
        logits = self.predict_dense(out)
        logits = logits.squeeze(1)
        
        out_dict = {
            "id_emb": id_emb_flatten,
            "dnn_out": dnn_out,
            "cross_out": cross_out
        }
        return logits, out_dict

