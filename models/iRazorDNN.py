'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 10:29:08
LastEditTime: 2021-09-22 10:36:34
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/models/iRazorDNN.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8
import torch

from layers.core import CrossNet, MLP
from layers.embeddings import IRazorEmbedding, SubRegionEmbedding

class IRazorDNN(torch.nn.Module):
    def __init__(self, field_num, id_feat_size, field_feat_nums, regions_dims, temperature,
                       mlp_dims, dropout,
                       emb_type="iRazor",
                       ):
        super(IRazorDNN, self).__init__()
        self.field_num = field_num
        self.id_feat_size = id_feat_size
        self.field_feat_nums = field_feat_nums
        self.regions_dims = regions_dims
        self.temperature = temperature

        self.mlp_dims = mlp_dims
        self.dropout = dropout
        if emb_type == "iRazor":
            self.embedding = IRazorEmbedding(field_num, id_feat_size, field_feat_nums, 
                                             regions_dims, temperature)
            self.embedding_output_dim = self.embedding.all_dim * self.field_num
        elif emb_type == "subRegion":
            self.embedding = SubRegionEmbedding(field_num, id_feat_size, field_feat_nums, 
                                                regions_dims, temperature)
            self.embedding_output_dim = sum(regions_dims) * self.field_num
        else:
            self.embedding = torch.nn.Embedding(self.id_vocab_size, sum(regions_dims), sparse=False)
            torch.nn.init.uniform_(self.embedding.weight, a=-0.05, b=0.05)
            self.embedding_output_dim = sum(regions_dims) * self.field_num
        # 

        self.mlp = MLP(self.embedding_output_dim, mlp_dims, dropout=dropout)
        self.predict_dense = torch.nn.Linear(self.mlp_dims[-1], 1)
        
    def forward(self, input_ids): 
        id_emb = self.embedding(input_ids)
        id_emb_flatten = torch.reshape(id_emb, [-1, self.embedding_output_dim])
        dnn_out = self.mlp(id_emb_flatten)
        logits = self.predict_dense(dnn_out)
        logits = torch.squeeze(logits)

        out_dict = {
            "id_emb": id_emb,
            "dnn_out": dnn_out,
        }
        return logits, out_dict
    
    def get_embedding_reg_loss(self):
        return self.embedding.reg_loss()

    def cpt_pruning(self, cpt=0.7):
        return self.embedding.cpt_pruning(cpt=cpt)
