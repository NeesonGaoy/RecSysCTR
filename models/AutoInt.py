'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 10:28:38
LastEditTime: 2021-09-22 10:28:38
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/models/AutoInt.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''

import torch


class AutoInt(torch.nn.Module):
    def __init__(self, with_residul, full_part, 
                       atte_emb_dim, num_heads, num_layers, atte_dropout):
        self.with_residul = with_residul
        self.full_part = full_part
        self.atte_emb_dim = atte_emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.atte_dropout = atte_dropout

        self.atte_output_dim = self.field_num * self.atte_emb_dim
        self.dnn_input_dim = self.field_num * self.emb_dim

        self.atten_embedding = torch.nn.Linear(self.emb_dim, self.atte_emb_dim)
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(self.atte_emb_dim, self.num_heads, dropout=self.atte_dropout) for _ in range(self.num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.with_residul:
            self.V_res_embedding = torch.nn.Linear(self.emb_dim, self.atte_emb_dim)

    def forward(self, input_x):
        xv = self.embedding(x)
        score = self.autoint_layer(x)
        if self.full_part:
            dnn_score = self.mlp.forward(xv.view(-1, self.embedding_output_dim))
            score = dnn_score + score
        return score.squeeze(1)

    def autoint_layer(self, xv):
        "Multi-head self-attention layer. "
        atten_x = self.atten_embedding(xv) # [bs, field, attn_dim]
        cross_term = atten_x.transpose(0, 1) # [field, bs, attn_dim]
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1) # [bs, field, dim]

        if self.with_residul:
            V_res = self.V_res_embedding(xv)
            cross_term += V_res
        cross_term = F.relu(cross_term).contigous().view(-1, self.atten_output_dim) # [bs, field * attn_dim]
        output = self.attn_fc(cross_term)
        return output
