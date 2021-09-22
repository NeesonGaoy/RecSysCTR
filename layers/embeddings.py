'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-18 14:45:19
LastEditTime: 2021-09-22 10:36:44
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/layers/embeddings.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8
import torch

from layers.utils_quant import QuantizeEmbedding

class IRazorEmbedding(torch.nn.Module):
    def __init__(self, field_num, id_feat_size, field_feat_nums, regions_dims, temperature=1.0):
        """
            i-Razor: A Neural Input Razor for Feature Selection and Dimension Search 
                in Large-Scale Recommender Systems.
        """
        super(IRazorEmbedding, self).__init__()
        self.field_num = field_num
        self.id_feat_size = id_feat_size
        self.field_feat_nums = field_feat_nums
        self.regions_dims = regions_dims
        
        self.temperature = temperature
        
        self.init_variables()

    def init_variables(self):
        # self.regions_dims  # [0, 1, 2, 4, 8]
        self.all_dim = sum(self.regions_dims)
        zero_emb_mask = torch.zeros([self.all_dim])
        zero_emb_mask = torch.unsqueeze(zero_emb_mask, 0) # [1, dim]
        reg_mask_list = [zero_emb_mask]

        dims_pre_sum = [0] + [sum(self.regions_dims[1:i]) 
                                for i in range(2, len(self.regions_dims) ) ] # [0, 1, 3, 7]
        for pre_sum,dim in zip(dims_pre_sum, self.regions_dims[1:]):
            prefix_mask = torch.zeros([pre_sum])
            reg_mask = torch.ones([dim])
            suffix_mask = torch.zeros([self.all_dim - pre_sum - dim])

            cat_mask = torch.cat([prefix_mask, reg_mask, suffix_mask], dim=0) # [dim]
            cat_mask = torch.unsqueeze(cat_mask, 0) # [1, dim]
            # print("cat_mask.size: {} ".format( cat_mask.size() ))
            reg_mask_list.append(cat_mask)
        self.region_mask = torch.unsqueeze(torch.cat(reg_mask_list, 0), 0) # [1, region, dim]

        self.embedding = torch.nn.Embedding(self.id_feat_size, self.all_dim, sparse=False)
        torch.nn.init.uniform_(self.embedding.weight, a=-0.05, b=0.05)

        self.batch_norm_layer = torch.nn.BatchNorm1d(self.field_num, affine=False)

        self.field_region_weights = torch.nn.Parameter(
                    torch.zeros([self.field_num, len(self.regions_dims), 1]))  # [field_num, region, 1]
        torch.nn.init.xavier_uniform(self.field_region_weights) 

        # for reg_loss
        field_feat_num_rate = [(1.0 * num) / sum(self.field_feat_nums) for num in self.field_feat_nums]
        self.field_rate = torch.FloatTensor(field_feat_num_rate) # [field_num]
        
        cm = [0] + [self.regions_dims[i] - self.regions_dims[i-1] 
                            for i in range(1, len(self.regions_dims))] # [region]
        self.cm = torch.unsqueeze(torch.FloatTensor(cm), dim=0) # [1, region] 
    
    def forward(self, input_ids):
        id_emb = self.embedding(input_ids) # [bs, field_num, dim]
        id_emb = self.batch_norm_layer(id_emb)

        id_emb = torch.unsqueeze(id_emb, dim=2) # [bs, field_num, 1, dim]
        id_region_emb = torch.mul(id_emb, torch.unsqueeze(self.region_mask, 0)) 
        # [bs, field_num, 1, dim] * [1, 1, region, dim] --> [bs, field_num, region, dim] 

        field_region_weights = F.softmax(self.field_region_weights / self.temperature, dim=1) # [field_num, region, 1]
        field_region_weights = torch.unsqueeze(field_region_weights, dim=0) # [1, field_num, region, 1]

        id_region_emb_out = torch.mul(id_region_emb, field_region_weights) # [bs, field_num, region, dim] 
        id_emb_out = torch.sum(id_region_emb_out, dim=2) # [bs, field_num, dim] 

        return id_emb_out

    def reg_loss(self):
        cm_weights = torch.mul(torch.squeeze(self.field_region_weights, dim=2), self.cm) # [field_num, region]
        cm_w = torch.sum(cm_weights, dim=1) # [field_num]

        cm_w_rate = torch.mul(self.field_rate, cm_w) # [field_num]
        reg_loss = torch.sum(cm_w_rate)
        return reg_loss

    def cpt_pruning(self, cpt=0.7):
        cm_val_list = self.cm.cpu().detach().numpy().flatten().tolist() #  [region]
        idx_l = [i for i in range(len(cm_val_list))] # region idx

        field_region_weights = F.softmax(self.field_region_weights / self.temperature, dim=1) # [field_num, region, 1]
        field_region_weights_list = field_region_weights.cpu().detach().numpy().squeeze(axis=2).tolist() # [field_num, region]
        field_dim_list = []
        for region_weights in field_region_weights_list:
            region_wts_cm_list = list(zip(idx_l, region_weights, cm_val_list))
            sorted_region_wts_cm = sorted(region_wts_cm_list, key=lambda x: x[1], reverse=True) # 
            region_idx = []
            field_dim = 0
            sum_wts = 0.0
            for i,wts,cm in sorted_region_wts_cm:
                region_idx.append(i)
                field_dim += cm
                sum_wts += wts
                if sum_wts >= cpt:
                    break
            field_dim_list.append( [region_idx, field_dim, sum_wts] )
        return field_dim_list

class SubRegionEmbedding(torch.nn.Module):
    def __init__(self, field_num, id_feat_size, field_feat_nums, regions_dims, temperature=1.0):
        """
            i-Razor: A Neural Input Razor for Feature Selection and Dimension Search 
                in Large-Scale Recommender Systems.
        """
        super(SubRegionEmbedding, self).__init__()
        self.field_num = field_num
        self.id_feat_size = id_feat_size
        self.field_feat_nums = field_feat_nums
        self.regions_dims = regions_dims

        self.out_dim = 64
        self.temperature = temperature
        self.init_variables()

    def init_variables(self):
        # self.regions_dims  # [0, 1, 2, 4, 8]
        self.region_embedding_list = [torch.nn.Embedding(self.id_feat_size, dim, sparse=False) \
                                        for dim in self.regions_dims]
        for region_embedding in self.region_embedding_list:
            torch.nn.init.uniform_(region_embedding.weight, a=-0.05, b=0.05)
        # 
        self.batch_norm_layer = torch.nn.BatchNorm1d(self.field_num, affine=False)

        self.field_weights_list = [torch.nn.Parameter(torch.zeros([self.field_num, 1])) \
                                        for _ in self.regions_dims]
        for field_weigts in self.field_weights_list:
            torch.nn.init.xavier_uniform(field_weigts) 

        region_weights = torch.nn.Parameter(torch.zeros([len(self.regions_dims), 1])) # [region, 1]
        torch.nn.init.xavier_uniform(self.region_weights)
        self.region_weights = F.softmax(region_weights / self.temperature, dim=0) # [region, 1]

        self.region_linear_layers = [torch.nn.Linear(dim * field_num, self.out_dim) for dim in self.regions_dims]
        
        # for reg_loss
        field_feat_num_rate = [(1.0 * num) / sum(self.field_feat_nums) for num in self.field_feat_nums]
        self.field_rate = torch.FloatTensor(field_feat_num_rate) # [field_num]
        
        cm = [0] + [self.regions_dims[i] - self.regions_dims[i-1] 
                            for i in range(1, len(self.regions_dims))] # [region]
        self.cm = torch.unsqueeze(torch.FloatTensor(cm), dim=0) # [1, region] 
    
    def forward(self, input_ids):
        id_flatten_emb_list = []
        for i,dim in enumerate(self.regions_dims):
            id_emb = self.region_embedding_list[i](input_ids)
            id_emb = self.batch_norm_layer(id_emb) # [bs, field, dim]
            id_field_emb = torch.mul(id_emb, torch.unsqueeze(self.field_weights_list[i], dim=0)) # 
            # [bs, field, dim]  * [1, field, 1]  -->  [bs, field, dim]
            id_flatten_emb = torch.reshape(id_field_emb, [-1, dim * self.field_num])

            id_flatten_emb = torch.mul(id_flatten_emb, torch.unsqueeze(self.region_weights[i], dim=0))
            
            id_flatten_emb_list.append( id_flatten_emb ) # [bs, field * dim]
        # 
        id_emb_out = torch.cat(id_flatten_emb_list, dim=1) # [bs, d] 
        return id_emb_out
