#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class AggrFunc():
    def __init__(self):
        pass

    @staticmethod
    def noise_deleted(pred_embs):
        '''
        To be implemented
        '''
        return torch.mean(pred_embs, dim=0)
    
    @staticmethod
    def mix_weighted_mean(pred_embs, r_list, query_r, rel_con_pro, e_list, e_freq, sqare, trans_type='sqr'):
        '''
        Mix the weighted_mean_by_rel_stat and the weighted_mean_by_ent_stat
        '''

        tmp_con_pro = rel_con_pro[:, query_r]
        weights_r1 = torch.index_select(
                tmp_con_pro, 
                dim=0, 
                index=r_list
            )
        tmp_con_pro = rel_con_pro[query_r, :]
        weights_r2 = torch.index_select(
                tmp_con_pro, 
                dim=0, 
                index=r_list
            )
        weights_r = weights_r1 + weights_r2
        if trans_type == 'sqr':
            weights_r = weights_r ** sqare
        elif trans_type == 'softmax': 
            weights_r = F.softmax(weights_r * sqare, dim=0)

        weights_e = torch.index_select(
                e_freq, 
                dim=0, 
                index=e_list
            )
        weights_e = torch.log(weights_e + 0.1)

        weights = weights_r * weights_e
        weights = weights / weights.sum(dim=0)

        weighted_emb = (pred_embs * weights.unsqueeze(dim=1)).sum(dim=0)
        
        return weighted_emb
    
    @staticmethod
    def weighted_mean_by_emb_sim(pred_embs, r_embs, query_r_emb):
        '''
        calculate weighted mean of pred_embs according to the embedding similarity
        '''
        # dot procucts of neighbor relations emb and query relation emb, shape: [num_neighbors]
        dot_products = (r_embs * query_r_emb).sum(dim=1)

        # weighting method 1
        weights = dot_products * dot_products
        weights = weights / weights.sum(dim=0)

        # weighting method 2
        # weights = torch.abs(dot_products)
        # weights = F.softmax(weights, dim=0)

        # weighting method 3
        # len_r_embs = query_r_emb.norm(p=2, dim=1)
        # len_query_r_emb = query_r_emb.norm(p=2, dim=1)
        # cos_weights = dot_products / len_r_embs / len_query_r_emb
        # cos_weights = torch.abs(cos_weights)
        # weights = F.softmax(cos_weights, dim=0)

        weighted_emb = (pred_embs * weights.unsqueeze(dim=1)).sum(dim=0)

        return weighted_emb
    
    @staticmethod
    def weighted_mean_by_rel_stat(pred_embs, r_list, query_r, rel_con_pro, sqare, trans_type='sqr'):
        '''
        calculate weighted mean of pred_embs according to the relation condictional probability statistical data
        '''
        
        tmp_con_pro = rel_con_pro[:, query_r]
        weights1 = torch.index_select(
                tmp_con_pro, 
                dim=0, 
                index=r_list
            )
        tmp_con_pro = rel_con_pro[query_r, :]
        weights2 = torch.index_select(
                tmp_con_pro, 
                dim=0, 
                index=r_list
            )
        
        weights = weights1 + weights2
        # exploring effects of neighbor numbers
        need_number = 10000  # 10000 means using all neighbors
        num_all_neighbors = weights.shape[0]
        if num_all_neighbors > need_number:
            index = torch.LongTensor(random.sample(range(num_all_neighbors), need_number)).cuda()
            weights = weights.index_select(dim=0, index=index)
            pred_embs = pred_embs.index_select(dim=0, index=index)

        if trans_type == 'sqr':
            weights = weights ** sqare
        elif trans_type == 'softmax': 
            weights = F.softmax(weights * sqare, dim=0)

        weights = weights / weights.sum(dim=0)
        weighted_emb = (pred_embs * weights.unsqueeze(dim=1)).sum(dim=0)
        
        return weighted_emb
    
    @staticmethod
    def weighted_mean_by_ent_stat(pred_embs, e_list, e_freq):
        '''
        calculate weighted mean of pred_embs according to the relation condictional probability statistical data
        '''
        
        weights = torch.index_select(
                e_freq, 
                dim=0, 
                index=e_list
            )
        
        weights = torch.log(weights + 0.1)
        weights = weights / weights.sum(dim=0)

        weighted_emb = (pred_embs * weights.unsqueeze(dim=1)).sum(dim=0)
        
        return weighted_emb
    
    @staticmethod
    def mean(pred_embs):
        '''
        Apply average pooling to all pred_embs
        '''
        return torch.mean(pred_embs, dim=0)
    
    @staticmethod
    def filtered_mean(pred_embs, ratio=0.9):
        '''
        Apply average pooling to 100*ratio percentage of pred_embs those are closest to the center
        '''
        center = torch.mean(pred_embs, dim=0)
        distances = torch.norm(pred_embs - center, p=2, dim=1)
        new_distances, ids = torch.sort(distances)
        num_neighbors = pred_embs.shape[0]
        new_center = torch.mean(pred_embs[ids[:math.ceil(num_neighbors * ratio)]], dim=0)
        return new_center
    
    @staticmethod
    def nearest_to_center(pred_embs):
        '''
        Directly return the pred_emb that is closest to the center 
        '''
        center = torch.mean(pred_embs, dim=0)
        distances = torch.norm(pred_embs - center, p=2, dim=1)
        minn, minn_id = torch.min(distances, dim=0)
        return pred_embs[minn_id, :]