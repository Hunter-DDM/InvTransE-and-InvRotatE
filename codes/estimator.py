#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Estimator():
    def __init__(self):
        pass

    @staticmethod
    def InvTransE(hort_embs, r_embs, pred_pos):
        if pred_pos == 't':
            h_embs = hort_embs
            h_plus_r = h_embs + r_embs
            return h_plus_r
        if pred_pos == 'h':
            t_embs = hort_embs
            t_minus_r = t_embs - r_embs
            return t_minus_r
        raise ValueError("pred_pos should be 't' or 'h'")
    
    @staticmethod
    def InvRotatE(model, hort_embs, r_embs, pred_pos):
        # hort_embs, r_embs shape: [num_neighbors, hidden_dim (may be 1x or 2x entity_dim or relation_dim)]
        pi = 3.14159265358979323846
        if pred_pos == 't':
            h_embs = hort_embs
            re_h_embs, im_h_embs = torch.chunk(h_embs, 2, dim=1)
            phase_r_embs = r_embs / (model.embedding_range.item() / pi)
            re_r_embs = torch.cos(phase_r_embs)
            im_r_embs = torch.sin(phase_r_embs)
            # x_h_embs, x_r_embs shape: [num_neighbors, 1x hidden_dim]
            re_t_embs = re_h_embs * re_r_embs - im_h_embs * im_r_embs
            im_t_embs = re_h_embs * im_r_embs + im_h_embs * re_r_embs
            h_times_r = torch.cat((re_t_embs, im_t_embs), dim=1)
            return h_times_r
        if pred_pos == 'h':
            t_embs = hort_embs
            re_t_embs, im_t_embs = torch.chunk(t_embs, 2, dim=1)
            phase_r_embs = r_embs / (model.embedding_range.item() / pi)
            re_r_embs = torch.cos(phase_r_embs)
            im_r_embs = torch.sin(phase_r_embs)
            # x_t_embs, x_r_embs shape: [num_neighbors, 1x hidden_dim]
            re_h_embs = re_t_embs * re_r_embs + im_t_embs * im_r_embs
            im_h_embs = im_t_embs * re_r_embs - re_t_embs * im_r_embs
            t_times_reciprocal_r = torch.cat((re_h_embs, im_h_embs), dim=1)
            return t_times_reciprocal_r
        raise ValueError("pred_pos should be 't' or 'h'")