#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import math

from aggregator import AggrFunc
from estimator import Estimator

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False, args=None):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.args = args
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples, 
        because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size = sample.shape[0]
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            # head shape: [batch_size, 1, entity_dim]
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            # relation shape: [batch_size, 1, relation_dim]
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            # tail shape: [batch_size, 1, entity_dim]
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.shape[0], head_part.shape[1]
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            # head shape: [batch_size, negative_sample_size, entity_dim]
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            # relation shape: [batch_size, 1, relation_dim]
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            # tail shape: [batch_size, 1, entity_dim]
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.shape[0], tail_part.shape[1]
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            # head shape: [batch_size, 1, entity_dim]
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            # relation shape: [batch_size, 1, relation_dim]
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            # tail shape: [batch_size, negative_sample_size, entity_dim]
            
        else:
            raise ValueError('mode %s not supported' % mode)
        
        # choose a score function
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        if self.args.eval_task == 'TC': 
            score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        elif self.args.eval_task == 'LP': 
            score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        # x_head, x_tail shape: [batch_size, x (1 or negative_sample_size), entity_dim]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item() / pi)

        # x_relation shape: [batch_size, 1, relation_dim]
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        # Make phases of entities and relations uniformly distributed in [-pi, pi]
        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()

        # positive_sample: a batch of positive triples, LongTensor, [batch_size, 3]
        # negative_sample: a batch of negative entities, LongTensor, [batch_size, negative_sample_size]
        # subsampling_weight: a batch of subsampling_weights, FloatTensor, [batch_size]
        # mode: sample mode, a str
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # negative score, shape: [batch_size, negative_sample_size]
        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # negative score, shape: [batch_size] 
            negative_score = (F.logsigmoid(-negative_score) * F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()).sum(dim = 1)
        else:
            # negative score, shape: [batch_size] 
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        # positive score, shape: [batch_size, 1]
        positive_score = model(positive_sample)

        # positive score, shape: [batch_size]
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            regularization = args.regularization * (model.entity_embedding.norm(p = 2) ** 2 + model.relation_embedding.norm(p = 2) ** 2)
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        # use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'head-batch'
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'tail-batch'
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    # positive_sample: a batch of positive triples, LongTensor, [batch_size, 3]
                    # negative_sample: a batch of all entities, LongTensor, [batch_size, nentity]
                    # filter_bias: a batch of filter_bias, valued from {0, -1}, FloatTensor, [batch_size, nentity]
                    # mode: sample mode, a str
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    # score for all entities, shape: [batch_size, nentity]
                    score = model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    # here argsort array contains the descending sorted indices of all triples by the model scores
                    argsort = torch.argsort(score, dim = 1, descending=True)

                    # choose the valid part (head or tail) that is linked to the sorted indices
                    # positive_arg, shape: [batch_size]
                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        # assert only one id is matched with the true id
                        assert ranking.shape[0] == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
    
    @staticmethod
    def process_aux(aux_triples): 
        '''
        Use aux_triples to link unseen test entities into the trained entities
        '''
        t2hr_by_aux = {}
        h2rt_by_aux = {}

        for head, relation, tail in aux_triples:
            if head not in h2rt_by_aux:
                h2rt_by_aux[head] = []
            h2rt_by_aux[head].append((relation, tail))
            if tail not in t2hr_by_aux:
                t2hr_by_aux[tail] = []
            t2hr_by_aux[tail].append((head, relation))

        for tail in t2hr_by_aux:
            t2hr_by_aux[tail] = list(set(t2hr_by_aux[tail]))
        for head in h2rt_by_aux:
            h2rt_by_aux[head] = list(set(h2rt_by_aux[head]))

        return [t2hr_by_aux, h2rt_by_aux]

    @staticmethod
    def process_weights_info(seen_triples, train_triples, nrelation, nentity): 
        '''
        calculate the conditional probablity of all relation pairs by their concurrence statistical data
        '''
        rel_set = set()
        rels_of_e = {}
        for head, relation, tail in seen_triples:
            rel_set.update([relation])
            if head not in rels_of_e: 
                rels_of_e[head] = set()
            rels_of_e[head].update([relation])
            if tail not in rels_of_e: 
                rels_of_e[tail] = set()
            rels_of_e[tail].update([relation])
        
        rel_set = list(rel_set)
        # rel_con_pro[i][j] is P(i|j) = P(ij) / P(j), shape: [nrelation, nrelation]
        rel_con_pro = [[0 for i in range(nrelation)] for j in range(nrelation)]
        pij = [[0 for i in range(nrelation)] for j in range(nrelation)]
        pj = [0 for i in range(nrelation)]

        for e, rels in rels_of_e.items():
            rels = list(rels)
            for rel1 in rels:
                for rel2 in rels:
                    pij[rel1][rel2] += 1
            for rel in rels:
                pj[rel] += 1
        
        for i in range(nrelation): 
            for j in range(nrelation): 
                if pj[j] == 0:
                    rel_con_pro[i][j] = 0
                else:
                    rel_con_pro[i][j] = pij[i][j] / pj[j]
        
        rel_con_pro = torch.FloatTensor(rel_con_pro).cuda()

        e_freq = [0 for i in range(nentity)]
        for head, relation, tail in train_triples: 
            e_freq[head] += 1
            e_freq[tail] += 1
        e_freq = torch.FloatTensor(e_freq).cuda()
        
        return rel_con_pro, e_freq
    
    @staticmethod
    def inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, query_r, e_freq, eval_task):
        # query_r, shape: 1 (scalar)
        h_list = torch.LongTensor([tp[0] for tp in hr_neighbors]).cuda()
        r1_list = torch.LongTensor([tp[1] for tp in hr_neighbors]).cuda()
        h_embs = torch.index_select(
                model.entity_embedding, 
                dim=0, 
                index=h_list
            )
        r1_embs = torch.index_select(
                model.relation_embedding, 
                dim=0, 
                index=r1_list
            )
        if model.model_name == 'TransE':
            pred_embs1 = Estimator().InvTransE(h_embs, r1_embs, 't')
        elif model.model_name == 'RotatE':
            pred_embs1 = Estimator().InvRotatE(model, h_embs, r1_embs, 't')
        else:
            raise ValueError("InversE now only supply TransE and RotatE. ")
        r2_list = torch.LongTensor([tp[0] for tp in rt_neighbors]).cuda()
        t_list = torch.LongTensor([tp[1] for tp in rt_neighbors]).cuda()
        r2_embs = torch.index_select(
                model.relation_embedding, 
                dim=0, 
                index=r2_list
            )
        t_embs = torch.index_select(
                model.entity_embedding, 
                dim=0, 
                index=t_list
            )
        if model.model_name == 'TransE':
            pred_embs2 = Estimator().InvTransE(t_embs, r2_embs, 'h')
        elif model.model_name == 'RotatE':
            pred_embs2 = Estimator().InvRotatE(model, t_embs, r2_embs, 'h')
        else:
            raise ValueError("InversE now only supply TransE and RotatE. ")
        # predicted embeddings by the neighbors, shape: [num_neighbors, hidden_dim]
        pred_embs = torch.cat((pred_embs1, pred_embs2), dim=0)
        # r_list shape: [num_neighbors]
        r_list = torch.cat((r1_list, r2_list))
        # e_list shape: [num_neighbors]
        e_list = torch.cat((h_list, t_list))
        # r_embs shape: [num_neighbors, hidden_dim]
        r_embs = torch.index_select(
                model.relation_embedding, 
                dim=0, 
                index=r_list
            )
        # r_embs shape: [1, hidden_dim]
        query_r_emb = torch.index_select(
                model.relation_embedding, 
                dim=0, 
                index=query_r
            )

        # aggregate all predicted embeddings, different for LP and TC
        if eval_task == 'TC': 
            if model.model_name == 'TransE':
                # aggregated_emb = AggrFunc.mean(pred_embs)
                # aggregated_emb = AggrFunc.weighted_mean_by_rel_stat(pred_embs, r_list, query_r, rel_con_pro, 1, trans_type='sqr')
                aggregated_emb = AggrFunc.weighted_mean_by_ent_stat(pred_embs, e_list, e_freq)
                # aggregated_emb = AggrFunc.mix_weighted_mean(pred_embs, r_list, query_r, rel_con_pro, e_list, e_freq, 1, trans_type='sqr')
            elif model.model_name == 'RotatE':
                # aggregated_emb = AggrFunc.mean(pred_embs)
                # aggregated_emb = AggrFunc.weighted_mean_by_rel_stat(pred_embs, r_list, query_r, rel_con_pro, 1, trans_type='sqr')
                aggregated_emb = AggrFunc.weighted_mean_by_ent_stat(pred_embs, e_list, e_freq)
                # aggregated_emb = AggrFunc.mix_weighted_mean(pred_embs, r_list, query_r, rel_con_pro, e_list, e_freq, 1, trans_type='sqr')
            else:
                raise ValueError("InversE now only supply TransE and RotatE. ")
        elif eval_task == 'LP':
            if model.model_name == 'TransE':
                # aggregated_emb = AggrFunc.mean(pred_embs)
                aggregated_emb = AggrFunc.weighted_mean_by_rel_stat(pred_embs, r_list, query_r, rel_con_pro, 4, trans_type='sqr')
                # aggregated_emb = AggrFunc.weighted_mean_by_ent_stat(pred_embs, e_list, e_freq)
                # aggregated_emb = AggrFunc.mix_weighted_mean(pred_embs, r_list, query_r, rel_con_pro, e_list, e_freq, 4, trans_type='sqr')
            elif model.model_name == 'RotatE':
                # aggregated_emb = AggrFunc.mean(pred_embs)
                aggregated_emb = AggrFunc.weighted_mean_by_rel_stat(pred_embs, r_list, query_r, rel_con_pro, 4, trans_type='sqr')
                # aggregated_emb = AggrFunc.weighted_mean_by_ent_stat(pred_embs, e_list, e_freq)
                # aggregated_emb = AggrFunc.mix_weighted_mean(pred_embs, r_list, query_r, rel_con_pro, e_list, e_freq, 4, trans_type='sqr')
            else:
                raise ValueError("InversE now only supply TransE and RotatE. ")

        return aggregated_emb

    @staticmethod
    def ookb_score(model, sample, aux_info, rel_con_pro, e_freq, mode):
        '''
        Modified from the KGEModel forward, aiming to calculate triple scores for ookb dataset. 
        '''
        [t2hr_by_aux, h2rt_by_aux] = aux_info
        if mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.shape[0], head_part.shape[1]
            
            relation = torch.index_select(
                model.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            # relation shape: [batch_size(==1), 1, relation_dim]
            
            head = torch.index_select(
                model.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            # head shape: [batch_size(==1), negative_sample_size, entity_dim]
            
            # calculate unseen entity embedding from the neighbors
            triple = tail_part[0]
            rt_neighbors = h2rt_by_aux[triple[2].item()] if triple[2].item() in h2rt_by_aux else []
            hr_neighbors = t2hr_by_aux[triple[2].item()] if triple[2].item() in t2hr_by_aux else []
            tail = model.inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, triple[1], e_freq, model.args.eval_task).unsqueeze(0).unsqueeze(0)
            # tail shape: [batch_size(==1), 1, entity_dim]
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.shape[0], tail_part.shape[1]
            
            relation = torch.index_select(
                model.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            # relation shape: [batch_size(==1), 1, relation_dim]
            
            tail = torch.index_select(
                model.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            # tail shape: [batch_size(==1), negative_sample_size, entity_dim]
            
            # calculate unseen entity embedding from the neighbors
            triple = head_part[0]
            rt_neighbors = h2rt_by_aux[triple[0].item()] if triple[0].item() in h2rt_by_aux else []
            hr_neighbors = t2hr_by_aux[triple[0].item()] if triple[0].item() in t2hr_by_aux else []
            head = model.inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, triple[1], e_freq, model.args.eval_task).unsqueeze(0).unsqueeze(0)
            # head shape: [batch_size(==1), 1, entity_dim]

        else:
            raise ValueError('mode %s not supported' % mode)
        
        # choose a score function
        model_func = {
            'TransE': model.TransE,
            'DistMult': model.DistMult,
            'ComplEx': model.ComplEx,
            'RotatE': model.RotatE,
            'pRotatE': model.pRotatE
        }
        
        if model.model_name in model_func:
            score = model_func[model.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % model.model_name)
        
        return score

    @staticmethod
    def test_ookb_step(model, test_triples, aux_triples, all_true_triples, train_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        # Prepare dataloader for evaluation
        # if the tail is unseen, the target for link prediction task is to predict the existing head
        if args.op: 
            test_dataloader = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=1,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
        # if the head is unseen, the target for link prediction task is to predict the existing tail
        if args.sp: 
            test_dataloader = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=1, 
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
        
        logs = []

        step = 0
        total_steps = len(test_dataloader)

        aux_info = model.process_aux(aux_triples)
        rel_con_pro, e_freq = model.process_weights_info(train_triples + aux_triples, train_triples, model.nrelation, model.nentity)

        with torch.no_grad():
            for positive_sample, negative_sample, filter_bias, mode in test_dataloader:
                # positive_sample: a batch of positive triples, LongTensor, [batch_size, 3]
                # negative_sample: a batch of all entities, LongTensor, [batch_size, nentity]
                # filter_bias: a batch of filter_bias, valued from {0, -1}, FloatTensor, [batch_size, nentity]
                # mode: sample mode, a str
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)

                # score for all entities, shape: [batch_size, nentity]
                score = model.ookb_score(model, (positive_sample, negative_sample), aux_info, rel_con_pro, e_freq, mode)
                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                # here argsort array contains the descending sorted indices of all triples by the model scores
                argsort = torch.argsort(score, dim = 1, descending=True)

                # choose the valid part (head or tail) that is linked to the sorted indices
                # positive_arg, shape: [batch_size]
                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    # assert only one id is matched with the true id
                    assert ranking.shape[0] == 1

                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
    
    @staticmethod
    def test_ookb_step_TC(model, valid_triples, test_triples, aux_triples, train_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        logs = []
        metrics = {}

        aux_info = model.process_aux(aux_triples)
        [t2hr_by_aux, h2rt_by_aux] = aux_info
        rel_con_pro, e_freq = model.process_weights_info(train_triples + aux_triples, train_triples, model.nrelation, model.nentity)
        
        # ===================================== valid ========================================

        correct_cnts_valid = [0 for i in range(model.nrelation)]
        cnts_valid = [0 for i in range(model.nrelation)]
        thresholds = [0.0 for i in range(model.nrelation)]
        scores_1_valid = [[] for i in range(model.nrelation)]
        scores_0_valid = [[] for i in range(model.nrelation)]        

        for triple in valid_triples:
            triple = torch.LongTensor(triple).cuda()
            h_emb = torch.index_select(
                    model.entity_embedding, 
                    dim=0, 
                    index=triple[0]
            ).unsqueeze(0)

            r_emb = torch.index_select(
                    model.relation_embedding, 
                    dim=0, 
                    index=triple[1]
            ).unsqueeze(0)
            
            t_emb = torch.index_select(
                    model.entity_embedding, 
                    dim=0, 
                    index=triple[2]
            ).unsqueeze(0)
            
            if model.model_name == 'TransE': 
                score = model.TransE(h_emb, r_emb, t_emb, 'single').squeeze()
            elif model.model_name == 'RotatE': 
                score = model.RotatE(h_emb, r_emb, t_emb, 'single').squeeze()
            else:
                raise ValueError('InversE should be InvTransE or InvRotatE')
            if triple[3] == 1: 
                scores_1_valid[triple[1].item()].append(score)
            else:
                scores_0_valid[triple[1].item()].append(score)
            cnts_valid[triple[1].item()] += 1

        for i in np.arange(-10, 10, 0.1): 
            for j in range(0, model.nrelation): 
                threshold = i
                correct_cnt = torch.FloatTensor(scores_1_valid[j]).ge(threshold).sum() + torch.FloatTensor(scores_0_valid[j]).lt(threshold).sum()
                if correct_cnt >= correct_cnts_valid[j]:
                    thresholds[j] = threshold
                    correct_cnts_valid[j] = correct_cnt
        
        print('valid_thresholds: ', thresholds)

        for j in range(0, model.nrelation): 
            print('rel %d: %f/%f, %f' % (j, correct_cnts_valid[j], cnts_valid[j], float(correct_cnts_valid[j]) / cnts_valid[j]))
        metrics['valid_acc'] = 100.0 * sum(correct_cnts_valid).item() / sum(cnts_valid)

        # ===================================== test ========================================

        correct_cnts_test = [0 for i in range(model.nrelation)]
        cnts_test = [0 for i in range(model.nrelation)]
        scores_1_test = [[] for i in range(model.nrelation)]
        scores_0_test = [[] for i in range(model.nrelation)]

        for triple in test_triples:
            triple = torch.LongTensor(triple).cuda()
            if args.sp or args.bp:
                rt_neighbors = h2rt_by_aux[triple[0].item()] if triple[0].item() in h2rt_by_aux else []
                hr_neighbors = t2hr_by_aux[triple[0].item()] if triple[0].item() in t2hr_by_aux else []
                h_emb = model.inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, triple[1], e_freq, args.eval_task).unsqueeze(0).unsqueeze(0)
            else:
                h_emb = torch.index_select(
                        model.entity_embedding, 
                        dim=0, 
                        index=triple[0]
                ).unsqueeze(0)
            
            r_emb = torch.index_select(
                    model.relation_embedding, 
                    dim=0, 
                    index=triple[1]
            ).unsqueeze(0)

            if args.op or args.bp:
                rt_neighbors = h2rt_by_aux[triple[2].item()] if triple[2].item() in h2rt_by_aux else []
                hr_neighbors = t2hr_by_aux[triple[2].item()] if triple[2].item() in t2hr_by_aux else []
                t_emb = model.inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, triple[1], e_freq, args.eval_task).unsqueeze(0).unsqueeze(0)
            else:
                t_emb = torch.index_select(
                        model.entity_embedding, 
                        dim=0, 
                        index=triple[2]
                ).unsqueeze(0)
            
            if model.model_name == 'TransE': 
                score = model.TransE(h_emb, r_emb, t_emb, 'single').squeeze()
            elif model.model_name == 'RotatE': 
                score = model.RotatE(h_emb, r_emb, t_emb, 'single').squeeze()
            else:
                raise ValueError('InversE should be InvTransE or InvRotatE')
            
            if triple[3] == 1: 
                scores_1_test[triple[1].item()].append(score)
            else:
                scores_0_test[triple[1].item()].append(score)
            cnts_test[triple[1].item()] += 1

        for j in range(0, model.nrelation): 
            threshold = thresholds[j]
            correct_cnts_test[j] = torch.FloatTensor(scores_1_test[j]).ge(threshold).sum() + torch.FloatTensor(scores_0_test[j]).lt(threshold).sum()

        metrics['test_acc'] = 100.0 * sum(correct_cnts_test).item() / sum(cnts_test)
        
        # # ============================================== calculate ideal test_acc ==============================================================

        # correct_cnts_test = [0 for i in range(model.nrelation)]
        # cnts_test = [0 for i in range(model.nrelation)]
        # thresholds = [0.0 for i in range(model.nrelation)]
        # scores_1_test = [[] for i in range(model.nrelation)]
        # scores_0_test = [[] for i in range(model.nrelation)]

        # for triple in test_triples[:len(test_triples) // 2]:
        #     triple = torch.LongTensor(triple).cuda()
        #     if args.sp or args.bp:
        #         rt_neighbors = h2rt_by_aux[triple[0].item()] if triple[0].item() in h2rt_by_aux else []
        #         hr_neighbors = t2hr_by_aux[triple[0].item()] if triple[0].item() in t2hr_by_aux else []
        #         h_emb = model.inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, triple[1], e_freq, args.eval_task).unsqueeze(0).unsqueeze(0)
        #     else:
        #         h_emb = torch.index_select(
        #                 model.entity_embedding, 
        #                 dim=0, 
        #                 index=triple[0]
        #         ).unsqueeze(0)
            
        #     r_emb = torch.index_select(
        #             model.relation_embedding, 
        #             dim=0, 
        #             index=triple[1]
        #     ).unsqueeze(0)

        #     if args.op or args.bp:
        #         rt_neighbors = h2rt_by_aux[triple[2].item()] if triple[2].item() in h2rt_by_aux else []
        #         hr_neighbors = t2hr_by_aux[triple[2].item()] if triple[2].item() in t2hr_by_aux else []
        #         t_emb = model.inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, triple[1], e_freq, args.eval_task).unsqueeze(0).unsqueeze(0)
        #     else:
        #         t_emb = torch.index_select(
        #                 model.entity_embedding, 
        #                 dim=0, 
        #                 index=triple[2]
        #         ).unsqueeze(0)
            
        #     if model.model_name == 'TransE': 
        #         score = model.TransE(h_emb, r_emb, t_emb, 'single').squeeze()
        #     elif model.model_name == 'RotatE': 
        #         score = model.RotatE(h_emb, r_emb, t_emb, 'single').squeeze()
        #     else:
        #         raise ValueError('InversE should be InvTransE or InvRotatE')
            
        #     if triple[3] == 1: 
        #         scores_1_test[triple[1].item()].append(score)
        #     else:
        #         scores_0_test[triple[1].item()].append(score)
        #     cnts_test[triple[1].item()] += 1
        
        # # grain into (0.1)^(num_grain - 1)
        # num_grain = 5
        # for grain in range(0, num_grain):
        #     tmp_thresholds = [thresholds[i] for i in range(model.nrelation)]
        #     for i in range(-20, 20, 1): 
        #         bias = i * math.pow(0.1, grain)
        #         for j in range(0, model.nrelation): 
        #             threshold = thresholds[j] + bias
        #             correct_cnt = torch.FloatTensor(scores_1_test[j]).ge(threshold).sum() + torch.FloatTensor(scores_0_test[j]).lt(threshold).sum()
        #             if correct_cnt > correct_cnts_test[j]:
        #                 tmp_thresholds[j] = threshold
        #                 correct_cnts_test[j] = correct_cnt
        #     for i in range(model.nrelation): 
        #         thresholds[i] = tmp_thresholds[i]
        #     logging.info('Evaluating the model on test set... (%d|%d)' % (grain, num_grain))
        
        # # >
        # print('ideal_test_thresholds: ', thresholds)

        # metrics['ideal_test_acc'] = 100.0 * sum(correct_cnts_test).item() / sum(cnts_test)
        
        # # ============================================== test the other test triplets using ideal threshold ==============================================================

        # correct_cnts_test = [0 for i in range(model.nrelation)]
        # cnts_test = [0 for i in range(model.nrelation)]
        # scores_1_test = [[] for i in range(model.nrelation)]
        # scores_0_test = [[] for i in range(model.nrelation)]

        # for triple in test_triples[len(test_triples) // 2:]:
        #     triple = torch.LongTensor(triple).cuda()
        #     if args.sp or args.bp:
        #         rt_neighbors = h2rt_by_aux[triple[0].item()] if triple[0].item() in h2rt_by_aux else []
        #         hr_neighbors = t2hr_by_aux[triple[0].item()] if triple[0].item() in t2hr_by_aux else []
        #         h_emb = model.inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, triple[1], e_freq, args.eval_task).unsqueeze(0).unsqueeze(0)
        #     else:
        #         h_emb = torch.index_select(
        #                 model.entity_embedding, 
        #                 dim=0, 
        #                 index=triple[0]
        #         ).unsqueeze(0)
            
        #     r_emb = torch.index_select(
        #             model.relation_embedding, 
        #             dim=0, 
        #             index=triple[1]
        #     ).unsqueeze(0)

        #     if args.op or args.bp:
        #         rt_neighbors = h2rt_by_aux[triple[2].item()] if triple[2].item() in h2rt_by_aux else []
        #         hr_neighbors = t2hr_by_aux[triple[2].item()] if triple[2].item() in t2hr_by_aux else []
        #         t_emb = model.inversE_embedding(model, hr_neighbors, rt_neighbors, rel_con_pro, triple[1], e_freq, args.eval_task).unsqueeze(0).unsqueeze(0)
        #     else:
        #         t_emb = torch.index_select(
        #                 model.entity_embedding, 
        #                 dim=0, 
        #                 index=triple[2]
        #         ).unsqueeze(0)
            
        #     if model.model_name == 'TransE': 
        #         score = model.TransE(h_emb, r_emb, t_emb, 'single').squeeze()
        #     elif model.model_name == 'RotatE': 
        #         score = model.RotatE(h_emb, r_emb, t_emb, 'single').squeeze()
        #     else:
        #         raise ValueError('InversE should be InvTransE or InvRotatE')
            
        #     if triple[3] == 1: 
        #         scores_1_test[triple[1].item()].append(score)
        #     else:
        #         scores_0_test[triple[1].item()].append(score)
        #     cnts_test[triple[1].item()] += 1
        
        # for j in range(0, model.nrelation): 
        #     threshold = thresholds[j]
        #     correct_cnts_test[j] = torch.FloatTensor(scores_1_test[j]).ge(threshold).sum() + torch.FloatTensor(scores_0_test[j]).lt(threshold).sum()

        # metrics['test_acc_using_ideal_threshold'] = 100.0 * sum(correct_cnts_test).item() / sum(cnts_test)

        return metrics
