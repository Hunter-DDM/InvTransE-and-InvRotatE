#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU is to use')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--ookb', action='store_true', help='is this a model to test ookb entities')
    parser.add_argument('-sp', action='store_true', help='test dataset is build by subject policy')
    parser.add_argument('-op', action='store_true', help='test dataset is build by object policy')
    parser.add_argument('-bp', action='store_true', help='test dataset is build by both policy')
    parser.add_argument('--eval_task', type=str, default=None, help='do what eval task (LP for FB15k and TC for WN11)')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=8, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--decay_rate', default=0.1, type=float)
    # parser.add_argument('--decay_steps', default=10000, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )


def read_triple(file_path, entity2id, relation2id, args):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    if args.eval_task == 'LP':
        with open(file_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
    elif args.eval_task == 'TC':
        with open(file_path) as fin:
            for line in fin:
                items = line.strip().split('\t')
                if len(items) == 3:
                    h, r, t = items
                    triples.append((entity2id[h], relation2id[r], entity2id[t]))
                else:
                    h, r, t, is_positive = items
                    triples.append((entity2id[h], relation2id[r], entity2id[t], int(is_positive)))
    else:
        raise ValueError('eval_task should be LP for FB15k or TC for WN11')
    
    return triples


def extract_true_triples(triples):
    '''
    Extract true triples to build all_true_triples
    '''
    true_triples = []
    for triple in triples:
        if len(triple) == 3:
            true_triples.append(triple)
        else:
            if triple[3] == 1:
                true_triples.append(triple[0:3])
    
    return true_triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    # Check program parameters
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed. ')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed. ')

    if args.do_train and args.save_path is None:
        raise ValueError('If do_train, you must specify a save_path to save your trained model. ')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    # Read entity and raletion mapping-to-id dict
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    # Read triples
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id, args)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id, args)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id, args)
    logging.info('#test: %d' % len(test_triples))
    if args.ookb:
        aux_triples = read_triple(os.path.join(args.data_path, 'aux.txt'), entity2id, relation2id, args)
        logging.info('#aux: %d' % len(aux_triples))
    
    # All true triples
    all_true_triples = extract_true_triples(train_triples) + extract_true_triples(valid_triples) + extract_true_triples(test_triples)
    if args.ookb:
        all_true_triples = all_true_triples + extract_true_triples(aux_triples)
    
    # Build model
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding, 
        args = args
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    # Build training dataloader iterator
    if args.do_train:
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        # It will return a head batch and a tail batch alternately
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    # Restore model from checkpoint directory if needed
    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('learning_rate = %.9f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %.9f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %.9f' % args.adversarial_temperature)
    
    if args.do_train:
        training_logs = []

        #Training Loop
        for step in range(init_step, args.max_steps + 1):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            # Check learning rate decay
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate * args.decay_rate
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                if args.eval_task == 'TC':
                    warm_up_steps = warm_up_steps * 2
                elif args.eval_task == 'LP': 
                    warm_up_steps = warm_up_steps * 3
            
            # Save model, optimizer, args and some runtime infomation as checkpoint per save_checkpoint_steps
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
            
            # Log training information per log_steps
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
            
            # Evaluate on valid set and log evaluating information per valid_steps
            if args.do_valid and step > 0 and step % args.valid_steps == 0:
                if args.eval_task == 'LP': 
                    logging.info('Evaluating on Valid Dataset...')
                    metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                    log_metrics('Valid', step, metrics)
                    # if ookb, test ookb on every evaluation step
                    if args.ookb:
                        logging.info('Evaluating on Test Dataset...')
                        metrics = kge_model.test_ookb_step(kge_model, test_triples, aux_triples, all_true_triples, train_triples, args)
                        log_metrics('Test', step, metrics)
                elif args.eval_task == 'TC': 
                    if args.ookb:
                        logging.info('Evaluating on Valid and Test Dataset for TC...')
                        metrics = kge_model.test_ookb_step_TC(kge_model, valid_triples, test_triples, aux_triples, train_triples, args)
                        log_metrics('Valid and Test for TC, ', step, metrics)
                        # if args.op or args.sp: 
                        #     logging.info('Evaluating on Valid Dataset...')
                        #     tmp_valid_triples = [triple for triple in valid_triples if triple[3] == 1]
                        #     tmp_valid_triples = list(map(lambda triple: triple[0:3], tmp_valid_triples))
                        #     metrics = kge_model.test_step(kge_model, tmp_valid_triples, all_true_triples, args)
                        #     log_metrics('Valid', step, metrics)
                            
                        #     logging.info('Evaluating on Test Dataset...')
                        #     tmp_test_triples = [triple for triple in test_triples if triple[3] == 1]
                        #     tmp_test_triples = list(map(lambda triple: triple[0:3], tmp_test_triples))
                        #     metrics = kge_model.test_ookb_step(kge_model, tmp_test_triples, aux_triples, all_true_triples, train_triples, args)
                        #     log_metrics('Test', step, metrics)
                else:
                    raise ValueError('eval_task should be LP or TC')
        
        # Save model, optimizer, args and some runtime infomation as checkpoint when the training ends
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
    
    # do evaluation on valid set, if eval_task is TC, do test only
    if args.do_valid and args.eval_task == 'LP':
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    # do evaluation on test set for LP
    if args.do_test and args.eval_task == 'LP':
        if args.ookb:
            logging.info('Evaluating on Test Dataset...')
            metrics = kge_model.test_ookb_step(kge_model, test_triples, aux_triples, all_true_triples, train_triples, args)
            log_metrics('Test', step, metrics)
        else:
            logging.info('Evaluating on Test Dataset...')
            metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
            log_metrics('Test', step, metrics)

    # do evaluation on test set for TC
    if args.do_test and args.eval_task == 'TC':
        logging.info('Evaluating on Valid and Test Dataset for TC...')
        metrics = kge_model.test_ookb_step_TC(kge_model, valid_triples, test_triples, aux_triples, train_triples, args)
        log_metrics('Valid and Test for TC, ', step, metrics)
        # if args.op or args.sp: 
        #     logging.info('Evaluating on Valid Dataset...')
        #     tmp_valid_triples = [triple for triple in valid_triples if triple[3] == 1]
        #     tmp_valid_triples = list(map(lambda triple: triple[0:3], tmp_valid_triples))
        #     metrics = kge_model.test_step(kge_model, tmp_valid_triples, all_true_triples, args)
        #     log_metrics('Valid', step, metrics)
        #     # if ookb, test ookb on every evaluation step
        #     if args.ookb:
        #         logging.info('Evaluating on Test Dataset...')
        #         tmp_test_triples = [triple for triple in test_triples if triple[3] == 1]
        #         tmp_test_triples = list(map(lambda triple: triple[0:3], tmp_test_triples))
        #         metrics = kge_model.test_ookb_step(kge_model, tmp_test_triples, aux_triples, all_true_triples, train_triples, args)
        #         log_metrics('Test', step, metrics)

    # do evaluation on training set
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())