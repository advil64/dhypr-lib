from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from dhypr.models.BaseModel import NCModel, LPModel, SPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics, save_results
import pdb
import torch.nn as nn


class LPModel(nn.Module):
    def __init__(self, args):
        super(LPModel, self).__init__()
        
        assert args.c is None
        self.c = nn.Parameter(torch.Tensor([1.])) 
        
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes       
        
        self.encoder = getattr(encoders, args.model)(self.c, args)  
        
        if not args.act:
            act = lambda x: x
        else:
            act = getattr(F, args.act)
            
        self.dc = GravityDecoder(
            self.manifold, args.dim, 1, self.c, act, args.bias, args.beta, args.lamb)  
        
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        
        self.fd_dc = FermiDiracDecoder(r=args.r, t=args.t)
    
        
    def encode(self, x, adj):
        h = self.encoder.forward(x, adj)
        return h
    
    
    def decode(self, h, idx): 
        emb_in = h[idx[:, 0], :]   
        emb_out = h[idx[:, 1], :]  
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)  
        # squared distance between pairs of nodes in the hyperbolic space
        probs, mass = self.dc.forward(h, idx, sqdist)
        return probs, mass
    
    
    def fd_decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.fd_dc.forward(sqdist)
        return probs
    

    def compute_metrics(self, args, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
            
        if args.wl1 > 0:
            # fermi dirac 
            pos_scores = self.fd_decode(embeddings, data[f'{split}_edges'])
            neg_scores = self.fd_decode(embeddings, edges_false)
            fd_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
            fd_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
            

        # gravity 
        pos_scores, mass = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores, mass = self.decode(embeddings, edges_false)
        g_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        g_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        
        if args.wl1 > 0:
            assert args.wl2 > 0
            loss = args.wl1 * fd_loss + args.wl2 * g_loss 
        else:
            loss = g_loss 
      
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    
    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    
    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])



# sets seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# set torch dtype
if int(args.double_precision):
    torch.set_default_dtype(torch.float64)
if int(args.cuda) >= 0:
    torch.cuda.manual_seed(args.seed)

# set the device that the model is trained on
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

# TODO: what is patience?
args.patience = args.epochs if not args.patience else  int(args.patience)

# set logging
logging.getLogger().setLevel(logging.INFO)

# Load data
data = load_data(args)
args.n_nodes, args.feat_dim = data['features'].shape

Model = LPModel # set the model as the one you want
args.nb_false_edges = len(data['train_edges_false'])
args.nb_edges = len(data['train_edges'])

# Model and optimizer
model = Model(args)
logging.info(str(model))

optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=int(args.lr_reduce_freq),
    gamma=float(args.gamma)
)
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
logging.info(f"Total number of parameters: {tot_params}")
if args.cuda is not None and int(args.cuda) >= 0 :
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    model = model.to(args.device)
    for x, val in data.items():
        if torch.is_tensor(data[x]):
            data[x] = data[x].to(args.device)
        if isinstance(val, dict):
            for vk, vval in val.items():
                if torch.is_tensor(data[x][vk]):
                    data[x][vk] = data[x][vk].to(args.device)

# Train model
t_total = time.time()
counter = 0
best_val_metrics = model.init_metric_dict()
best_test_metrics = None
best_emb = None
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if args.use_att:
        embeddings_all, attn_weights = model.encode(data['features'], data['adj_train_norm'])
    else:
        embeddings_all = model.encode(data['features'], data['adj_train_norm'])
    embeddings = embeddings_all[-1]
    train_metrics = model.compute_metrics(args, embeddings, data, 'train')
    train_metrics['loss'].backward()
    if args.grad_clip is not None:
        max_norm = float(args.grad_clip)
        all_params = list(model.parameters())
        for param in all_params:
            torch.nn.utils.clip_grad_norm_(param, max_norm)
    optimizer.step()
    lr_scheduler.step()
    if (epoch + 1) % args.log_freq == 0:
        logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                format_metrics(train_metrics, 'train'),
                                'time: {:.4f}s'.format(time.time() - t)
                                ]))
    if (epoch + 1) % args.eval_freq == 0:
        model.eval()
        if args.use_att:
            embeddings_all, attn_weights = model.encode(data['features'], data['adj_train_norm'])
        else:
            embeddings_all = model.encode(data['features'], data['adj_train_norm'])
        embeddings = embeddings_all[-1]
        val_metrics = model.compute_metrics(args, embeddings, data, 'val')
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
        if model.has_improved(best_val_metrics, val_metrics):
            test_metrics = model.compute_metrics(args, embeddings, data, 'test')
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(test_metrics, 'test')]))
            with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
                f.writelines(" ".join(["Test set results:", format_metrics(test_metrics, 'test')]))
            
            
            if args.save:
                if not args.savespace:
                    best_emb_all = embeddings_all
                    if args.use_att:
                        best_attn_weights = attn_weights
                    else:
                        best_attn_weights = None
                    save_results(save_dir, best_emb_all, best_attn_weights, print_statement=True)

                    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
                    logging.info(f"Saved model in {save_dir}")
                
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            counter = 0
        else:
            counter += 1
            if counter == args.patience: 
                logging.info("Early stopping")
                break

logging.info("Optimization Finished!")
logging.info("Total train time elapsed: {:.4f}s".format(time.time() - t_total))

logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

if args.save:
    with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
        f.writelines(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

    with open(os.path.join(save_dir, 'finished'), 'w') as f:
        f.writelines("Optimization Finished!\n" + 
                        "Total train time elapsed: {:.4f}s\n\n".format(time.time() - t_total))
    logging.info(f"Saved data/log in {save_dir}") 
else:
    logging.info(f"Saved log in ./log.txt") 