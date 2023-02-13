import os
import scipy.sparse as sp
import pdb
import os.path as osp
import numpy as np
import networkx as nx
import time
import pickle

K_max = 2

def compute_proximity_matrices(adj, K=2):
    A = dict()
    for k in range(1, K+1):
        t = time.time()
        compute_kth_diffusion_in(adj, k, A) 
        print('k={} {}   took {} s'.format(k, 'diffusion_in', time.time()-t))
        
        t = time.time()
        compute_kth_diffusion_out(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'diffusion_out', time.time()-t))
        
        t = time.time()
        compute_kth_neighbor_in(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'neighbor_in', time.time()-t))
        
        t = time.time()
        compute_kth_neighbor_out(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'neighbor_out', time.time()-t))
    
    return A

def compute_kth_diffusion_in(adj, k, A):
    if k == 1:
        A['a'+str(k)+'_d_i'] = adj.T
        
    if k > 1:
        A['a'+str(k)+'_d_i'] = np.where(np.dot(A['a'+str(k-1)+'_d_i'], A['a1_d_i']) > 0, 1, 0) 
    return 

def compute_kth_diffusion_out(adj, k, A):
    if k == 1:
        A['a'+str(k)+'_d_o'] = adj
        
    if k > 1:
        A['a'+str(k)+'_d_o'] = np.where(np.dot(A['a'+str(k-1)+'_d_o'], A['a1_d_o']) > 0, 1, 0) 
    return 

def compute_kth_neighbor_in(adj, k, A):
    tmp = np.dot(A['a'+str(k)+'_d_i'], A['a'+str(k)+'_d_o'])
    np.fill_diagonal(tmp, 0) 
    A['a'+str(k)+'_n_i'] = np.where(tmp + tmp.T - np.diag(tmp.diagonal()) > 0, 1, 0) 
    return 
    
def compute_kth_neighbor_out(adj, k, A):
    tmp = np.dot(A['a'+str(k)+'_d_o'], A['a'+str(k)+'_d_i'])
    np.fill_diagonal(tmp, 0) 
    A['a'+str(k)+'_n_o'] = np.where(tmp + tmp.T - np.diag(tmp.diagonal()) > 0, 1, 0) 
    return 


def get_k_order_lp_matrix(edgelist):
    G = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
    adj = nx.adjacency_matrix(G).toarray()

    A = compute_proximity_matrices(adj, K=K_max)
    A = {key: sp.csr_matrix(A[key]) for key in A}

    return A