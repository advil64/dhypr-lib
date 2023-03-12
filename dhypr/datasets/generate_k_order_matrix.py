import scipy.sparse as sp
import numpy as np
import networkx as nx
import time
from datasets.data_utils import process_adj

K_max = 2

# NOTE: the numpy dot product function uses an underlying C library called BLAS which is capable to speeding up 
# calculations significantly iff the values in the tensor are of a float type, this is why I convert each tensor to a 
# flow before doing dot products below

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
        A['a'+str(k)+'_d_i'] = np.where(np.dot(A['a'+str(k-1)+'_d_i'].astype(np.float32), A['a1_d_i'].astype(np.float32)).astype(np.int64) > 0, 1, 0) 
    return 

def compute_kth_diffusion_out(adj, k, A):
    if k == 1:
        A['a'+str(k)+'_d_o'] = adj
        
    if k > 1:
        A['a'+str(k)+'_d_o'] = np.where(np.dot(A['a'+str(k-1)+'_d_o'].astype(np.float32), A['a1_d_o'].astype(np.float32)).astype(np.int64) > 0, 1, 0) 
    return 

def compute_kth_neighbor_in(adj, k, A):
    tmp = np.dot(A['a'+str(k)+'_d_i'].astype(np.float32), A['a'+str(k)+'_d_o'].astype(np.float32)).astype(np.int64)
    np.fill_diagonal(tmp, 0) 
    A['a'+str(k)+'_n_i'] = np.where(tmp + tmp.T - np.diag(tmp.diagonal()) > 0, 1, 0) 
    return 
    
def compute_kth_neighbor_out(adj, k, A):
    tmp = np.dot(A['a'+str(k)+'_d_o'].astype(np.float32), A['a'+str(k)+'_d_i'].astype(np.float32)).astype(np.int64)
    np.fill_diagonal(tmp, 0) 
    A['a'+str(k)+'_n_o'] = np.where(tmp + tmp.T - np.diag(tmp.diagonal()) > 0, 1, 0) 
    return 


def get_k_order_lp_matrix(edgelist, K=K_max, normalize_adj=False):
    #TODO: Find a way to paralellize this computation
    G = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
    adj = nx.adjacency_matrix(G).toarray()

    A = compute_proximity_matrices(adj, K=K)
    A = {key: sp.csr_matrix(A[key]) for key in A}
    adj_matrix = process_adj(A, normalize_adj)

    return adj_matrix