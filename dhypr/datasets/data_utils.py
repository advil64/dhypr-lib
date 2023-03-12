'''
# return the transformed k sets of train, validation and test data in the ratio 8:1:1. 
##### Useful networkx methods include:
- read_edgelist
- adjacency_matrix
- from_scipy_sparse_matrix

##### Other:
- Gravity-Inspired Graph Autoencoders for Directed Link Prediction: https://arxiv.org/abs/1905.09570
- Benchmarking Graph Neural Networks: https://arxiv.org/abs/2003.00982
- scipy.sparse.csr.csr_matrix: https://stackoverflow.com/a/45786946
- sklearn train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

Note: the original input graph and generated train graph might not be a connect component by itself,
      which is reasonable (e.g., in Wiki-vote, user is node and edge is vote).
'''
import numpy as np
import math
import networkx as nx
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.filterwarnings("ignore")

import os
import scipy.sparse as sp
import torch
import pickle

verbose = False

def mask_edges_general_link_prediction(G, val_percent=0.1, test_percent=0.1, split_seed=1234):
    """
    Task 1: General Directed Link Prediction: get Train/Validation/Test

    :param G: networkx DiGraph()
    :param adj: complete sparse adjacency matrix of the graph
    :param val_percent: percentage of edges in validation set
    :param test_percent: percentage of edges in test set
    :return: train, validation and test edge sets
    """
    if verbose:
        print('\nTask 1: General Directed Link Prediction: kth fold cross validation split & save')
    
    # set random seed
    random.seed(split_seed)
    np.random.seed(split_seed)
    
    # We train models on incomplete versions of graphs where x% of edges were randomly removed. 
    # We take directionality into account in the masking process. 
    # In other words, if a link between node i and j is reciprocal, 
    # we can possibly remove the (i, j) edge 
    # but still observe the reverse (j,i) edge in the training incomplete graph. 
    # Then, we create validation and test sets from removed edges 
    # and form the same number of randomly sampled pairs of unconnected nodes. 
    
    toal_edges = G.number_of_edges()
    
    val_size = math.ceil(val_percent * toal_edges)
    test_size = math.ceil(test_percent * toal_edges)
    train_size = toal_edges - val_size - test_size
    
    # obtain train_pos_edges, val_test_pos_edges while making sure this incomplete version has all original nodes
    train_pos_edges = set()
    val_test_pos_edges = set()
    
    for node in G.nodes():
        try:
            train_pos_edges.add((node, list(G.neighbors(node))[0]))
        except IndexError:
            train_pos_edges.add(list(G.in_edges(node))[0])
        
    original_all_edges = list(G.edges())

    random.shuffle(original_all_edges)
    
    for (src, dst) in original_all_edges:
        if (src, dst) in train_pos_edges:
            continue
        else:
            if len(val_test_pos_edges) < (val_size + test_size):
                val_test_pos_edges.add((src, dst))
            else:
                train_pos_edges.add((src, dst))
    
    train_pos_edges = list(train_pos_edges)
    val_test_pos_edges = list(val_test_pos_edges)

    val_pos_edges, test_pos_edges  = train_test_split(val_test_pos_edges, test_size=test_size, 
        train_size=val_size, random_state=split_seed, shuffle=True, stratify=None)
    
    # sampling negative edges from unconnected node pairs
    val_test_neg_edges = set()
    
    while len(val_test_neg_edges) < (val_size + test_size):
        [src, dst] = random.sample(list(G.nodes()), 2)
        
        if (not G.has_edge(src, dst)) and (not G.has_edge(dst, src)):
            val_test_neg_edges.add((src, dst))
            
    val_test_neg_edges = list(val_test_neg_edges)
    
    val_neg_edges = val_test_neg_edges[:val_size]
    
    test_neg_edges = val_test_neg_edges[val_size:]
    
    '''assert'''
    assert not set(train_pos_edges) == set(val_pos_edges)
    assert not set(train_pos_edges) == set(val_neg_edges)
    assert not set(train_pos_edges) == set(test_pos_edges)
    assert not set(train_pos_edges) == set(test_neg_edges)
    assert not set(val_pos_edges) == set(val_neg_edges)
    assert not set(val_pos_edges) == set(test_pos_edges)
    assert not set(val_pos_edges) == set(test_neg_edges)
    assert not set(val_neg_edges) == set(test_pos_edges)
    assert not set(val_neg_edges) == set(test_neg_edges)
    assert not set(test_pos_edges) == set(test_neg_edges)
    
    assert len(train_pos_edges) == train_size
    assert len(val_pos_edges) == val_size
    assert len(val_neg_edges) == val_size
    assert len(test_pos_edges) == test_size
    assert len(test_neg_edges) == test_size

    G_train = nx.from_edgelist(train_pos_edges, create_using=nx.DiGraph)

    try:
        assert G_train.number_of_nodes() == G.number_of_nodes()
    except AssertionError:
        print("Task 1 -- Note: the training graph does not contain all nodes in the original graph!!!")
        
    return train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges


def mask_edges_bns_link_prediction(G, val_percent=0.1, test_percent=0.1, split_seed=1234):
    """
    Task 2: General Biased Negative Samples (B.N.S.) Directed Link Prediction: get Train/Validation/Test

    :param G: networkx DiGraph()
    :param val_percent: percentage of edges in validation set
    :param test_percent: percentage of edges in test set
    :return: train, validation and test edge sets
    """
    if verbose:
        print('\nTask 2: General Biased Negative Samples (B.N.S.) Directed Link Prediction: kth fold cross validation split & save')
    
    # set random seed
    random.seed(split_seed)
    np.random.seed(split_seed)
    
    # We train models on incomplete versions of graphs where x% of edges were randomly removed. 
    # In this setting, the reverse node pairs are included in validation and test sets and constitute negative samples. 
    # In other words, all node pairs from validation and test sets are included in both directions, and therefore 
    # we evaluate the ability of models to correctly reconstruct Aij = 1 and Aji = 0 simultaneously
    # (the ability to reconstruct asymmetric relationships).
    
    toal_edges = G.number_of_edges()
    
    val_size = math.ceil(val_percent * toal_edges)
    test_size = math.ceil(test_percent * toal_edges)
    train_size = toal_edges - val_size - test_size
    
    # get all possible removed edge set (i.e., node pair has the quality Aij = 1 and Aji = 0)
    # from all possible removed edges, put those with node out-degree 0 into train_pos_edges
    # to make sure the incomplete train version graph has all original nodes
    removed_edges = set()
    train_pos_edges = set()
    train_node_so_far = set()
    for (src, dst) in G.edges():
        if not (G.has_edge(dst, src)):  # all pair with Aij = 1 and Aji = 0
            if (len(list(G.out_edges(dst))) <= 0) or (len(list(G.in_edges(src))) <= 0):
                train_pos_edges.add((src, dst))
                train_node_so_far.add(src)
                train_node_so_far.add(dst)
            else:
                removed_edges.add((src, dst))
                
    # obtain train_pos_edges while making sure this incomplete version has all original nodes
    for node in G.nodes():
        if node in train_node_so_far:
            continue
            
        this_node_added = False
        for nei in list(G.neighbors(node)):
            if (node, nei) in removed_edges:
                continue
            else:
                train_pos_edges.add((node, nei))
                this_node_added = True
        
        if not this_node_added:
            nei = random.choice(list(G.neighbors(node)))
            # print(node, list(G.neighbors(node)), nei)
            # pdb.set_trace()
            train_pos_edges.add((node, nei))
            removed_edges.remove((node, nei))

    train_pos_edges = list(train_pos_edges)
    
    # obtain val_test_pos_edges 
    if len(removed_edges) < (val_size + test_size):
        print("Task 2 -- Note: the number of validation and test positive edges can not satisfy the user inputed percentage")
        val_test_pos_edges = set(removed_edges)
        test_size = math.ceil(len(removed_edges) / 2)
        val_size = len(removed_edges) - test_size
        train_size = toal_edges - val_size - test_size
    else:
        removed_edges = list(removed_edges)
        val_test_pos_edges = set(removed_edges[:(val_size + test_size)])
        train_pos_edges += removed_edges[(val_size + test_size):]
     
    # obtain rest train_pos_edges 
    for (src, dst) in G.edges():
        if not (src, dst) in val_test_pos_edges:
            train_pos_edges.append((src, dst))
            
    train_pos_edges = list(set(train_pos_edges))
    val_test_pos_edges = list(val_test_pos_edges)
                        
    val_pos_edges, test_pos_edges = train_test_split(val_test_pos_edges, test_size=test_size, 
        train_size=val_size, random_state=split_seed, shuffle=True, stratify=None)
    
    # obtain negative edges
    val_neg_edges = list()
    test_neg_edges = list()
    
    for (src, dst) in val_pos_edges:
        val_neg_edges.append((dst, src))
       
    for (src, dst) in test_pos_edges:
        test_neg_edges.append((dst, src))
    
    '''assert'''
    assert not set(train_pos_edges) == set(val_pos_edges)
    assert not set(train_pos_edges) == set(val_neg_edges)
    assert not set(train_pos_edges) == set(test_pos_edges)
    assert not set(train_pos_edges) == set(test_neg_edges)
    assert not set(val_pos_edges) == set(val_neg_edges)
    assert not set(val_pos_edges) == set(test_pos_edges)
    assert not set(val_pos_edges) == set(test_neg_edges)
    assert not set(val_neg_edges) == set(test_pos_edges)
    assert not set(val_neg_edges) == set(test_neg_edges)
    assert not set(test_pos_edges) == set(test_neg_edges)
    
    assert len(train_pos_edges) == train_size
    assert len(val_pos_edges) == val_size
    assert len(val_neg_edges) == val_size
    assert len(test_pos_edges) == test_size
    assert len(test_neg_edges) == test_size

    G_train = nx.from_edgelist(train_pos_edges, create_using=nx.DiGraph)

    try:
        assert G_train.number_of_nodes() == G.number_of_nodes()
    except AssertionError:
        print("Task 2 -- Note: the training graph does not contain all nodes in the original graph!!!")
        
    return train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges


def mask_edges_bidirectionality_link_prediction(G, split_seed=1234):
    """
    Task 3: Bidirectionality Prediction: get Train/Validation/Test

    :param G: networkx DiGraph()
    :return: train, validation and test edge sets
    """
    if verbose:
        print('\nTask 3: Bidirectionality Prediction: kth fold cross validation split & save')
    
    # set random seed
    random.seed(split_seed)
    np.random.seed(split_seed)

    # We evaluate the ability of models to discriminate bidirectional edges, 
    # i.e. reciprocal connections, from unidirectional edges. 
    # Specifically, we create an incomplete training graph 
    # by removing at random one of the two directions of all bidirectional edges. 
    # Therefore, the training graph only has unidirectional connections. 
    # Then, a binary classification problem is once again designed, 
    # aiming at retrieving bidirectional edges in a test set 
    # composed of their removed direction and of the same number of reverse directions 
    # from unidirectional edges (that are therefore fake edges). 
    # In other words, for each pair of nodes i, j from the test set, 
    # we observe a connection from j to i in the incomplete training graph, 
    # but only half of them are reciprocal.
    
    toal_edges = G.number_of_edges()
    
    # Step 1: find & remove at random one of the two directions of all bidirectional edges.
    
    train_pos_edges = set()
    val_test_pos_edges = set()
    val_test_neg_edges = set()
    
    for (src, dst) in G.edges():
        if G.has_edge(dst, src):  # reciprocal
            if random.choice([True, False]):
                dst, src = src, dst
                
            if (src, dst) not in val_test_pos_edges:
                train_pos_edges.add((src, dst))
                val_test_pos_edges.add((dst, src))
        else:
            train_pos_edges.add((src, dst))
            val_test_neg_edges.add((dst, src))  # fake edges
  
    val_test_pos_edges = list(val_test_pos_edges)
    val_test_neg_edges = list(val_test_neg_edges)
    train_pos_edges = list(train_pos_edges)
    
    try:
        assert len(val_test_pos_edges)==len(val_test_neg_edges)
    except AssertionError:
        random.shuffle(val_test_neg_edges)
        val_test_neg_edges = val_test_neg_edges[:len(val_test_pos_edges)]
        assert len(val_test_pos_edges)==len(val_test_neg_edges)
        
    train_size = len(train_pos_edges)
    test_size = math.ceil(len(val_test_pos_edges) / 2)
    val_size = len(val_test_pos_edges) - test_size
    
        
    val_pos_edges, test_pos_edges  = train_test_split(val_test_pos_edges, test_size=test_size, 
        train_size=val_size, random_state=split_seed, shuffle=True, stratify=None)
    
    val_neg_edges, test_neg_edges  = train_test_split(val_test_neg_edges, test_size=test_size, 
        train_size=val_size, random_state=split_seed, shuffle=True, stratify=None)
    
    if verbose:
        print("no of train edges: ", len(train_pos_edges))
        print("no of val edges: ", len(val_pos_edges))
        print("no of test edges: ", len(test_pos_edges))

        print("ratio of train edges to original graph edges: ", len(train_pos_edges) / toal_edges)
    
    '''assert'''
    assert not set(train_pos_edges) == set(val_pos_edges)
    assert not set(train_pos_edges) == set(val_neg_edges)
    assert not set(train_pos_edges) == set(test_pos_edges)
    assert not set(train_pos_edges) == set(test_neg_edges)
    assert not set(val_pos_edges) == set(val_neg_edges)
    assert not set(val_pos_edges) == set(test_pos_edges)
    assert not set(val_pos_edges) == set(test_neg_edges)
    assert not set(val_neg_edges) == set(test_pos_edges)
    assert not set(val_neg_edges) == set(test_neg_edges)
    assert not set(test_pos_edges) == set(test_neg_edges)
    
    assert len(train_pos_edges) == train_size
    assert len(val_pos_edges) == val_size
    assert len(val_neg_edges) == val_size
    assert len(test_pos_edges) == test_size
    assert len(test_neg_edges) == test_size

    G_train = nx.from_edgelist(train_pos_edges, create_using=nx.DiGraph)

    try:
        assert G_train.number_of_nodes() == G.number_of_nodes()
    except AssertionError:
        print("Task 3 -- Note: the training graph does not contain all nodes in the original graph!!!")
    
    return train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges

def mask_edges_link_sign_prediction(G, val_percent=0.4, test_percent=0.5, split_seed=1234):
    # set random seed
    random.seed(split_seed)
    np.random.seed(split_seed)

    edges = G.edges()
    random.shuffle(edges)
    
    val_size = int(val_percent*len(edges))
    test_size = int(test_percent*len(edges))
    
    test_edges = edges[:test_size]
    val_edges = edges[test_size:test_size+val_size]
    train_edges = edges[test_size+val_size:]
        
    return train_edges, val_edges, test_edges


def load_graph(filepath):
    G = nx.read_edgelist(os.path.join(filepath, 'train_edges.txt'), 
                         delimiter='\t', create_using=nx.DiGraph())
    adj = nx.adjacency_matrix(G)
    
    list_of_nodes = list(G.nodes())
    
    node_str2int = dict()
    for i in range(len(list_of_nodes)):
        node = list_of_nodes[i]
        node_str2int[node] = i

    def load_edges(txt_path):
        with open(txt_path, 'r') as f:
            data = f.readlines()
            
        edges = []
        for i in range(len(data)):
            [src, dst] = data[i].split()
            src, dst = node_str2int[src], node_str2int[dst]
            edges.append([src, dst])
            
        edges = np.array(edges)
        return edges
            
    val_edges = load_edges(os.path.join(filepath, 'val_pos_edges.txt'))
    val_edges_false = load_edges(os.path.join(filepath, 'val_neg_edges.txt'))
    test_edges = load_edges(os.path.join(filepath, 'test_pos_edges.txt'))
    test_edges_false = load_edges(os.path.join(filepath, 'test_neg_edges.txt'))
    
    train_edges = load_edges(os.path.join(filepath, 'train_edges.txt'))
    
    
    # get train_edges_false 
    train_edges_false = []
        
    for nodei in list_of_nodes:
        for nodej in list_of_nodes:
            if not G.has_edge(nodei, nodej):
                src, dst = node_str2int[nodei], node_str2int[nodej]
                train_edges_false.append([src, dst])
                    
    train_edges_false = np.array(train_edges_false)
                    
    assert train_edges_false.shape[0] + train_edges.shape[0] == len(G) * len(G)
        
    return G, adj, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, node_str2int
   

def load_data(args):
    if args.task == 'nc':
        data = load_data_nc_directed(args, args.dataset_path, args.dataset, args.use_feats)
        
        k_order_matrices = load_proximity_matrices(os.path.join(args.dataset_path, args.dataset, 
                                                                'Node_Classification', 
                                                                'train_graph_kth_order_matrices.pickle'))
        if args.wlp > 0:
            train_edges, train_edges_false = mask_train_edges(data['adj_train'], args.split_seed)
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false

            
    else:
        if args.task == 'sp':  
            data = load_data_sp_directed(args, args.dataset_path, args.dataset, args.use_feats)
        
            k_order_matrices = load_proximity_matrices(os.path.join(args.dataset_path, args.dataset, 
                                                                    'Link_Sign_Prediction', 
                                                                    'train_graph_kth_order_matrices.pickle'))
            if args.wlp > 0:
                train_edges, train_edges_false = mask_train_edges(data['adj_train'], args.split_seed)
                data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false

            
            
        if args.task == 'lp':
            G, adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, node_str2int = \
                load_graph(os.path.join(args.dataset_path, args.dataset, 'General_Directed_Link_Prediction',
                                        'fold_{}'.format(args.fold)))
            
            # If features are not used, replace feature matrix by identity matrix
            if not args.use_feats:
                features = sp.identity(adj_train.shape[0])
            
            train_edges_false = torch.LongTensor(train_edges_false).to(args.device)
            train_edges = torch.LongTensor(train_edges).to(args.device)
            val_edges = torch.LongTensor(val_edges).to(args.device)
            val_edges_false = torch.LongTensor(val_edges_false).to(args.device)
            test_edges = torch.LongTensor(test_edges).to(args.device)
            test_edges_false = torch.LongTensor(test_edges_false).to(args.device)
            
            data = dict()
            data['node_str2int'] = node_str2int
            data['node_int2str'] = dict()
            for node_str in node_str2int:
                data['node_int2str'][node_str2int[node_str]] = node_str
            data['adj_train'] = adj_train
            data['features'] = features
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
            
            k_order_matrices = load_proximity_matrices(os.path.join(args.dataset_path, args.dataset, 
                                                                    'General_Directed_Link_Prediction', 
                                                                    'fold_{}'.format(args.fold),
                                                                    'train_graph_kth_order_matrices.pickle'))
            
    data.update(k_order_matrices)
    
    data['features'] = process_feat(data['features'], args.normalize_feats)
    
    data = process_adj(data, args.normalize_adj)
    
    return data



def load_proximity_matrices(matrices_path):
    with open(matrices_path, 'rb') as f:
        return pickle.load(f)

    
def process_feat(features, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    
    return features


def process_adj(data, normalize_adj):
    data['adj_train_norm'] = dict()
    
    if normalize_adj:
        if 'adj_train' in data:
            data['adj_train_norm']['adj_train_norm'] = sparse_mx_to_torch_sparse_tensor(
                    normalize(data['adj_train'] + sp.eye(data['adj_train'].shape[0])))
        if 'a1_d_i' in data:
            data['adj_train_norm']['a1_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_d_i'] + sp.eye(data['a1_d_i'].shape[0])))
        if 'a1_d_o' in data:
            data['adj_train_norm']['a1_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_d_o'] + sp.eye(data['a1_d_o'].shape[0])))
        if 'a1_n_i' in data:
            data['adj_train_norm']['a1_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_n_i'] + sp.eye(data['a1_n_i'].shape[0])))
        if 'a1_n_o' in data:
            data['adj_train_norm']['a1_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_n_o'] + sp.eye(data['a1_n_o'].shape[0])))
        if 'a2_d_i' in data:
            data['adj_train_norm']['a2_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_d_i'] + sp.eye(data['a2_d_i'].shape[0])))
        if 'a2_d_o' in data:
            data['adj_train_norm']['a2_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_d_o'] + sp.eye(data['a2_d_o'].shape[0])))
        if 'a2_n_i' in data:
            data['adj_train_norm']['a2_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_n_i'] + sp.eye(data['a2_n_i'].shape[0])))
        if 'a2_n_o' in data:
            data['adj_train_norm']['a2_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_n_o'] + sp.eye(data['a2_n_o'].shape[0])))
        if 'a3_d_i' in data:
            data['adj_train_norm']['a3_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_d_i'] + sp.eye(data['a3_d_i'].shape[0])))
        if 'a3_d_o' in data:
            data['adj_train_norm']['a3_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_d_o'] + sp.eye(data['a3_d_o'].shape[0])))
        if 'a3_n_i' in data:
            data['adj_train_norm']['a3_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_n_i'] + sp.eye(data['a3_n_i'].shape[0])))
        if 'a3_n_o' in data:
            data['adj_train_norm']['a3_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_n_o'] + sp.eye(data['a3_n_o'].shape[0])))
    else:
        if 'adj_train' in data:
            data['adj_train_norm']['adj_train_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['adj_train'])
        if 'a1_d_i' in data:
            data['adj_train_norm']['a1_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_d_i'])
        if 'a1_d_o' in data:
            data['adj_train_norm']['a1_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_d_o'])
        if 'a1_n_i' in data:
            data['adj_train_norm']['a1_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_n_i'])
        if 'a1_n_o' in data:
            data['adj_train_norm']['a1_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a1_n_o'])
        if 'a2_d_i' in data:
            data['adj_train_norm']['a2_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_d_i'])
        if 'a2_d_o' in data:
            data['adj_train_norm']['a2_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_d_o'])
        if 'a2_n_i' in data:
            data['adj_train_norm']['a2_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_n_i'])
        if 'a2_n_o' in data:
            data['adj_train_norm']['a2_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_n_o'])
        if 'a3_d_i' in data:
            data['adj_train_norm']['a3_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_d_i'])
        if 'a3_d_o' in data:
            data['adj_train_norm']['a3_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_d_o'])
        if 'a3_n_i' in data:
            data['adj_train_norm']['a3_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_n_i'])
        if 'a3_n_o' in data:
            data['adj_train_norm']['a3_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_n_o'])
            
    return data


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def mask_train_edges(adj, seed):
    # get tp edges
    np.random.seed(seed)  
    x, y = adj.nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    
    # get tn edges
    x, y = sp.csr_matrix(1. - adj.toarray()).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)
    
    return torch.LongTensor(pos_edges), torch.LongTensor(neg_edges)


def load_edges_sp(txt_path, cuda_device, no_sign=False):
    
    with open(txt_path, 'r') as f:
        data = f.readlines()
            
    edges = []
    if not no_sign:
        signs = []
        
    for i in range(len(data)):
        if not no_sign:
            [src, dst, sign] = data[i].split()
            src = int(src)
            dst = int(dst)
            sign = int(sign)
            edges.append([src, dst])
            signs.append(sign)
        else:
            [src, dst] = data[i].split()
            src = int(src)
            dst = int(dst)
            edges.append([src, dst])
        
    edges = torch.LongTensor(np.array(edges)).to(cuda_device)
    
    if not no_sign:
        signs = torch.LongTensor(np.array(signs))
        return edges, signs
    else:
        return edges
    

def load_data_sp_directed(args, data_path, dataset, use_feats):
    if dataset in ['wiki']:
                
        graph = load_npz_dataset(os.path.join(data_path, dataset, 'Link_Sign_Prediction', dataset + '.npz'))
        
        adj = graph['A']
        features = graph['X']
        labels = graph['z']
        
        if dataset == 'wiki':
            # features = feature_normalize_wiki(features)
            features = sp.identity(adj.shape[0])
        
        train_edges, train_edges_signs = load_edges_sp(
            os.path.join(data_path, dataset, 'Link_Sign_Prediction', 'fold_{}'.format(args.fold), 'train_edges.txt'),
            args.device)
        val_edges, val_edges_signs = load_edges_sp(
            os.path.join(data_path, dataset, 'Link_Sign_Prediction', 'fold_{}'.format(args.fold), 'val_edges.txt'),
            args.device)
        test_edges, test_edges_signs = load_edges_sp(
            os.path.join(data_path, dataset, 'Link_Sign_Prediction', 'fold_{}'.format(args.fold), 'test_edges.txt'),
            args.device)
        
        
        data = dict()
        data['adj_train'] = adj
        data['features'] = features
        data['train_sign_edges'] = train_edges
        data['val_sign_edges'] = val_edges
        data['test_sign_edges'] = test_edges
        data['train_sign_labels'] = train_edges_signs
        data['val_sign_labels'] = val_edges_signs
        data['test_sign_labels'] = test_edges_signs
        
    else:
        print('undefined dataset!')
        os._exit(0)
        
    return data


def feature_normalize_wiki(features):
    features = features.toarray()
    
    # normalize features
    features_new = np.copy(features[:features.shape[0]]).astype('float64') 
    for dim in range(features_new.shape[1]):
        max_dim = float(max(features_new[:, dim]))
        min_dim = float(min(features_new[:, dim]))
        diff_dim = max_dim - min_dim
        # print('dim: {} max: {} min: {}'.format(dim, max_dim, min_dim))
        for i in range(features_new.shape[0]):
            features_new[i, dim] = (features[i, dim] - min_dim)/diff_dim

    features = sp.csr_matrix(features_new) 

    return features


def load_data_nc_directed(args, data_path, dataset, use_feats):
    if dataset in ['cora_ml', 'citeseer', 'wiki']:
        if not args.vnc:
            graph = load_npz_dataset(os.path.join(data_path, dataset, 'Node_Classification', dataset + '.npz'))
            
            with open(os.path.join(data_path, dataset, 'Node_Classification', 'fold_{}'.format(args.fold), 
                               dataset + '_splits.pkl'), 'rb') as f:
                splits = pickle.load(f)
        else:
            graph = load_npz_dataset(os.path.join(data_path, dataset, 'Vary_Labeled_Nodes', dataset + '.npz'))
            
            with open(os.path.join(data_path, dataset, 'Vary_Labeled_Nodes', 'fold_{}'.format(args.fold), 
                               dataset + '_splits.pkl'), 'rb') as f:
                splits = pickle.load(f)
            
        
        adj = graph['A']
        features = graph['X']
        labels = graph['z']
        
        if dataset == 'wiki':
            # features = feature_normalize_wiki(features)
            features = sp.identity(adj.shape[0])
        
        idx_train = list(np.where(splits['train_mask']>0)[0])
        idx_val = list(np.where(splits['val_mask']>0)[0])
        idx_test = list(np.where(splits['test_mask']>0)[0])
    else:
        print('undefined dataset!')
        os._exit(0)
        
    try:
        labels = torch.LongTensor(labels)
    except:
        labels = torch.LongTensor(labels.astype(int))
        
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val,
            'idx_test': idx_test}
    return data
    
def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        try:
            idx_to_node = loader.get('idx_to_node')
            if idx_to_node:
                idx_to_node = idx_to_node.tolist()
                graph['idx_to_node'] = idx_to_node

            idx_to_attr = loader.get('idx_to_attr')
            if idx_to_attr:
                idx_to_attr = idx_to_attr.tolist()
                graph['idx_to_attr'] = idx_to_attr

            idx_to_class = loader.get('idx_to_class')
            if idx_to_class:
                idx_to_class = idx_to_class.tolist()
                graph['idx_to_class'] = idx_to_class
        except:
            pass

        return graph
    
def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def my_train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    #print('number of training: {}'.format(len(train_indices)))
    #print('number of validation: {}'.format(len(val_indices)))
    #print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask



def wiki_datasets(path="./data", dataset='cora_ml', seed=1020):
    os.makedirs(path, exist_ok=True)
    dataset_path = os.path.join(path, '{}.npz'.format(dataset))
    g = load_npz_dataset(dataset_path)
    adj, features, labels = g['A'], g['X'], g['z']
    
    labels = labels[:adj.shape[1]]
    
    # Set new random splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * the rest for testing

    mask = train_test_split(labels, seed=seed, train_examples_per_class=20, val_size=500, test_size=None)
    
    
    print("this is just the original directed adj.")
    
    data = dict()
    data['train_mask'] = mask['train']
    data['val_mask'] = mask['val']
    data['test_mask'] = mask['test']
    
    return data
    