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
