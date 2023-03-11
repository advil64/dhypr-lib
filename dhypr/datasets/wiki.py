import torch

import os.path as osp
import networkx as nx

from torch_geometric.data import Data, Dataset, download_url, extract_gz
from torch_geometric.utils import convert
from torch_geometric.transforms import RandomLinkSplit
from generate_k_order_matrix import get_k_order_lp_matrix
from os import remove

class Wiki(Dataset):
    r"""An implementation of the Wikipedia vote network dataset as collected
    by the research group at Stanford, more information about this dataset can 
    be found here <https://snap.stanford.edu/data/wiki-Vote.html>

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): task that you want to do on the dataset
        folds (int): Number of folds to split this dataset into as dictated by the user
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        create_k_order (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            k_order neighborhood matrix. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self, root, name, folds=10, transform=None, pre_transform=RandomLinkSplit(num_val=0.4, num_test=0.5), 
                 create_k_order=get_k_order_lp_matrix, pre_filter=None):
        # TODO: add an argument to choose k for the k_order_matrix (dictates the number of neighbors)
        self.name = name.lower()
        self.folds = folds
        self.create_k_order = create_k_order

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['wiki-Vote.txt', 'wikiElec.ElecBs3.txt']

    @property
    def processed_file_names(self):
        return [f'fold_{i}.pt' for i in range(self.folds)]

    def download(self):
        edge_url = 'https://snap.stanford.edu/data/wiki-Vote.txt.gz'
        attr_url = 'https://snap.stanford.edu/data/wikiElec.ElecBs3.txt.gz'

        edge_path = download_url(edge_url, self.raw_dir)
        print(f'Extracting wikipedia vote edges dataset from {edge_path}')
        extract_gz(edge_path, self.raw_dir)
        remove(edge_path)

        attr_path = download_url(attr_url, self.raw_dir)
        print(f'Extracting wikipedia edges attributes dataset from {edge_path}')
        extract_gz(attr_path, self.raw_dir)
        remove(attr_path)
            

    def process(self):
        # create a graph with edge attributes
        edge_path = osp.join(self.raw_dir, self.raw_file_names[1])
        # G = nx.read_edgelist(edge_path,create_using=nx.DiGraph(),nodetype = str,data = False)
        G = nx.DiGraph()

        # with the graph created, find an attribute (sign) for each edge (link)
        with open(edge_path, 'r', encoding='latin-1') as f:
            data = f.read().split('\n')
            src, dest, edge_sign = '', '', ''

            # loop through the votes data to find edge signs
            for row in data:
                if row == '' or row[0] in ['#', 'E', 'N']:
                    continue # skip the line it's not useful for us
                else:
                    vote = row.split('\t') # info that we want
                    if vote[0] == 'U':
                        dest = vote[1]
                    elif vote[0] == 'V':
                        edge_sign = vote[1]
                        src = vote[2]
                        G.add_edge(int(src), int(dest), vote=int(edge_sign))
        
        #generate the k order matrix for the whole graph and save as pt file out of the folds
        k_order_matrix = self.create_k_order(G.edges)
        k_order_matrix_data = Data(k_order_matrix=k_order_matrix)
        torch.save(k_order_matrix_data, osp.join(self.processed_dir, f'k_order_matrix.pt'))

        # convert the networkx graph into a pyg data object
        pyg_g = convert.from_networkx(G)

        # grab the edges from link sign prediction transform
        for f in range(self.folds):
            print(f'Processing fold {f}')
            
            #TODO: figure out if random link split does exactly what I need here, the resulting test/train/val
            #splits don't add up when adding up the edges
            train_data, val_data, test_data = self.pre_transform(pyg_g)

            # save split edges as a pytorch file
            graph = Data(train_edge_index=train_data.edge_index, train_edge_vote=train_data.vote,
                         val_edge_index=val_data.edge_index, val_edge_vote=val_data.vote,
                         test_edge_index=test_data.edge_index, test_edge_vote=test_data.vote)
            torch.save(graph, osp.join(self.processed_dir, f'fold_{f}.pt'))


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'fold_{idx}.pt'))
        return data