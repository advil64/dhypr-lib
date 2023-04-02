import torch
import tarfile

import os.path as osp
import networkx as nx
import numpy as np

from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.utils import coalesce
from datasets.generate_k_order_matrix import get_k_order_lp_matrix
from datasets.data_utils import mask_edges_general_link_prediction
from torch_geometric.typing import SparseTensor
from os import remove

class Air(Dataset):
    r"""An implementation of the Air Traffic Control dataset as collected
    by the research group at mount sinai, more information about this dataset can 
    be found here <http://konect.cc/networks/maayan-faa/>

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): task that you want to do on the dataset
        folds (int): Number of folds to split this dataset into as dictated by the user
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self, name='air', transform=None, pre_transform=get_k_order_lp_matrix, 
                proximity=1, pre_filter=None, root=None):
        self.name = name.lower()
        self.proximity = proximity
        if root is None:
            root = osp.join(osp.dirname(osp.realpath(__file__)), 'air')

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['maayan-faa/out.maayan-faa']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = 'http://konect.cc/files/download.tsv.maayan-faa.tar.bz2'

        path = download_url(url, self.raw_dir)
        print(f'Extracting air traffic control dataset from {path}')
        
        my_tar = tarfile.open(path)
        remove(path)
        my_tar.extractall(self.raw_dir)
            

    def process(self):
        # create an empty networkx digraph
        G = nx.DiGraph()

        # populate the digraph accordingly
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'r') as f:
            data = f.read().split('\n')[1:-1]
            for row in data:
                src, dst = row.split()
                G.add_edge(int(src), int(dst))

        # generate a dummy features tensor
        adj_matrix = nx.adjacency_matrix(G)
        features = torch.eye(adj_matrix.shape[0])

        # get all edges
        original_all_edges = list(G.edges())
        src_pos, dest_pos = zip(*original_all_edges)
        edges = torch.tensor([src_pos, dest_pos])

        # generate the k order matrix
        k_order_matrix = self.pre_transform(original_all_edges, self.proximity)

        self.data = Data(edge_index=edges, x=features, num_nodes=features.shape[0], num_features=features.shape[1], k_order_matrix=k_order_matrix,)
        torch.save(self.data, self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data.pt'))
        return data