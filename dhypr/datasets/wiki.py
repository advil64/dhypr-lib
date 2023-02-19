import torch
import tarfile

import os.path as osp
import networkx as nx

from torch_geometric.data import Data, Dataset, download_url, extract_gz
from torch_geometric.utils import coalesce
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
    def __init__(self, root, name, folds=10, transform=None, pre_transform=None, create_k_order=None, pre_filter=None):
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
        # create a graph without edge attributes
        edge_path = osp.join(self.raw_dir, self.raw_file_names[0])
        G = nx.read_edgelist(edge_path,create_using=nx.DiGraph(),nodetype = str,data = False)

        # with the graph created, find an attribute (sign) for each edge
        print(G.number_of_nodes())
        print(G.number_of_edges())

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'fold_{idx}.pt'))
        return data