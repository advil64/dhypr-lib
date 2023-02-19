import torch
import tarfile

import os.path as osp
import networkx as nx

from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.utils import coalesce
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
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
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
        return ['maayan-faa/out.maayan-faa']

    @property
    def processed_file_names(self):
        return [f'fold_{i}.pt' for i in range(self.folds)]

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
        
        # split into k-folds and generate the task masking
        for f in range(self.folds):
            print(f'Processing fold {f}')

            train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges = \
                data if self.pre_transform is None else self.pre_transform(G, split_seed=f)

            # generate a k order matrix if the method is passed
            k_order_matrix = None if self.create_k_order is None else self.create_k_order(train_pos_edges)

            # convert each set of edges into tensors
            src_tr, dest_tr = zip(*train_pos_edges)
            train_pos_edges = torch.tensor([src_tr, dest_tr])

            src_val_pos, dest_val_pos = zip(*val_pos_edges)
            val_pos_edges = torch.tensor([src_val_pos, dest_val_pos])

            src_val_neg, dest_val_neg = zip(*val_neg_edges)
            val_neg_edges = torch.tensor([src_val_neg, dest_val_neg])

            src_test_pos, dest_test_pos = zip(*test_pos_edges)
            test_pos_edges = torch.tensor([src_test_pos, dest_test_pos])

            src_test_neg, dest_test_neg = zip(*test_neg_edges)
            test_neg_edges = torch.tensor([src_test_neg, dest_test_neg])

            # save as a pt file
            graph = Data(train_pos_edge_index=train_pos_edges, val_pos_edge_index=val_pos_edges, 
                val_neg_edge_index=val_neg_edges, test_pos_edge_index=test_pos_edges, 
                test_neg_edge_index=test_neg_edges, k_order_matrix=k_order_matrix)
            torch.save(graph, osp.join(self.processed_dir, f'fold_{f}.pt'))


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'fold_{idx}.pt'))
        return data