import os.path as osp

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import SNAPDataset, Planetoid
from torch_geometric.utils import negative_sampling

from models.manifolds.poincare import PoincareBall
from models.manifolds.base import Manifold
from models.encoders import DHYPR
from models.layers.layers import FermiDiracDecoder, GravityDecoder

from datasets.air import Air
from datasets.custom_transforms import GetKOrderMatrix, CreateDummyFeatures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.ToDevice(device),
    CreateDummyFeatures(),
    T.NormalizeFeatures(),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=False,
                      add_negative_train_samples=False),  # TODO: check if the random link split preserves the number of nodes in each set NOTE: (it doesn't)
    GetKOrderMatrix(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'datasets')
dataset = Air(root=path, name='Air', transform=transform)
# dataset = Planetoid(path, name='Cora', transform=transform)
# dataset = SNAPDataset(path, name='wiki-vote', transform=transform)

'''
NOTE: After applying the `RandomLinkSplit` transform, the data is transformed from
a data object to a list of tuples (train_data, val_data, test_data), with
each element representing the corresponding split.
'''

train_data, val_data, test_data = dataset[0]


class LPModel(nn.Module):

    def __init__(self, nnodes: int, feat_dim: int, proximity: int, num_layers: int = 2, dropout: float = 0.05,
                 momentum: float = 0.999, hidden: int = 64, device='cpu',
                 dim: int = 32, bias: int = 1, seed: int = 1234, epochs: int = 500, lamb: float = 0.1,
                 wl2: float = 0.1, beta: float = 1.0, r: float = 2.0, t: float = 1.0, alpha: float = 0.2):
        super(LPModel, self).__init__()

        self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = PoincareBall()
        self.act = nn.ReLU()

        self.encoder = DHYPR(self.c, self.manifold, num_layers, proximity, feat_dim, hidden, dim, dropout, bias, alpha, nnodes, self.act, device)
        self.dc = GravityDecoder(self.manifold, dim, 1, self.c, self.act, bias, beta, lamb)
        self.fd_dc = FermiDiracDecoder(r, t)

    def encode(self, x, adj, k_diffusion_in, k_diffusion_out, k_neighbor_in, k_neighbor_out):
        h = self.encoder.forward(x, adj, k_diffusion_in, k_diffusion_out, k_neighbor_in, k_neighbor_out)
        return h

    def decode(self, h, idx):
        emb_in = h[idx[0], :]
        emb_out = h[idx[1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        # squared distance between pairs of nodes in the hyperbolic space
        probs, mass = self.dc.forward(h, idx, sqdist)
        return probs, mass

    def fd_decode(self, h, idx):
        emb_in = h[idx[0], :]
        emb_out = h[idx[1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.fd_dc.forward(sqdist)
        return probs


model = LPModel(train_data.num_features, train_data.num_features, train_data.proximity).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.001, gamma=1.0)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.adj, train_data.k_diffusion_in,
                     train_data.k_diffusion_out, train_data.k_neighbor_in, train_data.k_neighbor_out)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat([
        train_data.edge_label_index,
        neg_edge_index
    ], dim=-1)

    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.fd_decode(z[-1], edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    # lr_scheduler.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index, data.k_diffusion_in,
                     data.k_diffusion_out, data.k_neighbor_in, data.k_neighbor_out)
    out = model.fd_decode(z[-1], data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


best_val_auc = final_test_auc = 0
for epoch in range(1, 500):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

# z = model.encode(test_data.x, test_data.edge_index)
# final_edge_index = model.decode_all(z)