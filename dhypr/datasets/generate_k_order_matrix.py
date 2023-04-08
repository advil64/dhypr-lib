import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


@functional_transform("k_order_matrix")
class KOrderMatrix(BaseTransform):
    r"""
    Implements a class to create the k order matrix from a pytorch geometric data object. Methodology inspired by
    code from here <https://github.com/hongluzhou/directed_graph_emb>
    TODO: convert to a sparse tensor if necessary <https://github.com/hongluzhou/dhypr/blob/85e50daba0fb5a1da729a0c41b48933e72a0cb13/code/baselines/utils/data_utils.py#L111>
    TODO: normalize the matrices if necessary
    """

    def __init__(self, is_undirected: bool = False, k: int = 2):
        assert k <= 5, "max value for k is 5"

        self.is_undirected = is_undirected
        self.k = k

        # populated in the call function
        self.adj = None

        # created by my functions
        self.k_diffusion_in = []
        self.k_diffusion_out = []
        self.k_neighbor_in = []
        self.k_neighbor_out = []

    def __call__(
        self,
        data: Data,
    ) -> Data:
        # TODO add on support for hetero data objects
        self.adj = to_dense_adj(data.edge_index)[0].float()
        self._compute_proximity_matrices()

        data.k_diffusion_in = self.k_diffusion_in
        data.k_diffusion_out = self.k_diffusion_out
        data.k_neighbor_in = self.k_neighbor_in
        data.k_neighbor_out = self.k_neighbor_out
        return data

    def _compute_proximity_matrices(self):
        self.k_diffusion_in.append(self.adj.T)
        self.k_diffusion_out.append(self.adj)

        for k in range(1, self.k):
            # diffusion matrices
            self.k_diffusion_in.append(
                torch.where(
                    torch.matmul(self.k_diffusion_in[k - 1], self.k_diffusion_in[0]) > 0.0, 1.0, 0.0,
                )
            )
            self.k_diffusion_out.append(
                torch.where(
                    torch.matmul(self.k_diffusion_out[k - 1], self.k_diffusion_out[0]) > 0.0, 1.0, 0.0,
                )
            )

            # neighbor matrices
            tmp = torch.matmul(self.k_diffusion_in[k], self.k_diffusion_out[k]).int()
            tmp.fill_diagonal_(0)
            self.k_neighbor_in.append(
                np.where(tmp + tmp.T - torch.diag(tmp.diagonal()) > 0.0, 1.0, 0.0)
            )

            tmp = torch.matmul(self.k_diffusion_out[k], self.k_diffusion_in[k]).int()
            tmp.fill_diagonal_(0)
            self.k_neighbor_out.append(
                np.where(tmp + tmp.T - torch.diag(tmp.diagonal()) > 0.0, 1.0, 0.0)
            )