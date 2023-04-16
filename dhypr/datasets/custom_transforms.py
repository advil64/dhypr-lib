import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj

@functional_transform("create_dummy_features")
class CreateDummyFeatures(BaseTransform):
    r"""adds a dummy feature matrix if one doesn't exist"""
    def __call__(self, data: Data) -> Data:
        assert data.num_nodes, 'data object must contain number of nodes'
        if data.x is None:
            features = torch.eye(data.num_nodes)
            data.x = features
        return data


@functional_transform("get_k_order_matrix")
class GetKOrderMatrix(BaseTransform):
    r"""
    Implements a class to create the k order matrix from a pytorch geometric data object. Methodology inspired by
    code from here <https://github.com/hongluzhou/directed_graph_emb>
    TODO: convert to a sparse tensor if necessary <https://github.com/hongluzhou/dhypr/blob/85e50daba0fb5a1da729a0c41b48933e72a0cb13/code/baselines/utils/data_utils.py#L111>
    TODO: normalize the matrices if necessary
    TODO add on support for hetero data objects
    """

    def __init__(self, is_undirected: bool = False, k: int = 2):
        assert k <= 5, "max value for k is 5"

        self.is_undirected = is_undirected
        self.k = k

        # populated in the call function
        self.adj = None

    def __call__(
        self,
        data: Data,
    ) -> Data:
        print("Generating k-order matrix...")
        self.adj = to_dense_adj(data.edge_label_index)[0].float()

        (
            data.k_diffusion_in,
            data.k_diffusion_out,
            data.k_neighbor_in,
            data.k_neighbor_out,
        ) = self._compute_proximity_matrices()
        data.proximity = self.k
        data.adj = self.adj
        print("Done!")
        return data

    def _compute_proximity_matrices(self):
        k_diffusion_in = []
        k_diffusion_out = []
        k_neighbor_in = []
        k_neighbor_out = []

        k_diffusion_in.append(self.adj.T)
        k_diffusion_out.append(self.adj)

        for k in range(1, self.k):
            # diffusion matrices
            k_diffusion_in.append(
                torch.where(
                    torch.matmul(k_diffusion_in[k - 1], k_diffusion_in[0]) > 0.0, 1.0, 0.0,
                )
            )
            k_diffusion_out.append(
                torch.where(
                    torch.matmul(k_diffusion_out[k - 1], k_diffusion_out[0]) > 0.0, 1.0, 0.0,
                )
            )

            # neighbor matrices
            tmp = torch.matmul(k_diffusion_in[k], k_diffusion_out[k]).int()
            tmp.fill_diagonal_(0)
            k_neighbor_in.append(
                np.where(tmp + tmp.T - torch.diag(tmp.diagonal()) > 0.0, 1.0, 0.0)
            )

            tmp = torch.matmul(k_diffusion_out[k], k_diffusion_in[k]).int()
            tmp.fill_diagonal_(0)
            k_neighbor_out.append(
                np.where(tmp + tmp.T - torch.diag(tmp.diagonal()) > 0.0, 1.0, 0.0)
            )
        return k_diffusion_in, k_diffusion_out, k_neighbor_in, k_diffusion_out
