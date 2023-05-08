import torch
import torch.nn.functional as f

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


@functional_transform("create_dummy_features")
class CreateDummyFeatures(BaseTransform):
    r"""adds a dummy feature matrix if one doesn't exist"""
    def __call__(self, data: Data) -> Data:
        # if data.x is None: TODO: think of a way to actually use the features
        if 'edge_label_index' in data.stores[0]:
            adj = to_dense_adj(data.edge_label_index)[0]
        else:
            adj = to_dense_adj(data.edge_index)[0]
        features = torch.eye(adj.shape[0], device=data.edge_index.device)
        data.x = features
        data.num_features = adj.shape[0]
        data.num_nodes = adj.shape[0]
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

    def __init__(self, normalize: bool = True, is_undirected: bool = False, k: int = 2):
        assert k <= 5, "max value for k is 5"

        self.is_undirected = is_undirected
        self.k = k
        self.normalize = normalize

        # populated in the call function
        self.adj = None

    def __call__(
        self,
        data: Data,
    ) -> Data:
        print("Generating k-order matrix...")
        self.adj = to_dense_adj(data.edge_label_index, max_num_nodes=data.num_nodes)[0].float()

        (
            data.k_diffusion_in,
            data.k_diffusion_out,
            data.k_neighbor_in,
            data.k_neighbor_out,
        ) = self._compute_proximity_matrices()
        data.proximity = self.k
        data.adj = self._normalize(self.adj) if self.normalize else self.adj
        print("Done!")
        return data

    def _normalize(self, mx: torch.tensor):
        mx += torch.eye(mx.size(0), device=mx.device)
        rowsum = mx.sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        return mx

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

        for k in range(self.k):
            # neighbor matrices
            tmp = torch.matmul(k_diffusion_in[k], k_diffusion_out[k]).int()
            tmp.fill_diagonal_(0)
            k_neighbor_in.append(
                torch.where(tmp + tmp.T - torch.diag(tmp.diagonal()) > 0.0, 1.0, 0.0)
            )

            tmp = torch.matmul(k_diffusion_out[k], k_diffusion_in[k]).int()
            tmp.fill_diagonal_(0)
            k_neighbor_out.append(
                torch.where(tmp + tmp.T - torch.diag(tmp.diagonal()) > 0.0, 1.0, 0.0)
            )

        # normalize if necessary also taken from the paper
        if self.normalize:
            k_diffusion_in = list(map(self._normalize, k_diffusion_in))
            k_diffusion_out = list(map(self._normalize, k_diffusion_out))
            k_neighbor_in = list(map(self._normalize, k_neighbor_in))
            k_neighbor_out = list(map(self._normalize, k_neighbor_out))

        # convert all matrices to sparse
        k_diffusion_in = list(map(lambda x: x.to_sparse(), k_diffusion_in))
        k_diffusion_out = list(map(lambda x: x.to_sparse(), k_diffusion_out))
        k_neighbor_in = list(map(lambda x: x.to_sparse(), k_neighbor_in))
        k_neighbor_out = list(map(lambda x: x.to_sparse(), k_neighbor_out))

        return k_diffusion_in, k_diffusion_out, k_neighbor_in, k_diffusion_out
