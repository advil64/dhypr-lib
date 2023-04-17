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

        # convert all matrices to sparse
        k_diffusion_in = list(map(lambda x: x.to_sparse(), k_diffusion_in))
        k_diffusion_out = list(map(lambda x: x.to_sparse(), k_diffusion_out))
        k_neighbor_in = list(map(lambda x: x.to_sparse(), k_neighbor_in))
        k_neighbor_out = list(map(lambda x: x.to_sparse(), k_neighbor_out))

        return k_diffusion_in, k_diffusion_out, k_neighbor_in, k_diffusion_out



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
