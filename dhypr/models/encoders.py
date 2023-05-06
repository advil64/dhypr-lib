import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from functools import reduce

import models.layers.hyp_layers as hyp_layers
from models.utils.utils import sparse_mx_to_torch_sparse_tensor, normalize


class DHYPR(nn.Module):
    def __init__(self, c, manifold, num_layers, proximity, feat_dim, hidden, dim, dropout, bias, alpha, n_nodes, act, device, n_heads: int = 4, use_att: bool = False):
        super(DHYPR, self).__init__()

        self.manifold = manifold
        self.c = c

        assert num_layers > 1
        self.dims, self.acts, self.curvatures = hyp_layers.get_dim_act_curv(num_layers, feat_dim, hidden, dim, act, c, device)
        self.curvatures.append(c)

        self.k_diffusion_in_layers = nn.ModuleList()
        self.k_diffusion_out_layers = nn.ModuleList()
        self.k_neighbor_in_layers = nn.ModuleList()
        self.k_neighbor_out_layers = nn.ModuleList()

        for i in range(proximity):
            # one layer for each k order matrix entry
            self.k_diffusion_in_layers.append(DHYPRLayer(dropout, bias, self.manifold, self.dims, self.acts, self.curvatures))
            self.k_diffusion_out_layers.append(DHYPRLayer(dropout, bias, self.manifold, self.dims, self.acts, self.curvatures))
            self.k_neighbor_in_layers.append(DHYPRLayer(dropout, bias, self.manifold, self.dims, self.acts, self.curvatures))
            self.k_neighbor_out_layers.append(DHYPRLayer(dropout, bias, self.manifold, self.dims, self.acts, self.curvatures))

        self.embed_agg = hyp_layers.HypAttnAgg(self.manifold, c, self.dims[-1], dropout, alpha, use_att, n_heads)
        self.proximity = proximity
        self.nnodes = n_nodes
        self.nrepre = proximity * 4 + 1
        self.embed_agg_adj_size = n_nodes * self.nrepre
        self.use_att = use_att

        embed_agg_adj = np.zeros((self.embed_agg_adj_size, self.embed_agg_adj_size), dtype=np.float32)
        for n in range(self.nnodes):
            block_start = n * self.nrepre
            embed_agg_adj[block_start][block_start + 1: block_start + self.nrepre] = 1

        embed_agg_adj = sp.csr_matrix(embed_agg_adj)
        self.embed_agg_adj = sparse_mx_to_torch_sparse_tensor(
            normalize(embed_agg_adj + sp.eye(embed_agg_adj.shape[0]))).to(device)

    def forward(self, x, adj, k_diffusion_in, k_diffusion_out, k_neighbor_in, k_neighbor_out):

        target_context_list = []
        target_weights_list = []
        embeddings = []

        for i in range(self.proximity):
            # TODO: use normalized versions of the k order matrices
            # TODO: add some comments here ask Honglu for these
            d_is = self.k_diffusion_in_layers[i].encode(x, k_diffusion_in[i])
            d_os = self.k_diffusion_out_layers[i].encode(x, k_diffusion_out[i])
            n_is = self.k_neighbor_in_layers[i].encode(x, k_neighbor_in[i])
            n_os = self.k_neighbor_out_layers[i].encode(x, k_neighbor_out[i])

            # input dimension?
            d_i = d_is[-1]
            d_o = d_os[-1]
            n_i = n_is[-1]
            n_o = n_os[-1]

            # target embedding
            d_iw = self.manifold.mobius_mulscaler(1.0 / 8, d_i, self.c)
            d_ow = self.manifold.mobius_mulscaler(1.0 / 8, d_o, self.c)
            n_iw = self.manifold.mobius_mulscaler(1.0 / 8, n_i, self.c)
            n_ow = self.manifold.mobius_mulscaler(1.0 / 8, n_o, self.c)

            target_context_list.extend([d_i, d_o, n_i, n_o])
            target_weights_list.extend([d_iw, d_ow, n_iw, n_ow])
            embeddings.extend([d_is, d_os, n_is, n_os])

        target_context = torch.stack(target_context_list)
        target = reduce(lambda a, b: self.manifold.mobius_add(a, b, self.c), target_weights_list)

        target_context_feat = torch.cat((target_context, target.unsqueeze(0)), dim=0).permute(1, 0, 2)
        target_context_feat = target_context_feat.reshape(self.nnodes * self.nrepre, self.dims[-1])

        if self.use_att:
            output, output_attn = self.embed_agg.forward(target_context_feat, self.embed_agg_adj)
            output = output.reshape(
                self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

            embeddings.append(output)

            return embeddings, output_attn
        else:
            output = self.embed_agg.forward(target_context_feat, self.embed_agg_adj).reshape(
                self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

            embeddings.append(output)

            return embeddings


class DHYPRLayer(nn.Module):
    def __init__(self, dropout, bias, manifold, dims, acts, curvatures):
        super(DHYPRLayer, self).__init__()
        self.manifold = manifold
        self.curvatures = curvatures
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out,
                    dropout, act, bias
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        embeddings = []
        input = (x_hyp, adj)
        for layer in self.layers:
            x_hyp = layer.forward(input)
            input = (x_hyp, adj)
            embeddings.append(x_hyp)

        return embeddings
