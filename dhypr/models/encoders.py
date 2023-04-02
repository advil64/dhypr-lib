import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.layers.hyp_layers as hyp_layers
from models.layers.layers import Linear
import pdb
from models.utils.utils import sparse_mx_to_torch_sparse_tensor, normalize
import os

        
class DHYPR(nn.Module):
    # TODO: what is c??
    def __init__(self, c, manifold, num_layers, proximity, feat_dim, hidden, dim, dropout, bias, alpha, n_nodes, use_att: bool=False):
        super(DHYPR, self).__init__()
        
        self.manifold = manifold
        self.c = c
        
        assert num_layers > 1
        self.dims, self.acts, self.curvatures = hyp_layers.get_dim_act_curv(num_layers, 
                                                                            feat_dim, hidden, dim)
        self.curvatures.append(c)
        
        if proximity == 1:
            self.model1_d_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_d_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_n_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_n_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
        elif proximity == 2:
            self.model1_d_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_d_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_n_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_n_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model2_d_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model2_d_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model2_n_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model2_n_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
        elif proximity == 3:
            self.model1_d_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_d_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_n_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model1_n_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model2_d_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model2_d_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model2_n_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model2_n_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model3_d_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model3_d_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model3_n_i = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
            self.model3_n_o = DHYPRLayer(self.manifold, self.dims, self.acts, self.curvatures, dropout, bias)
        else:
            os._exit(0)

        n_heads = 1
        
        self.embed_agg = hyp_layers.HypAttnAgg(self.manifold, c, self.dims[-1], dropout, alpha, use_att, n_heads)
        
        self.proximity = proximity
        self.nnodes = n_nodes
        self.nrepre = proximity*4+1
        self.embed_agg_adj_size = n_nodes * self.nrepre
        
        embed_agg_adj = np.zeros((self.embed_agg_adj_size, self.embed_agg_adj_size))
        for n in range(self.nnodes):
            block_start = n*self.nrepre
            embed_agg_adj[block_start][block_start+1: block_start+self.nrepre] = 1

        embed_agg_adj = sp.csr_matrix(embed_agg_adj)
        # TODO: need to normalize 
        self.embed_agg_adj = sparse_mx_to_torch_sparse_tensor(
            normalize(embed_agg_adj + sp.eye(embed_agg_adj.shape[0])))
    
        
    def forward(self, x, adj):
        if self.proximity == 1:
            # x1_d_is = self.model1_d_i.encode(x, adj['a1_d_i_norm'])
            # x1_d_os = self.model1_d_o.encode(x, adj['a1_d_o_norm'])
            # x1_n_is = self.model1_n_i.encode(x, adj['a1_n_i_norm'])
            # x1_n_os = self.model1_n_o.encode(x, adj['a1_n_o_norm'])

            x1_d_is = self.model1_d_i.encode(x, adj['a1_d_i'])
            x1_d_os = self.model1_d_o.encode(x, adj['a1_d_o'])
            x1_n_is = self.model1_n_i.encode(x, adj['a1_n_i'])
            x1_n_os = self.model1_n_o.encode(x, adj['a1_n_o'])
        
            x1_d_i = x1_d_is[-1]
            x1_d_o = x1_d_os[-1]
            x1_n_i = x1_n_is[-1]
            x1_n_o = x1_n_os[-1]
           
            ### target embedding
            target_context = torch.stack((x1_d_i, x1_d_o, x1_n_i, x1_n_o))
            x1_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_i, self.c)
            x1_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_o, self.c)
            x1_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_i, self.c)
            x1_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_o, self.c)
            
            target = self.manifold.mobius_add(
                self.manifold.mobius_add(
                    self.manifold.mobius_add(x1_d_i_w, x1_d_o_w, self.c), 
                    x1_n_i_w, self.c), 
                x1_n_o_w, self.c)

            target_context_feat = torch.cat((target_context, target.unsqueeze(0)), dim=0).permute(1, 0, 2)
            target_context_feat = target_context_feat.reshape(self.nnodes*self.nrepre, self.dims[-1])

            if self.args.use_att:
                output, output_attn = self.embed_agg.forward(target_context_feat, self.embed_agg_adj)
                output = output.reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, output]

                return embeddings, output_attn
            else:
                output = self.embed_agg.forward(target_context_feat, self.embed_agg_adj).reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, output]

                return embeddings
            
        elif self.proximity == 2:
            x1_d_is = self.model1_d_i.encode(x, adj['a1_d_i_norm'])
            x1_d_os = self.model1_d_o.encode(x, adj['a1_d_o_norm'])
            x1_n_is = self.model1_n_i.encode(x, adj['a1_n_i_norm'])
            x1_n_os = self.model1_n_o.encode(x, adj['a1_n_o_norm'])
            x2_d_is = self.model2_d_i.encode(x, adj['a2_d_i_norm'])
            x2_d_os = self.model2_d_o.encode(x, adj['a2_d_o_norm'])
            x2_n_is = self.model2_n_i.encode(x, adj['a2_n_i_norm'])
            x2_n_os = self.model2_n_o.encode(x, adj['a2_n_o_norm'])

            x1_d_i = x1_d_is[-1]
            x1_d_o = x1_d_os[-1]
            x1_n_i = x1_n_is[-1]
            x1_n_o = x1_n_os[-1]
            x2_d_i = x2_d_is[-1]
            x2_d_o = x2_d_os[-1]
            x2_n_i = x2_n_is[-1]
            x2_n_o = x2_n_os[-1]

            ### target embedding
            target_context = torch.stack((x1_d_i, x1_d_o, x1_n_i, x1_n_o, x2_d_i, x2_d_o, x2_n_i, x2_n_o))
            x1_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_i, self.c)
            x1_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_o, self.c)
            x1_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_i, self.c)
            x1_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_o, self.c)
            x2_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x2_d_i, self.c)
            x2_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x2_d_o, self.c)
            x2_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x2_n_i, self.c)
            x2_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x2_n_o, self.c)

            target = self.manifold.mobius_add(
                self.manifold.mobius_add(
                    self.manifold.mobius_add(
                        self.manifold.mobius_add(
                            self.manifold.mobius_add(
                                self.manifold.mobius_add(
                                    self.manifold.mobius_add(x1_d_i_w, x1_d_o_w, self.c), 
                                    x1_n_i_w, self.c), 
                                x1_n_o_w, self.c), 
                            x2_d_i_w, self.c), 
                        x2_d_o_w, self.c), 
                    x2_n_i_w, self.c), 
                x2_n_o_w, self.c)  

            target_context_feat = torch.cat((target_context, target.unsqueeze(0)), dim=0).permute(1, 0, 2)
            target_context_feat = target_context_feat.reshape(self.nnodes*self.nrepre, self.dims[-1])

            if self.args.use_att:
                output, output_attn = self.embed_agg.forward(target_context_feat, self.embed_agg_adj)
                output = output.reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, x2_d_is, x2_d_os, x2_n_is, x2_n_os, output]

                return embeddings, output_attn
            else:
                output = self.embed_agg.forward(target_context_feat, self.embed_agg_adj).reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, x2_d_is, x2_d_os, x2_n_is, x2_n_os, output]

                return embeddings
            
        elif self.proximity == 3:
            x1_d_is = self.model1_d_i.encode(x, adj['a1_d_i_norm'])
            x1_d_os = self.model1_d_o.encode(x, adj['a1_d_o_norm'])
            x1_n_is = self.model1_n_i.encode(x, adj['a1_n_i_norm'])
            x1_n_os = self.model1_n_o.encode(x, adj['a1_n_o_norm'])
            x2_d_is = self.model2_d_i.encode(x, adj['a2_d_i_norm'])
            x2_d_os = self.model2_d_o.encode(x, adj['a2_d_o_norm'])
            x2_n_is = self.model2_n_i.encode(x, adj['a2_n_i_norm'])
            x2_n_os = self.model2_n_o.encode(x, adj['a2_n_o_norm'])
            x3_d_is = self.model3_d_i.encode(x, adj['a3_d_i_norm'])
            x3_d_os = self.model3_d_o.encode(x, adj['a3_d_o_norm'])
            x3_n_is = self.model3_n_i.encode(x, adj['a3_n_i_norm'])
            x3_n_os = self.model3_n_o.encode(x, adj['a3_n_o_norm'])

            x1_d_i = x1_d_is[-1]
            x1_d_o = x1_d_os[-1]
            x1_n_i = x1_n_is[-1]
            x1_n_o = x1_n_os[-1]
            x2_d_i = x2_d_is[-1]
            x2_d_o = x2_d_os[-1]
            x2_n_i = x2_n_is[-1]
            x2_n_o = x2_n_os[-1]
            x3_d_i = x3_d_is[-1]
            x3_d_o = x3_d_os[-1]
            x3_n_i = x3_n_is[-1]
            x3_n_o = x3_n_os[-1]

            ### target embedding
            target_context = torch.stack((x1_d_i, x1_d_o, x1_n_i, x1_n_o, 
                                          x2_d_i, x2_d_o, x2_n_i, x2_n_o, 
                                          x3_d_i, x3_d_o, x3_n_i, x3_n_o))
            x1_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_i, self.c)
            x1_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_d_o, self.c)
            x1_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_i, self.c)
            x1_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x1_n_o, self.c)
            x2_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x2_d_i, self.c)
            x2_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x2_d_o, self.c)
            x2_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x2_n_i, self.c)
            x2_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x2_n_o, self.c)
            x3_d_i_w = self.manifold.mobius_mulscaler(1.0/8, x3_d_i, self.c)
            x3_d_o_w = self.manifold.mobius_mulscaler(1.0/8, x3_d_o, self.c)
            x3_n_i_w = self.manifold.mobius_mulscaler(1.0/8, x3_n_i, self.c)
            x3_n_o_w = self.manifold.mobius_mulscaler(1.0/8, x3_n_o, self.c)

            target = self.manifold.mobius_add(
                self.manifold.mobius_add(
                    self.manifold.mobius_add(
                        self.manifold.mobius_add(
                            self.manifold.mobius_add(
                                self.manifold.mobius_add(
                                    self.manifold.mobius_add(
                                        self.manifold.mobius_add(
                                            self.manifold.mobius_add(
                                                self.manifold.mobius_add(
                                                    self.manifold.mobius_add(x1_d_i_w, x1_d_o_w, self.c), 
                                                    x1_n_i_w, self.c), 
                                                x1_n_o_w, self.c), 
                                            x2_d_i_w, self.c), 
                                        x2_d_o_w, self.c), 
                                    x2_n_i_w, self.c), 
                                x2_n_o_w, self.c), 
                            x3_d_i_w, self.c), 
                        x3_d_o_w, self.c), 
                    x3_n_i_w, self.c), 
                x3_n_o_w, self.c)
                                                                                                                         
            target_context_feat = torch.cat((target_context, target.unsqueeze(0)), dim=0).permute(1, 0, 2)
            target_context_feat = target_context_feat.reshape(self.nnodes*self.nrepre, self.dims[-1])

            if self.args.use_att:
                output, output_attn = self.embed_agg.forward(target_context_feat, self.embed_agg_adj)
                output = output.reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, 
                              x2_d_is, x2_d_os, x2_n_is, x2_n_os, 
                              x3_d_is, x3_d_os, x3_n_is, x3_n_os, output]

                return embeddings, output_attn
            else:
                output = self.embed_agg.forward(target_context_feat, self.embed_agg_adj).reshape(
                    self.nnodes, self.nrepre, self.dims[-1])[:, 0, :]

                embeddings = [x1_d_is, x1_d_os, x1_n_is, x1_n_os, 
                              x2_d_is, x2_d_os, x2_n_is, x2_n_os,
                              x3_d_is, x3_d_os, x3_n_is, x3_n_os, output]

                return embeddings
        
        
    
    
class DHYPRLayer(nn.Module):
    def __init__(self, manifold, dims, acts, curvatures, dropout, bias):
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
