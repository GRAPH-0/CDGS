import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn

from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import dense_to_sparse
from .transformer_layers import EdgeGateTransLayer


class HybridMPBlock(nn.Module):
    """Local MPNN + fully-connected attention-based message passing layer. Inspired by GPSLayer."""

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 temb_dim=None, act=None, dropout=0.0, attn_dropout=0.0):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        if act is None:
            self.act = nn.ReLU()
        else:
            self.act = act

        # time embedding
        if temb_dim is not None:
            self.t_node = nn.Linear(temb_dim, dim_h)
            self.t_edge = nn.Linear(temb_dim, dim_h)

        # local message-passing model
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), nn.ReLU(), Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'LocalTrans_1':
            self.local_model = EdgeGateTransLayer(dim_h, dim_h // num_heads, num_heads, edge_dim=dim_h)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'FullTrans_1':
            self.self_attn = EdgeGateTransLayer(dim_h, dim_h // num_heads, num_heads, edge_dim=dim_h)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")

        # Normalization for MPNN and Self-Attention representations.
        self.norm1_local = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)
        self.norm1_attn = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        # Feed Forward block -> node.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.norm2_node = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        # Feed Forward block -> edge.
        self.ff_linear3 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear4 = nn.Linear(dim_h * 2, dim_h)
        self.norm2_edge = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

    def _ff_block_node(self, x):
        """Feed Forward block.
        """
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def _ff_block_edge(self, x):
        """Feed Forward block.
        """
        x = self.dropout(self.act(self.ff_linear3(x)))
        return self.dropout(self.ff_linear4(x))

    def forward(self, x, edge_index, dense_edge, dense_index, node_mask, adj_mask, temb=None):
        """
        Args:
            x: node feature [B*N, dim_h]
            edge_index: [2, edge_length]
            dense_edge: edge features in dense form [B, N, N, dim_h]
            dense_index: indices for valid edges [B, N, N, 1]
            node_mask: [B, N]
            adj_mask: [B, N, N, 1]
            temb: time conditional embedding [B, temb_dim]
        Returns:
            h
            edge
        """

        B, N, _, _ = dense_edge.shape
        h_in1 = x
        h_in2 = dense_edge

        if temb is not None:
            h_edge = (dense_edge + self.t_edge(self.act(temb))[:, None, None, :]) * adj_mask
            temb = temb.unsqueeze(1).repeat(1, N, 1)
            temb = temb.reshape(-1, temb.size(-1))
            h = (x + self.t_node(self.act(temb))) * node_mask.reshape(-1, 1)

        h_out_list = []
        # Local MPNN with edge attributes
        if self.local_model is not None:
            edge_attr = h_edge[dense_index]
            h_local = self.local_model(h, edge_index, edge_attr) * node_mask.reshape(-1, 1)
            h_local = h_in1 + self.dropout(h_local)
            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention
        if self.self_attn is not None:
            if 'FullTrans' in self.global_model_type:
                # extract full connect edge_index and edge_attr
                dense_index_full = adj_mask.squeeze(-1).nonzero(as_tuple=True)
                edge_index_full, _ = dense_to_sparse(adj_mask.squeeze(-1))
                edge_attr_full = h_edge[dense_index_full]
                h_attn = self.self_attn(h, edge_index_full, edge_attr_full)
            else:
                raise ValueError(f"Unsupported global transformer layer")
            h_attn = h_in1 + self.dropout(h_attn)
            h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs
        assert len(h_out_list) > 0
        h = sum(h_out_list) * node_mask.reshape(-1, 1)
        h_dense = h.reshape(B, N, -1)
        h_edge = h_dense.unsqueeze(1) + h_dense.unsqueeze(2)

        # Feed Forward block
        h = h + self._ff_block_node(h)
        h = self.norm2_node(h) * node_mask.reshape(-1, 1)

        h_edge = h_in2 + self._ff_block_edge(h_edge)
        h_edge = self.norm2_edge(h_edge.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * adj_mask

        return h, h_edge

