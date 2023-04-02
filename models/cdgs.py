import torch.nn as nn
import torch
import functools
from torch_geometric.utils import dense_to_sparse

from . import utils, layers
from .hmpb import HybridMPBlock

get_act = layers.get_act
conv1x1 = layers.conv1x1


@utils.register_model(name='CDGS')
class CDGS(nn.Module):
    """
    Graph Noise Prediction Model.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.act = act = get_act(config)

        # get input channels(data.num_channels), hidden channels(model.nf), number of blocks(model.num_res_blocks)
        self.nf = nf = config.model.nf
        self.num_gnn_layers = num_gnn_layers = config.model.num_gnn_layers
        dropout = config.model.dropout
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        self.conditional = conditional = config.model.conditional
        self.edge_th = config.model.edge_th
        self.rw_depth = rw_depth = config.model.rw_depth

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 2))
            modules.append(nn.Linear(nf * 2, nf))

        atom_ch = config.data.atom_channels
        bond_ch = config.data.bond_channels
        temb_dim = nf

        # project bond features
        assert bond_ch == 2
        bond_se_ch = int(nf * 0.4)
        bond_type_ch = int(0.5 * (nf - bond_se_ch))
        modules.append(conv1x1(1, bond_type_ch))
        modules.append(conv1x1(1, bond_type_ch))
        modules.append(conv1x1(rw_depth + 1, bond_se_ch))
        modules.append(nn.Linear(bond_se_ch + 2 * bond_type_ch, nf))

        # project atom features
        atom_se_ch = int(nf * 0.2)
        atom_type_ch = nf - 2 * atom_se_ch
        modules.append(nn.Linear(bond_ch, atom_se_ch))
        modules.append(nn.Linear(atom_ch, atom_type_ch))
        modules.append(nn.Linear(rw_depth, atom_se_ch))
        modules.append(nn.Linear(atom_type_ch + 2 * atom_se_ch, nf))
        self.x_ch = nf

        # gnn network
        cat_dim = (nf * 2) // num_gnn_layers
        for _ in range(num_gnn_layers):
            modules.append(HybridMPBlock(nf, config.model.graph_layer, "FullTrans_1", config.model.heads,
                                         temb_dim=temb_dim, act=act, dropout=dropout, attn_dropout=dropout))
            modules.append(nn.Linear(nf, cat_dim))
            modules.append(nn.Linear(nf, cat_dim))

        # atom output
        modules.append(nn.Linear(cat_dim * num_gnn_layers + atom_type_ch, nf))
        modules.append(nn.Linear(nf, nf // 2))
        modules.append(nn.Linear(nf // 2, atom_ch))

        # bond output
        modules.append(conv1x1(cat_dim * num_gnn_layers + bond_type_ch, nf))
        modules.append(conv1x1(nf, nf // 2))
        modules.append(conv1x1(nf // 2, 1))

        # structure output
        modules.append(conv1x1(cat_dim * num_gnn_layers + bond_type_ch, nf))
        modules.append(conv1x1(nf, nf // 2))
        modules.append(conv1x1(nf // 2, 1))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond, *args, **kwargs):

        atom_feat, bond_feat = x
        atom_mask = kwargs['atom_mask']
        bond_mask = kwargs['bond_mask']

        edge_exist = bond_feat[:, 1:, :, :]
        edge_cate = bond_feat[:, 0:1, :, :]

        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.data.centered:
            # rescale the input data to [-1, 1]
            atom_feat = atom_feat * 2. - 1.
            bond_feat = bond_feat * 2. - 1.

        # discretize dense adj
        with torch.no_grad():
            adj = edge_exist.squeeze(1).clone()  # [B, N, N]
            adj[adj >= 0.] = 1.
            adj[adj < 0.] = 0.
            adj = adj * bond_mask.squeeze(1)

        # extract RWSE and Shortest-Path Distance
        rw_landing, spd_onehot = utils.get_rw_feat(self.rw_depth, adj)

        # construct edge feature [B, N, N, F]
        adj_mask = bond_mask.permute(0, 2, 3, 1)
        dense_cate = modules[m_idx](edge_cate).permute(0, 2, 3, 1) * adj_mask
        m_idx += 1
        dense_exist = modules[m_idx](edge_exist).permute(0, 2, 3, 1) * adj_mask
        m_idx += 1
        dense_spd = modules[m_idx](spd_onehot).permute(0, 2, 3, 1) * adj_mask
        m_idx += 1
        dense_edge = modules[m_idx](torch.cat([dense_cate, dense_exist, dense_spd], dim=-1)) * adj_mask
        m_idx += 1

        # Use Degree as atom feature
        atom_degree = torch.sum(bond_feat, dim=-1).permute(0, 2, 1)  # [B, N, C]
        atom_degree = modules[m_idx](atom_degree)  # [B, N, nf]
        m_idx += 1
        atom_cate = modules[m_idx](atom_feat)
        m_idx += 1
        x_rwl = modules[m_idx](rw_landing)
        m_idx += 1
        x_atom = modules[m_idx](torch.cat([atom_degree, atom_cate, x_rwl], dim=-1))
        m_idx += 1
        h_atom = x_atom.reshape(-1, self.x_ch)
        # Dense to sparse node [BxN, -1]

        dense_index = adj.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj)
        h_dense_edge = dense_edge

        # Run GNN layers
        atom_hids = []
        bond_hids = []
        for _ in range(self.num_gnn_layers):
            h_atom, h_dense_edge = modules[m_idx](h_atom, edge_index, h_dense_edge, dense_index,
                                                  atom_mask, adj_mask, temb)
            m_idx += 1
            atom_hids.append(modules[m_idx](h_atom.reshape(x_atom.shape)))
            m_idx += 1
            bond_hids.append(modules[m_idx](h_dense_edge))
            m_idx += 1

        atom_hids = torch.cat(atom_hids, dim=-1)
        bond_hids = torch.cat(bond_hids, dim=-1)

        # Output
        atom_score = self.act(modules[m_idx](torch.cat([atom_cate, atom_hids], dim=-1))) \
                     * atom_mask.unsqueeze(-1)
        m_idx += 1
        atom_score = self.act(modules[m_idx](atom_score))
        m_idx += 1
        atom_score = modules[m_idx](atom_score)
        m_idx += 1

        bond_score = self.act(modules[m_idx](torch.cat([dense_cate, bond_hids], dim=-1).permute(0, 3, 1, 2))) \
                     * bond_mask
        m_idx += 1
        bond_score = self.act(modules[m_idx](bond_score))
        m_idx += 1
        bond_score = modules[m_idx](bond_score)
        m_idx += 1

        exist_score = self.act(modules[m_idx](torch.cat([dense_exist, bond_hids], dim=-1).permute(0, 3, 1, 2))) \
                      * bond_mask
        m_idx += 1
        exist_score = self.act(modules[m_idx](exist_score))
        m_idx += 1
        exist_score = modules[m_idx](exist_score)
        m_idx += 1

        # make score symmetric
        bond_score = torch.cat([bond_score, exist_score], dim=1)
        bond_score = (bond_score + bond_score.transpose(2, 3)) / 2.

        assert m_idx == len(modules)

        atom_score = atom_score * atom_mask.unsqueeze(-1)
        bond_score = bond_score * bond_mask

        return atom_score, bond_score
