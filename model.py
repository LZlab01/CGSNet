import torch
import numpy as np
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.conv import MessagePassing
from utils import clones, attention, padding_mask, TSTransformerEncoder, device


class MultiAttn(nn.Module):
    def __init__(self, input_dim, d_model, n_views, dropout=0):
        super(MultiAttn, self).__init__()
        self.projs = clones(nn.Linear(input_dim, d_model), 3)
        self.view_projs = clones(self.projs, n_views)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        attn_list = [self.cal_att(projs, query, key, value) for projs in self.view_projs]

        return torch.stack(attn_list)

    def cal_att(self, projs, query, key, value):
        query, key, value = [l(x) for l, x in zip(projs, (query, key, value))]
        _, attn = attention(query, key, value, dropout=self.dropout)

        return attn


class Transformer(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, n_layers, ff_dim, dropout):
        super(Transformer, self).__init__()
        self.max_len = max_len
        self.tst = TSTransformerEncoder(feat_dim, max_len, d_model, n_heads, n_layers, ff_dim, dropout, activation='relu')

    def forward(self, x, sorted_length):
        mask_pad = padding_mask(torch.from_numpy(np.array(sorted_length)).to(device) ,self.max_len)
        output = self.tst(x, mask_pad)

        return output


class GNN(MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)

        return out


class Model(nn.Module):
    def __init__(self, x_dim, base_dim, d_model_att, n_views, seq_len, d_model_tf, n_heads, n_layers, ff_dim,
                 dropout):
        super(Model, self).__init__()
        self.n_views = n_views
        self.seq_len = seq_len
        self.d_model_tf = d_model_tf
        self.x_dim = x_dim

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.matt = MultiAttn(input_dim=x_dim + base_dim, d_model=d_model_att, n_views=n_views, dropout=dropout)
        self.phi = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gnn = GNN()
        self.transformer = Transformer(feat_dim=x_dim + base_dim, max_len=seq_len, d_model=d_model_tf,
                                       n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim, dropout=dropout)

        self.W_pre = nn.Linear(d_model_tf, 2)
        self.W_imp = nn.Linear(d_model_tf, x_dim + base_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_base, sorted_length):
        batch_size = x.size(0)

        x_ = self.pool(x.permute(0, 2, 1)).squeeze(2)
        x_all = torch.cat((x_, x_base), 1)
        attns = self.matt(x_all, x_all, x_all)
        attns = torch.where(attns > self.phi, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))

        x_base_seq = x_base.unsqueeze(1).expand(x.shape[0], x.shape[1], x_base.shape[1])
        x_all_seq = torch.cat((x, x_base_seq), 2)
        x_all_seq = self.transformer(x_all_seq, sorted_length)
        x_all_seq = self.dropout(x_all_seq)
        tf_out = x_all_seq
        x_all_seq = x_all_seq.reshape(batch_size, -1)

        gnn_views = torch.stack(
            [self.gnn(x_all_seq, torch.nonzero(attns[i] == 1).T).reshape(batch_size, self.seq_len, self.d_model_tf)
             for i in range(self.n_views)])

        tf_out_ = self.pool(tf_out.permute(0, 2, 1)).squeeze(2)
        out_pre = self.softmax(self.W_pre(tf_out_))
        out_imp = self.W_imp(tf_out)

        gnn_views = torch.stack([self.pool(gnn_views[i].permute(0, 2, 1)).squeeze(2) for i in range(self.n_views)])

        return out_pre, out_imp[:, :, 0:self.x_dim], gnn_views, attns

