import os.path as os
from torch.nn import Linear
from torch_geometric.nn import TransformerConv
from torch import Tensor
from torch_scatter import scatter
from torch_scatter import scatter_softmax, scatter_max

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class AttentionAggregator(nn.Module):
    def __init__(self, msg_dim, time_dim):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(msg_dim + 1, 128), 
            nn.ReLU(),
            nn.Linear(128, 1) 
        )

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        if msg.size(0) == 0 or t.size(0) == 0:
            return torch.zeros((dim_size, msg.size(-1)), device=msg.device)
        last_update = scatter(t, index, dim=0, dim_size=dim_size, reduce='max')  
        delta_t = last_update[index] - t 
        combined = torch.cat([msg, delta_t.unsqueeze(-1)], dim=-1)  
        attn_logits = self.attn_mlp(combined).squeeze(-1)  
        attn_weights = scatter_softmax(attn_logits, index, dim=0)  
        weighted_msg = attn_weights.unsqueeze(-1) * msg 
        aggregated = scatter(weighted_msg, index, dim=0, dim_size=dim_size, reduce="sum")  

class SumAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        if msg.size(0) == 0:
            return torch.zeros((dim_size, msg.size(-1)), device=msg.device)
        aggregated = scatter(msg, index, dim=0, dim_size=dim_size, reduce="sum")
        return aggregated


class MaxAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        if msg.size(0) == 0:
            return torch.zeros((dim_size, msg.size(-1)), device=msg.device)
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='max')


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg, return_attention=False):
        rel_t = t - last_update[edge_index[0]]
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        print("edge_attr std:", edge_attr.std(dim=0))
        if return_attention:
            out, attn_weights = self.conv(x, edge_index, edge_attr, return_attention_weights=True)
            return out, attn_weights  # return both
        else:
            return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)
