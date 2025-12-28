import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    MeanAggregator,
)
from collections import Counter
import pandas as pd
import numpy as np
from torch_geometric.data import TemporalData
from torch_scatter import scatter_softmax
from NN import GraphAttentionEmbedding, LinkPredictor, AttentionAggregator, SumAggregator, MaxAggregator
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

feature_list = pd.read_csv("fea_list.csv")
s_list = pd.read_csv("src_list.csv").values.tolist()
d_list = pd.read_csv("des_list.csv").values.tolist()
length = 1000
num_nodes = 4
num_time_steps = length / 4
num_edges_per_step = 1

src_list = []
des_list = []

for i in range(length):
    src_list.append(s_list[i][0])
    des_list.append(d_list[i][0])
edge_index = torch.tensor([src_list, des_list], dtype=torch.int64)
edge_attr_list = []
for i in range(length):
    edge_attr_list.append(
        [feature_list.iloc[i, 0], feature_list.iloc[i, 1], feature_list.iloc[i, 2], feature_list.iloc[i, 3]])

edge_attr_np = np.array(edge_attr_list)
edge_attr = torch.from_numpy(edge_attr_np)
edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

t = torch.arange(num_time_steps)
t = torch.repeat_interleave(t, repeats=4)
t = torch.tensor(t, dtype=torch.int64)

data = TemporalData(
    src=edge_index[0],
    dst=edge_index[1],
    t=t,
    msg=edge_attr,
)

data = data.to(device)

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

train_loader = TemporalDataLoader(
    train_data,
    batch_size=128,
    neg_sampling_ratio=1.0,
)
val_loader = TemporalDataLoader(
    val_data,
    batch_size=128,
    neg_sampling_ratio=1.0,
)
test_loader = TemporalDataLoader(
    test_data,
    batch_size=128,
    neg_sampling_ratio=1.0,
)

neighbor_loader = LastNeighborLoader(data.num_nodes, size=20, device=device)
memory_dim = time_dim = 100
embedding_dim = 100

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=AttentionAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)
optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.00001)
criterion = torch.nn.BCEWithLogitsLoss()
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
target_node = 27

def train():
    memory.train()
    gnn.train()
    link_pred.train()
    memory.reset_state()
    neighbor_loader.reset_state()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        z, attn_weights = gnn(
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
            return_attention=True,
        )
        edge_idx, alpha = attn_weights  

        num_edges = edge_idx.size(1)
        num_heads = 2
        alpha = alpha.view(num_edges, num_heads)
        dst_nodes = edge_idx[1]  
        dst_counts = Counter(edge_idx[1].tolist())
        print('dst_counts is:', dst_counts)
        alpha_norm = scatter_softmax(alpha, dst_nodes, dim=0)
        for i in range(num_edges):
            src = edge_idx[0, i].item()
            dst = edge_idx[1, i].item()

            if dst == target_node:
                attn_per_head = alpha[i]
                attn_mean = attn_per_head.mean().item()
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()
    torch.manual_seed(12345) 
    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        z, attn_weights = gnn(
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
            return_attention=True,
        )
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)
        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


loss_list = []
val_auc_list = []
test_auc_list = []
for epoch in tqdm(range(1, 100)):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_loader)
    test_ap, test_auc = test(test_loader)
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
    loss_list.append(loss)
    val_auc_list.append(val_auc)
    test_auc_list.append(test_auc)
pd.DataFrame(loss_list).to_csv("max_loss.csv")
pd.DataFrame(val_auc_list).to_csv("max_val.csv")
pd.DataFrame(test_auc_list).to_csv("max_test.csv")
