"""Inference-only endpoint recovery; reuses existing DDGN and BNN weights."""
import torch
from torch.nn import functional as F
from .metrics import combine_predictive_samples


@torch.inference_mode()
def endpoint_recovery(model, batch, edge_index, *, sample=True):
    values, mask = batch['input_values'], batch['input_mask']
    edges, glob = batch['edge_values'], batch['global_values']
    b, t, n, _ = values.shape
    delta = batch.get('time_delta', batch.get('node_aoi'))
    if delta is None:
        time = values.new_ones((b, t, n, 1))
    elif delta.ndim == 3:
        time = delta.unsqueeze(2).expand(-1, -1, n, -1)
    else:
        time = delta
    context = glob.unsqueeze(2).expand(-1, -1, n, -1)
    encoded = model.ddgn.input_projection(torch.cat((values, mask.float(), context, time), -1))
    # These graph operations have no inter-time state. This combines time
    # positions inside ONE causal window, not different online requests.
    state = encoded.reshape(b * t, n, -1)
    edge_batch = edges.reshape(b * t, edges.shape[2], edges.shape[3])
    for layer in model.ddgn.graph_layers:
        state = layer(state, edge_batch, edge_index)
    spatial = state.reshape(b, t, n, -1)
    recurrent, _ = model.ddgn.temporal(spatial.permute(0, 2, 1, 3).reshape(b * n, t, -1))
    hidden = recurrent.reshape(b, n, t, -1).permute(0, 2, 1, 3)
    proposal = model.ddgn.decoder(hidden)
    completed = torch.where(mask, values, proposal)
    ctx = model.bnn.context(torch.cat((completed, mask.float(), context), -1))
    recurrent, _ = model.bnn.temporal(ctx.permute(0, 2, 1, 3).reshape(b * n, t, -1))
    last = recurrent.reshape(b, n, t, -1)[:, :, -1]
    # Weight draws use the same variational distributions and draw order as
    # the reference reconstruction head. Forecast and training KL are unused.
    latent = F.silu(model.bnn.reconstruction_hidden(last, sample=sample))
    residual = model.bnn.reconstruction_head(latent, sample=sample).chunk(2, -1)[0]
    return torch.where(mask[:, -1], values[:, -1], completed[:, -1] + residual)


def encode(encoder,batch,edge,legacy):
    values,mask=batch['input_values'],batch['input_mask']
    b,t,n,_=values.shape
    delta=batch.get('time_delta',batch.get('node_aoi'))
    if delta is None:delta=values.new_ones((b,t,n,1))
    elif delta.ndim==3:delta=delta.unsqueeze(2).expand(-1,-1,n,-1)
    glob=batch['global_values'].unsqueeze(2).expand(-1,-1,n,-1)
    encoded=encoder.input_projection(torch.cat((values,mask.float(),glob,delta),-1))
    state=encoded.reshape(b*t,n,-1)
    edges=batch['edge_values'].reshape(b*t,batch['edge_values'].shape[2],-1)
    for layer in encoder.graph_layers:
        state=layer(state,edges,edge) if legacy else layer(state,edge,edges)
    temporal,_=encoder.temporal(state.reshape(b,t,n,-1).permute(0,2,1,3).reshape(b*n,t,-1))
    return temporal.reshape(b,n,t,-1).permute(0,2,1,3),glob

@torch.inference_mode()
def endpoint_stats(model,batch,edge,*,sample=True,preserve_rng=False):
    values,mask=batch['input_values'],batch['input_mask']
    hidden,glob=encode(model.ddgn,batch,edge,True)
    completed=torch.where(mask,values,model.ddgn.decoder(hidden))
    ctx=model.bnn.context(torch.cat((completed,mask.float(),glob),-1))
    b,t,n,_=ctx.shape
    temporal,_=model.bnn.temporal(ctx.permute(0,2,1,3).reshape(b*n,t,-1))
    last=temporal.reshape(b,n,t,-1)[:,:,-1]
    latent=F.silu(model.bnn.reconstruction_hidden(last,sample=sample))
    residual,raw=model.bnn.reconstruction_head(latent,sample=sample).chunk(2,-1)
    mean=torch.where(mask[:,-1],values[:,-1],completed[:,-1]+residual)
    if preserve_rng and sample:
        # The reference evaluates two unused variational forecast layers after
        # the reconstruction head. Preserve their random-number advancement
        # when reproducing the original MC accuracy evaluation stream.
        for layer in (model.bnn.forecast_hidden,model.bnn.forecast_head):
            torch.randn_like(layer.weight_mu)
            torch.randn_like(layer.bias_mu)
    return mean,F.softplus(raw)+model.bnn.predictive_scale_floor

@torch.inference_mode()
def mc_endpoint(model, batch, edge_index, *, samples: int, preserve_rng: bool):
    """Moment-match predictive means/scales with an explicit MC count.

    Optional RNG advancement aligns repeated draws with full-forward evaluation.
    """
    if samples < 1:
        raise ValueError("samples must be positive.")
    means, scales = [], []
    for _ in range(samples):
        mean, scale = endpoint_stats(model, batch, edge_index, sample=True, preserve_rng=preserve_rng)
        means.append(mean)
        scales.append(scale)
    return combine_predictive_samples(means, scales)
