"""A causal graph-temporal Bayesian state estimator for incomplete IIoT data."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    """Variational Bayesian linear layer with a standard-normal weight prior."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))
        nn.init.kaiming_uniform_(self.weight_mu, a=5**0.5)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-5

    def forward(self, values: torch.Tensor, sample: bool = True) -> torch.Tensor:
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)
        if sample:
            weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(values, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        def standard_normal_kl(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            return 0.5 * (mu.square() + sigma.square() - 1.0 - 2.0 * sigma.log()).sum()

        return standard_normal_kl(self.weight_mu, self._sigma(self.weight_rho)) + standard_normal_kl(
            self.bias_mu, self._sigma(self.bias_rho)
        )


class EdgeConditionedGraphLayer(nn.Module):
    """Message passing with time-varying edge attributes and directed topology."""

    def __init__(self, hidden_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, node_hidden: torch.Tensor, edge_index: torch.Tensor, edge_values: torch.Tensor
    ) -> torch.Tensor:
        # node_hidden: [batch, node, hidden], edge_values: [batch, edge, edge_feature]
        source_index, destination_index = edge_index
        source = node_hidden[:, source_index, :]
        destination = node_hidden[:, destination_index, :]
        edge_hidden = self.edge_encoder(edge_values)
        message = self.message(torch.cat([source, edge_hidden], dim=-1))
        gate = self.gate(torch.cat([source, destination, edge_hidden], dim=-1))
        message = gate * message

        aggregate = torch.zeros_like(node_hidden)
        aggregate.index_add_(1, destination_index, message)
        degree = torch.zeros(node_hidden.size(1), device=node_hidden.device, dtype=node_hidden.dtype)
        degree.index_add_(0, destination_index, torch.ones_like(destination_index, dtype=node_hidden.dtype))
        aggregate = aggregate / degree.clamp_min(1.0).view(1, -1, 1)

        batch_size, node_count, hidden_dim = node_hidden.shape
        updated = self.update(
            aggregate.reshape(batch_size * node_count, hidden_dim),
            node_hidden.reshape(batch_size * node_count, hidden_dim),
        )
        return self.norm(updated.reshape(batch_size, node_count, hidden_dim) + node_hidden)


class CausalGraphTemporalEncoder(nn.Module):
    """Spatial graph propagation followed by a forward-only temporal recurrence."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        graph_layers: int,
    ) -> None:
        super().__init__()
        # The input mask is a first-class feature, not just a preprocessing detail.
        self.input_projection = nn.Sequential(
            nn.Linear(2 * node_feature_dim + global_feature_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_layers = nn.ModuleList(
            EdgeConditionedGraphLayer(hidden_dim, edge_feature_dim) for _ in range(graph_layers)
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        input_values: torch.Tensor,
        input_mask: torch.Tensor,
        edge_values: torch.Tensor,
        global_values: torch.Tensor,
        edge_index: torch.Tensor,
        time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, time_steps, node_count, _ = input_values.shape
        if time_delta is None:
            time_context = torch.ones(
                batch_size, time_steps, node_count, 1, dtype=input_values.dtype, device=input_values.device
            )
        elif time_delta.ndim == 3 and time_delta.shape == (batch_size, time_steps, 1):
            time_context = time_delta.unsqueeze(2).expand(-1, -1, node_count, -1)
        elif time_delta.ndim == 4 and time_delta.shape == (batch_size, time_steps, node_count, 1):
            time_context = time_delta
        else:
            raise ValueError(
                "time_delta must have shape [batch, time, 1] or [batch, time, node, 1]."
            )
        global_context = global_values.unsqueeze(2).expand(-1, -1, node_count, -1)
        hidden = self.input_projection(
            torch.cat([input_values, input_mask.float(), global_context, time_context], dim=-1)
        )

        spatial_states = []
        for time_index in range(time_steps):
            state = hidden[:, time_index]
            for graph_layer in self.graph_layers:
                state = graph_layer(state, edge_index, edge_values[:, time_index])
            spatial_states.append(state)
        spatial = torch.stack(spatial_states, dim=1)

        temporal_input = spatial.permute(0, 2, 1, 3).reshape(batch_size * node_count, time_steps, -1)
        temporal_output, _ = self.temporal(temporal_input)
        return temporal_output.reshape(batch_size, node_count, time_steps, -1).permute(0, 2, 1, 3)


class TwinOutput(NamedTuple):
    reconstruction_mean: torch.Tensor
    reconstruction_std: torch.Tensor
    completed_mean: torch.Tensor
    forecast_mean: torch.Tensor
    forecast_std: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_std: torch.Tensor
    latent_kl: torch.Tensor
    bnn_kl: torch.Tensor


class EndToEndBayesianDT(nn.Module):
    """Jointly reconstructs private/missing states and forecasts their future evolution.

    The Bayesian output head models epistemic uncertainty through a variational weight
    posterior. The sampled latent state models conditional state uncertainty. Together
    they provide a distribution for a black-box device's intermediate state.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        latent_dim: int,
        graph_layers: int,
    ) -> None:
        """Build the public HEIDI architecture.

        The release intentionally does not provide the paper-run dimensions or
        layer count.  They are part of the private experiment protocol.  Any
        positive dimensions can be supplied for a toy smoke test or an
        independently configured application.
        """
        dimensions = {
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "global_feature_dim": global_feature_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "graph_layers": graph_layers,
        }
        if any(value < 1 for value in dimensions.values()):
            raise ValueError(f"All architecture dimensions must be positive: {dimensions}")
        super().__init__()
        self.encoder = CausalGraphTemporalEncoder(
            node_feature_dim, edge_feature_dim, global_feature_dim, hidden_dim, graph_layers
        )
        self.posterior_mean = BayesianLinear(hidden_dim, latent_dim)
        self.posterior_raw_std = BayesianLinear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * node_feature_dim)
        )
        self.forecaster = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * node_feature_dim)
        )
        self.node_feature_dim = node_feature_dim

    def forward(
        self,
        input_values: torch.Tensor,
        input_mask: torch.Tensor,
        edge_values: torch.Tensor,
        global_values: torch.Tensor,
        edge_index: torch.Tensor,
        time_delta: torch.Tensor | None = None,
        sample_latent: bool = True,
        sample_weights: bool = True,
    ) -> TwinOutput:
        hidden = self.encoder(
            input_values, input_mask, edge_values, global_values, edge_index, time_delta=time_delta
        )
        posterior_mean = self.posterior_mean(hidden, sample=sample_weights)
        posterior_std = F.softplus(self.posterior_raw_std(hidden, sample=sample_weights)) + 1e-4
        latent = posterior_mean + posterior_std * torch.randn_like(posterior_std) if sample_latent else posterior_mean
        decoded = self.decoder(torch.cat([hidden, latent], dim=-1))
        reconstruction_mean, reconstruction_raw_std = decoded.chunk(2, dim=-1)
        reconstruction_std = F.softplus(reconstruction_raw_std) + 1e-4
        forecast = self.forecaster(torch.cat([hidden, latent], dim=-1))
        forecast_mean, forecast_raw_std = forecast.chunk(2, dim=-1)
        forecast_std = F.softplus(forecast_raw_std) + 1e-4

        # Only fresh measurements, not delayed packets, are copied into the DT state.
        completed_mean = torch.where(input_mask, input_values, reconstruction_mean)
        latent_kl = 0.5 * (
            posterior_mean.square() + posterior_std.square() - 1.0 - 2.0 * posterior_std.log()
        ).mean()
        bnn_kl = self.posterior_mean.kl_divergence() + self.posterior_raw_std.kl_divergence()
        return TwinOutput(
            reconstruction_mean=reconstruction_mean,
            reconstruction_std=reconstruction_std,
            completed_mean=completed_mean,
            forecast_mean=forecast_mean,
            forecast_std=forecast_std,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            latent_kl=latent_kl,
            bnn_kl=bnn_kl,
        )
