"""DDGN proposal and Bayesian residual refinement, with caller-supplied settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class ModelNumerics:
    """Explicit initialization/stability settings; no paper defaults are provided."""
    initial_weight_rho: float
    weight_scale_floor: float
    predictive_scale_floor: float

    def __post_init__(self):
        if self.weight_scale_floor <= 0 or self.predictive_scale_floor <= 0:
            raise ValueError("Scale floors must be positive.")


class VariationalLinear(nn.Module):
    """Factorized Gaussian linear layer with a standard-normal prior."""

    def __init__(self, input_dim: int, output_dim: int, *, numerics: ModelNumerics) -> None:
        super().__init__()
        self.scale_floor = numerics.weight_scale_floor
        self.weight_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_rho = nn.Parameter(torch.full((output_dim, input_dim), numerics.initial_weight_rho))
        self.bias_mu = nn.Parameter(torch.zeros(output_dim))
        self.bias_rho = nn.Parameter(torch.full((output_dim,), numerics.initial_weight_rho))
        nn.init.kaiming_uniform_(self.weight_mu, a=5**0.5)

    def _scale(self, rho: torch.Tensor) -> torch.Tensor:
        return functional.softplus(rho) + self.scale_floor

    def forward(self, values: torch.Tensor, *, sample: bool) -> torch.Tensor:
        if sample:
            weight = self.weight_mu + self._scale(self.weight_rho) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + self._scale(self.bias_rho) * torch.randn_like(self.bias_mu)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return functional.linear(values, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        def gaussian_kl(mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return 0.5 * (mean.square() + scale.square() - 1.0 - 2.0 * scale.log()).sum()

        return gaussian_kl(self.weight_mu, self._scale(self.weight_rho)) + gaussian_kl(
            self.bias_mu, self._scale(self.bias_rho)
        )


class DirectedDynamicGraphLayer(nn.Module):
    """One directed message-passing update conditioned on current edge attributes."""

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
        self,
        node_state: torch.Tensor,
        edge_values: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source_index, destination_index = edge_index
        source = node_state[:, source_index]
        destination = node_state[:, destination_index]
        edge_state = self.edge_encoder(edge_values)
        message = self.message(torch.cat((source, edge_state), dim=-1))
        message = message * self.gate(torch.cat((source, destination, edge_state), dim=-1))
        aggregate = torch.zeros_like(node_state)
        aggregate.index_add_(1, destination_index, message)
        degree = torch.zeros(node_state.size(1), dtype=node_state.dtype, device=node_state.device)
        degree.index_add_(0, destination_index, torch.ones_like(destination_index, dtype=node_state.dtype))
        batch, nodes, hidden = node_state.shape
        updated = self.update(
            (aggregate / degree.clamp_min(1.0).view(1, -1, 1)).reshape(batch * nodes, hidden),
            node_state.reshape(batch * nodes, hidden),
        ).reshape(batch, nodes, hidden)
        return self.norm(updated + node_state)


class DDGNGenerator(nn.Module):
    """Causal directed dynamic graph generator that preserves observed values."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        graph_layers: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(2 * node_feature_dim + global_feature_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_layers = nn.ModuleList(
            DirectedDynamicGraphLayer(hidden_dim, edge_feature_dim) for _ in range(graph_layers)
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_feature_dim))

    def forward(
        self,
        input_values: torch.Tensor,
        input_mask: torch.Tensor,
        edge_values: torch.Tensor,
        global_values: torch.Tensor,
        edge_index: torch.Tensor,
        time_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, nodes, _ = input_values.shape
        if time_delta is None:
            time_context = input_values.new_ones((batch, steps, nodes, 1))
        elif time_delta.ndim == 3:
            time_context = time_delta.unsqueeze(2).expand(-1, -1, nodes, -1)
        elif time_delta.ndim == 4:
            time_context = time_delta
        else:
            raise ValueError("time_delta must have shape [batch, time, 1] or [batch, time, node, 1].")
        global_context = global_values.unsqueeze(2).expand(-1, -1, nodes, -1)
        encoded = self.input_projection(
            torch.cat((input_values, input_mask.float(), global_context, time_context), dim=-1)
        )
        spatial_steps: list[torch.Tensor] = []
        for index in range(steps):
            state = encoded[:, index]
            for layer in self.graph_layers:
                state = layer(state, edge_values[:, index], edge_index)
            spatial_steps.append(state)
        spatial = torch.stack(spatial_steps, dim=1)
        temporal_input = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        temporal_state, _ = self.temporal(temporal_input)
        hidden = temporal_state.reshape(batch, nodes, steps, -1).permute(0, 2, 1, 3)
        proposal = self.decoder(hidden)
        completed = torch.where(input_mask, input_values, proposal)
        return proposal, completed, hidden


class ConditionalWGANSequenceCritic(nn.Module):
    """Conditional graph-temporal Wasserstein critic for full state trajectories."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        *,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(3 * node_feature_dim + global_feature_dim, hidden_dim), nn.LeakyReLU(negative_slope)
        )
        self.graph = DirectedDynamicGraphLayer(hidden_dim, edge_feature_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        candidate_values: torch.Tensor,
        observed_values: torch.Tensor,
        observed_mask: torch.Tensor,
        edge_values: torch.Tensor,
        global_values: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, nodes, _ = candidate_values.shape
        global_context = global_values.unsqueeze(2).expand(-1, -1, nodes, -1)
        encoded = self.input_projection(
            torch.cat((candidate_values, observed_values, observed_mask.float(), global_context), dim=-1)
        )
        graph_steps = [self.graph(encoded[:, index], edge_values[:, index], edge_index) for index in range(steps)]
        graph_state = torch.stack(graph_steps, dim=1)
        temporal_input = graph_state.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        temporal_state, _ = self.temporal(temporal_input)
        pooled = temporal_state[:, -1].reshape(batch, nodes, -1).mean(dim=1)
        return self.score(pooled).squeeze(-1)


class BayesianResidualRefiner(nn.Module):
    """Frozen-DDGN conditional BNN for residual reconstruction and forecasting."""

    def __init__(self, node_feature_dim: int, global_feature_dim: int, hidden_dim: int, *, numerics: ModelNumerics) -> None:
        super().__init__()
        self.predictive_scale_floor = numerics.predictive_scale_floor
        self.context = nn.Sequential(
            nn.Linear(2 * node_feature_dim + global_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.reconstruction_hidden = VariationalLinear(hidden_dim, hidden_dim, numerics=numerics)
        self.reconstruction_head = VariationalLinear(hidden_dim, 2 * node_feature_dim, numerics=numerics)
        self.forecast_hidden = VariationalLinear(hidden_dim, hidden_dim, numerics=numerics)
        self.forecast_head = VariationalLinear(hidden_dim, 2 * node_feature_dim, numerics=numerics)

    def forward(
        self,
        completed: torch.Tensor,
        input_values: torch.Tensor,
        input_mask: torch.Tensor,
        global_values: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, nodes, _ = completed.shape
        global_context = global_values.unsqueeze(2).expand(-1, -1, nodes, -1)
        context = self.context(torch.cat((completed, input_mask.float(), global_context), dim=-1))
        temporal_input = context.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        hidden, _ = self.temporal(temporal_input)
        hidden = hidden.reshape(batch, nodes, steps, -1).permute(0, 2, 1, 3)
        reconstruction = self.reconstruction_head(functional.silu(self.reconstruction_hidden(hidden, sample=sample)), sample=sample)
        forecast = self.forecast_head(functional.silu(self.forecast_hidden(hidden, sample=sample)), sample=sample)
        residual, reconstruction_raw_std = reconstruction.chunk(2, dim=-1)
        forecast_mean, forecast_raw_std = forecast.chunk(2, dim=-1)
        reconstruction_mean = torch.where(input_mask, input_values, completed + residual)
        reconstruction_std = functional.softplus(reconstruction_raw_std) + self.predictive_scale_floor
        forecast_std = functional.softplus(forecast_raw_std) + self.predictive_scale_floor
        kl = sum(layer.kl_divergence() for layer in self._bayesian_layers())
        return reconstruction_mean, reconstruction_std, forecast_mean, forecast_std, kl

    def _bayesian_layers(self) -> tuple[VariationalLinear, ...]:
        return (
            self.reconstruction_hidden,
            self.reconstruction_head,
            self.forecast_hidden,
            self.forecast_head,
        )


class TwinOutput(NamedTuple):
    reconstruction_mean: torch.Tensor
    reconstruction_std: torch.Tensor
    completed_mean: torch.Tensor
    forecast_mean: torch.Tensor
    forecast_std: torch.Tensor
    ddgn_proposal: torch.Tensor
    bnn_kl: torch.Tensor


class HEIDI(nn.Module):
    """DDGN -> Bayesian residual composite; internal state-dict names are retained."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        graph_layers: int,
        *,
        numerics: ModelNumerics,
    ) -> None:
        super().__init__()
        self.ddgn = DDGNGenerator(
            node_feature_dim, edge_feature_dim, global_feature_dim, hidden_dim, graph_layers
        )
        self.bnn = BayesianResidualRefiner(node_feature_dim, global_feature_dim, hidden_dim, numerics=numerics)

    def forward(
        self,
        input_values: torch.Tensor,
        input_mask: torch.Tensor,
        edge_values: torch.Tensor,
        global_values: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        time_delta: torch.Tensor | None = None,
        sample_bnn: bool = True,
    ) -> TwinOutput:
        proposal, completed, _ = self.ddgn(
            input_values, input_mask, edge_values, global_values, edge_index, time_delta=time_delta
        )
        reconstruction_mean, reconstruction_std, forecast_mean, forecast_std, bnn_kl = self.bnn(
            completed, input_values, input_mask, global_values, sample=sample_bnn
        )
        return TwinOutput(
            reconstruction_mean,
            reconstruction_std,
            torch.where(input_mask, input_values, reconstruction_mean),
            forecast_mean,
            forecast_std,
            proposal,
            bnn_kl,
        )
