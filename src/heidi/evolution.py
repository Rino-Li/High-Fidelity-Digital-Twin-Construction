"""Shared evolution Q and offline consistency F; fitted settings are caller-supplied."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable
import torch
from torch import nn

class _EvolutionEncoder(nn.Module):
    """A compact one-step state-evolution predictor used as Q."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim))

    def forward(self, flattened_state: torch.Tensor) -> torch.Tensor:
        return self.network(flattened_state)

@dataclass
class SharedQF:
    """Frozen, common Q/F module for one scenario and random seed."""
    q_model: _EvolutionEncoder
    node_count: int
    feature_count: int
    acceptance_threshold: float
    softmax_temperature: float

    @property
    def parameter_count(self) -> int:
        return sum((parameter.numel() for parameter in self.q_model.parameters()))

    @torch.no_grad()
    def q_predict(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError('Q expects state tensors with shape [batch, node, feature].')
        if state.shape[1:] != (self.node_count, self.feature_count):
            raise ValueError('Q received a state with incompatible node or feature dimensions.')
        return self.q_model(state.reshape(state.size(0), -1)).reshape_as(state)

    @torch.no_grad()
    def true_probability(self, consistency_error: torch.Tensor) -> torch.Tensor:
        """Return F(True) from a fixed two-logit Softmax classifier."""
        margin = (self.acceptance_threshold - consistency_error) / self.softmax_temperature
        logits = torch.stack((-margin, margin), dim=-1)
        return torch.softmax(logits, dim=-1)[..., 1]

    def checkpoint(self) -> dict[str, object]:
        return {'q_state_dict': self.q_model.state_dict(), 'node_count': self.node_count, 'feature_count': self.feature_count, 'acceptance_threshold': self.acceptance_threshold, 'softmax_temperature': self.softmax_temperature}

def _transitions(data: object, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    episodes = getattr(data, 'normalized_episodes', {}).get(split)
    if not episodes:
        raise ValueError(f'Q/F fitting requires a prepared {split!r} split.')
    states, next_states = ([], [])
    for episode in episodes:
        states.append(episode.target_values[:-1])
        next_states.append(episode.target_values[1:])
    return (torch.cat(states), torch.cat(next_states))

def _per_window_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).abs().mean(dim=(1, 2))

def fit_shared_qf(data: object, *, epochs: int, learning_rate: float, hidden_dim: int, weight_decay: float, batch_size: int, calibration_quantile: float, temperature_factor: float, temperature_floor: float) -> SharedQF:
    """Fit Q on clean training transitions and calibrate F on clean validation; settings are explicit."""
    if epochs < 1 or batch_size < 1 or hidden_dim < 1 or temperature_floor <= 0:
        raise ValueError("Epochs, batch size, hidden dimension and temperature floor must be positive.")
    if not 0 <= calibration_quantile <= 1 or temperature_factor < 0:
        raise ValueError("Invalid calibration parameters.")
    train_state, train_next = _transitions(data, 'train')
    node_count, feature_count = train_state.shape[1:]
    state_dim = node_count * feature_count
    model = _EvolutionEncoder(state_dim, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    flattened_state = train_state.reshape(train_state.size(0), -1)
    flattened_next = train_next.reshape(train_next.size(0), -1)
    model.train()
    for _ in range(epochs):
        for start in range(0, flattened_state.size(0), batch_size):
            stop = min(start + batch_size, flattened_state.size(0))
            prediction = model(flattened_state[start:stop])
            loss = nn.functional.mse_loss(prediction, flattened_next[start:stop])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    validation_state, validation_next = _transitions(data, 'val')
    with torch.no_grad():
        validation_prediction = model(validation_state.reshape(validation_state.size(0), -1)).reshape_as(validation_next)
        validation_error = _per_window_mae(validation_prediction, validation_next)
    threshold = float(torch.quantile(validation_error, calibration_quantile))
    temperature = max(threshold * temperature_factor, temperature_floor)
    return SharedQF(q_model=model, node_count=node_count, feature_count=feature_count, acceptance_threshold=threshold, softmax_temperature=temperature)

@torch.no_grad()
def evaluate_shared_qf(qf: SharedQF, predictor: Callable[[dict[str, torch.Tensor]], object], loader: Iterable[dict[str, torch.Tensor]], *, acceptance_probability: float, strict_acceptance: bool) -> dict[str, float]:
    """Measure Q prediction quality and F-valid virtual DT updates."""
    q_errors, q_squared, true_probabilities = ([], [], [])
    for raw_batch in loader:
        output = predictor(raw_batch)
        completed = output.completed_mean.detach().cpu()[:, -1]
        next_state = raw_batch['forecast_target'][:, -1].detach().cpu()
        q_prediction = qf.q_predict(completed)
        error = _per_window_mae(q_prediction, next_state)
        q_errors.append((q_prediction - next_state).abs().reshape(-1))
        q_squared.append((q_prediction - next_state).square().reshape(-1))
        true_probabilities.append(qf.true_probability(error))
    if not q_errors:
        raise ValueError('Cannot evaluate Q/F on an empty loader.')
    true_probability = torch.cat(true_probabilities)
    accepted = true_probability > acceptance_probability if strict_acceptance else true_probability >= acceptance_probability
    return {'q_evolution_mae': float(torch.cat(q_errors).mean()), 'q_evolution_rmse': float(torch.cat(q_squared).mean().sqrt()), 'f_valid_update_count': float(accepted.sum()), 'f_valid_update_rate': float(accepted.float().mean()), 'f_mean_true_probability': float(true_probability.mean()), 'f_acceptance_threshold': float(qf.acceptance_threshold)}
