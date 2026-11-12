"""Two-stage trainer for the DDGN + BNN composite."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, NamedTuple
import torch
from .losses import bnn_loss, ddgn_critic_loss, ddgn_generator_loss
from .model import ConditionalWGANSequenceCritic, HEIDI

@dataclass(frozen=True)
class FitConfig:
    ddgn_epochs: int
    bnn_epochs: int
    learning_rate: float
    critic_learning_rate: float
    weight_decay: float
    critic_steps: int
    gradient_penalty_weight: float
    ddgn_reconstruction_weight: float
    ddgn_adversarial_weight: float
    bnn_reconstruction_weight: float
    bnn_forecast_weight: float
    bnn_kl_weight: float
    validation_mc_samples: int
    test_mc_samples: int
    critic_beta1: float
    critic_beta2: float
    gradient_clip_norm: float

class FittedHEIDI(NamedTuple):
    model: HEIDI
    critic: ConditionalWGANSequenceCritic
    validation_metrics: dict[str, float]
    history: list[dict[str, float | str]]
    fit_wall_clock_seconds: float

def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}

def _time_delta(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    return batch.get('time_delta', batch.get('node_aoi'))

def _prediction(model: HEIDI, batch: dict[str, torch.Tensor], edge_index: torch.Tensor) -> object:
    return model(batch['input_values'], batch['input_mask'], batch['edge_values'], batch['global_values'], edge_index, time_delta=_time_delta(batch), sample_bnn=True)

def fit_two_stage(model: HEIDI, critic: ConditionalWGANSequenceCritic, *, train_loader: Iterable[dict[str, torch.Tensor]], validation_loader: Iterable[dict[str, torch.Tensor]], edge_index: torch.Tensor, device: torch.device, config: FitConfig, evaluate: Callable[[Callable[[dict[str, torch.Tensor]], object], Iterable[dict[str, torch.Tensor]], int], dict[str, float]], select_validation: Callable[[dict[str, float]], float]) -> FittedHEIDI:
    """Fit DDGN with WGAN-GP, freeze it, then select the BNN using a caller-supplied validation criterion."""
    if config.ddgn_epochs < 1 or config.bnn_epochs < 1 or config.critic_steps < 1:
        raise ValueError('ddgn_epochs, bnn_epochs, and critic_steps must be positive.')
    model = model.to(device)
    critic = critic.to(device)
    ddgn_optimizer = torch.optim.AdamW(model.ddgn.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate, betas=(config.critic_beta1, config.critic_beta2))
    bnn_optimizer = torch.optim.AdamW(model.bnn.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float | str]] = []
    started = perf_counter()
    for epoch in range(1, config.ddgn_epochs + 1):
        model.ddgn.train()
        critic.train()
        total_generator = total_critic = total_reconstruction = total_adversarial = 0.0
        batches = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            for _ in range(config.critic_steps):
                with torch.no_grad():
                    _, fake_completed, _ = model.ddgn(batch['input_values'], batch['input_mask'], batch['edge_values'], batch['global_values'], edge_index, time_delta=_time_delta(batch))
                critic_optimizer.zero_grad(set_to_none=True)
                critic_objective = ddgn_critic_loss(critic, real_values=batch['target_values'], fake_values=fake_completed, observed_values=batch['input_values'], observed_mask=batch['input_mask'], edge_values=batch['edge_values'], global_values=batch['global_values'], edge_index=edge_index, gradient_penalty_weight=config.gradient_penalty_weight)
                critic_objective.backward()
                critic_optimizer.step()
            ddgn_optimizer.zero_grad(set_to_none=True)
            proposal, completed, _ = model.ddgn(batch['input_values'], batch['input_mask'], batch['edge_values'], batch['global_values'], edge_index, time_delta=_time_delta(batch))
            generator_objective, reconstruction, adversarial = ddgn_generator_loss(critic, proposal=proposal, completed=completed, target=batch['target_values'], hidden_mask=batch['hidden_target_mask'], observed_values=batch['input_values'], observed_mask=batch['input_mask'], edge_values=batch['edge_values'], global_values=batch['global_values'], edge_index=edge_index, reconstruction_weight=config.ddgn_reconstruction_weight, adversarial_weight=config.ddgn_adversarial_weight)
            generator_objective.backward()
            torch.nn.utils.clip_grad_norm_(model.ddgn.parameters(), max_norm=config.gradient_clip_norm)
            ddgn_optimizer.step()
            total_generator += float(generator_objective.detach().cpu())
            total_critic += float(critic_objective.detach().cpu())
            total_reconstruction += float(reconstruction.detach().cpu())
            total_adversarial += float(adversarial.detach().cpu())
            batches += 1
        if not batches:
            raise ValueError('Cannot fit an empty training loader.')
        history.append({'stage': 'ddgn_wgan_gp', 'epoch': float(epoch), 'ddgn_generator_loss': total_generator / batches, 'ddgn_critic_loss': total_critic / batches, 'ddgn_reconstruction_mse': total_reconstruction / batches, 'ddgn_adversarial_loss': total_adversarial / batches})
    model.ddgn.eval()
    for parameter in model.ddgn.parameters():
        parameter.requires_grad_(False)
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    best_score = float('inf')
    best_bnn_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float] | None = None
    for epoch in range(1, config.bnn_epochs + 1):
        model.bnn.train()
        total_loss = total_reconstruction = total_forecast = total_kl = 0.0
        batches = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            bnn_optimizer.zero_grad(set_to_none=True)
            output = _prediction(model, batch, edge_index)
            total, reconstruction, forecast, kl = bnn_loss(output, batch, reconstruction_weight=config.bnn_reconstruction_weight, forecast_weight=config.bnn_forecast_weight, kl_weight=config.bnn_kl_weight)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.bnn.parameters(), max_norm=config.gradient_clip_norm)
            bnn_optimizer.step()
            total_loss += float(total.detach().cpu())
            total_reconstruction += float(reconstruction.detach().cpu())
            total_forecast += float(forecast.detach().cpu())
            total_kl += float(kl.detach().cpu())
            batches += 1
        if not batches:
            raise ValueError('Cannot fit an empty training loader.')
        model.eval()
        validation = evaluate(lambda batch: _prediction(model, batch, edge_index), validation_loader, config.validation_mc_samples)
        selection_score = float(select_validation(validation))
        history.append({'stage': 'bnn_refiner', 'epoch': float(epoch), 'bnn_total_loss': total_loss / batches, 'bnn_reconstruction_nll': total_reconstruction / batches, 'bnn_forecast_nll': total_forecast / batches, 'bnn_normalized_kl': total_kl / batches, 'validation_endpoint_reconstruction_rmse': validation['endpoint_reconstruction_rmse'], 'validation_endpoint_forecast_rmse': validation['endpoint_forecast_rmse'], 'validation_selection_score': selection_score})
        if selection_score < best_score:
            best_score = selection_score
            best_bnn_state = {name: value.detach().cpu() for name, value in copy.deepcopy(model.bnn.state_dict()).items()}
            best_validation = validation
    if best_bnn_state is None or best_validation is None:
        raise RuntimeError('BNN training did not produce a validation checkpoint.')
    model.bnn.load_state_dict(best_bnn_state)
    model.eval()
    return FittedHEIDI(model, critic, best_validation, history, perf_counter() - started)
