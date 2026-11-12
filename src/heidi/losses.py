"""Stage-specific objectives for the two-stage DDGN + BNN pipeline."""

from __future__ import annotations

import torch

from .model import ConditionalWGANSequenceCritic, TwinOutput


LOG_2PI = float(torch.log(torch.tensor(2.0 * torch.pi)))


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((prediction - target).square(), mask)


def masked_gaussian_nll(
    mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    normalized_error = (target - mean) / std.clamp_min(1e-6)
    nll = 0.5 * (normalized_error.square() + 2.0 * std.clamp_min(1e-6).log() + LOG_2PI)
    return masked_mean(nll, mask)


def gradient_penalty(
    critic: ConditionalWGANSequenceCritic,
    *,
    real_values: torch.Tensor,
    fake_values: torch.Tensor,
    observed_values: torch.Tensor,
    observed_mask: torch.Tensor,
    edge_values: torch.Tensor,
    global_values: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real_values.size(0), 1, 1, 1, device=real_values.device, dtype=real_values.dtype)
    interpolated = (alpha * real_values + (1.0 - alpha) * fake_values).requires_grad_(True)
    # WGAN-GP differentiates this score with respect to both the input and
    # critic parameters. CuDNN RNN kernels do not support that double backward.
    with torch.backends.cudnn.flags(enabled=False):
        scores = critic(
            interpolated, observed_values, observed_mask, edge_values, global_values, edge_index
        )
    gradients = torch.autograd.grad(
        scores,
        interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (gradients.reshape(gradients.size(0), -1).norm(2, dim=1) - 1.0).square().mean()


def ddgn_critic_loss(
    critic: ConditionalWGANSequenceCritic,
    *,
    real_values: torch.Tensor,
    fake_values: torch.Tensor,
    observed_values: torch.Tensor,
    observed_mask: torch.Tensor,
    edge_values: torch.Tensor,
    global_values: torch.Tensor,
    edge_index: torch.Tensor,
    gradient_penalty_weight: float,
) -> torch.Tensor:
    real_score = critic(real_values, observed_values, observed_mask, edge_values, global_values, edge_index).mean()
    fake_score = critic(fake_values, observed_values, observed_mask, edge_values, global_values, edge_index).mean()
    penalty = gradient_penalty(
        critic,
        real_values=real_values,
        fake_values=fake_values,
        observed_values=observed_values,
        observed_mask=observed_mask,
        edge_values=edge_values,
        global_values=global_values,
        edge_index=edge_index,
    )
    return fake_score - real_score + gradient_penalty_weight * penalty


def ddgn_generator_loss(
    critic: ConditionalWGANSequenceCritic,
    *,
    proposal: torch.Tensor,
    completed: torch.Tensor,
    target: torch.Tensor,
    hidden_mask: torch.Tensor,
    observed_values: torch.Tensor,
    observed_mask: torch.Tensor,
    edge_values: torch.Tensor,
    global_values: torch.Tensor,
    edge_index: torch.Tensor,
    reconstruction_weight: float,
    adversarial_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = masked_mse(proposal, target, hidden_mask)
    adversarial = -critic(
        completed, observed_values, observed_mask, edge_values, global_values, edge_index
    ).mean()
    return reconstruction_weight * reconstruction + adversarial_weight * adversarial, reconstruction, adversarial


def bnn_loss(
    output: TwinOutput,
    batch: dict[str, torch.Tensor],
    *,
    reconstruction_weight: float,
    forecast_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = masked_gaussian_nll(
        output.reconstruction_mean,
        output.reconstruction_std,
        batch["target_values"],
        batch["hidden_target_mask"],
    )
    forecast = masked_gaussian_nll(
        output.forecast_mean,
        output.forecast_std,
        batch["forecast_target"],
        torch.ones_like(batch["forecast_target"], dtype=torch.bool),
    )
    normalized_kl = output.bnn_kl / (batch["target_values"].size(0) * batch["target_values"].size(1))
    total = reconstruction_weight * reconstruction + forecast_weight * forecast + kl_weight * normalized_kl
    return total, reconstruction, forecast, normalized_kl
