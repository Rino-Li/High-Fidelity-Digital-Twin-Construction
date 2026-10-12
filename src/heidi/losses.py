"""Joint losses for end-to-end Bayesian digital-twin training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import TwinOutput


LOG_2PI = float(torch.log(torch.tensor(2.0 * torch.pi)))


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_gaussian_nll(
    mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    normalized_error = (target - mean) / std.clamp_min(1e-6)
    nll = 0.5 * (normalized_error.square() + 2.0 * std.clamp_min(1e-6).log() + LOG_2PI)
    return masked_mean(nll, mask)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    reconstruction_nll: torch.Tensor
    forecast_nll: torch.Tensor
    latent_kl: torch.Tensor
    bnn_kl: torch.Tensor


def joint_loss(
    output: TwinOutput,
    batch: dict[str, torch.Tensor],
    *,
    reconstruction_weight: float,
    forecast_weight: float,
    latent_kl_weight: float,
    bnn_kl_weight: float,
) -> LossBreakdown:
    """Compute the joint objective with caller-supplied private weights."""
    # The reconstruction loss is deliberately evaluated only where observations are absent.
    reconstruction_nll = masked_gaussian_nll(
        output.reconstruction_mean,
        output.reconstruction_std,
        batch["target_values"],
        batch["hidden_target_mask"],
    )
    forecast_mask = torch.ones_like(batch["forecast_target"], dtype=torch.bool)
    forecast_nll = masked_gaussian_nll(
        output.forecast_mean, output.forecast_std, batch["forecast_target"], forecast_mask
    )
    bnn_kl = output.bnn_kl / (batch["target_values"].size(0) * batch["target_values"].size(1))
    total = reconstruction_weight * reconstruction_nll + forecast_weight * forecast_nll
    total = total + latent_kl_weight * output.latent_kl + bnn_kl_weight * bnn_kl
    return LossBreakdown(total, reconstruction_nll, forecast_nll, output.latent_kl, bnn_kl)
