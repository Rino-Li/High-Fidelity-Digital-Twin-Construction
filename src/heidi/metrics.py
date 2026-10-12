"""Accuracy and uncertainty-calibration metrics for hidden DT states."""

from __future__ import annotations

import math

import torch

from .losses import masked_gaussian_nll, masked_mean


def combine_predictive_samples(
    means: list[torch.Tensor], stds: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_stack = torch.stack(means)
    std_stack = torch.stack(stds)
    predictive_mean = mean_stack.mean(dim=0)
    epistemic_variance = mean_stack.var(dim=0, unbiased=False)
    aleatoric_variance = std_stack.square().mean(dim=0)
    return predictive_mean, (epistemic_variance + aleatoric_variance).sqrt()


def reconstruction_metrics(
    mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    error = mean - target
    mae = masked_mean(error.abs(), mask)
    rmse = masked_mean(error.square(), mask).sqrt()
    nll = masked_gaussian_nll(mean, std, target, mask)
    scale = std.clamp_min(1e-6)
    standardized = (target - mean) / scale
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=mean.dtype, device=mean.device),
        torch.ones((), dtype=mean.dtype, device=mean.device),
    )
    normal_pdf = torch.exp(-0.5 * standardized.square()) / math.sqrt(2.0 * math.pi)
    normal_cdf = normal.cdf(standardized)
    crps = scale * (
        standardized * (2.0 * normal_cdf - 1.0) + 2.0 * normal_pdf - 1.0 / math.sqrt(math.pi)
    )
    z90 = 1.6448536269514722
    lower, upper = mean - z90 * std, mean + z90 * std
    coverage = masked_mean(((target >= lower) & (target <= upper)).float(), mask)
    width = masked_mean(upper - lower, mask)
    return {
        "masked_mae": float(mae.detach().cpu()),
        "masked_rmse": float(rmse.detach().cpu()),
        "masked_nll": float(nll.detach().cpu()),
        "masked_crps": float(masked_mean(crps, mask).detach().cpu()),
        "interval_90_coverage": float(coverage.detach().cpu()),
        "interval_90_width": float(width.detach().cpu()),
    }


def empirical_reconstruction_metrics(samples: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Score samples directly, without imposing a Gaussian approximation.

    ``samples`` has shape ``[draw, scalar]`` after an evaluation mask has been
    applied. The CRPS identity uses the empirical distribution represented by
    those draws and the central 90% interval uses empirical quantiles.
    """
    if samples.ndim != 2 or target.ndim != 1 or samples.size(1) != target.numel():
        raise ValueError("samples must be [draw, scalar] and align with the one-dimensional target.")
    if samples.size(0) < 2:
        raise ValueError("Empirical CRPS and intervals require at least two predictive draws.")
    mean = samples.mean(dim=0)
    error = mean - target
    draws = samples.size(0)
    sorted_samples = samples.sort(dim=0).values
    coefficients = (
        2 * torch.arange(draws, dtype=samples.dtype, device=samples.device) - draws + 1
    ).view(draws, 1)
    expected_pairwise_distance = 2.0 * (coefficients * sorted_samples).sum(dim=0) / float(draws * draws)
    crps = (samples - target.unsqueeze(0)).abs().mean(dim=0) - 0.5 * expected_pairwise_distance
    lower = torch.quantile(samples, 0.05, dim=0)
    upper = torch.quantile(samples, 0.95, dim=0)
    return {
        "masked_mae": float(error.abs().mean().detach().cpu()),
        "masked_rmse": float(error.square().mean().sqrt().detach().cpu()),
        "masked_crps": float(crps.mean().detach().cpu()),
        "interval_90_coverage": float(((target >= lower) & (target <= upper)).float().mean().detach().cpu()),
        "interval_90_width": float((upper - lower).mean().detach().cpu()),
    }


def forecast_rmse(mean: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((mean - target).square()).sqrt().detach().cpu())
