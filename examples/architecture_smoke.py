"""Synthetic interface check. Every numeric setting here is TOY ONLY."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from heidi import HEIDI, ModelNumerics, endpoint_recovery, endpoint_stats

def toy():
    numerics = ModelNumerics(initial_weight_rho=-2.7, weight_scale_floor=2e-5, predictive_scale_floor=3e-5)
    model = HEIDI(node_feature_dim=2, edge_feature_dim=1, global_feature_dim=1,
                  hidden_dim=7, graph_layers=1, numerics=numerics).eval()
    values = torch.randn(1, 4, 3, 2)
    mask = torch.rand_like(values) > 0.37
    mask[..., 0] = True
    mask[:, -1, :, 1] = False
    batch = {"input_values": torch.where(mask, values, torch.zeros_like(values)),
             "input_mask": mask, "edge_values": torch.randn(1, 4, 3, 1),
             "global_values": torch.randn(1, 4, 1)}
    edge = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    return model, batch, edge

def main():
    model, batch, edge = toy()
    with torch.inference_mode():
        full = model(**batch, edge_index=edge, sample_bnn=False)
        mean, scale = endpoint_stats(model, batch, edge, sample=False)
        torch.testing.assert_close(mean, full.completed_mean[:, -1], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(scale, full.reconstruction_std[:, -1], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(endpoint_recovery(model, batch, edge, sample=False), mean, atol=3e-6, rtol=3e-6)
    print("Synthetic DDGN + Bayesian residual interface check passed.")

if __name__ == "__main__":
    main()
