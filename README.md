# HEIDI: High-Fidelity Digital Twin Construction

This public implementation exposes the DDGN proposal and Bayesian residual
refinement path used by the experimental implementation, its two-stage training
logic, and optimized endpoint recovery. It supersedes the earlier
EndToEndBayesianDT architecture-only interface.

**Release scope:** architecture, training mechanics, generic evaluation
utilities, and synthetic interface tests. Paper-run settings are intentionally
withheld. This repository is not a turnkey reproduction of the paper's figures
and tables, and synthetic checks do not validate the reported performance.

## Included

- Directed edge-conditioned graph messages, causal GRU history, and DDGN proposals.
- Bayesian residual reconstruction/forecasting that preserves received measurements.
- A Wasserstein critic, stage-specific losses, and training that freezes DDGN
  and the critic before fitting the Bayesian refiner.
- Endpoint-only inference with window-local spatial vectorization, predictive
  scales, Monte Carlo aggregation, and optional reference-RNG alignment.
- Configurable packet primitives, shared evolution Q, offline consistency F,
  reconstruction/uncertainty metrics, and deadline rates.
- Synthetic examples/tests and a placeholder-only configuration template.

## Deliberately not included

Real model widths/depths, initialization settings, optimizer settings, loss
weights, epochs, batch sizes, seeds, MC settings, checkpoint selection, channel
probabilities/delays, visibility thresholds, feature mappings, normalization,
data splits, topology assets, simulator coefficients, acceptance calibration,
and measured results/timing files are not provided. Datasets, fitted weights,
experiment launchers, baselines, and private provenance are also excluded.
No private settings are loaded automatically.

Source code reveals algorithm structure and tensor interfaces. Withholding
experiment settings does not prevent independent reimplementation.

## Quick check

Use Python 3.10 or later and a suitable PyTorch installation.

~~~bash
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python examples/architecture_smoke.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
~~~

Every numeric choice in examples/tests is **toy only**, not a paper setting,
recommended configuration, data sample, or measured result.

Supply model/training fields explicitly for private use. The template is
documentation, not an executable experiment configuration.

## Interfaces

Generic input shapes are:

- node observations and boolean visibility: [batch, time, node, feature];
- edge features: [batch, time, edge, edge_feature];
- global context: [batch, time, global_feature];
- directed edge indices: [source/destination, edge].

Mask unavailable observations before inference. Clean targets belong only in
offline training/evaluation. See [architecture](docs/architecture.md) and
[evaluation semantics](docs/protocol.md).

Internal DDGN/BNN state-dict module names are retained. Checkpoint loading
requires private dimensions/numerical settings. No checkpoints are included.
This interface replaces the previous public EndToEndBayesianDT constructor.

## Communication and data

The packet primitive exposes independent jitter, independent per-device burst
states, and shared congestion. All probabilities, delay ranges, seeds, and
visibility deadlines are caller-supplied. Virtual steps do not imply physical
milliseconds without an external calibration.

The experimental FFP line uses recorded observations. The BPLP line uses
Bosch-derived topology with simulated dynamic states. Neither private adapter
nor the BPLP process simulator is distributed here. A generic packet primitive
does not replace that process simulator.

Original sources:

- [Future Factories analog dataset](https://www.kaggle.com/datasets/ramyharik/ff-2023-12-12-analog-dataset)
- [Bosch Production Line Performance](https://www.kaggle.com/competitions/bosch-production-line-performance)

## Citation and license

Please cite the HEIDI paper when using this implementation. The final
bibliographic record can be added after publication. See [MIT license](LICENSE).

## Updating an existing checkout

Copy all contents into the repository root, including hidden files, replacing
matching files. The previous model, architecture documentation/PDF, and test
are replaced together. Do not mix files from the two interfaces. Keep private
settings and experimental data outside the upload.
