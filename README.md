# HEIDI: High-Fidelity Digital Twin Construction

This package contains the architecture-level public release of **HEIDI** for
incomplete IIoT observations. It exposes the causal graph-temporal encoder,
edge-conditioned message passing, Bayesian latent/output heads, uncertainty
outputs, the joint loss interface, and generic evaluation metrics.

HEIDI is designed for digital-twin construction when industrial observations
are incomplete, delayed, or unavailable. The public code presents the method's
overall architecture and tensor interfaces; it is intended for academic
inspection and citation.

The release is intentionally limited to the overall method structure. It does
not contain the private experiment configuration needed to reproduce the paper
protocol.

This boundary reduces direct reproducibility, but it is not a cryptographic or
legal guarantee that the method cannot be independently reimplemented.

## Included

- HEIDI architecture code under `src/heidi/`;
- a toy-only shape smoke test under `examples/`;
- a public configuration template with withheld values;
- a conceptual architecture description and the architecture figure.

## Not included

- experimental or industrial datasets;
- checkpoints, normalization statistics, results, logs, and provenance files;
- comparison/baseline methods;
- formal experiment launchers and the paper-run data schema;
- private hyperparameters, seeds, missingness masks, simulator settings, and
  other protocol details.

The public smoke test is deliberately configured with toy dimensions and is not
the paper experiment. It verifies tensor interfaces only.

## Datasets

The datasets are not redistributed in this repository. Please obtain them from
their original sources and follow their respective terms of use:

- **FFP / Future Factories analog dataset:**
  [Kaggle dataset](https://www.kaggle.com/datasets/ramyharik/ff-2023-12-12-analog-dataset)
- **Bosch Production Line Performance:**
  [Kaggle competition](https://www.kaggle.com/competitions/bosch-production-line-performance)

These links are provided for provenance only. Dataset access, preprocessing,
feature mapping, normalization statistics, and the exact experiment protocol
are not included in this release.

## Quick smoke test

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python examples/architecture_smoke.py
```

This is not a reproduction recipe. The paper-run parameter handover sheet is
maintained outside this public directory and is provided only after a direct
request and an appropriate citation agreement.

## Citation

Please cite the HEIDI paper whenever this architecture or a derivative is used.
The final bibliographic record will be added here after publication.

## License

This repository is released under the MIT License; see
[`LICENSE`](LICENSE). A license alone cannot enforce academic citation, so
please also cite the HEIDI paper when using or extending this code.
