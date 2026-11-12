# HEIDI: High-Fidelity Digital Twin Construction

Digital Twin (DT) enables real-time monitoring, parallel evolution, and feedback-driven optimization in Industrial Internet of Things (IIoT) production systems. However, most existing studies rely on overly idealized assumptions about data availability and timeliness. In practical IIoT-enabled production systems, DT construction is often hindered by incomplete measurements caused by limited measurement capability and excessive transmission latency. To address these issues, we propose the High-Fidelity DT Construction (HEIDI) method, which comprises the REaL-time uncertAinTy infErence (RELATE) module and a data-driven DT construction module. To recover missing information under changing production conditions, RELATE uses a Double-Dimension Graph Network (DDGN) to capture temporal interactions and spatial dependencies among equipment. For production states that cannot be directly measured, a Bayesian Inference Generator (BIG) infers their distributions from observable production evidence through Bayesian inference and adversarial learning. The DT construction module merges inferred information with received measurements to construct the twin state for subsequent evolution and feedback. Finally, experiments demonstrate that HEIDI reduces Mean Absolute Error (MAE) by 4.1% and increases construction rates by 8.3% on average over the strongest baselines, enabling timely DT construction while maintaining high construction accuracy

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

Please cite the HEIDI paper when using this implementation. The final bibliographic record can be added after publication. 


