# High-Fidelity Digital Twin Construction  Under Incomplete Measurements
![Static Badge](https://img.shields.io/badge/Bayesian%20Inference-red)![Static Badge](https://img.shields.io/badge/Windows%2011-blue)![Static Badge](https://img.shields.io/badge/Data%20Generation-grey)![Static Badge](https://img.shields.io/badge/Generative%20Adversarial%20Network-yellow)![Static Badge](https://img.shields.io/badge/Graph%20Neural%20Network-brightgreen)![Static Badge](https://img.shields.io/badge/License-MIT-purple)

## Abstract
Owing to the capabilities of real-time replication, parallel evolution, and feedback-driven optimization, Digital Twin (DT) is recognized as a promising paradigm for resilient production. In Industrial Internet of Things (IIoT) environments, production data are acquired by distributed sensors and transmitted to the DT in real time to support twin construction, state synchronization, and evolution correction. However, most existing studies rely on overly idealized assumptions about data availability and timeliness. In practical IIoT-enabled production systems, DT construction is often hindered by incomplete measurements caused by limited measurement capability and excessive transmission latency. To address these issues, we propose the High-Fidelity DT Construction(HEIDI) method, which includes the REaL-time uncertAinTy infErence (RELATE) module and the data-driven DT Construction module. These two modules respectively address the missing data problem caused by latency and measurement limitations, and enable the parallel evolution and feedback of the twin. In particular, RELATE consists of two core modules: a temporal graph module that enables missing data completion to adapt to time-varying production changes, and a data generation module, which explicitly models uncertainty and infers latent production states that cannot be directly measured, such as equipment wear conditions. The DT construction module includes two neural network models respond to the parallel evolution and feedback decision. Finally, we construct a simplified IIoT-enabled manufacturing DT based on HEIDI and conduct extensive experiments to validate its effectiveness.

## RELATE Framework Figure

![Framework](Architecture.png "It is the first version of the framework figure, but not the final.")

## How to run the code?
The required packages can be installed directly by the env.yaml file.

BIG code: `BIG_Network.py` and `BIG_Training.py`

DDGN code: `DDGN.py` and `GNN_Network.py`

The data can be found in

 - https://www.kaggle.com/datasets/ramyharik/ff-2023-12-12-analog-dataset 


 - https://www.kaggle.com/competitions/bosch-production-line-performance.


**<font color=red>Note:</font>** Some personal infomation are deleted, please note these problems in running. 
