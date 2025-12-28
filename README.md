# Towards Real Time Data Generation and Uncertainty Inference for Digital Twin Construction
![Static Badge](https://img.shields.io/badge/Bayesian%20Inference-red)![Static Badge](https://img.shields.io/badge/Windows%2011-blue)![Static Badge](https://img.shields.io/badge/Data%20Generation-grey)![Static Badge](https://img.shields.io/badge/Generative%20Adversarial%20Network-yellow)![Static Badge](https://img.shields.io/badge/Graph%20Neural%20Network-brightgreen)![Static Badge](https://img.shields.io/badge/License-MIT-purple)

## Abstract
Owing to the real-time replication, analytical prediction, and solution feedback capacities, Digital Twin (DT) is recognized as the key paradigm to enable resilient industrial production. Existing studies on digital twins primarily focus on the applications that can be realized through DT, while rarely investigating the issues associated with DT construction. Therefore, this study focuses on exploring the problems arising in the DT construction process in an industrial scenario. In the production process, some data may not be gathered fully in real time due to the limited measurement capacity or the excessive transmission latency. To address these problems, this study proposes the REaL-time uncertAinTy infErence (RELATE) for data generation, which resolves the mentioned problems by integrating real-time data generation and uncertainty inference functions. The data generation function adopts an improved TGN algorithm, namely Double-Dimension Graph Network (DDGN), to achieve real-time analysis of production states from both temporal and spatial dimensions. DDGN aims to extract features that can guide subsequent data generation. To infer the unmeasurable data, we design a Bayesian Inference Generator (BIG) by imitating the training process of Generative Adversarial Networks (GANs) to realize the inference and generation, and the Evidence Lower Bound (ELBO) maximization method is employed as the objective during the model training. Finally, extensive experiments are conducted to validate the effectiveness of the DDGN algorithm in feature aggregation and analysis, as well as the performance of BIG in data inference and generation.

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