# iiot-wsn-explainable-ids
Explainable intrusion detection pipeline for IIoT/WSN using OMNeT++/INET simulation, machine learning, Bayesian Network, SHAP, and external validation.


# Explainable 

This repository contains datasets, scripts, figures, and reproducibility materials for the paper:

"Um Pipeline Explicável de Ciência de Dados para Detecção de Ataques em Redes IIoT/WSN com Simulação e Validação Externa"

## Datasets

- OMNeT++/INET simulated dataset: 20,000 samples, 5 traffic classes, 4 network densities.
- Trust-Aware IIoT Routing Dataset: external validation dataset used for multiclass and binary anomaly detection.

## Models

- Random Forest
- XGBoost
- Bayesian Network with 3 and 5 discretization bins

## Evaluation protocols

- Stratified 80/20 split
- Leave-one-topology-out validation
- Noise robustness analysis
- SHAP explainability
- External validation with Trust-Aware IIoT

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/01_limpeza_dados.py
python scripts/02_analise_exploratoria.py
python scripts/04_random_forest_model.py
python scripts/08_xgboost_model.py
python scripts/15_rede_bayesiana_5bins.py
