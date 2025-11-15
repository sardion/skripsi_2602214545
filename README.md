# Comparative Analysis of LSTM, TCN, and TFT for Financial Time Series Forecasting  
### *Accuracy & Computational Efficiency Study on the Indonesia Stock Exchange (IDX)*  
**Researcher:** Sardion Maranatha (2602214545)  
**University:** BINUS University  
**Year:** 2025  

---

## Project Overview

This repository contains the full implementation of the research titled:

**“Analisis Komparatif LSTM, TCN dan TFT untuk Peramalan Deret Waktu Keuangan: Akurasi dan Efisiensi Komputasi”**

The study compares **three deep learning architectures**:

- **LSTM** (Long Short-Term Memory)  
- **TCN** (Temporal Convolutional Network)  
- **TFT** (Temporal Fusion Transformer – simplified version)**  

across **accuracy**, **training efficiency**, and **computational resource usage** in predicting:

- **Close Price (next day)**
- **Arithmetic Mean Price (OHLC average)**

on **five IDX tickers** (BBCA, ANTM, ICBP, TLKM, ASII) using a **55-feature unified feature set**.

---

## Key Features of This Repository

- **Complete data pipeline**
  - Raw data ➝ calendar-aligned ➝ engineered features ➝ sliding windows
- **Independent training pipelines**
  - `/lstm/`, `/tcn/`, `/tft/` with 3 scenarios each:
    - Early Stopping + Embargo  
    - Early Stopping (no embargo)  
    - Efficiency (100 epochs fixed)
- **Deterministic training setup**
  - Fixed seeds, reproducible dataloaders, CPU fallback
- **Automatic metrics logging & plots**
  - RMSE, MAE, MAPE  
  - Computational cost: elapsed time, CPU util %, memory usage  
- **Metrics collector script**
  - Generates a unified CSV table for chapter 4 analysis
- **Notebook + scripts compatible**
- **Fully CPU-compatible for AWS c7i / r6i instances**

---

## Project Structure

skripsi_2602214545/
│
├── data/
│ ├── raw/ # Raw IDX, Yahoo Finance, Investing.com data
│ ├── raw_features/ # Processed calendar TS + technical indicators
│ ├── processed/ # Per-ticker aligned feature tables
│ └── features/ # Sliding-window supervised datasets
│
├── lstm/
│ ├── stopping_embargo/
│ ├── stopping_no_embargo/
│ └── efficiency/
│
├── tcn/
│ ├── stopping_embargo/
│ ├── stopping_no_embargo/
│ └── efficiency/
│
├── tft/
│ ├── stopping_embargo/
│ ├── stopping_no_embargo/
│ └── efficiency/
│
├── config/ # Global configuration
├── splitter/ # Train-val-test split logic (+ embargo)
├── figures/ # Plotting utilities
├── metrics/ # Eval + metrics collector
├── scripts/ # Data preprocessing scripts
└── results/ # All metrics, figures, logs, checkpoints

## Installation
git clone https://github.com/sardion/thesis2602214545.git
cd skripsi_2602214545

Create virtual environment (recommended):
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

Install dependencies:
pip install -r requirements.txt

## Running Training Scripts
Each architecture has independent training scripts.
Example: 
1. Train LSTM (close price, early stopping + embargo)
python lstm/stopping_embargo/train_close.py --ticker BBCA
2. Train TCN (mean price, efficiency 100 epochs)
python tcn/efficiency/train_mean.py --ticker TLKM

All results are automatically saved to:
results/metrics/
results/figures/
results/logs/
results/checkpoints/

## Collecting Metrics
python collect_all_metrics.py
This generates: all_metrics_summary.csv

## Computational Environment (AWS)
This research was executed on AWS:
Instance Type: r6i.2xlarge
OS: Ubuntu 24.04
Python: 3.10
PyTorch: 2.8 (CPU mode)
Environment: Virtual environment (venv)

## Access to Full Code (Private Repository Policy)

This repository may be set to private to protect academic integrity.
Examiners can request access by contacting:
Email: sardion.maranatha@gmail.coom

