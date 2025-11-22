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
```txt
skripsi_2602214545/
│
├── data/
│   ├── raw/                     # Raw IDX, Yahoo Finance, Investing.com data
│   ├── raw_features/            # Output: calendar TS + technical indicators
│   ├── processed/               # Per-ticker aligned feature tables
│   └── features/                # Sliding-window supervised datasets
│
├── lstm/
│   ├── stopping_embargo/
│   ├── stopping_no_embargo/
│   └── efficiency/
│
├── tcn/
│   ├── stopping_embargo/
│   ├── stopping_no_embargo/
│   └── efficiency/
│
├── tft/
    ├── stopping_embargo/
    ├── stopping_no_embargo/
    └── efficiency/

```

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
- **1. Train LSTM (close price, early stopping + embargo)
- **python lstm/stopping_embargo/train_close.py --ticker BBCA
- **2. Train TCN (mean price, efficiency 100 epochs)
- **python tcn/efficiency/train_mean.py --ticker TLKM

All results are automatically saved to:
- **results/metrics/
- **results/figures/
- **results/logs/
- **results/checkpoints/

## Collecting Metrics
python collect_all_metrics.py
This generates: all_metrics_summary.csv

## Computational Environment (AWS)
This research was executed on AWS:
- **Instance Type: r6i.2xlarge
- **OS: Ubuntu 24.04
- **Python: 3.10
- **PyTorch: 2.8 (CPU mode)
- **Environment: Virtual environment (venv)

## 📦 Raw Data Policy

This repository provides the complete preprocessing and training pipeline, but **does not include large datasets** due to size constraints and licensing rules.

### 1. Sample Raw Data Included
The directory:

```
data/raw/
```

contains **only minimal sample files** illustrating the expected input structure  
(e.g., trimmed IDX, Investing.com, Yahoo Finance samples).

These samples allow reviewers to understand the format without distributing full datasets.

### 2. Full Raw Data Not Included
The full raw datasets (Jan 2020 – Aug 2025) are *not* included because:
- They exceed GitHub’s storage limits
- Redistribution violates data source terms & conditions
- Total size reaches multiple GB across all tickers and macro sources

All download instructions and file formats are documented in Lampiran A–C of the thesis.

### 🗂 3. Processed Data Included (Except Sliding-Window)
The following directories **are included** because they are lightweight and safe to publish:

```
data/raw_features/
data/processed/
```

These contain:
- calendar-aligned time series  
- engineered features  
- normalized variants  
- integrated ticker feature tables  

These are sufficient for reviewers to inspect feature engineering.

### 4. Sliding-Window Supervised Datasets Not Included
The directory:

```
data/features/
```

is **left empty on purpose**, because:
- Sliding-window supervised learning datasets are extremely large  
- Sizes range from hundreds of MB to several GB  
- Regeneration is deterministic using the provided preprocessing scripts

Users can regenerate the full supervised learning datasets by running:

```
python scripts/transform_sliding_window.py
```

(as documented in Lampiran E).

### 5. Full Dataset Regeneration
All scripts needed to rebuild the entire dataset pipeline are provided under:

```
scripts/
```

This ensures full reproducibility without distributing copyrighted raw data.

### 6. Optional Private Repository Access
If required, the repository may be set to private, with access granted manually to reviewers.

## Large Log & Profiler Files Not Included

During efficiency experiments, each architecture (LSTM, TCN, TFT) produces detailed
profiling logs inside:

```
lstm/efficiency/results/logs/
tcn/efficiency/results/logs/
tft/efficiency/results/logs/
```

These include files such as:

```
{TICKER}_close_trace_cpu.json
{TICKER}_mean_trace_cpu.json
```

### Why These Files Are Excluded
These JSON trace files:
- Frequently exceed **100–300 MB** per ticker
- Surpass GitHub’s **100 MB per file** limit
- Contain low-level CPU event traces not essential for report evaluation
- Are fully regenerable from the public training scripts

### How to Regenerate Them
If needed, they can be reproduced by running the efficiency training scripts:

```
python lstm/efficiency/train_close.py --ticker BBCA
python tcn/efficiency/train_close.py --ticker BBCA
python tft/efficiency/train_close.py --ticker BBCA
```

The profiler will automatically create new trace files in:

```
results/logs/
```

These profiler traces are **not required** to reproduce metrics or conclusions of the thesis.

## Access to Full Code (Private Repository Policy)

This repository may be set to private to protect academic integrity.
Examiners can request access by contacting:
Email: sardion.maranatha@binus.ac.id / sardion.maranatha@gmail.com

