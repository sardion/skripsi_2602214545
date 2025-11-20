# TFT Scaffold — Global `data/features/`

All scenarios read from the **project-level** path:
`<PROJECT_ROOT>/data/features/`

Expected project layout:
```
<PROJECT_ROOT>/
├── data/
│   └── features/   # <- unified features live here
└── tft/
    ├── stopping_embargo/
    ├── stopping_no_embargo/
    └── efficiency/
```
Each scenario is fully independent (no shared helpers).

## Hyperparameters
- **Accuracy (stopping_embargo / stopping_no_embargo)**: lookback 120, hidden_size=32, enc/dec LSTM layers=1, heads=4, dropout=0.2, Adam lr=1e-3, batch=32, ES on
- **Efficiency**: lookback 30, hidden_size=32, enc/dec LSTM layers=1, heads=4, dropout=0.2, Adam lr=1e-3, batch=32, AMP on, epochs=100 fixed (no ES)

to run:
cd tft/stopping_embargo
python train_close.py --ticker BBCA