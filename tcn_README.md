# TCN Scaffold — Global `data/features/`

All scenarios read from the **project-level** path:
`<PROJECT_ROOT>/data/features/`

Expected project layout:
```
<PROJECT_ROOT>/
├── data/
│   ├── raw/
│   ├── raw_features/
│   ├── processed/
│   └── features/   # <- unified features live here
└── tcn/
    ├── stopping_embargo/
    ├── stopping_no_embargo/
    └── efficiency/
```

Each scenario is fully independent (no shared helpers).

## Hyperparameters
- **Accuracy (stopping_embargo / stopping_no_embargo)**: lookback 120, residual_blocks=4, kernel_size=3, channels=64, dropout=0.2, Adam lr=2e-3, grad clip=0.5, batch=32
- **Efficiency**: lookback 30, residual_blocks=4, kernel_size=3, channels=64, dilations [1,2,4,8], Adam lr=1e-3, grad clip=0.5, batch=32, AMP on, epochs=100 fixed (no ES)

To run:

cd lstm/stopping_embargo
python train_close.py --ticker BBCA