# LSTM Scaffold — Global `data/features/`

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
└── lstm/
    ├── stopping_embargo/
    ├── stopping_no_embargo/
    └── efficiency/
```

Each scenario has independent scripts (no shared helpers).

To run
cd lstm/stopping_embargo
python train_close.py --ticker BBCA