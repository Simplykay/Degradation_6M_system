# Cotton Seed Degradation Prediction Intelligence System

Predicts cotton seed quality degradation using a CT score threshold of 60%.

Primary deliverable: **M6 survival modeling**, which estimates how many growing seasons remain before a lot's CT score drops below 60.

## Project Layout

```text
Data/raw/      Raw CSV files, excluded from git because several exceed GitHub size limits
models/        Saved deployment model artifacts
src/           Reusable loading, feature, training, and prediction code
outputs/       Generated reports and plots, excluded from git
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Place the nine source CSV files in `Data/raw/`.

## Train Models

```bash
python -m src.train_m1_m5
python -m src.train_m6
```

Current validation:

- M1 AUC: `0.9707`
- M2 RMSE: `8.04`
- M3 Macro-F1: `0.7359`
- M4 AUC: `0.6365`, below target and needs iteration
- M5 RMSE: `22.22`, below target and needs iteration
- M6 Cox PH C-index: `0.8974`
- M6 AFT median predicted shelf-life on test: `4.12` seasons

## Predict Shelf Life

```python
import pandas as pd
from src.predict import predict_shelf_life

lots = pd.DataFrame([{
    "WG_Current": 90,
    "CT_Initial": 70,
    "Moisture": 7,
    "Mechanical_Damage": 5,
    "Actual_Seed_Per_LB": 4300,
    "Stage": 4,
    "SEASON_YR": 2024,
    "Origin_Region": "AZ",
}])

print(predict_shelf_life(lots))
```
