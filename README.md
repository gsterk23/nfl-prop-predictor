# 🏈 NFL Prop Predictor

A machine learning tool that predicts the probability of NFL players hitting 
their prop lines using XGBoost and normal distribution modeling.

## Features
- Supports RB, WR, TE, and QB positions
- Pulls 4 seasons of real NFL data (2022-2025)
- Engineers 7-12 predictive features per position including:
  - Weighted recent averages (L5 games)
  - Usage rates (carries/targets/attempts)
  - Opponent defensive ratings
  - Home/away splits
  - EPA efficiency metrics
  - Target share and WOPR (WR/TE)
- Outputs over/under probability using normal distribution
- Interactive web UI built with Streamlit

## Model Performance
| Position | MAE | R² |
|---|---|---|
| RB | 19.58 yards | 0.434 |
| WR | 21.71 yards | 0.311 |
| QB | 63.66 yards | 0.317 |

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the web app
```bash
streamlit run app.py
```

### Run the CLI version
```bash
python nfl_prop_predictor.py
```

## Example Output
Enter a player, opponent, and prop line to get:
- Predicted yards
- Over/under probability
- Normal distribution visualization
- Defensive context

## Tech Stack
- Python 3.11
- XGBoost
- Scikit-learn
- Pandas / NumPy
- SciPy
- Streamlit
- nflreadpy