# ⬡ WC2026 Neural Betting Oracle

> **Disclaimer**: All predictions are for educational and entertainment purposes only. This is not financial advice. Gamble responsibly.

A state-of-the-art World Cup 2026 betting predictor built with Dixon-Coles, ELO, and XGBoost — wrapped in a cyberpunk Streamlit UI. Runs fully in **demo mode** (no API keys needed) with bundled 2026 WC schedule, results, and realistic odds.

---

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/pawanshetty-9/wc2026-oracle.git
cd wc2026-oracle

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run (no API keys needed — demo mode)
streamlit run app.py
# Opens at http://localhost:8501

# 5. Run tests
python -m pytest tests/ -v
```

---

## Architecture

```
wc2026-oracle/
├── app.py                      # Streamlit entry point (5 pages)
├── requirements.txt
├── .streamlit/
│   ├── config.toml             # Cyberpunk dark theme
│   └── secrets.toml.example    # API key template
├── data/
│   ├── wc2026_teams.py         # 48 teams, schedule, demo odds (OFFLINE)
│   ├── historical.py           # Loads CSV; synthetic fallback for CI
│   ├── loader.py               # football-data.org + The Odds API clients
│   └── cache.py                # SQLite TTL cache (thread-safe)
├── models/
│   ├── elo.py                  # World Football ELO (K=40/60)
│   ├── dixon_coles.py          # Bivariate Poisson + tau correction
│   ├── xgboost_model.py        # XGBoost 3-class classifier (23 features)
│   ├── ensemble.py             # 60% DC + 40% XGB → MatchPrediction
│   └── calibration.py          # Isotonic regression post-hoc calibration
├── features/
│   ├── form.py                 # Form, goals stats, H2H, days rest
│   └── engineering.py          # Walk-forward feature builder (no lookahead)
├── strategy/
│   ├── ev.py                   # EV = model_prob × (odds−1) − (1−model_prob)
│   ├── kelly.py                # 25% fractional Kelly stake sizing
│   └── portfolio.py            # Max 20% exposure, max 5 bets, dedup
├── ui/
│   ├── sidebar.py              # Controls: bankroll, Kelly %, EV filter
│   ├── predictions.py          # Match cards with probability bars
│   ├── betting_advice.py       # Bet signals + P&L tracker
│   └── charts.py               # Plotly: ELO grid, feature importance, P&L
└── tests/                      # pytest suite (no API keys needed)
```

---

## Prediction Models

### 1. Dixon-Coles (60% weight by default)
Bivariate Poisson model with τ low-score correction for draws (0-0, 1-0, 0-1, 1-1). MLE optimized via L-BFGS-B with time decay (ξ=0.002) — recent matches count more. Outputs an 11×11 score probability matrix from which all market probabilities are derived.

### 2. ELO Rating System (used by XGBoost as features)
World Football ELO with K=40 (friendly) / K=60 (World Cup). Draw probability = 0.30 × exp(−|Δelo|/200). All World Cup matches are treated as neutral-venue games.

### 3. XGBoost Classifier (40% weight by default)
23-feature gradient boosted classifier trained on 1993–2022 international results. Walk-forward construction prevents lookahead bias. Features include ELO diff, FIFA rank diff, form (5/10 match windows), H2H record, goals averages, days rest, and tournament round encoding.

### 4. Ensemble + Calibration
Weighted average of Dixon-Coles and XGBoost probabilities (adjustable via sidebar). Isotonic regression post-hoc calibration ensures probabilities are well-calibrated against historical data.

---

## Betting Strategy

### Expected Value
```
EV = model_prob × (decimal_odds − 1) − (1 − model_prob)
```
Only bets with EV ≥ 3% (adjustable) are shown as recommendations.

### Kelly Criterion
```
f* = (decimal_odds × model_prob − 1) / (decimal_odds − 1)
stake = 0.25 × f* × bankroll   (25% fractional Kelly)
```

### Portfolio Rules
- Maximum 20% of bankroll at risk across all open bets
- Maximum 5 simultaneous bets
- Only one market per match (highest EV wins; no correlated bets)
- Bets sorted by EV descending

**Verification example**: 60% model probability vs 2.0 decimal odds → EV = +20%, Full Kelly = 20% → 25% fractional = 5% of bankroll.

---

## Data Sources

| Source | Purpose | Free? | TTL Cache |
|--------|---------|-------|-----------|
| `data/wc2026_teams.py` | Schedule, demo odds | Built-in | — |
| Bundled CSV | Model training (CC0 license) | Yes | — |
| football-data.org | Live WC results | Free (10 req/min) | 30 min |
| The Odds API | Live bookmaker odds | 500 req/month | 30 min |

---

## App Pages

| Page | What You See |
|------|-------------|
| ⚡ NEURAL GRID | Today's upcoming matches with bet signals |
| ◈ TOURNAMENT ORACLE | All group stage predictions + standings |
| ⊕ PROFIT MATRIX | All value bets ranked by EV, bet tracker, P&L |
| ◉ SIGNAL ANALYSIS | ELO power rankings, XGBoost feature importance |
| ◆ DATASTREAM | Full methodology explanation with math |

---

## Optional: Live Data (API Keys)

Without keys, the app runs identically in demo mode. To enable live data:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `secrets.toml`:
```toml
FOOTBALL_DATA_API_KEY = "your_key"   # football-data.org (free)
ODDS_API_KEY = "your_key"            # the-odds-api.com (500 req/month free)
```

**Get free keys**:
- football-data.org: Register at `https://www.football-data.org/client/register`
- The Odds API: Sign up at `https://the-odds-api.com/#get-access`

---

## Streamlit Community Cloud Deployment

### One-time setup
1. **Push to GitHub** (already done if you're reading this)

2. **Go to** `https://share.streamlit.io` → sign in with GitHub (pawanshetty-9)

3. **New App** → configure:
   ```
   Repository:  pawanshetty-9/wc2026-oracle
   Branch:      main
   Main file:   app.py
   ```

4. **(Optional) Add secrets** — App settings → Secrets:
   ```toml
   FOOTBALL_DATA_API_KEY = "..."
   ODDS_API_KEY = "..."
   ```

5. **Deploy** — first build takes ~3 minutes (installing deps + cold-start model training)

### Auto-redeploy
Every `git push` to `main` triggers an automatic redeploy (takes ~60 seconds).

### Updating World Cup results (demo mode)
After each match day, edit `data/wc2026_teams.py` → `RESULTS_SO_FAR` list, then push:
```bash
git add data/wc2026_teams.py
git commit -m "chore: add WC2026 Matchday 2 results"
git push
```

---

## Running Tests

```bash
# From repo root (no API keys needed)
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_elo.py -v
python -m pytest tests/test_dixon_coles.py -v
python -m pytest tests/test_kelly.py -v
python -m pytest tests/test_features.py -v
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run from repo root: `streamlit run app.py` |
| App shows "training models" forever | Check `data/raw/` — bundled CSV may be missing |
| Odds show "Demo" even with API key | Verify secret name is exactly `ODDS_API_KEY` in Streamlit Cloud UI |
| XGBoost fails to train | Delete `data/raw/xgb_model.joblib` and restart |
| Streamlit Cloud can't find requirements | Confirm `requirements.txt` exists at repo root |
| `sqlite3.OperationalError` | Check that `data/raw/` directory exists (gitignored but auto-created) |

---

## Key Design Decisions

- **Self-contained**: no external module dependencies beyond `requirements.txt`
- **Demo-first**: App is fully functional without any external API calls
- **Walk-forward only**: All feature computation strictly avoids lookahead bias
- **Thread-safe cache**: SQLite uses `threading.local()` for Streamlit's multi-threaded server
- **`@st.cache_resource`**: Models train once and are shared across all browser sessions
- **SQLite gitignored**: `*.db` and `*.joblib` are in `.gitignore` (auto-created on startup)

---

*WC2026 Neural Betting Oracle v1.0 · 2026 · For educational use only · Not financial advice*
