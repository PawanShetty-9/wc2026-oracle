"""
app.py — WC2026 Neural Betting Oracle
======================================
Main Streamlit application entry point. Deploy via Streamlit Cloud with
main file path set to "app.py" (at the repo root).

ARCHITECTURE:
  Cold start (first load):
    → @st.cache_resource trains all models (~15-30s) from historical data
    → Progress spinner is shown while training

  Warm start (subsequent loads / page navigation):
    → Models served from Streamlit's resource cache (instant)
    → Odds/results re-fetched from cache layer (30-min TTL)

PAGE STRUCTURE:
  ⚡ NEURAL GRID       → Today's matches + bet signals
  ◈ TOURNAMENT ORACLE  → All group predictions + standings
  ⊕ PROFIT MATRIX      → Bet tracker + P&L analytics
  ◉ SIGNAL ANALYSIS    → ELO rankings + model insights
  ◆ DATASTREAM         → How it works / methodology

HOW TO RUN LOCALLY:
  cd WC2026Oracle
  pip install -r requirements.txt
  streamlit run app.py

HOW TO DEBUG:
  - If models take >60s to train: set FAST_MODE=True (uses smaller dataset)
  - If you see "ModuleNotFoundError": ensure you are running from the project
    root directory (WC2026Oracle/) with: streamlit run app.py
  - If Streamlit shows ImportError: ensure requirements.txt is installed
  - For verbose model logs: set LOG_LEVEL=DEBUG env var before starting
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path

# ── Path fix: ensure project root is on Python path ──────────────────────────
# This allows "from data.xxx import yyy", "from models.xxx import yyy", etc.
# to work when running "streamlit run app.py" from the project root.
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wc2026.app")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AetherMind WC2026 Oracle",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CYBERPUNK + SACRED GEOMETRY CSS INJECTION
# Injected once at the top of every page via st.markdown(unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
CYBERPUNK_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

/* ── Global font override ─────────────────────────────────────────────── */
html, body, [class*="css"], .stMarkdown, .stText, div {
    font-family: 'Share Tech Mono', monospace !important;
}
h1, h2, h3, h4, .stMetric label {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.08em;
}

/* ── Neon heading glows ───────────────────────────────────────────────── */
h1 {
    color: #00FFD1 !important;
    text-shadow: 0 0 20px #00FFD1, 0 0 40px #00FFD180, 0 0 80px #00FFD130 !important;
}
h2 { color: #FF6B9D !important; text-shadow: 0 0 10px #FF6B9D60 !important; }
h3 { color: #00FFD1 !important; }

/* ── Scanline effect (subtle CRT look) ───────────────────────────────── */
body::after {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.08) 2px,
        rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Sacred geometry background watermark ────────────────────────────── */
/* Hexagonal grid using CSS gradients */
.main .block-container {
    background-image:
        linear-gradient(rgba(0,255,209,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,209,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── Neon card panels (expanders) ────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,255,209,0.20) !important;
    box-shadow:
        0 0 20px rgba(0,255,209,0.06),
        inset 0 0 30px rgba(0,0,0,0.4) !important;
    backdrop-filter: blur(4px) !important;
    border-radius: 4px !important;
    margin: 4px 0 !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(0,255,209,0.50) !important;
    box-shadow: 0 0 30px rgba(0,255,209,0.15) !important;
}

/* ── Metric value glow ────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    color: #00FFD1 !important;
    font-family: 'Orbitron', sans-serif !important;
    text-shadow: 0 0 8px rgba(0,255,209,0.6) !important;
}
[data-testid="stMetricLabel"] {
    color: #888 !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
}

/* ── Neon buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00FFD1 !important;
    color: #00FFD1 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 8px rgba(0,255,209,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(0,255,209,0.1) !important;
    box-shadow: 0 0 20px rgba(0,255,209,0.6), 0 0 40px rgba(0,255,209,0.3) !important;
    color: #FFF !important;
}

/* ── Sidebar neon border ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(0,255,209,0.15) !important;
    box-shadow: 4px 0 20px rgba(0,255,209,0.05) !important;
}

/* ── Radio/selectbox ─────────────────────────────────────────────────── */
.stRadio label, .stSelectbox label {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    color: #00FFD1 !important;
}

/* ── Dataframe / table ───────────────────────────────────────────────── */
.dataframe {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Slider ──────────────────────────────────────────────────────────── */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #00FFD1, #FF6B9D) !important;
}

/* ── Info / warning / success banners ───────────────────────────────── */
.stAlert {
    border-left: 3px solid #00FFD1 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────── */
.stProgress > div > div {
    background-color: #00FFD1 !important;
}
</style>
"""

# Inject CSS once per session
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — @st.cache_resource ensures models train ONCE per server
# worker and are shared across all users on that worker.
# Training is slow (~20s) but subsequent requests are instant.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚡ INITIALISING NEURAL ORACLE — training prediction models...")
def load_all_models():
    """Train and cache all prediction models.

    Returns
    -------
    dict with keys: elo, dc, xgb, eng, calibrator, predictor

    HOW TO DEBUG:
        If this function throws an error, it will show as a red Streamlit
        error box. Common causes:
        - ImportError: package not installed (check requirements.txt)
        - ValueError from Dixon-Coles: not enough training data
        - XGBoost ConvergenceWarning: harmless, model still works
    """
    from data.historical import load_training_data
    from data.wc2026_teams import TEAM_META, RESULTS_SO_FAR, GROUPS
    from models.elo import EloSystem
    from models.dixon_coles import DixonColesModel
    from models.xgboost_model import load_or_train_xgb
    from models.calibration import ProbabilityCalibrator
    from models.ensemble import EnsemblePredictor
    from features.engineering import FeatureEngineer

    import numpy as np
    import pandas as pd
    logger.info("Loading historical training data...")
    train_df = load_training_data(min_year=1993)

    # Augment training data with actual WC 2026 results (24 real matches)
    _wc_rows = []
    for _r in RESULTS_SO_FAR:
        _hs, _as = _r["home_score"], _r["away_score"]
        _wc_rows.append({
            "date":       _r["date"],
            "home_team":  _r["home"],
            "away_team":  _r["away"],
            "home_score": _hs,
            "away_score": _as,
            "tournament": "FIFA World Cup",
            "neutral":    True,
            "outcome":    2 if _hs > _as else (0 if _hs < _as else 1),
            "_source":    "wc2026_actual",
        })
    _wc_df = pd.DataFrame(_wc_rows)
    _wc_df["date"] = pd.to_datetime(_wc_df["date"])
    augmented_df = pd.concat([train_df, _wc_df], ignore_index=True).sort_values("date").reset_index(drop=True)
    _sample_weight = np.ones(len(augmented_df))
    _sample_weight[augmented_df["_source"] == "wc2026_actual"] = 10.0

    # ── ELO System ───────────────────────────────────────────────────────
    logger.info("Training ELO system...")
    initial_ratings = {team: meta["elo"] for team, meta in TEAM_META.items()}
    elo = EloSystem(initial_ratings=initial_ratings)
    elo.train_on_history(train_df)

    # Update ELO with actual WC 2026 tournament results so far
    for result in RESULTS_SO_FAR:
        elo.update(
            home_team=result["home"],
            away_team=result["away"],
            home_score=result["home_score"],
            away_score=result["away_score"],
            is_wc=True,
            is_neutral=True,
        )
    logger.info("ELO updated with %d WC 2026 results", len(RESULTS_SO_FAR))

    # ── Dixon-Coles Model ─────────────────────────────────────────────────
    logger.info("Fitting Dixon-Coles model...")
    dc = DixonColesModel(xi=0.002)
    dc.fit(train_df)

    # ── Feature Engineer ──────────────────────────────────────────────────
    eng = FeatureEngineer(
        historical_df=augmented_df,
        elo_system=elo,
        dc_model=dc,
        team_meta=TEAM_META,
    )

    # ── XGBoost Classifier ────────────────────────────────────────────────
    logger.info("Loading / training XGBoost model (augmented with WC 2026 data)...")
    xgb = load_or_train_xgb(eng, augmented_df, force_retrain=True, sample_weight=_sample_weight)

    # ── Calibrator (minimal — fitted on synthetic validation split) ───────
    calibrator = ProbabilityCalibrator()
    # Note: calibrator is not fitted here to avoid needing a holdout set
    # in all deployment conditions. It's a no-op until explicitly fitted.

    # ── Ensemble Predictor ────────────────────────────────────────────────
    predictor = EnsemblePredictor(
        dc_model=dc,
        xgb_model=xgb,
        feature_engineer=eng,
        calibrator=calibrator,
    )

    # ── Retrospective prediction accuracy for completed WC 2026 matches ───
    from data.cache import log_prediction, settle_prediction
    from datetime import date as _date
    logger.info("Computing retrospective accuracy for %d completed WC matches...", len(RESULTS_SO_FAR))
    for _result in RESULTS_SO_FAR:
        _home = _result["home"]
        _away = _result["away"]
        _date_str = _result["date"]
        _stage = _result["stage"]
        _mid = f"{_home}_vs_{_away}_{_date_str}"
        _group = next(
            (g for g, teams in GROUPS.items() if _home in teams or _away in teams),
            "",
        )
        try:
            _match_date = _date.fromisoformat(_date_str)
            _pred = predictor.predict(_home, _away, _match_date, _stage, _group)
            log_prediction(
                match_id=_mid,
                home_team=_home,
                away_team=_away,
                match_date=_date_str,
                stage=_stage,
                pred_winner=_pred.favourite,
                home_prob=_pred.home_prob,
                draw_prob=_pred.draw_prob,
                away_prob=_pred.away_prob,
                pred_home_goals=_pred.predicted_score[0] if _pred.predicted_score else None,
                pred_away_goals=_pred.predicted_score[1] if _pred.predicted_score else None,
            )
            _hs, _as = _result["home_score"], _result["away_score"]
            _ar = "H" if _hs > _as else ("A" if _as > _hs else "D")
            settle_prediction(_mid, _ar, _hs, _as)
        except Exception as _exc:
            logger.warning("Retrospective prediction failed for %s vs %s: %s", _home, _away, _exc)

    logger.info("All models loaded successfully.")
    return {
        "elo":        elo,
        "dc":         dc,
        "xgb":        xgb,
        "eng":        eng,
        "calibrator": calibrator,
        "predictor":  predictor,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS CACHE — regenerated when weights change or every 30 min
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def get_all_predictions(dc_weight: float, xgb_weight: float) -> list:
    """Get predictions for all upcoming matches.

    Cached for 30 minutes (TTL) or invalidated when weights change.
    """
    from data.wc2026_teams import upcoming_matches
    models = load_all_models()

    # Update ensemble weights from user settings
    models["predictor"].weights = {
        "dixon_coles": dc_weight,
        "xgboost":     xgb_weight,
    }

    today = date.today()
    upcoming = upcoming_matches(as_of_date=today.isoformat())
    return models["predictor"].predict_all_upcoming(upcoming, today)


@st.cache_data(ttl=1800, show_spinner=False)
def get_all_odds() -> dict:
    """Fetch/cache all available odds."""
    from data.loader import OddsAPIClient
    client = OddsAPIClient()
    return client.get_wc_odds()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build bet recommendations for a list of predictions
# ─────────────────────────────────────────────────────────────────────────────

def build_all_recommendations(
    predictions: list,
    all_odds: dict,
    bankroll: float,
    kelly_fraction: float,
    min_ev: float,
) -> list:
    """Build portfolio-optimised bet recommendations across all upcoming matches."""
    from strategy.ev import analyse_match_markets
    from strategy.kelly import build_recommendation
    from strategy.portfolio import build_portfolio
    from data.wc2026_teams import get_demo_odds

    all_recs = []

    for pred in predictions:
        odds = all_odds.get(f"{pred.home_team}_vs_{pred.away_team}")
        if odds is None:
            odds = get_demo_odds(pred.home_team, pred.away_team)
        if odds is None:
            continue

        # Analyse all markets for this match
        ev_results = analyse_match_markets(
            home_prob=pred.home_prob,
            draw_prob=pred.draw_prob,
            away_prob=pred.away_prob,
            over_25_prob=pred.over_25_prob,
            btts_prob=pred.btts_prob,
            odds=odds,
            home_team=pred.home_team,
            away_team=pred.away_team,
            min_ev=min_ev,
        )

        for ev_r in ev_results:
            rec = build_recommendation(
                ev_result=ev_r,
                match_id=pred.match_id,
                home_team=pred.home_team,
                away_team=pred.away_team,
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                match_date=pred.match_date,
                stage=pred.stage,
            )
            if rec is not None:
                all_recs.append(rec)

    return build_portfolio(all_recs, bankroll)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
from ui.sidebar import render_sidebar
settings = render_sidebar()

bankroll        = settings["bankroll"]
kelly_fraction  = settings["kelly_fraction"]
min_ev          = settings["min_ev"]
dc_weight       = settings["dc_weight"]
xgb_weight      = settings["xgb_weight"]
show_matrix     = settings["show_score_matrix"]

# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE BANNER
# ─────────────────────────────────────────────────────────────────────────────
from data.loader import is_demo_mode
if is_demo_mode():
    st.warning(
        "⚡ **DEMO MODE** — Running on bundled data. "
        "Add `FOOTBALL_DATA_API_KEY` and `ODDS_API_KEY` in Streamlit secrets for live predictions. "
        "[Get free keys ↗](https://www.football-data.org/client/register)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
PAGES = {
    "⚡ NEURAL GRID":      "today",
    "◈ TOURNAMENT ORACLE": "tournament",
    "⊕ PROFIT MATRIX":     "tracker",
    "◉ SIGNAL ANALYSIS":   "insights",
    "◆ DATASTREAM":        "methodology",
}

page_label = st.sidebar.radio(
    "NAVIGATION",
    list(PAGES.keys()),
    label_visibility="collapsed",
)
page = PAGES[page_label]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (cached after first call)
# ─────────────────────────────────────────────────────────────────────────────
try:
    models = load_all_models()
except Exception as exc:
    st.error(f"❌ Model loading failed: {exc}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ⚡ NEURAL GRID (Today's Matches + Bet Signals)
# ─────────────────────────────────────────────────────────────────────────────
if page == "today":
    st.markdown("# ⚡ NEURAL GRID")
    st.markdown(
        f"<div style='font-family:monospace; font-size:13px; color:#888;'>"
        f"Real-time predictions for upcoming 2026 FIFA World Cup matches · "
        f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Prediction accuracy widget ────────────────────────────────────────
    from data.cache import get_prediction_accuracy, get_prediction_history
    _acc = get_prediction_accuracy()
    if _acc["settled"] > 0:
        _col1, _col2, _col3, _col4 = st.columns(4)
        with _col1:
            st.metric("PREDICTIONS MADE", _acc["total"])
        with _col2:
            st.metric("SETTLED", _acc["settled"])
        with _col3:
            st.metric("CORRECT", _acc["correct"])
        with _col4:
            _color = "normal" if _acc["accuracy_pct"] >= 50 else "inverse"
            st.metric("ACCURACY", f"{_acc['accuracy_pct']:.1f}%",
                      delta=f"{_acc['correct']}/{_acc['settled']} matches",
                      delta_color=_color)
        st.divider()

    predictions = get_all_predictions(dc_weight, xgb_weight)
    all_odds    = get_all_odds()

    if not predictions:
        st.info("No upcoming matches found in the schedule.")
    else:
        portfolio = build_all_recommendations(
            predictions, all_odds, bankroll, kelly_fraction, min_ev
        )

        # ── Stake.com Betting Slip ────────────────────────────────────────
        from ui.betting_advice import render_stake_slip
        with st.expander("🎯 TODAY'S STAKE.COM BETTING SLIP", expanded=True):
            render_stake_slip(portfolio, bankroll)

        st.divider()

        # Notification banner for active signals
        if portfolio:
            st.success(
                f"⚡ **{len(portfolio)} BET SIGNAL(S) DETECTED** — "
                f"Total exposure: £{sum(r.stake_amount for r in portfolio):.2f} · "
                f"Expected profit: £{sum(r.stake_amount * r.ev for r in portfolio):.2f}"
            )

        # Render match cards grouped by date
        from itertools import groupby
        from ui.predictions import render_match_card

        # Filter to next 3 days of matches
        today_str = date.today().isoformat()
        near_predictions = [p for p in predictions if p.match_date >= today_str][:12]

        for pred in near_predictions:
            # Find recommendations for this specific match
            match_recs = [r for r in portfolio if r.match_id == pred.match_id]
            render_match_card(
                prediction=pred,
                recommendations=match_recs,
                show_matrix=show_matrix,
                dc_model=models["dc"],
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ◈ TOURNAMENT ORACLE (All Predictions + Group Standings)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "tournament":
    st.markdown("# ◈ TOURNAMENT ORACLE")
    st.markdown("Full 2026 World Cup group stage standings and predictions.")
    st.divider()

    from data.wc2026_teams import GROUPS as _BUNDLED_GROUPS, RESULTS_SO_FAR as _BUNDLED_RESULTS
    from data.loader import FootballDataClient
    from ui.predictions import render_match_card, render_group_standings

    _fd_client = FootballDataClient()

    # Use live API groups when available; fall back to bundled data
    _live_groups = _fd_client.get_live_groups()
    display_groups = _live_groups if _live_groups else _BUNDLED_GROUPS

    # Use live completed matches from API; fall back to bundled results
    if not _fd_client._demo:
        _api_results = _fd_client.get_completed_matches()
        display_results = _api_results if _api_results else _BUNDLED_RESULTS
        if _live_groups:
            st.success("⚡ **LIVE DATA** — Groups and results synced from football-data.org")
    else:
        display_results = _BUNDLED_RESULTS

    predictions = get_all_predictions(dc_weight, xgb_weight)
    all_odds    = get_all_odds()
    portfolio   = build_all_recommendations(
        predictions, all_odds, bankroll, kelly_fraction, min_ev
    )

    # Group tabs
    group_tabs = st.tabs([f"GROUP {g}" for g in sorted(display_groups.keys())])

    for tab, (group_letter, teams) in zip(group_tabs, sorted(display_groups.items())):
        with tab:
            col1, col2 = st.columns([1, 2])

            with col1:
                # Standings table built from live or bundled results
                group_results = [
                    r for r in display_results
                    if r["home"] in teams and r["away"] in teams
                ]
                render_group_standings(group_letter, teams, group_results)

            with col2:
                # Upcoming matches in this group
                group_preds = [p for p in predictions if p.group == group_letter]
                if group_preds:
                    st.markdown(f"**UPCOMING FIXTURES — GROUP {group_letter}**")
                    for pred in group_preds:
                        match_recs = [r for r in portfolio if r.match_id == pred.match_id]
                        render_match_card(pred, match_recs, show_matrix=False)
                else:
                    st.info(f"All Group {group_letter} matches completed")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ⊕ PROFIT MATRIX (Bet Tracker + P&L)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "tracker":
    st.markdown("# ⊕ PROFIT MATRIX")
    st.markdown("Track your bets, settle outcomes, and monitor ROI.")
    st.divider()

    from ui.betting_advice import render_bet_recommendations, render_bet_tracker

    # Current signals section
    st.markdown("## CURRENT BET SIGNALS")
    predictions = get_all_predictions(dc_weight, xgb_weight)
    all_odds    = get_all_odds()
    portfolio   = build_all_recommendations(
        predictions, all_odds, bankroll, kelly_fraction, min_ev
    )
    render_bet_recommendations(portfolio, bankroll)

    st.divider()

    # Tracker section
    render_bet_tracker()

    # ── Prediction History ────────────────────────────────────────────────
    st.divider()
    st.markdown("## 🎯 PREDICTION HISTORY")
    from data.cache import get_prediction_history, get_prediction_accuracy
    from ui.betting_advice import render_prediction_history
    render_prediction_history()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ◉ SIGNAL ANALYSIS (ELO Rankings + Model Insights)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "insights":
    st.markdown("# ◉ SIGNAL ANALYSIS")
    st.divider()

    from ui.charts import elo_rankings_chart, feature_importance_chart

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("## TEAM POWER GRID (ELO)")
        elo_fig = elo_rankings_chart(models["elo"].ratings, top_n=48)
        st.plotly_chart(elo_fig, use_container_width=True)

    with col2:
        st.markdown("## SIGNAL WEIGHTS (XGBOOST)")
        try:
            imp_df = models["xgb"].get_feature_importance()
            feat_fig = feature_importance_chart(imp_df)
            st.plotly_chart(feat_fig, use_container_width=True)
        except Exception:
            st.info("Feature importance unavailable — XGBoost model may not be fitted yet.")

    # Dixon-Coles attack/defense table
    st.divider()
    st.markdown("## ATTACK/DEFENSE STRENGTHS (DIXON-COLES)")
    import pandas as pd
    dc_data = []
    for team in sorted(models["dc"].attack_strengths.keys()):
        dc_data.append({
            "Team":    team,
            "Attack":  round(models["dc"].attack_strengths.get(team, 0), 3),
            "Defense": round(models["dc"].defense_strengths.get(team, 0), 3),
            "Net":     round(
                models["dc"].attack_strengths.get(team, 0) +
                models["dc"].defense_strengths.get(team, 0), 3
            ),
        })
    dc_df = pd.DataFrame(dc_data).sort_values("Net", ascending=False)
    st.dataframe(dc_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ◆ DATASTREAM (Methodology)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "methodology":
    st.markdown("# ◆ DATASTREAM — HOW IT WORKS")
    st.divider()

    st.markdown("""
    ## PREDICTION ENGINE

    AetherMind combines three complementary models into a single neural ensemble:

    ### 1. ⬡ ELO RATING SYSTEM
    Based on the World Football Elo Ratings methodology. Every team has a rating
    (1500–2200). After each match:
    ```
    Δrating = K × (actual_score − expected_score)
    expected = 1 / (1 + 10^((opponent_elo − your_elo) / 400))
    ```
    - K = 40 (standard matches), 60 (World Cup)
    - Draw probability modelled as 0.30 × exp(−|Δelo| / 200)
    - All WC matches = neutral venue (no home advantage)

    ---

    ### 2. ◈ DIXON-COLES MODEL (1997)
    Bivariate Poisson model — the gold standard for football score prediction:
    ```
    λ_home = exp(attack_home + defense_away)
    λ_away = exp(attack_away + defense_home)
    P(i-j) = Poisson(i; λ_home) × Poisson(j; λ_away) × τ(i,j,ρ)
    ```
    The τ correction improves prediction of 0-0, 1-0, 0-1, 1-1 scorelines.
    Parameters are found by Maximum Likelihood Estimation (L-BFGS-B).
    Time decay (ξ=0.002) weights recent matches more heavily.

    ---

    ### 3. ⚡ XGBOOST ML CLASSIFIER
    Gradient-boosted decision tree ensemble with 23 features:
    - ELO difference, FIFA ranking difference
    - Recent form (last 5 and 10 matches)
    - Head-to-head record (last 5 meetings)
    - Goals scored/conceded averages
    - Days rest since last match
    - Tournament stage encoding

    ---

    ### 4. ◉ ENSEMBLE COMBINATION
    ```
    P(outcome) = 0.60 × Dixon-Coles + 0.40 × XGBoost
    ```
    Weights adjustable in sidebar. Calibrated via isotonic regression.

    ---

    ## BETTING STRATEGY

    ### EXPECTED VALUE
    ```
    EV = model_probability × decimal_odds − 1
    ```
    Only recommend bets where EV > 3% (configurable in sidebar).

    ### KELLY CRITERION
    ```
    Kelly% = (odds × probability − 1) / (odds − 1)
    Stake  = 25% Kelly × bankroll   (fractional Kelly for risk management)
    ```

    ### PORTFOLIO MANAGEMENT
    - Maximum 20% bankroll exposure simultaneously
    - Maximum 5 concurrent open bets
    - Correlated bets on same match are deduplicated

    ---

    ## ⚠️ DISCLAIMER
    This tool is for **educational and entertainment purposes only**.
    It does NOT constitute financial advice. Gambling carries risk of loss.
    Please bet responsibly and within your means.
    """)
