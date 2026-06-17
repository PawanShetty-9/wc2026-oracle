"""
ui/predictions.py — Match Prediction Card Renderer
===================================================
Renders individual match prediction cards in the Streamlit UI.
Each card uses the cyberpunk aesthetic: dark background, neon accents,
sacred-geometry-inspired borders, and monospace data displays.

Each match card shows:
  - Match header (date, stage, group, venue)
  - Team names with flag emojis
  - Probability bar chart (home/draw/away)
  - Predicted scoreline
  - Confidence indicator
  - Bet recommendations (if available)
  - Over/Under and BTTS probabilities

HOW TO DEBUG:
  - If a card doesn't render: check that prediction is a MatchPrediction
    dataclass (not a dict or None)
  - If the probability bar is missing: check that plotly is installed
    and charts.probability_bar_chart() returns a valid Figure
"""

from __future__ import annotations

import streamlit as st

from models.ensemble import MatchPrediction
from strategy.kelly import BetRecommendation
from ui.charts import probability_bar_chart, score_matrix_heatmap

# Emoji flags for common teams (for visual flair in cyberpunk UI)
TEAM_FLAGS: dict[str, str] = {
    "ENGLAND":      "🏴󠁧󠁢󠁥󠁮󠁧󠁿",  "USA":         "🇺🇸",
    "GERMANY":      "🇩🇪",  "FRANCE":      "🇫🇷",
    "SPAIN":        "🇪🇸",  "BRAZIL":      "🇧🇷",
    "ARGENTINA":    "🇦🇷",  "NETHERLANDS": "🇳🇱",
    "PORTUGAL":     "🇵🇹",  "BELGIUM":     "🇧🇪",
    "ITALY":        "🇮🇹",  "CROATIA":     "🇭🇷",
    "MEXICO":       "🇲🇽",  "CANADA":      "🇨🇦",
    "JAPAN":        "🇯🇵",  "SOUTH KOREA": "🇰🇷",
    "MOROCCO":      "🇲🇦",  "SENEGAL":     "🇸🇳",
    "COLOMBIA":     "🇨🇴",  "URUGUAY":     "🇺🇾",
    "DENMARK":      "🇩🇰",  "SWITZERLAND": "🇨🇭",
    "AUSTRIA":      "🇦🇹",  "POLAND":      "🇵🇱",
    "SERBIA":       "🇷🇸",  "AUSTRALIA":   "🇦🇺",
    "IRAN":         "🇮🇷",  "NIGERIA":     "🇳🇬",
    "CAMEROON":     "🇨🇲",  "EGYPT":       "🇪🇬",
    "CHILE":        "🇨🇱",  "ECUADOR":     "🇪🇨",
    "SAUDI ARABIA": "🇸🇦",  "IVORY COAST": "🇨🇮",
    "HUNGARY":      "🇭🇺",  "ROMANIA":     "🇷🇴",
    "ALGERIA":      "🇩🇿",  "MALI":        "🇲🇱",
    "JORDAN":       "🇯🇴",  "VENEZUELA":   "🇻🇪",
    "HONDURAS":     "🇭🇳",  "PANAMA":      "🇵🇦",
    "COSTA RICA":   "🇨🇷",  "PARAGUAY":    "🇵🇾",
    "UZBEKISTAN":   "🇺🇿",  "SOUTH AFRICA":"🇿🇦",
    "JAMAICA":      "🇯🇲",  "PERU":        "🇵🇪",
    "NEW ZEALAND":  "🇳🇿",
}


def get_flag(team: str) -> str:
    """Return the flag emoji for a team, or a generic football emoji."""
    return TEAM_FLAGS.get(team.upper(), "⚽")


def render_match_card(
    prediction: MatchPrediction,
    recommendations: list[BetRecommendation],
    show_matrix: bool = False,
    dc_model=None,
) -> None:
    """Render a single match prediction card.

    Parameters
    ----------
    prediction      : MatchPrediction — from EnsemblePredictor.predict()
    recommendations : list[BetRecommendation] — from build_portfolio()
                      (empty list = no bet recommendations for this match)
    show_matrix     : bool — if True, show the score probability heatmap
    dc_model        : DixonColesModel — needed if show_matrix=True
    """
    home = prediction.home_team
    away = prediction.away_team
    home_flag = get_flag(home)
    away_flag = get_flag(away)

    # Build a key for the expandable card header
    conf_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(prediction.confidence_label, "⚪")
    bet_badge  = f" · **{len(recommendations)} BET SIGNAL(S)**" if recommendations else ""
    header = (
        f"{conf_color} {home_flag} **{home}** vs **{away}** {away_flag}  "
        f"· {prediction.match_date} · {prediction.stage}"
        f"{bet_badge}"
    )

    with st.expander(header, expanded=bool(recommendations)):
        # ── Row 1: Match header info ──────────────────────────────────────
        cols = st.columns([2, 1, 2])
        with cols[0]:
            st.markdown(f"### {home_flag} {home}")
        with cols[1]:
            st.markdown(
                f"<div style='text-align:center; color:#FFB800; font-family:Orbitron; font-size:18px; padding-top:6px;'>VS</div>",
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(f"### {away} {away_flag}")

        # Venue and stage info
        if prediction.venue:
            st.caption(f"📍 {prediction.venue}  ·  Stage: {prediction.stage}  ·  Group {prediction.group}")

        st.divider()

        # ── Row 2: Probability bar ─────────────────────────────────────────
        fig = probability_bar_chart(
            home, away,
            prediction.home_prob,
            prediction.draw_prob,
            prediction.away_prob,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Row 3: Key metrics ─────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("PREDICTED SCORE", prediction.implied_score_str)
        with m2:
            st.metric("CONFIDENCE", prediction.confidence_label, f"{prediction.confidence*100:.0f}%")
        with m3:
            st.metric("OVER 2.5 GOALS", f"{prediction.over_25_prob*100:.0f}%")
        with m4:
            st.metric("BOTH SCORE", f"{prediction.btts_prob*100:.0f}%")

        # ── Row 4: Model breakdown (collapsible) ──────────────────────────
        with st.expander("🔍 Model Breakdown", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Dixon-Coles**")
                dc = prediction.dc_probs
                st.markdown(
                    f"<div style='font-family:monospace; font-size:12px; color:#00FFD1'>"
                    f"Home: {dc.get('home', 0)*100:.1f}%<br>"
                    f"Draw: {dc.get('draw', 0)*100:.1f}%<br>"
                    f"Away: {dc.get('away', 0)*100:.1f}%"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown("**XGBoost ML**")
                xgb = prediction.xgb_probs
                st.markdown(
                    f"<div style='font-family:monospace; font-size:12px; color:#FF6B9D'>"
                    f"Home: {xgb.get('home', 0)*100:.1f}%<br>"
                    f"Draw: {xgb.get('draw', 0)*100:.1f}%<br>"
                    f"Away: {xgb.get('away', 0)*100:.1f}%"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Row 5: Score probability heatmap (optional) ────────────────────
        if show_matrix and dc_model is not None:
            try:
                matrix = dc_model.predict_score_matrix(home, away)
                fig_m = score_matrix_heatmap(matrix, home, away, max_goals=6)
                st.plotly_chart(fig_m, use_container_width=True)
            except Exception:
                pass

        # ── Row 6: Bet recommendations ─────────────────────────────────────
        if recommendations:
            st.divider()
            st.markdown("#### ⚡ BET SIGNALS DETECTED")
            for rec in recommendations:
                render_bet_signal_inline(rec)


def render_bet_signal_inline(rec: BetRecommendation) -> None:
    """Render a single bet recommendation as a compact neon badge row."""
    tier_color = {"STRONG": "#00FF88", "MEDIUM": "#FFB800", "WEAK": "#FF6B9D"}.get(
        rec.confidence_tier, "#E0E0FF"
    )
    tier_icon  = {"STRONG": "◉", "MEDIUM": "◎", "WEAK": "○"}.get(rec.confidence_tier, "○")

    st.markdown(
        f"""
        <div style="
            border: 1px solid {tier_color};
            border-left: 4px solid {tier_color};
            padding: 10px 14px;
            margin: 6px 0;
            background: rgba(13, 13, 43, 0.8);
            box-shadow: 0 0 12px {tier_color}30;
            font-family: 'Share Tech Mono', monospace;
        ">
            <span style="color:{tier_color}; font-size:16px;">{tier_icon} {rec.confidence_tier}</span>
            &nbsp;&nbsp;
            <span style="color:#E0E0FF; font-size:14px;"><b>{rec.outcome_label}</b></span>
            &nbsp;&nbsp;
            <span style="color:#FFB800;">@ {rec.decimal_odds:.2f}</span>
            <br>
            <span style="color:#888; font-size:11px;">
                Model: {rec.model_prob*100:.1f}% ·
                Implied: {rec.implied_prob*100:.1f}% ·
                EV: <span style="color:{tier_color};">{rec.ev_pct_str}</span> ·
                Stake: <span style="color:#00FFD1;">£{rec.stake_amount:.2f}</span> ·
                Potential profit: <span style="color:#00FF88;">£{rec.potential_profit:.2f}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_group_standings(
    group: str,
    teams: list[str],
    results: list[dict],
) -> None:
    """Render a group table (W/D/L/GD/Pts) for a given group.

    Parameters
    ----------
    group   : str  — group letter ("A", "B", etc.)
    teams   : list — 4 team names in this group
    results : list — all completed matches (from RESULTS_SO_FAR)
    """
    import pandas as pd

    # Build standings from results
    standings: dict[str, dict] = {
        t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
        for t in teams
    }

    for r in results:
        home, away = r["home"], r["away"]
        if home not in standings or away not in standings:
            continue

        hs, as_ = int(r["home_score"]), int(r["away_score"])
        standings[home]["P"]  += 1
        standings[away]["P"]  += 1
        standings[home]["GF"] += hs
        standings[home]["GA"] += as_
        standings[away]["GF"] += as_
        standings[away]["GA"] += hs

        if hs > as_:
            standings[home]["W"]   += 1
            standings[home]["Pts"] += 3
            standings[away]["L"]   += 1
        elif hs < as_:
            standings[away]["W"]   += 1
            standings[away]["Pts"] += 3
            standings[home]["L"]   += 1
        else:
            standings[home]["D"]   += 1
            standings[home]["Pts"] += 1
            standings[away]["D"]   += 1
            standings[away]["Pts"] += 1

    # Build DataFrame
    rows = []
    for team, s in standings.items():
        rows.append({
            "Team":     team,
            "P":  s["P"],
            "W":  s["W"],
            "D":  s["D"],
            "L":  s["L"],
            "GF": s["GF"],
            "GA": s["GA"],
            "GD": s["GF"] - s["GA"],
            "Pts": s["Pts"],
        })

    df = pd.DataFrame(rows).sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-indexed position

    st.markdown(
        f"<h4 style='color:#00FFD1; font-family:Orbitron; letter-spacing:0.1em;'>GROUP {group}</h4>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=False,
    )
