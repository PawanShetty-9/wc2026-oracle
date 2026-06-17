"""
ui/betting_advice.py — Bet Tracker and Advice Renderer
=======================================================
Two main sections:
  1. render_bet_recommendations() — top bets sorted by EV with stakes
  2. render_bet_tracker()         — log bets, settle outcomes, show P&L

HOW TO DEBUG:
  - If bet tracker is empty: check that add_bet() writes to the correct
    SQLite file (betting/data/raw/betting_cache.db)
  - If P&L chart is blank: ensure at least one bet has been settled
  - If stakes seem too high: check that bankroll in session_state is set
    correctly (should default to £1000)
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from data.cache import add_bet, get_all_bets, get_pnl_summary, settle_bet
from strategy.kelly import BetRecommendation
from strategy.portfolio import portfolio_summary
from ui.charts import pnl_chart, ev_scatter_chart

TIER_CONFIG = {
    "STRONG": {"color": "#00FF88", "icon": "◉", "border": "#00FF88"},
    "MEDIUM": {"color": "#FFB800", "icon": "◎", "border": "#FFB800"},
    "WEAK":   {"color": "#FF6B9D", "icon": "○", "border": "#FF6B9D"},
}


def render_bet_recommendations(
    recommendations: list[BetRecommendation],
    bankroll: float,
) -> None:
    """Display all current bet recommendations with stakes and EV.

    Parameters
    ----------
    recommendations : list[BetRecommendation] — from build_portfolio()
    bankroll        : float — current bankroll
    """
    if not recommendations:
        st.markdown(
            """
            <div style="
                border: 1px dashed #00FFD130;
                padding: 30px;
                text-align: center;
                font-family: Orbitron, sans-serif;
                color: #555;
                margin: 20px 0;
            ">
                <div style="font-size: 24px; color: #00FFD140;">◉</div>
                <div style="margin-top: 10px;">NO VALUE BETS DETECTED</div>
                <div style="font-size: 12px; margin-top: 8px; font-family: monospace; color: #444;">
                    Lower the MIN EDGE FILTER in the sidebar to see more candidates
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Portfolio summary metrics
    summary = portfolio_summary(recommendations, bankroll)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ACTIVE SIGNALS", summary["n_bets"])
    with c2:
        st.metric("TOTAL AT RISK", f"£{summary['total_stake']:.2f}",
                  f"{summary['total_exposure_pct']:.1f}% of bankroll")
    with c3:
        sign = "+" if summary["expected_profit"] >= 0 else ""
        st.metric("EXPECTED PROFIT", f"£{sign}{summary['expected_profit']:.2f}")
    with c4:
        st.metric("BEST EV", f"+{summary['best_ev']*100:.1f}%")

    st.divider()

    # EV scatter chart
    if len(recommendations) > 1:
        fig = ev_scatter_chart(recommendations)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚡ ACTIVE BET SIGNALS")

    for i, rec in enumerate(recommendations):
        cfg = TIER_CONFIG.get(rec.confidence_tier, TIER_CONFIG["MEDIUM"])

        with st.container():
            st.markdown(
                f"""
                <div style="
                    border: 1px solid {cfg['border']};
                    border-left: 5px solid {cfg['border']};
                    padding: 14px 18px;
                    margin: 8px 0;
                    background: rgba(13,13,43,0.9);
                    box-shadow: 0 0 20px {cfg['border']}25;
                    font-family: 'Share Tech Mono', monospace;
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="color:{cfg['color']}; font-size:18px; font-family:Orbitron;">{cfg['icon']} {rec.confidence_tier}</span>
                            &nbsp;&nbsp;
                            <span style="color:#E0E0FF; font-size:15px;"><b>{rec.outcome_label}</b></span>
                        </div>
                        <div style="text-align:right;">
                            <span style="color:#FFB800; font-size:20px; font-family:Orbitron;">@ {rec.decimal_odds:.2f}</span>
                        </div>
                    </div>
                    <div style="margin-top:8px; color:#888; font-size:12px;">
                        {rec.home_team} vs {rec.away_team} · {rec.match_date}
                    </div>
                    <div style="margin-top:10px; display:flex; gap:24px; font-size:13px; flex-wrap:wrap;">
                        <span>Model: <b style="color:{cfg['color']};">{rec.model_prob*100:.1f}%</b></span>
                        <span>Implied: <b style="color:#888;">{rec.implied_prob*100:.1f}%</b></span>
                        <span>Edge: <b style="color:{cfg['color']};">{rec.edge_pct_str}</b></span>
                        <span>EV: <b style="color:{cfg['color']};">{rec.ev_pct_str}</b></span>
                        <span>Stake: <b style="color:#00FFD1;">£{rec.stake_amount:.2f}</b></span>
                        <span>If wins: <b style="color:#00FF88;">+£{rec.potential_profit:.2f}</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Quick-log button
            btn_col, _ = st.columns([1, 4])
            with btn_col:
                if st.button(f"📋 LOG BET", key=f"log_{i}_{rec.match_id}"):
                    bet_id = add_bet(
                        match_id=rec.match_id,
                        home_team=rec.home_team,
                        away_team=rec.away_team,
                        market=rec.market,
                        outcome_label=rec.outcome_label,
                        decimal_odds=rec.decimal_odds,
                        stake=rec.stake_amount,
                        model_prob=rec.model_prob,
                        ev=rec.ev,
                    )
                    st.success(f"✓ Bet #{bet_id} logged to tracker")


def render_bet_tracker() -> None:
    """Render the full bet tracker UI: open bets, settle outcomes, P&L chart."""
    st.markdown("### 📊 BET TRACKER")

    bets = get_all_bets()
    summary = get_pnl_summary()

    # ── Summary metrics ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("TOTAL BETS", summary["total_bets"])
    with c2:
        st.metric("OPEN", summary["open_bets"], delta="active")
    with c3:
        st.metric("SETTLED", summary["settled_bets"])
    with c4:
        st.metric("TOTAL STAKED", f"£{summary['total_staked']:.2f}")
    with c5:
        pnl = summary["total_pnl"]
        roi = summary["roi_pct"]
        sign = "+" if pnl >= 0 else ""
        st.metric("P&L", f"£{sign}{pnl:.2f}", f"ROI: {sign}{roi:.1f}%",
                  delta_color="normal" if pnl >= 0 else "inverse")

    # ── P&L chart ─────────────────────────────────────────────────────────
    fig = pnl_chart(bets)
    st.plotly_chart(fig, use_container_width=True)

    # ── Open bets to settle ───────────────────────────────────────────────
    open_bets = [b for b in bets if b["result"] is None]
    if open_bets:
        st.markdown("#### 🔴 OPEN BETS — SETTLE OUTCOMES")
        for bet in open_bets:
            cols = st.columns([3, 1, 1, 1])
            with cols[0]:
                st.markdown(
                    f"**{bet['outcome_label']}** @ {bet['decimal_odds']:.2f}  "
                    f"· £{bet['stake']:.2f} · *{bet['home_team']} vs {bet['away_team']}*"
                )
            with cols[1]:
                if st.button("✅ WIN", key=f"win_{bet['id']}"):
                    profit = settle_bet(bet["id"], "WIN")
                    st.success(f"+£{profit:.2f}")
                    st.rerun()
            with cols[2]:
                if st.button("❌ LOSS", key=f"loss_{bet['id']}"):
                    settle_bet(bet["id"], "LOSS")
                    st.warning(f"-£{bet['stake']:.2f}")
                    st.rerun()
            with cols[3]:
                if st.button("↩ VOID", key=f"void_{bet['id']}"):
                    settle_bet(bet["id"], "VOID")
                    st.info("Bet voided")
                    st.rerun()

    # ── Settled bets history ──────────────────────────────────────────────
    settled = [b for b in bets if b["result"] is not None]
    if settled:
        st.markdown("#### ✓ SETTLED HISTORY")
        df = pd.DataFrame(settled)[[
            "placed_at", "outcome_label", "home_team", "away_team",
            "decimal_odds", "stake", "result", "pnl"
        ]]
        df.columns = ["Date", "Bet", "Home", "Away", "Odds", "Stake", "Result", "P&L"]
        df["P&L"] = df["P&L"].apply(lambda x: f"+£{x:.2f}" if x >= 0 else f"-£{abs(x):.2f}")

        st.dataframe(df, use_container_width=True, hide_index=True)

    if not bets:
        st.info("No bets logged yet. Use the 'LOG BET' button on any signal card.")


def render_prediction_history() -> None:
    """Render prediction history table with accuracy stats."""
    from data.cache import get_prediction_history, get_prediction_accuracy

    history = get_prediction_history()
    acc = get_prediction_accuracy()

    if not history:
        st.info("Prediction history will appear here after the first model load. Refresh the page.")
        return

    # Accuracy summary
    settled = [h for h in history if h["actual_result"] is not None]
    pending = [h for h in history if h["actual_result"] is None]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("TOTAL PREDICTED", acc["total"])
    with c2:
        st.metric("SETTLED", acc["settled"])
    with c3:
        st.metric("CORRECT", acc["correct"])
    with c4:
        color = "normal" if acc["accuracy_pct"] >= 50 else "inverse"
        st.metric(
            "MODEL ACCURACY",
            f"{acc['accuracy_pct']:.1f}%",
            delta=f"{acc['correct']}/{acc['settled']} correct" if acc["settled"] else "pending",
            delta_color=color,
        )

    if settled:
        st.markdown("#### ✓ SETTLED PREDICTIONS")
        rows = []
        for h in settled:
            actual_score = f"{h['actual_home_goals']} – {h['actual_away_goals']}" if h["actual_home_goals"] is not None else "?"
            pred_score = f"{h['pred_home_goals']} – {h['pred_away_goals']}" if h["pred_home_goals"] is not None else "?"
            result_icon = "✅" if h["correct"] else "❌"
            rows.append({
                "Date":      h["match_date"],
                "Match":     f"{h['home_team']} vs {h['away_team']}",
                "Predicted": h["pred_winner"],
                "Pred Score": pred_score,
                "Actual":    actual_score,
                "Correct":   result_icon,
                "H%":        f"{h['home_prob']*100:.0f}",
                "D%":        f"{h['draw_prob']*100:.0f}",
                "A%":        f"{h['away_prob']*100:.0f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    if pending:
        st.markdown("#### ⏳ PENDING PREDICTIONS")
        rows = []
        for h in pending:
            pred_score = f"{h['pred_home_goals']} – {h['pred_away_goals']}" if h["pred_home_goals"] is not None else "?"
            rows.append({
                "Date":      h["match_date"],
                "Match":     f"{h['home_team']} vs {h['away_team']}",
                "Predicted": h["pred_winner"],
                "Pred Score": pred_score,
                "H%":        f"{h['home_prob']*100:.0f}",
                "D%":        f"{h['draw_prob']*100:.0f}",
                "A%":        f"{h['away_prob']*100:.0f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_stake_slip(
    recommendations: list[BetRecommendation],
    bankroll: float,
) -> None:
    """Render a Stake.com formatted betting slip with navigation guide."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0D0D2B 0%, #060611 100%);
            border: 2px solid #00FFD1;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: 0 0 30px #00FFD130;
        ">
            <div style="font-family:Orbitron,sans-serif; color:#00FFD1; font-size:18px; font-weight:700; letter-spacing:0.1em;">
                ⚡ STAKE.COM BETTING SLIP
            </div>
            <div style="font-family:monospace; color:#888; font-size:12px; margin-top:6px;">
                Navigate: Stake.com → Sports → Soccer → FIFA World Cup 2026
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not recommendations:
        st.markdown(
            "<div style='text-align:center; color:#555; padding:20px; font-family:monospace;'>"
            "NO VALUE BETS TODAY — check back after odds update</div>",
            unsafe_allow_html=True,
        )
        return

    total_stake = sum(r.stake_amount for r in recommendations)
    total_exp   = sum(r.stake_amount * r.ev for r in recommendations)

    st.markdown(
        f"<div style='font-family:monospace; color:#888; font-size:13px; margin-bottom:12px;'>"
        f"📋 {len(recommendations)} bet(s) · "
        f"Total stake: <b style='color:#00FFD1;'>£{total_stake:.2f}</b> · "
        f"Expected profit: <b style='color:#00FF88;'>+£{total_exp:.2f}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    market_nav = {
        "1X2":    "Match Result",
        "over_25": "Goals → Over/Under → Over 2.5",
        "btts":    "Both Teams to Score → Yes",
    }

    for i, rec in enumerate(recommendations, 1):
        cfg = TIER_CONFIG.get(rec.confidence_tier, TIER_CONFIG["MEDIUM"])
        nav_path = market_nav.get(rec.market, rec.market)

        st.markdown(
            f"""
            <div style="
                border: 1px solid {cfg['border']}60;
                border-left: 4px solid {cfg['border']};
                background: rgba(6,6,17,0.95);
                padding: 14px 18px;
                margin: 8px 0;
                font-family: 'Share Tech Mono', monospace;
                box-shadow: 0 0 15px {cfg['border']}15;
            ">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <span style="color:{cfg['color']}; font-family:Orbitron; font-size:14px; font-weight:700;">
                            {cfg['icon']} BET {i} — {rec.confidence_tier}
                        </span>
                        <div style="margin-top:6px; color:#E0E0FF; font-size:15px; font-weight:bold;">
                            {rec.outcome_label}
                        </div>
                        <div style="margin-top:3px; color:#888; font-size:12px;">
                            {rec.home_team} vs {rec.away_team} · {rec.match_date}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#FFB800; font-family:Orbitron; font-size:22px; font-weight:700;">@ {rec.decimal_odds:.2f}</div>
                        <div style="color:#00FFD1; font-size:16px; font-weight:bold; margin-top:2px;">Stake: £{rec.stake_amount:.2f}</div>
                        <div style="color:#00FF88; font-size:13px;">If wins: +£{rec.potential_profit:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:10px; padding-top:10px; border-top:1px solid #1A1F2E;">
                    <div style="color:#00FFD180; font-size:11px; text-transform:uppercase; letter-spacing:0.05em;">
                        📍 Stake.com path: Sports → Soccer → FIFA World Cup 2026 → {rec.home_team} vs {rec.away_team} → {nav_path}
                    </div>
                    <div style="margin-top:6px; display:flex; gap:20px; font-size:12px; color:#666;">
                        <span>Model: <b style="color:{cfg['color']};">{rec.model_prob*100:.1f}%</b></span>
                        <span>Bookmaker: <b style="color:#888;">{rec.implied_prob*100:.1f}%</b></span>
                        <span>Edge: <b style="color:{cfg['color']};">{rec.edge_pct_str}</b></span>
                        <span>EV: <b style="color:{cfg['color']};">{rec.ev_pct_str}</b></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="
            background: rgba(0,255,209,0.05);
            border: 1px solid #00FFD130;
            border-radius: 4px;
            padding: 10px 14px;
            font-family: monospace;
            font-size: 11px;
            color: #666;
            margin-top: 12px;
        ">
            ⚠️ Responsible gambling: Never bet more than you can afford to lose.
            Stakes shown are Kelly Criterion recommendations based on your bankroll setting.
            Verify odds on Stake.com before placing — live odds may differ from displayed values.
        </div>
        """,
        unsafe_allow_html=True,
    )
