"""
ui/sidebar.py — Streamlit Sidebar Configuration Panel
======================================================
The sidebar provides user controls for:
  - Bankroll amount (affects all Kelly stake calculations)
  - Kelly fraction (risk level: 10-50%)
  - Minimum EV threshold (bet filter: 1-10%)
  - Model weights (Dixon-Coles % vs XGBoost %)
  - API status indicators
  - Data refresh button

All settings are stored in Streamlit's session_state so they persist
across page navigation within the same browser session.

HOW TO DEBUG:
  - If settings don't persist across pages, ensure they're stored with
    st.session_state["key"] = value (not local variables)
  - If sliders have wrong ranges, adjust the min_value/max_value args
  - If API status shows wrong state, check is_demo_mode() in loader.py
"""

from __future__ import annotations

import streamlit as st

from data.loader import is_demo_mode


def render_sidebar() -> dict:
    """Render the full sidebar and return a dict of user settings.

    Returns
    -------
    dict with keys:
        bankroll         : float — total bankroll in currency units
        kelly_fraction   : float — 0.10 to 0.50
        min_ev           : float — 0.01 to 0.10
        dc_weight        : float — 0.0 to 1.0 (Dixon-Coles weight)
        xgb_weight       : float — 1.0 - dc_weight
        show_score_matrix: bool  — show Dixon-Coles heatmap on cards
        refresh_clicked  : bool  — True if user clicked the refresh button
    """
    st.sidebar.markdown(
        """
        <div style="
            text-align: center;
            font-family: Orbitron, sans-serif;
            font-size: 20px;
            color: #00FFD1;
            text-shadow: 0 0 15px #00FFD1, 0 0 30px #00FFD180;
            letter-spacing: 0.15em;
            padding: 10px 0;
        ">
        ⬡ AETHERMIND<br>
        <span style="font-size:11px; color:#E0E0FF; letter-spacing:0.3em;">NEURAL BETTING ORACLE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    # ── API Status ───────────────────────────────────────────────────────────
    demo = is_demo_mode()
    if demo:
        st.sidebar.markdown(
            "<div style='font-family:monospace; font-size:12px; color:#FFB800;'>"
            "🔴 LIVE ODDS · OFFLINE<br>"
            "🔴 MATCH DATA · BUNDLED<br>"
            "<span style='color:#888;'>Add API keys for live data →</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "<div style='font-family:monospace; font-size:12px; color:#00FF88;'>"
            "🟢 LIVE ODDS · CONNECTED<br>"
            "🟢 MATCH DATA · LIVE"
            "</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.divider()

    # ── Bankroll ─────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='color:#00FFD1; font-family:Orbitron; font-size:12px; letter-spacing:0.1em;'>BANKROLL</div>",
        unsafe_allow_html=True,
    )
    bankroll = st.sidebar.number_input(
        "Total bankroll (£)",
        min_value=10.0,
        max_value=100_000.0,
        value=float(st.session_state.get("bankroll", 1000.0)),
        step=50.0,
        format="%.2f",
        label_visibility="collapsed",
    )
    st.session_state["bankroll"] = bankroll

    st.sidebar.divider()

    # ── Kelly Fraction ───────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='color:#00FFD1; font-family:Orbitron; font-size:12px; letter-spacing:0.1em;'>KELLY FRACTION</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Risk level: 10% = conservative, 50% = aggressive")
    kelly_pct = st.sidebar.slider(
        "Kelly fraction (%)",
        min_value=10,
        max_value=50,
        value=int(st.session_state.get("kelly_pct", 25)),
        step=5,
        label_visibility="collapsed",
    )
    st.session_state["kelly_pct"] = kelly_pct
    kelly_fraction = kelly_pct / 100.0

    # ── Minimum EV Threshold ─────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='color:#00FFD1; font-family:Orbitron; font-size:12px; letter-spacing:0.1em; margin-top:8px;'>MIN EDGE FILTER</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Only show bets with at least this EV edge")
    min_ev_pct = st.sidebar.slider(
        "Min EV (%)",
        min_value=1,
        max_value=10,
        value=int(st.session_state.get("min_ev_pct", 3)),
        step=1,
        label_visibility="collapsed",
    )
    st.session_state["min_ev_pct"] = min_ev_pct
    min_ev = min_ev_pct / 100.0

    # ── Model Weights ─────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='color:#00FFD1; font-family:Orbitron; font-size:12px; letter-spacing:0.1em;'>MODEL WEIGHTS</div>",
        unsafe_allow_html=True,
    )
    dc_pct = st.sidebar.slider(
        "Dixon-Coles weight (%)",
        min_value=0,
        max_value=100,
        value=int(st.session_state.get("dc_pct", 60)),
        step=10,
        help="Remaining weight goes to XGBoost ML model",
    )
    st.session_state["dc_pct"] = dc_pct
    dc_weight  = dc_pct / 100.0
    xgb_weight = 1.0 - dc_weight

    st.sidebar.caption(
        f"Dixon-Coles: **{dc_pct}%** · XGBoost: **{100-dc_pct}%**"
    )

    # ── Display Options ───────────────────────────────────────────────────────
    st.sidebar.divider()
    show_matrix = st.sidebar.checkbox(
        "Show score probability matrix",
        value=False,
        help="Show Dixon-Coles heatmap on each match card (slower rendering)",
    )

    # ── Data Refresh ──────────────────────────────────────────────────────────
    refresh_clicked = st.sidebar.button(
        "⟳ REFRESH DATA",
        use_container_width=True,
        help="Force re-fetch of odds and match results from APIs",
    )

    if refresh_clicked:
        from data.cache import clear_cache
        cleared = clear_cache()
        st.sidebar.success(f"✓ Cleared {cleared} cache entries")

    # ── Info footer ───────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='font-family:monospace; font-size:10px; color:#555; text-align:center;'>"
        "AETHERMIND v1.0 · 2026<br>"
        "NOT FINANCIAL ADVICE<br>"
        "For educational use only"
        "</div>",
        unsafe_allow_html=True,
    )

    return {
        "bankroll":          bankroll,
        "kelly_fraction":    kelly_fraction,
        "min_ev":            min_ev,
        "dc_weight":         dc_weight,
        "xgb_weight":        xgb_weight,
        "show_score_matrix": show_matrix,
        "refresh_clicked":   refresh_clicked,
    }
