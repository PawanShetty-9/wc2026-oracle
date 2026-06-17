"""
ui/charts.py — Cyberpunk Plotly Visualisations
===============================================
All charts use a consistent dark cyberpunk theme:
  - Background: #060611 (near-black deep space)
  - Primary accent: #00FFD1 (neon cyan)
  - Secondary: #FF6B9D (neon pink)
  - Warning / amber: #FFB800
  - Text: #E0E0FF (cool white)

Each function returns a plotly Figure object.
Call fig = some_chart(...) then st.plotly_chart(fig, use_container_width=True).

HOW TO DEBUG:
  - If chart appears blank: check that the input data is not empty
  - If colours look wrong: Plotly's CSS inheritance from Streamlit may
    override theme — use explicit colors in all trace and layout calls
  - If chart is too small: always pass use_container_width=True to st.plotly_chart
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Cyberpunk colour palette ──────────────────────────────────────────────────
BG       = "#060611"
PANEL_BG = "#0D0D2B"
CYAN     = "#00FFD1"
PINK     = "#FF6B9D"
AMBER    = "#FFB800"
RED      = "#FF4444"
GREEN    = "#00FF88"
TEXT     = "#E0E0FF"
GRID     = "#1A1F2E"

_LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL_BG,
    font=dict(family="Share Tech Mono, monospace", color=TEXT, size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Reusable axis style — merge into per-chart xaxis/yaxis dicts as needed
_AX = dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT)


def probability_bar_chart(
    home_team: str,
    away_team: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
) -> go.Figure:
    """Horizontal stacked bar showing 1X2 probability split.

    Example: BRAZIL 45% | DRAW 25% | FRANCE 30%
    """
    fig = go.Figure()

    # Home win — neon cyan
    fig.add_trace(go.Bar(
        x=[home_prob * 100],
        y=[""],
        orientation="h",
        name=home_team,
        marker=dict(color=CYAN, line=dict(color=CYAN, width=1)),
        text=f"<b>{home_team}</b><br>{home_prob*100:.1f}%",
        textposition="inside",
        textfont=dict(color=BG, size=11, family="Orbitron"),
        hovertemplate=f"{home_team} Win: {home_prob*100:.1f}%<extra></extra>",
    ))

    # Draw — amber
    fig.add_trace(go.Bar(
        x=[draw_prob * 100],
        y=[""],
        orientation="h",
        name="Draw",
        marker=dict(color=AMBER, line=dict(color=AMBER, width=1)),
        text=f"DRAW<br>{draw_prob*100:.1f}%",
        textposition="inside",
        textfont=dict(color=BG, size=11, family="Orbitron"),
        hovertemplate=f"Draw: {draw_prob*100:.1f}%<extra></extra>",
    ))

    # Away win — neon pink
    fig.add_trace(go.Bar(
        x=[away_prob * 100],
        y=[""],
        orientation="h",
        name=away_team,
        marker=dict(color=PINK, line=dict(color=PINK, width=1)),
        text=f"<b>{away_team}</b><br>{away_prob*100:.1f}%",
        textposition="inside",
        textfont=dict(color=BG, size=11, family="Orbitron"),
        hovertemplate=f"{away_team} Win: {away_prob*100:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="stack",
        showlegend=False,
        height=80,
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return fig


def elo_rankings_chart(ratings: dict[str, float], top_n: int = 32) -> go.Figure:
    """Horizontal bar chart of ELO ratings for the top N teams."""
    sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    teams  = [t[0] for t in reversed(sorted_teams)]
    ratings_vals = [t[1] for t in reversed(sorted_teams)]

    # Colour teams by ELO tier
    colours = []
    for r in ratings_vals:
        if r >= 2050:
            colours.append(CYAN)
        elif r >= 1900:
            colours.append(GREEN)
        elif r >= 1750:
            colours.append(AMBER)
        else:
            colours.append(PINK)

    fig = go.Figure(go.Bar(
        x=ratings_vals,
        y=teams,
        orientation="h",
        marker=dict(
            color=colours,
            line=dict(color=GRID, width=0.5),
        ),
        text=[f"{r:.0f}" for r in ratings_vals],
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="<b>%{y}</b><br>ELO: %{x:.0f}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="⚡ NEURAL ELO POWER GRID", font=dict(color=CYAN, size=14, family="Orbitron"), x=0.5),
        height=max(400, top_n * 18),
        xaxis=dict(
            title="ELO Rating",
            range=[min(ratings_vals) - 50, max(ratings_vals) + 80],
            gridcolor=GRID,
            color=TEXT,
        ),
        yaxis=dict(
            tickfont=dict(size=10, color=TEXT),
            gridcolor=GRID,
        ),
    )
    return fig


def feature_importance_chart(feature_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importances."""
    # Show top 15 features
    top = feature_df.head(15).copy()
    top = top.sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker=dict(
            color=CYAN,
            opacity=0.85,
            line=dict(color=CYAN, width=0.5),
        ),
        text=[f"{v:.3f}" for v in top["importance"]],
        textposition="outside",
        textfont=dict(color=CYAN, size=10),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="⚙ SIGNAL WEIGHTS (XGBOOST GAIN)", font=dict(color=CYAN, size=14, family="Orbitron"), x=0.5),
        height=420,
        xaxis=dict(title="Feature Importance (Gain)", gridcolor=GRID, color=TEXT),
        yaxis=dict(tickfont=dict(size=10, color=TEXT)),
    )
    return fig


def pnl_chart(bets: list[dict]) -> go.Figure:
    """Cumulative P&L line chart over time.

    `bets` is a list of dicts from cache.get_all_bets() with 'pnl' set.
    """
    settled = [b for b in bets if b.get("pnl") is not None]

    if not settled:
        # Return empty chart with instruction
        fig = go.Figure()
        fig.add_annotation(
            text="NO SETTLED BETS YET<br><span style='font-size:12px'>Place bets in the PROFIT MATRIX tab</span>",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=CYAN, family="Orbitron"),
        )
        fig.update_layout(**_LAYOUT_BASE, height=300)
        return fig

    cumulative_pnl = 0.0
    dates, values, labels = [], [], []

    for b in sorted(settled, key=lambda x: x.get("placed_at", "")):
        cumulative_pnl += b["pnl"]
        dates.append(b.get("placed_at", "")[:10])
        values.append(round(cumulative_pnl, 2))
        outcome_icon = "✓" if b["pnl"] > 0 else "✗"
        labels.append(f"{outcome_icon} {b['outcome_label']} @ {b['decimal_odds']}")

    # Color the line based on final P&L
    line_color = GREEN if cumulative_pnl >= 0 else RED

    fig = go.Figure()

    # Zero line (break-even reference)
    fig.add_hline(y=0, line_dash="dash", line_color=GRID, line_width=1)

    # Shaded area under curve
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        fill="tozeroy",
        fillcolor=f"rgba{tuple(int(line_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}",
        line=dict(color=line_color, width=2),
        mode="lines+markers",
        marker=dict(color=line_color, size=6, line=dict(color=BG, width=1)),
        text=labels,
        hovertemplate="<b>%{text}</b><br>Cumulative P&L: %{y:.2f}<extra></extra>",
    ))

    final_pnl = values[-1] if values else 0
    sign = "+" if final_pnl >= 0 else ""

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f"⚡ CUMULATIVE P&L: {sign}{final_pnl:.2f}",
            font=dict(color=line_color, size=14, family="Orbitron"),
            x=0.5,
        ),
        height=300,
        xaxis=dict(title="Date", gridcolor=GRID, color=TEXT),
        yaxis=dict(title="P&L (£)", gridcolor=GRID, color=TEXT),
    )
    return fig


def ev_scatter_chart(recommendations: list) -> go.Figure:
    """Scatter plot: EV% vs Model Probability, sized by recommended stake.

    `recommendations` is a list of BetRecommendation dataclass instances.
    """
    if not recommendations:
        fig = go.Figure()
        fig.add_annotation(
            text="NO VALUE BETS DETECTED",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=AMBER, family="Orbitron"),
        )
        fig.update_layout(**_LAYOUT_BASE, height=300)
        return fig

    x_vals     = [r.model_prob * 100 for r in recommendations]
    y_vals     = [r.ev * 100 for r in recommendations]
    sizes      = [max(10, r.recommended_stake * 500) for r in recommendations]
    labels     = [f"{r.outcome_label}<br>{r.home_team} vs {r.away_team}" for r in recommendations]
    colours    = [GREEN if r.ev >= 0.05 else AMBER for r in recommendations]

    fig = go.Figure(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colours,
            opacity=0.85,
            line=dict(color=BG, width=1),
        ),
        text=[r.outcome_label for r in recommendations],
        textposition="top center",
        textfont=dict(size=9, color=TEXT),
        hovertemplate="<b>%{text}</b><br>Prob: %{x:.1f}%<br>EV: %{y:.1f}%<extra></extra>",
    ))

    # Break-even line (EV=0)
    fig.add_hline(y=0, line_dash="dash", line_color=RED, line_width=1,
                  annotation_text="BREAK EVEN", annotation_font_color=RED)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="⊕ VALUE BET SIGNAL MAP", font=dict(color=CYAN, size=14, family="Orbitron"), x=0.5),
        height=350,
        xaxis=dict(title="Model Probability (%)", gridcolor=GRID, color=TEXT),
        yaxis=dict(title="Expected Value (%)", gridcolor=GRID, color=TEXT),
    )
    return fig


def score_matrix_heatmap(
    matrix: list[list[float]],
    home_team: str,
    away_team: str,
    max_goals: int = 6,
) -> go.Figure:
    """Heatmap of the score probability matrix from Dixon-Coles.

    Shows P(home_goals=i, away_goals=j) for i,j in 0..max_goals.
    """
    import numpy as np
    m = matrix[:max_goals+1, :max_goals+1]

    fig = go.Figure(go.Heatmap(
        z=m,
        x=[str(j) for j in range(max_goals + 1)],
        y=[str(i) for i in range(max_goals + 1)],
        colorscale=[
            [0.0, BG],
            [0.3, "#003355"],
            [0.6, "#0088AA"],
            [1.0, CYAN],
        ],
        text=[[f"{m[i,j]*100:.1f}%" for j in range(m.shape[1])] for i in range(m.shape[0])],
        texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT),
        showscale=False,
        hovertemplate=f"{home_team} %{{y}} – %{{x}} {away_team}: %{{z:.3f}}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f"◈ SCORE PROBABILITY MATRIX: {home_team} vs {away_team}",
            font=dict(color=CYAN, size=13, family="Orbitron"),
            x=0.5,
        ),
        height=380,
        xaxis=dict(title=f"{away_team} Goals", color=TEXT),
        yaxis=dict(title=f"{home_team} Goals", color=TEXT, autorange="reversed"),
    )
    return fig
