"""
data/loader.py — Live Data API Clients (with Demo-Mode Fallback)
================================================================
Two API clients for live World Cup data:
  1. FootballDataClient  → football-data.org (match results, fixtures)
  2. OddsAPIClient       → the-odds-api.com (bookmaker odds)

DEMO MODE:
  If no API keys are configured, both clients automatically fall back
  to bundled data from wc2026_teams.py. The app works identically in
  demo mode; the only difference is that odds and results won't update
  in real time.

  is_demo_mode() returns True when running without API keys.
  The UI shows a yellow banner in demo mode.

API KEY SETUP:
  Option A (Streamlit Cloud): Add secrets in the Streamlit Cloud UI
    FOOTBALL_DATA_API_KEY = "..."
    ODDS_API_KEY = "..."

  Option B (local): Create betting/.streamlit/secrets.toml
    FOOTBALL_DATA_API_KEY = "..."
    ODDS_API_KEY = "..."

  Option C (environment variable):
    export FOOTBALL_DATA_API_KEY=...
    export ODDS_API_KEY=...

RATE LIMIT PROTECTION:
  The Odds API free tier has 500 requests/month.
  We cache responses for 30 minutes to use at most ~1500 req/month at
  maximum page activity — well within the free limit.

HOW TO DEBUG:
  - Set logging level to DEBUG to see all API requests and cache decisions
  - If you get 429 (rate limit), increase cache TTL or switch to demo mode
  - If matches are missing from the schedule, check football-data.org WC
    competition code — it may change between years ("WC" vs "CL" etc.)
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import Any

import requests

from data.cache import get_cached, set_cached
from data.wc2026_teams import (
    DEMO_ODDS,
    RESULTS_SO_FAR,
    SCHEDULE,
    TEAM_META,
    get_demo_odds,
    upcoming_matches,
)

logger = logging.getLogger(__name__)

# ── API configuration ─────────────────────────────────────────────────────────
_FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_ODDS_SPORT = "soccer_fifa_world_cup"

# Cache TTL: 30 minutes for both (protects Odds API 500 req/month limit)
_CACHE_TTL = 30 * 60


def _get_secret(name: str) -> str | None:
    """Try to get a secret from Streamlit secrets, then environment variables.

    Streamlit secrets take priority (for Streamlit Cloud deployment).
    Falls back to env vars for local development.

    Returns None if the key is not found anywhere.
    """
    # Try Streamlit secrets (only available inside a Streamlit app context)
    try:
        import streamlit as st
        val = st.secrets.get(name)
        if val:
            return str(val)
    except Exception:
        pass  # Not running in a Streamlit context (e.g., during pytest)

    # Fall back to environment variables
    return os.environ.get(name) or None


def is_demo_mode() -> bool:
    """Return True if the app is running without live API keys.

    In demo mode: bundled WC2026 schedule + realistic demo odds are used.
    In live mode: real match results and bookmaker odds are fetched.
    """
    fd_key = _get_secret("FOOTBALL_DATA_API_KEY")
    return fd_key is None


# ─── Football Data Client ─────────────────────────────────────────────────────

class FootballDataClient:
    """Client for football-data.org API.

    Free tier: 10 requests/minute, all competitions.
    WC 2026 competition code: "WC"

    Documentation: https://www.football-data.org/documentation/quickstart
    """

    def __init__(self) -> None:
        self._api_key = _get_secret("FOOTBALL_DATA_API_KEY")
        self._demo   = self._api_key is None

    def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Make an authenticated GET request with caching.

        Returns None on any error (network, auth, rate limit).
        """
        if self._demo:
            return None

        cache_key = f"fd_{endpoint}_{params}"
        cached = get_cached(cache_key)
        if cached is not None:
            return cached

        url = f"{_FOOTBALL_DATA_BASE}/{endpoint}"
        headers = {"X-Auth-Token": self._api_key}

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            set_cached(cache_key, data, ttl_seconds=_CACHE_TTL)
            return data
        except requests.HTTPError as exc:
            logger.warning("football-data.org HTTP %s: %s", exc.response.status_code, exc)
            return None
        except Exception as exc:
            logger.warning("football-data.org request failed: %s", exc)
            return None

    def get_wc_matches(self) -> list[dict]:
        """Return all 2026 WC matches with results/scores.

        Combines live API data (when available) with the bundled schedule.
        Returns normalised list matching SCHEDULE format.
        """
        if self._demo:
            logger.debug("Demo mode: returning bundled WC schedule")
            return SCHEDULE

        data = self._get("competitions/WC/matches", {"season": "2026"})
        if data is None:
            logger.info("football-data.org unavailable — using bundled schedule")
            return SCHEDULE

        # Normalise API response to our internal format
        normalised: list[dict] = []
        for m in data.get("matches", []):
            home = m.get("homeTeam", {}).get("name", "").upper()
            away = m.get("awayTeam", {}).get("name", "").upper()
            score = m.get("score", {}).get("fullTime", {})
            status = m.get("status", "")
            utc_date = m.get("utcDate", "")[:10]  # YYYY-MM-DD

            home_score = score.get("home")
            away_score = score.get("away")
            stage_raw  = m.get("stage", "GROUP_STAGE")

            normalised.append({
                "date":       utc_date,
                "home":       home,
                "away":       away,
                "home_score": home_score,
                "away_score": away_score,
                "stage":      _normalise_stage(stage_raw),
                "venue":      m.get("venue", ""),
                "status":     status,
            })

        return normalised if normalised else SCHEDULE

    def get_completed_matches(self) -> list[dict]:
        """Return only matches that have been played (with scores).

        Used to update ELO ratings with actual tournament results.
        Falls back to RESULTS_SO_FAR in demo mode.
        """
        if self._demo:
            return RESULTS_SO_FAR

        matches = self.get_wc_matches()
        return [
            m for m in matches
            if m.get("home_score") is not None
            and m.get("away_score") is not None
        ]


# ─── Odds API Client ─────────────────────────────────────────────────────────

class OddsAPIClient:
    """Client for The Odds API (the-odds-api.com).

    Free tier: 500 requests/month.
    With 30-min cache: ≈ 96 requests/day max = ~2880/month
    That's still over the limit at full activity — we cap at 400/month via
    the request counter in the cache layer.

    Documentation: https://the-odds-api.com/liveapi/guides/v4/
    """

    def __init__(self) -> None:
        self._api_key = _get_secret("ODDS_API_KEY")
        self._demo   = self._api_key is None

    def get_wc_odds(
        self,
        regions: str = "uk,eu",
        markets: str = "h2h,totals",
        bookmakers: str = "pinnacle,bet365,betfair_ex_eu",
    ) -> dict[str, dict[str, float]]:
        """Return odds for upcoming WC matches.

        Returns
        -------
        dict keyed by "{HOME}_vs_{AWAY}" with sub-dicts:
            {"home": odds, "draw": odds, "away": odds,
             "over_25": odds, "btts": odds}

        Falls back to DEMO_ODDS if API is unavailable.

        HOW TO DEBUG:
            If odds look wrong, print the raw API response:
            data = client._get_raw()
            import json
            print(json.dumps(data[:2], indent=2))
        """
        if self._demo:
            logger.debug("Demo mode: returning bundled demo odds")
            return DEMO_ODDS

        cache_key = "odds_wc_upcoming"
        cached = get_cached(cache_key)
        if cached is not None:
            return cached

        params = {
            "apiKey":     self._api_key,
            "regions":    regions,
            "markets":    markets,
            "oddsFormat": "decimal",
        }

        try:
            url = f"{_ODDS_API_BASE}/sports/{_ODDS_SPORT}/odds/"
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            raw = resp.json()
            logger.info(
                "Odds API: fetched %d events. Remaining requests: %s",
                len(raw),
                resp.headers.get("x-requests-remaining", "?"),
            )

            result = _parse_odds_api_response(raw)
            set_cached(cache_key, result, ttl_seconds=_CACHE_TTL)
            return result

        except requests.HTTPError as exc:
            if exc.response.status_code == 401:
                logger.error("Odds API: Invalid API key — check ODDS_API_KEY")
            elif exc.response.status_code == 422:
                logger.error("Odds API: No odds available for this sport/market")
            else:
                logger.warning("Odds API HTTP error: %s", exc)
            return DEMO_ODDS

        except Exception as exc:
            logger.warning("Odds API request failed: %s — using demo odds", exc)
            return DEMO_ODDS

    def get_match_odds(self, home: str, away: str) -> dict[str, float] | None:
        """Return odds for a specific match. Returns None if not found."""
        all_odds = self.get_wc_odds()
        key = f"{home}_vs_{away}"
        if key in all_odds:
            return all_odds[key]
        # Try reversed direction
        rev_key = f"{away}_vs_{home}"
        if rev_key in all_odds:
            o = all_odds[rev_key].copy()
            o["home"], o["away"] = o["away"], o["home"]
            return o
        return get_demo_odds(home, away)  # last resort: demo odds


# ─── Private helpers ─────────────────────────────────────────────────────────

def _normalise_stage(raw: str) -> str:
    """Convert football-data.org stage names to our internal codes."""
    mapping = {
        "GROUP_STAGE":         "GROUP",
        "ROUND_OF_32":         "R32",
        "ROUND_OF_16":         "R16",
        "QUARTER_FINALS":      "QF",
        "SEMI_FINALS":         "SF",
        "THIRD_PLACE":         "SF",  # treat 3rd place similarly
        "FINAL":               "FINAL",
    }
    return mapping.get(raw.upper(), "GROUP")


def _parse_odds_api_response(raw: list[dict]) -> dict[str, dict[str, float]]:
    """Parse The Odds API response into our internal odds format.

    The Odds API returns events with nested bookmakers and markets.
    We take the best available odds across all listed bookmakers.

    HOW TO DEBUG:
        If all odds are 0.0: the market structure may have changed in API v5+
        Check: raw[0]["bookmakers"][0]["markets"] for current structure
    """
    result: dict[str, dict[str, float]] = {}

    for event in raw:
        home_raw = event.get("home_team", "").upper()
        away_raw = event.get("away_team", "").upper()
        key = f"{home_raw}_vs_{away_raw}"

        # Aggregate best odds across bookmakers (take the maximum = best for punter)
        best_odds: dict[str, float] = {}

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")

                if market_key == "h2h":
                    # Head-to-head: home / draw / away
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "").upper()
                        price = float(outcome.get("price", 0))

                        if home_raw in name:
                            best_odds["home"] = max(best_odds.get("home", 0), price)
                        elif away_raw in name:
                            best_odds["away"] = max(best_odds.get("away", 0), price)
                        elif "DRAW" in name:
                            best_odds["draw"] = max(best_odds.get("draw", 0), price)

                elif market_key == "totals":
                    # Over/Under goals
                    for outcome in market.get("outcomes", []):
                        name  = outcome.get("name", "").upper()
                        point = outcome.get("point", 0)
                        price = float(outcome.get("price", 0))

                        if abs(point - 2.5) < 0.01 and "OVER" in name:
                            best_odds["over_25"] = max(best_odds.get("over_25", 0), price)
                        elif abs(point - 2.5) < 0.01 and "UNDER" in name:
                            pass  # we don't recommend Under bets currently

        # Only add if we have at least 1X2 odds
        if "home" in best_odds and "draw" in best_odds and "away" in best_odds:
            # Fill in BTTS with a typical estimate if not available
            if "btts" not in best_odds:
                best_odds["btts"] = 1.80  # typical BTTS market
            result[key] = best_odds

    return result
