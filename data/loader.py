"""
data/loader.py — Live Data Clients
===================================
Three data sources, in priority order:

  1. ESPNClient (PRIMARY — no API key, always works)
     → Live scores, group standings, full fixture list
     → ESPN's public endpoints require zero registration
     → Used for all match data by default

  2. FootballDataClient (SECONDARY — optional football-data.org key)
     → Adds group labels to fixtures (GROUP_A, GROUP_B, etc.)
     → Useful if ESPN group inference fails
     → 10 requests/minute free tier

  3. OddsAPIClient (OPTIONAL — the-odds-api.com key)
     → Live bookmaker odds
     → 500 requests/month free tier
     → Falls back to ELO-based DEMO_ODDS if no key

DEMO MODE:
  is_demo_mode() now returns False — ESPN always provides live data.
  The only "demo" element is odds when no ODDS_API_KEY is set.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import requests

from data.cache import get_cached, set_cached
from data.wc2026_teams import (
    DEMO_ODDS,
    GROUPS,
    RESULTS_SO_FAR,
    SCHEDULE,
    TEAM_META,
    get_demo_odds,
    upcoming_matches,
)

logger = logging.getLogger(__name__)

_ODDS_API_BASE    = "https://api.the-odds-api.com/v4"
_ODDS_SPORT       = "soccer_fifa_world_cup"
_FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

_CACHE_TTL        = 30 * 60   # 30-minute cache for all live calls
_ESPN_SCOREBOARD  = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
_ESPN_STANDINGS   = "https://site.api.espn.com/apis/v2/sports/soccer/fifa.world/standings"

# WC 2026 date range for fetching the full group stage schedule
_WC_START = "20260611"
_WC_END   = "20260703"


def _get_secret(name: str) -> str | None:
    """Return secret from Streamlit secrets or environment variable."""
    try:
        import streamlit as st
        val = st.secrets.get(name)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(name) or None


def is_demo_mode() -> bool:
    """Always False — ESPN provides live data with no key required.
    Kept for backwards compatibility; UI no longer shows a demo banner.
    """
    return False


# ─── Team name normalisation ──────────────────────────────────────────────────

_TEAM_NAME_MAP: dict[str, str] = {
    # ESPN / football-data.org → internal names
    "UNITED STATES":                "USA",
    "KOREA REPUBLIC":               "SOUTH KOREA",
    "REPUBLIC OF KOREA":            "SOUTH KOREA",
    "IR IRAN":                      "IRAN",
    "ISLAMIC REPUBLIC OF IRAN":     "IRAN",
    "CÔTE D'IVOIRE":                "IVORY COAST",
    "COTE D'IVOIRE":                "IVORY COAST",
    "CURAÇAO":                      "CURACAO",
    "CURACAO":                      "CURACAO",
    "CAPE VERDE":                   "CAPE VERDE ISLANDS",
    "CAPE VERDE ISLANDS":           "CAPE VERDE ISLANDS",
    "CONGO DR":                     "CONGO DR",
    "DR CONGO":                     "CONGO DR",
    "DEMOCRATIC REPUBLIC OF CONGO": "CONGO DR",
    "BOSNIA-HERZEGOVINA":           "BOSNIA-HERZEGOVINA",
    "BOSNIA AND HERZEGOVINA":       "BOSNIA-HERZEGOVINA",
    "TÜRKIYE":                      "TURKEY",
    "TURKEY":                       "TURKEY",
    "NORTH MACEDONIA":              "NORTH MACEDONIA",
    "TRINIDAD AND TOBAGO":          "TRINIDAD & TOBAGO",
    "REPUBLIC OF IRELAND":          "IRELAND",
}


def _normalize_team(name: str) -> str:
    upper = name.upper().strip()
    return _TEAM_NAME_MAP.get(upper, upper)


def _normalize_group(raw: str) -> str:
    """'GROUP_A' or 'Group A' → 'A'. Returns '' for non-group stages."""
    if not raw:
        return ""
    upper = raw.upper().replace("GROUP_", "").replace("GROUP ", "").strip()
    return upper if len(upper) == 1 and upper.isalpha() else ""


def _normalise_stage(raw: str) -> str:
    mapping = {
        "GROUP_STAGE":    "GROUP",
        "ROUND_OF_32":    "R32",
        "ROUND_OF_16":    "R16",
        "QUARTER_FINALS": "QF",
        "SEMI_FINALS":    "SF",
        "THIRD_PLACE":    "SF",
        "FINAL":          "FINAL",
    }
    return mapping.get(raw.upper(), "GROUP")


def _team_to_group(team: str, groups: dict[str, list[str]]) -> str:
    """Reverse-lookup group letter for a team name."""
    for letter, teams in groups.items():
        if team in teams:
            return letter
    return ""


# ─── ESPN Client (PRIMARY — no API key) ───────────────────────────────────────

class ESPNClient:
    """Client for ESPN's public (undocumented but stable) football endpoints.

    No API key or registration required. Rate-limits are generous for a
    single-app use case; results are cached for 30 minutes.

    Endpoints used:
      Scoreboard: site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard
      Standings:  site.api.espn.com/apis/v2/sports/soccer/fifa.world/standings
    """

    def _get(self, url: str, params: dict | None = None, cache_key: str | None = None) -> dict | None:
        key = cache_key or f"espn_{url}_{params}"
        cached = get_cached(key)
        if cached is not None:
            return cached
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            set_cached(key, data, ttl_seconds=_CACHE_TTL)
            return data
        except Exception as exc:
            logger.warning("ESPN request failed (%s): %s", url, exc)
            return None

    def get_wc_standings(self) -> dict[str, list[dict]] | None:
        """Return live group standings.

        Returns dict keyed by group letter ("A"…"L"), each value a list
        of team dicts sorted by standing position:
          {"team": "ARGENTINA", "pts": 3, "gp": 1, "gd": 3, "gf": 3, "ga": 0}

        Returns None on failure (fall back to RESULTS_SO_FAR).
        """
        data = self._get(_ESPN_STANDINGS, cache_key="espn_wc_standings")
        if not data:
            return None

        result: dict[str, list[dict]] = {}
        for group in data.get("children", []):
            raw_name = group.get("name", "")  # e.g. "Group A"
            letter = raw_name.replace("Group ", "").strip()
            if not letter or len(letter) != 1:
                continue

            entries = []
            for e in group.get("standings", {}).get("entries", []):
                team_name = _normalize_team(e.get("team", {}).get("displayName", ""))
                stats = {
                    s["name"]: s.get("displayValue", s.get("value", 0))
                    for s in e.get("stats", [])
                }
                entries.append({
                    "team": team_name,
                    "pts":  int(float(stats.get("points", 0))),
                    "gp":   int(float(stats.get("gamesPlayed", 0))),
                    "gf":   int(float(stats.get("pointsFor", 0))),
                    "ga":   int(float(stats.get("pointsAgainst", 0))),
                    "gd":   int(float(stats.get("pointDifferential", 0))),
                    "w":    int(float(stats.get("wins", 0))),
                    "d":    int(float(stats.get("ties", 0))),
                    "l":    int(float(stats.get("losses", 0))),
                })
            result[letter] = entries

        logger.info("ESPN standings: %d groups loaded", len(result))
        return result if result else None

    def get_live_groups(self) -> dict[str, list[str]] | None:
        """Return group assignments derived from live standings."""
        standings = self.get_wc_standings()
        if not standings:
            return None
        return {letter: [e["team"] for e in entries] for letter, entries in standings.items()}

    def get_wc_matches(self) -> list[dict]:
        """Return all WC 2026 group stage fixtures with live scores.

        Fetches the full date range at once and normalises to our internal
        format. Results are cached for 30 minutes.
        """
        data = self._get(
            _ESPN_SCOREBOARD,
            params={"dates": f"{_WC_START}-{_WC_END}", "limit": 200},
            cache_key="espn_wc_matches_full",
        )
        if not data:
            logger.warning("ESPN scoreboard unavailable — using bundled schedule")
            return SCHEDULE

        # Build a reverse-lookup from live standings so we can tag matches with group
        live_groups = self.get_live_groups() or GROUPS

        normalised: list[dict] = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            status_type = comp.get("status", {}).get("type", {})

            home_data = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away_data = next((c for c in competitors if c.get("homeAway") == "away"), {})

            home = _normalize_team(home_data.get("team", {}).get("displayName", ""))
            away = _normalize_team(away_data.get("team", {}).get("displayName", ""))
            if not home or not away:
                continue

            completed = status_type.get("completed", False)
            home_score = int(home_data.get("score", 0)) if completed else None
            away_score = int(away_data.get("score", 0)) if completed else None

            utc_date = event.get("date", "")[:10]
            venue = comp.get("venue", {}).get("fullName", "")

            group = _team_to_group(home, live_groups) or _team_to_group(away, live_groups)

            normalised.append({
                "date":       utc_date,
                "home":       home,
                "away":       away,
                "home_score": home_score,
                "away_score": away_score,
                "stage":      "GROUP",
                "group":      group,
                "venue":      venue,
                "status":     status_type.get("name", ""),
            })

        logger.info("ESPN: fetched %d WC fixtures", len(normalised))
        return normalised if normalised else SCHEDULE

    def get_completed_matches(self) -> list[dict]:
        """Return only finished matches with scores."""
        matches = self.get_wc_matches()
        return [m for m in matches if m.get("home_score") is not None]


# ─── Football Data Client (SECONDARY — optional enrichment) ──────────────────

class FootballDataClient:
    """Client for football-data.org API (optional — adds group labels).

    If FOOTBALL_DATA_API_KEY is not set, this client is a no-op and
    ESPNClient handles all match data.
    """

    def __init__(self) -> None:
        self._api_key = _get_secret("FOOTBALL_DATA_API_KEY")
        self._demo    = self._api_key is None

    def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        if self._demo:
            return None
        cache_key = f"fd_{endpoint}_{params}"
        cached = get_cached(cache_key)
        if cached is not None:
            return cached
        url = f"{_FOOTBALL_DATA_BASE}/{endpoint}"
        try:
            resp = requests.get(url, headers={"X-Auth-Token": self._api_key}, params=params, timeout=10)
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
        if self._demo:
            return SCHEDULE
        data = self._get("competitions/WC/matches", {"season": "2026"})
        if data is None:
            return SCHEDULE
        normalised: list[dict] = []
        for m in data.get("matches", []):
            home = _normalize_team(m.get("homeTeam", {}).get("name", ""))
            away = _normalize_team(m.get("awayTeam", {}).get("name", ""))
            score = m.get("score", {}).get("fullTime", {})
            normalised.append({
                "date":       m.get("utcDate", "")[:10],
                "home":       home,
                "away":       away,
                "home_score": score.get("home"),
                "away_score": score.get("away"),
                "stage":      _normalise_stage(m.get("stage", "GROUP_STAGE")),
                "group":      _normalize_group(m.get("group", "")),
                "venue":      m.get("venue", ""),
                "status":     m.get("status", ""),
            })
        return normalised if normalised else SCHEDULE

    def get_completed_matches(self) -> list[dict]:
        if self._demo:
            return RESULTS_SO_FAR
        matches = self.get_wc_matches()
        return [m for m in matches if m.get("home_score") is not None]

    def get_live_groups(self) -> dict[str, list[str]] | None:
        if self._demo:
            return None
        matches = self.get_wc_matches()
        groups: dict[str, set[str]] = {}
        for m in matches:
            g = m.get("group", "")
            if not g or m.get("stage") != "GROUP":
                continue
            groups.setdefault(g, set()).add(m["home"])
            groups.setdefault(g, set()).add(m["away"])
        if not groups:
            return None
        return {k: sorted(v) for k, v in sorted(groups.items())}


# ─── Odds API Client ──────────────────────────────────────────────────────────

class OddsAPIClient:
    """Client for The Odds API (the-odds-api.com). 500 requests/month free."""

    def __init__(self) -> None:
        self._api_key = _get_secret("ODDS_API_KEY")
        self._demo    = self._api_key is None

    def get_wc_odds(
        self,
        regions: str = "uk,eu",
        markets: str = "h2h,totals",
    ) -> dict[str, dict[str, float]]:
        """Return bookmaker odds. Falls back to DEMO_ODDS if no key."""
        if self._demo:
            return DEMO_ODDS

        cache_key = "odds_wc_upcoming"
        cached = get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            resp = requests.get(
                f"{_ODDS_API_BASE}/sports/{_ODDS_SPORT}/odds/",
                params={"apiKey": self._api_key, "regions": regions,
                        "markets": markets, "oddsFormat": "decimal"},
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()
            logger.info("Odds API: %d events, remaining: %s",
                        len(raw), resp.headers.get("x-requests-remaining", "?"))
            result = _parse_odds_api_response(raw)
            set_cached(cache_key, result, ttl_seconds=_CACHE_TTL)
            return result
        except requests.HTTPError as exc:
            logger.warning("Odds API HTTP error: %s", exc)
            return DEMO_ODDS
        except Exception as exc:
            logger.warning("Odds API failed: %s", exc)
            return DEMO_ODDS

    def get_match_odds(self, home: str, away: str) -> dict[str, float] | None:
        all_odds = self.get_wc_odds()
        key = f"{home}_vs_{away}"
        if key in all_odds:
            return all_odds[key]
        rev = f"{away}_vs_{home}"
        if rev in all_odds:
            o = all_odds[rev].copy()
            o["home"], o["away"] = o["away"], o["home"]
            return o
        return get_demo_odds(home, away)


# ─── Private helpers ──────────────────────────────────────────────────────────

def _parse_odds_api_response(raw: list[dict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for event in raw:
        home_raw = _normalize_team(event.get("home_team", ""))
        away_raw = _normalize_team(event.get("away_team", ""))
        key = f"{home_raw}_vs_{away_raw}"
        best: dict[str, float] = {}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                mk = market.get("key", "")
                if mk == "h2h":
                    for outcome in market.get("outcomes", []):
                        n = _normalize_team(outcome.get("name", ""))
                        p = float(outcome.get("price", 0))
                        if n == home_raw:
                            best["home"] = max(best.get("home", 0), p)
                        elif n == away_raw:
                            best["away"] = max(best.get("away", 0), p)
                        elif "DRAW" in outcome.get("name", "").upper():
                            best["draw"] = max(best.get("draw", 0), p)
                elif mk == "totals":
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point", 0)
                        if abs(point - 2.5) < 0.01 and "OVER" in outcome.get("name", "").upper():
                            best["over_25"] = max(best.get("over_25", 0), float(outcome.get("price", 0)))
        if "home" in best and "draw" in best and "away" in best:
            best.setdefault("btts", 1.80)
            result[key] = best
    return result
