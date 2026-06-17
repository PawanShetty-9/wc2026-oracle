"""
data/wc2026_teams.py — 2026 FIFA World Cup Tournament Data (OFFLINE)
====================================================================
This module is the foundation of DEMO MODE. All data is hardcoded so
the app runs fully offline without any API keys. It encodes:

  - 48 teams across 12 groups (A–L)
  - Full group-stage schedule (72 matches)
  - Match results so far (updated through June 17, 2026 — Day 7)
  - Team metadata: FIFA ranking, pre-tournament ELO, WC appearances
  - Realistic demo odds for upcoming matches (derived from ELO gap)

2026 FORMAT:
  - 48 teams, 12 groups of 4, top 2 advance + 8 best 3rd-place teams
  - Group stage: June 11–July 3, 2026
  - Round of 32: July 4–7  |  Round of 16: July 10–12
  - Quarterfinals: July 14–15  |  Semifinals: July 17–18
  - Third place: July 21  |  Final: July 19, 2026
  - All World Cup matches treated as NEUTRAL venue (no home advantage)

HOW TO UPDATE THIS FILE:
  After each match day, add results to RESULTS_SO_FAR:
    {"date": "2026-06-18", "home": "ENGLAND", "away": "USA",
     "home_score": 1, "away_score": 1, "stage": "GROUP"}

  The ELO model auto-recalculates from this list on each app startup.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: GROUP ASSIGNMENTS
# 12 groups × 4 teams = 48 teams
# Seeding: Pot 1 = hosts + ranked nations; geographic/confederation spread
# ─────────────────────────────────────────────────────────────────────────────
GROUPS: dict[str, list[str]] = {
    "A": ["ENGLAND",     "USA",         "CAMEROON",    "IRAN"],
    "B": ["GERMANY",     "CANADA",      "SOUTH KOREA", "NIGERIA"],
    "C": ["SPAIN",       "MEXICO",      "AUSTRALIA",   "IVORY COAST"],
    "D": ["FRANCE",      "ARGENTINA",   "SAUDI ARABIA","NEW ZEALAND"],
    "E": ["BRAZIL",      "CROATIA",     "JAPAN",       "MOROCCO"],
    "F": ["PORTUGAL",    "COLOMBIA",    "SENEGAL",     "ECUADOR"],
    "G": ["NETHERLANDS", "URUGUAY",     "EGYPT",       "PANAMA"],
    "H": ["BELGIUM",     "CHILE",       "IRAN",        "COSTA RICA"],  # Iran removed from A
    "I": ["ITALY",       "SWITZERLAND", "ALGERIA",     "COSTA RICA"],
    "J": ["DENMARK",     "ROMANIA",     "MALI",        "VENEZUELA"],
    "K": ["AUSTRIA",     "HUNGARY",     "JORDAN",      "HONDURAS"],
    "L": ["SERBIA",      "POLAND",      "PARAGUAY",    "UZBEKISTAN"],
}

# Fix: Iran appeared in both A and H — corrected below with deduplicated groups
GROUPS = {
    "A": ["ENGLAND",     "USA",          "CAMEROON",     "IRAN"],
    "B": ["GERMANY",     "CANADA",       "SOUTH KOREA",  "NIGERIA"],
    "C": ["SPAIN",       "MEXICO",       "AUSTRALIA",    "IVORY COAST"],
    "D": ["FRANCE",      "ARGENTINA",    "SAUDI ARABIA", "NEW ZEALAND"],
    "E": ["BRAZIL",      "CROATIA",      "JAPAN",        "MOROCCO"],
    "F": ["PORTUGAL",    "COLOMBIA",     "SENEGAL",      "ECUADOR"],
    "G": ["NETHERLANDS", "URUGUAY",      "EGYPT",        "PANAMA"],
    "H": ["BELGIUM",     "SWITZERLAND",  "ALGERIA",      "COSTA RICA"],
    "I": ["ITALY",       "DENMARK",      "SOUTH AFRICA", "JAMAICA"],
    "J": ["NETHERLANDS", "ROMANIA",      "MALI",         "VENEZUELA"],
    "K": ["AUSTRIA",     "HUNGARY",      "JORDAN",       "HONDURAS"],
    "L": ["SERBIA",      "POLAND",       "PARAGUAY",     "UZBEKISTAN"],
}

# Final corrected groups (no duplicates, all 48 unique teams):
GROUPS = {
    "A": ["ENGLAND",      "USA",          "CAMEROON",     "IRAN"],
    "B": ["GERMANY",      "CANADA",       "SOUTH KOREA",  "NIGERIA"],
    "C": ["SPAIN",        "MEXICO",       "AUSTRALIA",    "IVORY COAST"],
    "D": ["FRANCE",       "ARGENTINA",    "SAUDI ARABIA", "NEW ZEALAND"],
    "E": ["BRAZIL",       "CROATIA",      "JAPAN",        "MOROCCO"],
    "F": ["PORTUGAL",     "COLOMBIA",     "SENEGAL",      "ECUADOR"],
    "G": ["NETHERLANDS",  "URUGUAY",      "EGYPT",        "PANAMA"],
    "H": ["BELGIUM",      "SWITZERLAND",  "ALGERIA",      "COSTA RICA"],
    "I": ["ITALY",        "DENMARK",      "SOUTH AFRICA", "JAMAICA"],
    "J": ["CHILE",        "ROMANIA",      "MALI",         "VENEZUELA"],
    "K": ["AUSTRIA",      "HUNGARY",      "JORDAN",       "HONDURAS"],
    "L": ["SERBIA",       "POLAND",       "PARAGUAY",     "UZBEKISTAN"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TEAM METADATA
# fifa_rank : FIFA ranking as of May 2026 (lower = better)
# elo       : Pre-tournament ELO rating (World Football Elo scale)
# wc_apps   : Number of previous World Cup appearances (not counting 2026)
# region    : Confederation for display purposes
# ─────────────────────────────────────────────────────────────────────────────
TEAM_META: dict[str, dict] = {
    # ── Group A ──────────────────────────────────────────────────────────────
    "ENGLAND":      {"fifa_rank": 4,  "elo": 2025, "wc_apps": 16, "region": "UEFA"},
    "USA":          {"fifa_rank": 11, "elo": 1858, "wc_apps": 11, "region": "CONCACAF"},
    "CAMEROON":     {"fifa_rank": 43, "elo": 1673, "wc_apps": 8,  "region": "CAF"},
    "IRAN":         {"fifa_rank": 22, "elo": 1790, "wc_apps": 6,  "region": "AFC"},
    # ── Group B ──────────────────────────────────────────────────────────────
    "GERMANY":      {"fifa_rank": 3,  "elo": 2030, "wc_apps": 19, "region": "UEFA"},
    "CANADA":       {"fifa_rank": 41, "elo": 1710, "wc_apps": 2,  "region": "CONCACAF"},
    "SOUTH KOREA":  {"fifa_rank": 23, "elo": 1788, "wc_apps": 11, "region": "AFC"},
    "NIGERIA":      {"fifa_rank": 39, "elo": 1683, "wc_apps": 7,  "region": "CAF"},
    # ── Group C ──────────────────────────────────────────────────────────────
    "SPAIN":        {"fifa_rank": 2,  "elo": 2080, "wc_apps": 15, "region": "UEFA"},
    "MEXICO":       {"fifa_rank": 15, "elo": 1827, "wc_apps": 17, "region": "CONCACAF"},
    "AUSTRALIA":    {"fifa_rank": 24, "elo": 1783, "wc_apps": 6,  "region": "AFC"},
    "IVORY COAST":  {"fifa_rank": 33, "elo": 1720, "wc_apps": 4,  "region": "CAF"},
    # ── Group D ──────────────────────────────────────────────────────────────
    "FRANCE":       {"fifa_rank": 2,  "elo": 2075, "wc_apps": 16, "region": "UEFA"},
    "ARGENTINA":    {"fifa_rank": 1,  "elo": 2133, "wc_apps": 18, "region": "CONMEBOL"},
    "SAUDI ARABIA": {"fifa_rank": 56, "elo": 1630, "wc_apps": 7,  "region": "AFC"},
    "NEW ZEALAND":  {"fifa_rank": 98, "elo": 1518, "wc_apps": 2,  "region": "OFC"},
    # ── Group E ──────────────────────────────────────────────────────────────
    "BRAZIL":       {"fifa_rank": 5,  "elo": 2070, "wc_apps": 22, "region": "CONMEBOL"},
    "CROATIA":      {"fifa_rank": 10, "elo": 1888, "wc_apps": 6,  "region": "UEFA"},
    "JAPAN":        {"fifa_rank": 16, "elo": 1836, "wc_apps": 8,  "region": "AFC"},
    "MOROCCO":      {"fifa_rank": 13, "elo": 1852, "wc_apps": 7,  "region": "CAF"},
    # ── Group F ──────────────────────────────────────────────────────────────
    "PORTUGAL":     {"fifa_rank": 6,  "elo": 2015, "wc_apps": 8,  "region": "UEFA"},
    "COLOMBIA":     {"fifa_rank": 9,  "elo": 1912, "wc_apps": 7,  "region": "CONMEBOL"},
    "SENEGAL":      {"fifa_rank": 17, "elo": 1830, "wc_apps": 4,  "region": "CAF"},
    "ECUADOR":      {"fifa_rank": 44, "elo": 1670, "wc_apps": 4,  "region": "CONMEBOL"},
    # ── Group G ──────────────────────────────────────────────────────────────
    "NETHERLANDS":  {"fifa_rank": 7,  "elo": 1985, "wc_apps": 11, "region": "UEFA"},
    "URUGUAY":      {"fifa_rank": 14, "elo": 1848, "wc_apps": 14, "region": "CONMEBOL"},
    "EGYPT":        {"fifa_rank": 36, "elo": 1698, "wc_apps": 3,  "region": "CAF"},
    "PANAMA":       {"fifa_rank": 70, "elo": 1598, "wc_apps": 2,  "region": "CONCACAF"},
    # ── Group H ──────────────────────────────────────────────────────────────
    "BELGIUM":      {"fifa_rank": 3,  "elo": 2035, "wc_apps": 14, "region": "UEFA"},
    "SWITZERLAND":  {"fifa_rank": 19, "elo": 1810, "wc_apps": 12, "region": "UEFA"},
    "ALGERIA":      {"fifa_rank": 35, "elo": 1702, "wc_apps": 4,  "region": "CAF"},
    "COSTA RICA":   {"fifa_rank": 50, "elo": 1651, "wc_apps": 6,  "region": "CONCACAF"},
    # ── Group I ──────────────────────────────────────────────────────────────
    "ITALY":        {"fifa_rank": 8,  "elo": 1952, "wc_apps": 18, "region": "UEFA"},
    "DENMARK":      {"fifa_rank": 12, "elo": 1858, "wc_apps": 5,  "region": "UEFA"},
    "SOUTH AFRICA": {"fifa_rank": 60, "elo": 1625, "wc_apps": 4,  "region": "CAF"},
    "JAMAICA":      {"fifa_rank": 55, "elo": 1637, "wc_apps": 1,  "region": "CONCACAF"},
    # ── Group J ──────────────────────────────────────────────────────────────
    "CHILE":        {"fifa_rank": 30, "elo": 1740, "wc_apps": 9,  "region": "CONMEBOL"},
    "ROMANIA":      {"fifa_rank": 46, "elo": 1662, "wc_apps": 7,  "region": "UEFA"},
    "MALI":         {"fifa_rank": 57, "elo": 1625, "wc_apps": 0,  "region": "CAF"},
    "VENEZUELA":    {"fifa_rank": 68, "elo": 1601, "wc_apps": 0,  "region": "CONMEBOL"},
    # ── Group K ──────────────────────────────────────────────────────────────
    "AUSTRIA":      {"fifa_rank": 20, "elo": 1805, "wc_apps": 7,  "region": "UEFA"},
    "HUNGARY":      {"fifa_rank": 28, "elo": 1757, "wc_apps": 9,  "region": "UEFA"},
    "JORDAN":       {"fifa_rank": 73, "elo": 1582, "wc_apps": 0,  "region": "AFC"},
    "HONDURAS":     {"fifa_rank": 82, "elo": 1560, "wc_apps": 3,  "region": "CONCACAF"},
    # ── Group L ──────────────────────────────────────────────────────────────
    "SERBIA":       {"fifa_rank": 25, "elo": 1780, "wc_apps": 13, "region": "UEFA"},
    "POLAND":       {"fifa_rank": 26, "elo": 1774, "wc_apps": 9,  "region": "UEFA"},
    "PARAGUAY":     {"fifa_rank": 63, "elo": 1617, "wc_apps": 9,  "region": "CONMEBOL"},
    "UZBEKISTAN":   {"fifa_rank": 74, "elo": 1580, "wc_apps": 0,  "region": "AFC"},
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GROUP STAGE SCHEDULE (72 matches)
# stage: "GROUP"
# Venues listed for atmosphere; all treated as neutral for modelling
# HOW TO READ: home/away labels are just seeding order, NOT actual home teams
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULE: list[dict] = [
    # ── MATCHDAY 1 (June 11–17) ─────────────────────────────────────────────
    # Group A
    {"date": "2026-06-11", "group": "A", "matchday": 1, "home": "ENGLAND",    "away": "CAMEROON",    "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-11", "group": "A", "matchday": 1, "home": "USA",        "away": "IRAN",        "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    # Group B
    {"date": "2026-06-12", "group": "B", "matchday": 1, "home": "GERMANY",    "away": "NIGERIA",     "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    {"date": "2026-06-12", "group": "B", "matchday": 1, "home": "CANADA",     "away": "SOUTH KOREA", "venue": "BC Place, Vancouver",           "stage": "GROUP"},
    # Group C
    {"date": "2026-06-12", "group": "C", "matchday": 1, "home": "SPAIN",      "away": "AUSTRALIA",   "venue": "Estadio Azteca, Mexico City",   "stage": "GROUP"},
    {"date": "2026-06-12", "group": "C", "matchday": 1, "home": "MEXICO",     "away": "IVORY COAST", "venue": "Estadio Azteca, Mexico City",   "stage": "GROUP"},
    # Group D
    {"date": "2026-06-13", "group": "D", "matchday": 1, "home": "FRANCE",     "away": "NEW ZEALAND", "venue": "Levi's Stadium, San Francisco", "stage": "GROUP"},
    {"date": "2026-06-13", "group": "D", "matchday": 1, "home": "ARGENTINA",  "away": "SAUDI ARABIA","venue": "Hard Rock Stadium, Miami",      "stage": "GROUP"},
    # Group E
    {"date": "2026-06-13", "group": "E", "matchday": 1, "home": "BRAZIL",     "away": "JAPAN",       "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-13", "group": "E", "matchday": 1, "home": "CROATIA",    "away": "MOROCCO",     "venue": "Lincoln Financial, Philadelphia","stage": "GROUP"},
    # Group F
    {"date": "2026-06-14", "group": "F", "matchday": 1, "home": "PORTUGAL",   "away": "SENEGAL",     "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    {"date": "2026-06-14", "group": "F", "matchday": 1, "home": "COLOMBIA",   "away": "ECUADOR",     "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    # Group G
    {"date": "2026-06-14", "group": "G", "matchday": 1, "home": "NETHERLANDS","away": "PANAMA",      "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    {"date": "2026-06-14", "group": "G", "matchday": 1, "home": "URUGUAY",    "away": "EGYPT",       "venue": "Arrowhead Stadium, Kansas City", "stage": "GROUP"},
    # Group H
    {"date": "2026-06-15", "group": "H", "matchday": 1, "home": "BELGIUM",    "away": "COSTA RICA",  "venue": "NRG Stadium, Houston",          "stage": "GROUP"},
    {"date": "2026-06-15", "group": "H", "matchday": 1, "home": "SWITZERLAND","away": "ALGERIA",     "venue": "Empower Field, Denver",         "stage": "GROUP"},
    # Group I
    {"date": "2026-06-15", "group": "I", "matchday": 1, "home": "ITALY",      "away": "SOUTH AFRICA","venue": "Gillette Stadium, Boston",      "stage": "GROUP"},
    {"date": "2026-06-15", "group": "I", "matchday": 1, "home": "DENMARK",    "away": "JAMAICA",     "venue": "BMO Field, Toronto",            "stage": "GROUP"},
    # Group J
    {"date": "2026-06-16", "group": "J", "matchday": 1, "home": "CHILE",      "away": "VENEZUELA",   "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    {"date": "2026-06-16", "group": "J", "matchday": 1, "home": "ROMANIA",    "away": "MALI",        "venue": "State Farm Stadium, Phoenix",   "stage": "GROUP"},
    # Group K
    {"date": "2026-06-16", "group": "K", "matchday": 1, "home": "AUSTRIA",    "away": "HONDURAS",    "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    {"date": "2026-06-16", "group": "K", "matchday": 1, "home": "HUNGARY",    "away": "JORDAN",      "venue": "Empower Field, Denver",         "stage": "GROUP"},
    # Group L
    {"date": "2026-06-17", "group": "L", "matchday": 1, "home": "SERBIA",     "away": "PARAGUAY",    "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-17", "group": "L", "matchday": 1, "home": "POLAND",     "away": "UZBEKISTAN",  "venue": "Estadio BBVA, Monterrey",       "stage": "GROUP"},

    # ── MATCHDAY 2 (June 18–22) — UPCOMING ──────────────────────────────────
    # Group A
    {"date": "2026-06-18", "group": "A", "matchday": 2, "home": "ENGLAND",    "away": "USA",         "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-18", "group": "A", "matchday": 2, "home": "CAMEROON",   "away": "IRAN",        "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    # Group B
    {"date": "2026-06-18", "group": "B", "matchday": 2, "home": "GERMANY",    "away": "SOUTH KOREA", "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    {"date": "2026-06-18", "group": "B", "matchday": 2, "home": "CANADA",     "away": "NIGERIA",     "venue": "BC Place, Vancouver",           "stage": "GROUP"},
    # Group C
    {"date": "2026-06-19", "group": "C", "matchday": 2, "home": "SPAIN",      "away": "MEXICO",      "venue": "Estadio Azteca, Mexico City",   "stage": "GROUP"},
    {"date": "2026-06-19", "group": "C", "matchday": 2, "home": "AUSTRALIA",  "away": "IVORY COAST", "venue": "Estadio BBVA, Monterrey",       "stage": "GROUP"},
    # Group D
    {"date": "2026-06-19", "group": "D", "matchday": 2, "home": "FRANCE",     "away": "ARGENTINA",   "venue": "Hard Rock Stadium, Miami",      "stage": "GROUP"},
    {"date": "2026-06-19", "group": "D", "matchday": 2, "home": "SAUDI ARABIA","away": "NEW ZEALAND", "venue": "Levi's Stadium, San Francisco","stage": "GROUP"},
    # Group E
    {"date": "2026-06-20", "group": "E", "matchday": 2, "home": "BRAZIL",     "away": "MOROCCO",     "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-20", "group": "E", "matchday": 2, "home": "CROATIA",    "away": "JAPAN",       "venue": "Lincoln Financial, Philadelphia","stage": "GROUP"},
    # Group F
    {"date": "2026-06-20", "group": "F", "matchday": 2, "home": "PORTUGAL",   "away": "ECUADOR",     "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    {"date": "2026-06-20", "group": "F", "matchday": 2, "home": "COLOMBIA",   "away": "SENEGAL",     "venue": "NRG Stadium, Houston",          "stage": "GROUP"},
    # Group G
    {"date": "2026-06-21", "group": "G", "matchday": 2, "home": "NETHERLANDS","away": "EGYPT",       "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    {"date": "2026-06-21", "group": "G", "matchday": 2, "home": "URUGUAY",    "away": "PANAMA",      "venue": "Arrowhead Stadium, Kansas City", "stage": "GROUP"},
    # Group H
    {"date": "2026-06-21", "group": "H", "matchday": 2, "home": "BELGIUM",    "away": "ALGERIA",     "venue": "Empower Field, Denver",         "stage": "GROUP"},
    {"date": "2026-06-21", "group": "H", "matchday": 2, "home": "SWITZERLAND","away": "COSTA RICA",  "venue": "NRG Stadium, Houston",          "stage": "GROUP"},
    # Group I
    {"date": "2026-06-21", "group": "I", "matchday": 2, "home": "ITALY",      "away": "JAMAICA",     "venue": "Gillette Stadium, Boston",      "stage": "GROUP"},
    {"date": "2026-06-21", "group": "I", "matchday": 2, "home": "DENMARK",    "away": "SOUTH AFRICA","venue": "BMO Field, Toronto",            "stage": "GROUP"},
    # Group J
    {"date": "2026-06-22", "group": "J", "matchday": 2, "home": "CHILE",      "away": "MALI",        "venue": "State Farm Stadium, Phoenix",   "stage": "GROUP"},
    {"date": "2026-06-22", "group": "J", "matchday": 2, "home": "ROMANIA",    "away": "VENEZUELA",   "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    # Group K
    {"date": "2026-06-22", "group": "K", "matchday": 2, "home": "AUSTRIA",    "away": "JORDAN",      "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    {"date": "2026-06-22", "group": "K", "matchday": 2, "home": "HUNGARY",    "away": "HONDURAS",    "venue": "Empower Field, Denver",         "stage": "GROUP"},
    # Group L
    {"date": "2026-06-22", "group": "L", "matchday": 2, "home": "SERBIA",     "away": "UZBEKISTAN",  "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-22", "group": "L", "matchday": 2, "home": "POLAND",     "away": "PARAGUAY",    "venue": "Estadio BBVA, Monterrey",       "stage": "GROUP"},

    # ── MATCHDAY 3 (June 26–30) — UPCOMING ──────────────────────────────────
    # Group A
    {"date": "2026-06-26", "group": "A", "matchday": 3, "home": "ENGLAND",    "away": "IRAN",        "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-26", "group": "A", "matchday": 3, "home": "USA",        "away": "CAMEROON",    "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    # Group B
    {"date": "2026-06-26", "group": "B", "matchday": 3, "home": "GERMANY",    "away": "CANADA",      "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    {"date": "2026-06-26", "group": "B", "matchday": 3, "home": "SOUTH KOREA","away": "NIGERIA",     "venue": "BC Place, Vancouver",           "stage": "GROUP"},
    # Group C
    {"date": "2026-06-27", "group": "C", "matchday": 3, "home": "SPAIN",      "away": "IVORY COAST", "venue": "Estadio Azteca, Mexico City",   "stage": "GROUP"},
    {"date": "2026-06-27", "group": "C", "matchday": 3, "home": "AUSTRALIA",  "away": "MEXICO",      "venue": "Estadio BBVA, Monterrey",       "stage": "GROUP"},
    # Group D
    {"date": "2026-06-27", "group": "D", "matchday": 3, "home": "FRANCE",     "away": "SAUDI ARABIA","venue": "Hard Rock Stadium, Miami",      "stage": "GROUP"},
    {"date": "2026-06-27", "group": "D", "matchday": 3, "home": "NEW ZEALAND","away": "ARGENTINA",   "venue": "Levi's Stadium, San Francisco", "stage": "GROUP"},
    # Group E
    {"date": "2026-06-28", "group": "E", "matchday": 3, "home": "BRAZIL",     "away": "CROATIA",     "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-28", "group": "E", "matchday": 3, "home": "JAPAN",      "away": "MOROCCO",     "venue": "Lincoln Financial, Philadelphia","stage": "GROUP"},
    # Group F
    {"date": "2026-06-28", "group": "F", "matchday": 3, "home": "PORTUGAL",   "away": "COLOMBIA",    "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    {"date": "2026-06-28", "group": "F", "matchday": 3, "home": "SENEGAL",    "away": "ECUADOR",     "venue": "NRG Stadium, Houston",          "stage": "GROUP"},
    # Group G
    {"date": "2026-06-29", "group": "G", "matchday": 3, "home": "NETHERLANDS","away": "URUGUAY",     "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    {"date": "2026-06-29", "group": "G", "matchday": 3, "home": "EGYPT",      "away": "PANAMA",      "venue": "Arrowhead Stadium, Kansas City", "stage": "GROUP"},
    # Group H
    {"date": "2026-06-29", "group": "H", "matchday": 3, "home": "BELGIUM",    "away": "SWITZERLAND", "venue": "Empower Field, Denver",         "stage": "GROUP"},
    {"date": "2026-06-29", "group": "H", "matchday": 3, "home": "ALGERIA",    "away": "COSTA RICA",  "venue": "NRG Stadium, Houston",          "stage": "GROUP"},
    # Group I
    {"date": "2026-06-29", "group": "I", "matchday": 3, "home": "ITALY",      "away": "DENMARK",     "venue": "Gillette Stadium, Boston",      "stage": "GROUP"},
    {"date": "2026-06-29", "group": "I", "matchday": 3, "home": "SOUTH AFRICA","away": "JAMAICA",    "venue": "BMO Field, Toronto",            "stage": "GROUP"},
    # Group J
    {"date": "2026-06-30", "group": "J", "matchday": 3, "home": "CHILE",      "away": "ROMANIA",     "venue": "State Farm Stadium, Phoenix",   "stage": "GROUP"},
    {"date": "2026-06-30", "group": "J", "matchday": 3, "home": "MALI",       "away": "VENEZUELA",   "venue": "Sofi Stadium, Los Angeles",     "stage": "GROUP"},
    # Group K
    {"date": "2026-06-30", "group": "K", "matchday": 3, "home": "AUSTRIA",    "away": "HUNGARY",     "venue": "AT&T Stadium, Dallas",          "stage": "GROUP"},
    {"date": "2026-06-30", "group": "K", "matchday": 3, "home": "JORDAN",     "away": "HONDURAS",    "venue": "Empower Field, Denver",         "stage": "GROUP"},
    # Group L
    {"date": "2026-06-30", "group": "L", "matchday": 3, "home": "SERBIA",     "away": "POLAND",      "venue": "MetLife Stadium, New York",     "stage": "GROUP"},
    {"date": "2026-06-30", "group": "L", "matchday": 3, "home": "PARAGUAY",   "away": "UZBEKISTAN",  "venue": "Estadio BBVA, Monterrey",       "stage": "GROUP"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RESULTS SO FAR (Matchday 1, June 11–17)
# UPDATE THIS LIST after every match!
# Format: home/away are the seeded team labels from SCHEDULE above
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_SO_FAR: list[dict] = [
    # Group A — June 11
    {"date": "2026-06-11", "home": "ENGLAND",     "away": "CAMEROON",    "home_score": 3, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-11", "home": "USA",          "away": "IRAN",        "home_score": 2, "away_score": 1, "stage": "GROUP"},
    # Group B — June 12
    {"date": "2026-06-12", "home": "GERMANY",      "away": "NIGERIA",     "home_score": 4, "away_score": 1, "stage": "GROUP"},
    {"date": "2026-06-12", "home": "CANADA",       "away": "SOUTH KOREA", "home_score": 0, "away_score": 1, "stage": "GROUP"},
    # Group C — June 12
    {"date": "2026-06-12", "home": "SPAIN",        "away": "AUSTRALIA",   "home_score": 2, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-12", "home": "MEXICO",       "away": "IVORY COAST", "home_score": 1, "away_score": 0, "stage": "GROUP"},
    # Group D — June 13
    {"date": "2026-06-13", "home": "FRANCE",       "away": "NEW ZEALAND", "home_score": 3, "away_score": 1, "stage": "GROUP"},
    {"date": "2026-06-13", "home": "ARGENTINA",    "away": "SAUDI ARABIA","home_score": 2, "away_score": 0, "stage": "GROUP"},
    # Group E — June 13
    {"date": "2026-06-13", "home": "BRAZIL",       "away": "JAPAN",       "home_score": 2, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-13", "home": "CROATIA",      "away": "MOROCCO",     "home_score": 1, "away_score": 1, "stage": "GROUP"},
    # Group F — June 14
    {"date": "2026-06-14", "home": "PORTUGAL",     "away": "SENEGAL",     "home_score": 2, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-14", "home": "COLOMBIA",     "away": "ECUADOR",     "home_score": 3, "away_score": 1, "stage": "GROUP"},
    # Group G — June 14
    {"date": "2026-06-14", "home": "NETHERLANDS",  "away": "PANAMA",      "home_score": 3, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-14", "home": "URUGUAY",      "away": "EGYPT",       "home_score": 1, "away_score": 0, "stage": "GROUP"},
    # Group H — June 15
    {"date": "2026-06-15", "home": "BELGIUM",      "away": "COSTA RICA",  "home_score": 3, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-15", "home": "SWITZERLAND",  "away": "ALGERIA",     "home_score": 2, "away_score": 1, "stage": "GROUP"},
    # Group I — June 15
    {"date": "2026-06-15", "home": "ITALY",        "away": "SOUTH AFRICA","home_score": 2, "away_score": 2, "stage": "GROUP"},
    {"date": "2026-06-15", "home": "DENMARK",      "away": "JAMAICA",     "home_score": 3, "away_score": 0, "stage": "GROUP"},
    # Group J — June 16
    {"date": "2026-06-16", "home": "CHILE",        "away": "VENEZUELA",   "home_score": 2, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-16", "home": "ROMANIA",      "away": "MALI",        "home_score": 1, "away_score": 1, "stage": "GROUP"},
    # Group K — June 16
    {"date": "2026-06-16", "home": "AUSTRIA",      "away": "HONDURAS",    "home_score": 2, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-16", "home": "HUNGARY",      "away": "JORDAN",      "home_score": 1, "away_score": 2, "stage": "GROUP"},  # UPSET!
    # Group L — June 17
    {"date": "2026-06-17", "home": "SERBIA",       "away": "PARAGUAY",    "home_score": 1, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-17", "home": "POLAND",       "away": "UZBEKISTAN",  "home_score": 2, "away_score": 0, "stage": "GROUP"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: DEMO ODDS (for upcoming Matchday 2 matches)
# Used when no Odds API key is set. Derived from ELO gap estimates.
# Format: {"{HOME}_vs_{AWAY}": {"home": float, "draw": float, "away": float,
#                                "over_25": float, "btts": float}}
# HOW TO GENERATE: Take ELO win probability → convert to decimal odds,
# then add ~8% bookmaker margin (typical overround)
# ─────────────────────────────────────────────────────────────────────────────
DEMO_ODDS: dict[str, dict[str, float]] = {
    # Group A Matchday 2 (June 18) — ENGLAND vs USA (blockbuster)
    "ENGLAND_vs_USA":         {"home": 1.95, "draw": 3.50, "away": 4.20, "over_25": 1.80, "btts": 1.85},
    "CAMEROON_vs_IRAN":       {"home": 2.80, "draw": 3.00, "away": 2.60, "over_25": 2.10, "btts": 2.20},
    # Group B Matchday 2 (June 18)
    "GERMANY_vs_SOUTH KOREA": {"home": 1.65, "draw": 3.80, "away": 5.50, "over_25": 1.65, "btts": 1.80},
    "CANADA_vs_NIGERIA":      {"home": 2.20, "draw": 3.20, "away": 3.40, "over_25": 1.90, "btts": 1.95},
    # Group C Matchday 2 (June 19) — SPAIN vs MEXICO (massive game)
    "SPAIN_vs_MEXICO":        {"home": 1.72, "draw": 3.60, "away": 5.00, "over_25": 1.78, "btts": 1.88},
    "AUSTRALIA_vs_IVORY COAST":{"home":2.50, "draw": 3.10, "away": 2.90, "over_25": 2.00, "btts": 2.10},
    # Group D Matchday 2 (June 19) — FRANCE vs ARGENTINA (the dream match)
    "FRANCE_vs_ARGENTINA":    {"home": 2.25, "draw": 3.40, "away": 3.10, "over_25": 1.82, "btts": 1.88},
    "SAUDI ARABIA_vs_NEW ZEALAND":{"home":1.70,"draw":3.70,"away":5.50, "over_25": 2.10, "btts": 2.20},
    # Group E Matchday 2 (June 20)
    "BRAZIL_vs_MOROCCO":      {"home": 1.60, "draw": 3.80, "away": 6.00, "over_25": 1.85, "btts": 1.95},
    "CROATIA_vs_JAPAN":       {"home": 2.00, "draw": 3.30, "away": 3.90, "over_25": 1.88, "btts": 1.98},
    # Group F Matchday 2 (June 20)
    "PORTUGAL_vs_ECUADOR":    {"home": 1.50, "draw": 4.20, "away": 7.00, "over_25": 1.75, "btts": 1.92},
    "COLOMBIA_vs_SENEGAL":    {"home": 1.85, "draw": 3.50, "away": 4.50, "over_25": 1.85, "btts": 1.95},
    # Group G Matchday 2 (June 21)
    "NETHERLANDS_vs_EGYPT":   {"home": 1.45, "draw": 4.50, "away": 8.00, "over_25": 1.72, "btts": 1.95},
    "URUGUAY_vs_PANAMA":      {"home": 1.40, "draw": 4.80, "away": 9.00, "over_25": 1.85, "btts": 2.10},
    # Group H Matchday 2 (June 21)
    "BELGIUM_vs_ALGERIA":     {"home": 1.45, "draw": 4.50, "away": 8.00, "over_25": 1.68, "btts": 1.88},
    "SWITZERLAND_vs_COSTA RICA":{"home":1.55,"draw":4.00, "away":6.50, "over_25": 1.78, "btts": 1.98},
    # Group I Matchday 2 (June 21)
    "ITALY_vs_JAMAICA":       {"home": 1.30, "draw": 5.50, "away": 10.0,"over_25": 1.68, "btts": 1.98},
    "DENMARK_vs_SOUTH AFRICA":{"home": 1.45, "draw": 4.40, "away": 7.50, "over_25": 1.78, "btts": 1.92},
    # Group J Matchday 2 (June 22)
    "CHILE_vs_MALI":          {"home": 1.62, "draw": 3.70, "away": 6.00, "over_25": 1.88, "btts": 2.00},
    "ROMANIA_vs_VENEZUELA":   {"home": 1.75, "draw": 3.50, "away": 5.00, "over_25": 2.00, "btts": 2.10},
    # Group K Matchday 2 (June 22)
    "AUSTRIA_vs_JORDAN":      {"home": 1.40, "draw": 4.50, "away": 9.00, "over_25": 1.80, "btts": 2.05},
    "HUNGARY_vs_HONDURAS":    {"home": 1.70, "draw": 3.60, "away": 5.50, "over_25": 1.90, "btts": 2.00},
    # Group L Matchday 2 (June 22)
    "SERBIA_vs_UZBEKISTAN":   {"home": 1.45, "draw": 4.20, "away": 8.00, "over_25": 1.78, "btts": 1.95},
    "POLAND_vs_PARAGUAY":     {"home": 1.60, "draw": 3.80, "away": 6.00, "over_25": 1.85, "btts": 1.98},
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Which stage we're currently in (update as tournament progresses)
CURRENT_STAGE: str = "GROUP"

# Tournament stages in order (for UI display and model feature encoding)
STAGE_ORDER: list[str] = ["GROUP", "R32", "R16", "QF", "SF", "FINAL"]
STAGE_LABELS: dict[str, str] = {
    "GROUP": "Group Stage",
    "R32":   "Round of 32",
    "R16":   "Round of 16",
    "QF":    "Quarterfinal",
    "SF":    "Semifinal",
    "FINAL": "Final",
}

# All 48 team names (derived from GROUPS for easy iteration)
ALL_TEAMS: list[str] = sorted({t for teams in GROUPS.values() for t in teams})


def get_team_group(team: str) -> str | None:
    """Return the group letter for a given team name, or None if not found."""
    for group, teams in GROUPS.items():
        if team in teams:
            return group
    return None


def get_demo_odds(home: str, away: str) -> dict[str, float] | None:
    """Look up demo odds for a match by team names.

    Tries both {HOME}_vs_{AWAY} and {AWAY}_vs_{HOME} (flipped markets).
    Returns None if no demo odds are available for this fixture.
    """
    key = f"{home}_vs_{away}"
    if key in DEMO_ODDS:
        return DEMO_ODDS[key]

    # Try reversed order (some fixtures may be stored in opposite order)
    rev_key = f"{away}_vs_{home}"
    if rev_key in DEMO_ODDS:
        # Flip home/away odds
        odds = DEMO_ODDS[rev_key].copy()
        odds["home"], odds["away"] = odds["away"], odds["home"]
        return odds

    return None


def upcoming_matches(as_of_date: str = "2026-06-17") -> list[dict]:
    """Return matches not yet in RESULTS_SO_FAR.

    Parameters
    ----------
    as_of_date : str — ISO date string; only return matches on/after this date

    Returns list of schedule entries for unplayed matches.
    """
    played = {
        (r["home"], r["away"], r["date"])
        for r in RESULTS_SO_FAR
    }
    result = []
    for m in SCHEDULE:
        key = (m["home"], m["away"], m["date"])
        if key not in played and m["date"] >= as_of_date:
            result.append(m)
    return result
