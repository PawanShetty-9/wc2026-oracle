"""
data/wc2026_teams.py — 2026 FIFA World Cup Tournament Data (OFFLINE / DEMO FALLBACK)
=====================================================================================
This module is the DEMO MODE foundation — all data is pulled from the real
football-data.org API when FOOTBALL_DATA_API_KEY is configured.

Groups and results are sourced from the December 5 2024 FIFA draw (Coral Gables, Miami).
Results updated through June 17 2026 (end of Matchday 1 / start of MD2).

2026 FORMAT:
  - 48 teams, 12 groups of 4, top 2 + 8 best 3rd-place teams advance
  - Group stage: June 11–July 3, 2026
  - Round of 32: July 4–7  |  Round of 16: July 10–12
  - Quarterfinals: July 14–15  |  Semifinals: July 17–18
  - Third place: July 21  |  Final: July 19, 2026

HOW TO UPDATE:
  After each match day, add results to RESULTS_SO_FAR:
    {"date": "2026-06-18", "home": "CZECHIA", "away": "SOUTH AFRICA",
     "home_score": 2, "away_score": 0, "stage": "GROUP"}
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: GROUP ASSIGNMENTS (real FIFA WC 2026 draw — December 5 2024)
# 12 groups × 4 teams = 48 teams
# ─────────────────────────────────────────────────────────────────────────────
GROUPS: dict[str, list[str]] = {
    "A": ["MEXICO",      "SOUTH AFRICA",       "SOUTH KOREA",       "CZECHIA"],
    "B": ["CANADA",      "BOSNIA-HERZEGOVINA", "QATAR",             "SWITZERLAND"],
    "C": ["BRAZIL",      "HAITI",              "MOROCCO",           "SCOTLAND"],
    "D": ["USA",         "PARAGUAY",           "TURKEY",            "AUSTRALIA"],
    "E": ["GERMANY",     "CURACAO",            "IVORY COAST",       "ECUADOR"],
    "F": ["NETHERLANDS", "JAPAN",              "SWEDEN",            "TUNISIA"],
    "G": ["BELGIUM",     "EGYPT",              "IRAN",              "NEW ZEALAND"],
    "H": ["SPAIN",       "CAPE VERDE ISLANDS", "SAUDI ARABIA",      "URUGUAY"],
    "I": ["FRANCE",      "IRAQ",               "NORWAY",            "SENEGAL"],
    "J": ["ARGENTINA",   "ALGERIA",            "AUSTRIA",           "JORDAN"],
    "K": ["PORTUGAL",    "CONGO DR",           "COLOMBIA",          "UZBEKISTAN"],
    "L": ["ENGLAND",     "CROATIA",            "GHANA",             "PANAMA"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TEAM METADATA
# fifa_rank : FIFA ranking as of May 2026
# elo       : Pre-tournament ELO rating (World Football Elo scale)
# wc_apps   : Previous World Cup appearances (not counting 2026)
# region    : Confederation
# ─────────────────────────────────────────────────────────────────────────────
TEAM_META: dict[str, dict] = {
    # ── Group A ──────────────────────────────────────────────────────────────
    "MEXICO":             {"fifa_rank": 15, "elo": 1827, "wc_apps": 17, "region": "CONCACAF"},
    "SOUTH AFRICA":       {"fifa_rank": 66, "elo": 1600, "wc_apps": 3,  "region": "CAF"},
    "SOUTH KOREA":        {"fifa_rank": 23, "elo": 1788, "wc_apps": 11, "region": "AFC"},
    "CZECHIA":            {"fifa_rank": 37, "elo": 1680, "wc_apps": 4,  "region": "UEFA"},
    # ── Group B ──────────────────────────────────────────────────────────────
    "CANADA":             {"fifa_rank": 41, "elo": 1710, "wc_apps": 2,  "region": "CONCACAF"},
    "BOSNIA-HERZEGOVINA": {"fifa_rank": 55, "elo": 1630, "wc_apps": 1,  "region": "UEFA"},
    "QATAR":              {"fifa_rank": 62, "elo": 1590, "wc_apps": 1,  "region": "AFC"},
    "SWITZERLAND":        {"fifa_rank": 19, "elo": 1810, "wc_apps": 12, "region": "UEFA"},
    # ── Group C ──────────────────────────────────────────────────────────────
    "BRAZIL":             {"fifa_rank": 5,  "elo": 2070, "wc_apps": 22, "region": "CONMEBOL"},
    "HAITI":              {"fifa_rank": 80, "elo": 1545, "wc_apps": 1,  "region": "CONCACAF"},
    "MOROCCO":            {"fifa_rank": 13, "elo": 1852, "wc_apps": 7,  "region": "CAF"},
    "SCOTLAND":           {"fifa_rank": 38, "elo": 1660, "wc_apps": 8,  "region": "UEFA"},
    # ── Group D ──────────────────────────────────────────────────────────────
    "USA":                {"fifa_rank": 11, "elo": 1858, "wc_apps": 11, "region": "CONCACAF"},
    "PARAGUAY":           {"fifa_rank": 63, "elo": 1617, "wc_apps": 9,  "region": "CONMEBOL"},
    "TURKEY":             {"fifa_rank": 33, "elo": 1690, "wc_apps": 2,  "region": "UEFA"},
    "AUSTRALIA":          {"fifa_rank": 24, "elo": 1783, "wc_apps": 6,  "region": "AFC"},
    # ── Group E ──────────────────────────────────────────────────────────────
    "GERMANY":            {"fifa_rank": 3,  "elo": 2030, "wc_apps": 19, "region": "UEFA"},
    "CURACAO":            {"fifa_rank": 72, "elo": 1560, "wc_apps": 0,  "region": "CONCACAF"},
    "IVORY COAST":        {"fifa_rank": 33, "elo": 1720, "wc_apps": 4,  "region": "CAF"},
    "ECUADOR":            {"fifa_rank": 44, "elo": 1670, "wc_apps": 4,  "region": "CONMEBOL"},
    # ── Group F ──────────────────────────────────────────────────────────────
    "NETHERLANDS":        {"fifa_rank": 7,  "elo": 1985, "wc_apps": 11, "region": "UEFA"},
    "JAPAN":              {"fifa_rank": 16, "elo": 1836, "wc_apps": 8,  "region": "AFC"},
    "SWEDEN":             {"fifa_rank": 26, "elo": 1760, "wc_apps": 12, "region": "UEFA"},
    "TUNISIA":            {"fifa_rank": 42, "elo": 1660, "wc_apps": 5,  "region": "CAF"},
    # ── Group G ──────────────────────────────────────────────────────────────
    "BELGIUM":            {"fifa_rank": 3,  "elo": 2035, "wc_apps": 14, "region": "UEFA"},
    "EGYPT":              {"fifa_rank": 36, "elo": 1698, "wc_apps": 3,  "region": "CAF"},
    "IRAN":               {"fifa_rank": 22, "elo": 1790, "wc_apps": 6,  "region": "AFC"},
    "NEW ZEALAND":        {"fifa_rank": 98, "elo": 1518, "wc_apps": 2,  "region": "OFC"},
    # ── Group H ──────────────────────────────────────────────────────────────
    "SPAIN":              {"fifa_rank": 2,  "elo": 2080, "wc_apps": 15, "region": "UEFA"},
    "CAPE VERDE ISLANDS": {"fifa_rank": 78, "elo": 1555, "wc_apps": 0,  "region": "CAF"},
    "SAUDI ARABIA":       {"fifa_rank": 56, "elo": 1630, "wc_apps": 7,  "region": "AFC"},
    "URUGUAY":            {"fifa_rank": 14, "elo": 1848, "wc_apps": 14, "region": "CONMEBOL"},
    # ── Group I ──────────────────────────────────────────────────────────────
    "FRANCE":             {"fifa_rank": 2,  "elo": 2075, "wc_apps": 16, "region": "UEFA"},
    "IRAQ":               {"fifa_rank": 65, "elo": 1595, "wc_apps": 1,  "region": "AFC"},
    "NORWAY":             {"fifa_rank": 36, "elo": 1670, "wc_apps": 3,  "region": "UEFA"},
    "SENEGAL":            {"fifa_rank": 17, "elo": 1830, "wc_apps": 4,  "region": "CAF"},
    # ── Group J ──────────────────────────────────────────────────────────────
    "ARGENTINA":          {"fifa_rank": 1,  "elo": 2133, "wc_apps": 18, "region": "CONMEBOL"},
    "ALGERIA":            {"fifa_rank": 35, "elo": 1702, "wc_apps": 4,  "region": "CAF"},
    "AUSTRIA":            {"fifa_rank": 20, "elo": 1805, "wc_apps": 7,  "region": "UEFA"},
    "JORDAN":             {"fifa_rank": 73, "elo": 1582, "wc_apps": 0,  "region": "AFC"},
    # ── Group K ──────────────────────────────────────────────────────────────
    "PORTUGAL":           {"fifa_rank": 6,  "elo": 2015, "wc_apps": 8,  "region": "UEFA"},
    "CONGO DR":           {"fifa_rank": 61, "elo": 1610, "wc_apps": 2,  "region": "CAF"},
    "COLOMBIA":           {"fifa_rank": 9,  "elo": 1912, "wc_apps": 7,  "region": "CONMEBOL"},
    "UZBEKISTAN":         {"fifa_rank": 74, "elo": 1580, "wc_apps": 0,  "region": "AFC"},
    # ── Group L ──────────────────────────────────────────────────────────────
    "ENGLAND":            {"fifa_rank": 4,  "elo": 2025, "wc_apps": 16, "region": "UEFA"},
    "CROATIA":            {"fifa_rank": 10, "elo": 1888, "wc_apps": 6,  "region": "UEFA"},
    "GHANA":              {"fifa_rank": 53, "elo": 1640, "wc_apps": 4,  "region": "CAF"},
    "PANAMA":             {"fifa_rank": 70, "elo": 1598, "wc_apps": 2,  "region": "CONCACAF"},
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GROUP STAGE SCHEDULE (72 matches — all three matchdays)
# Used as fallback when Football Data API is unavailable.
# stage: "GROUP"
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULE: list[dict] = [
    # ── MATCHDAY 1 (June 11–17) ──────────────────────────────────────────────
    # Group A
    {"date": "2026-06-11", "group": "A", "matchday": 1, "home": "MEXICO",             "away": "SOUTH AFRICA",       "venue": "SoFi Stadium, Los Angeles",       "stage": "GROUP"},
    {"date": "2026-06-12", "group": "A", "matchday": 1, "home": "SOUTH KOREA",        "away": "CZECHIA",            "venue": "Levi's Stadium, San Jose",        "stage": "GROUP"},
    # Group B
    {"date": "2026-06-12", "group": "B", "matchday": 1, "home": "CANADA",             "away": "BOSNIA-HERZEGOVINA", "venue": "BC Place, Vancouver",             "stage": "GROUP"},
    {"date": "2026-06-13", "group": "B", "matchday": 1, "home": "QATAR",              "away": "SWITZERLAND",        "venue": "BMO Field, Toronto",              "stage": "GROUP"},
    # Group C
    {"date": "2026-06-13", "group": "C", "matchday": 1, "home": "BRAZIL",             "away": "MOROCCO",            "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    {"date": "2026-06-14", "group": "C", "matchday": 1, "home": "HAITI",              "away": "SCOTLAND",           "venue": "Hard Rock Stadium, Miami",         "stage": "GROUP"},
    # Group D
    {"date": "2026-06-13", "group": "D", "matchday": 1, "home": "USA",               "away": "PARAGUAY",           "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-14", "group": "D", "matchday": 1, "home": "AUSTRALIA",          "away": "TURKEY",             "venue": "Arrowhead Stadium, Kansas City",   "stage": "GROUP"},
    # Group E
    {"date": "2026-06-14", "group": "E", "matchday": 1, "home": "GERMANY",            "away": "CURACAO",            "venue": "Lincoln Financial, Philadelphia",  "stage": "GROUP"},
    {"date": "2026-06-14", "group": "E", "matchday": 1, "home": "IVORY COAST",        "away": "ECUADOR",            "venue": "NRG Stadium, Houston",            "stage": "GROUP"},
    # Group F
    {"date": "2026-06-14", "group": "F", "matchday": 1, "home": "NETHERLANDS",        "away": "JAPAN",              "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-15", "group": "F", "matchday": 1, "home": "SWEDEN",             "away": "TUNISIA",            "venue": "Empower Field, Denver",           "stage": "GROUP"},
    # Group G
    {"date": "2026-06-15", "group": "G", "matchday": 1, "home": "BELGIUM",            "away": "EGYPT",              "venue": "Gillette Stadium, Boston",        "stage": "GROUP"},
    {"date": "2026-06-16", "group": "G", "matchday": 1, "home": "IRAN",               "away": "NEW ZEALAND",        "venue": "State Farm Stadium, Glendale",    "stage": "GROUP"},
    # Group H
    {"date": "2026-06-15", "group": "H", "matchday": 1, "home": "SPAIN",              "away": "CAPE VERDE ISLANDS", "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    {"date": "2026-06-15", "group": "H", "matchday": 1, "home": "SAUDI ARABIA",       "away": "URUGUAY",            "venue": "SoFi Stadium, Los Angeles",       "stage": "GROUP"},
    # Group I
    {"date": "2026-06-16", "group": "I", "matchday": 1, "home": "FRANCE",             "away": "SENEGAL",            "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-16", "group": "I", "matchday": 1, "home": "IRAQ",               "away": "NORWAY",             "venue": "Estadio Azteca, Mexico City",      "stage": "GROUP"},
    # Group J
    {"date": "2026-06-17", "group": "J", "matchday": 1, "home": "ARGENTINA",          "away": "ALGERIA",            "venue": "Hard Rock Stadium, Miami",         "stage": "GROUP"},
    {"date": "2026-06-17", "group": "J", "matchday": 1, "home": "AUSTRIA",            "away": "JORDAN",             "venue": "NRG Stadium, Houston",            "stage": "GROUP"},
    # Group K
    {"date": "2026-06-17", "group": "K", "matchday": 1, "home": "PORTUGAL",           "away": "CONGO DR",           "venue": "Arrowhead Stadium, Kansas City",   "stage": "GROUP"},
    {"date": "2026-06-17", "group": "K", "matchday": 1, "home": "UZBEKISTAN",         "away": "COLOMBIA",           "venue": "Estadio BBVA, Monterrey",         "stage": "GROUP"},
    # Group L
    {"date": "2026-06-17", "group": "L", "matchday": 1, "home": "ENGLAND",            "away": "CROATIA",            "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    {"date": "2026-06-17", "group": "L", "matchday": 1, "home": "GHANA",              "away": "PANAMA",             "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},

    # ── MATCHDAY 2 (June 18–24) ──────────────────────────────────────────────
    # Group A
    {"date": "2026-06-18", "group": "A", "matchday": 2, "home": "CZECHIA",            "away": "SOUTH AFRICA",       "venue": "Levi's Stadium, San Jose",        "stage": "GROUP"},
    {"date": "2026-06-19", "group": "A", "matchday": 2, "home": "MEXICO",             "away": "SOUTH KOREA",        "venue": "SoFi Stadium, Los Angeles",       "stage": "GROUP"},
    # Group B
    {"date": "2026-06-18", "group": "B", "matchday": 2, "home": "SWITZERLAND",        "away": "BOSNIA-HERZEGOVINA", "venue": "Empower Field, Denver",           "stage": "GROUP"},
    {"date": "2026-06-18", "group": "B", "matchday": 2, "home": "CANADA",             "away": "QATAR",              "venue": "BC Place, Vancouver",             "stage": "GROUP"},
    # Group C
    {"date": "2026-06-19", "group": "C", "matchday": 2, "home": "SCOTLAND",           "away": "MOROCCO",            "venue": "Hard Rock Stadium, Miami",         "stage": "GROUP"},
    {"date": "2026-06-20", "group": "C", "matchday": 2, "home": "BRAZIL",             "away": "HAITI",              "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    # Group D
    {"date": "2026-06-19", "group": "D", "matchday": 2, "home": "USA",               "away": "AUSTRALIA",          "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-20", "group": "D", "matchday": 2, "home": "TURKEY",             "away": "PARAGUAY",           "venue": "Arrowhead Stadium, Kansas City",   "stage": "GROUP"},
    # Group E
    {"date": "2026-06-20", "group": "E", "matchday": 2, "home": "GERMANY",            "away": "IVORY COAST",        "venue": "Lincoln Financial, Philadelphia",  "stage": "GROUP"},
    {"date": "2026-06-21", "group": "E", "matchday": 2, "home": "ECUADOR",            "away": "CURACAO",            "venue": "NRG Stadium, Houston",            "stage": "GROUP"},
    # Group F
    {"date": "2026-06-20", "group": "F", "matchday": 2, "home": "NETHERLANDS",        "away": "SWEDEN",             "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-21", "group": "F", "matchday": 2, "home": "TUNISIA",            "away": "JAPAN",              "venue": "Empower Field, Denver",           "stage": "GROUP"},
    # Group G
    {"date": "2026-06-21", "group": "G", "matchday": 2, "home": "BELGIUM",            "away": "IRAN",               "venue": "Gillette Stadium, Boston",        "stage": "GROUP"},
    {"date": "2026-06-22", "group": "G", "matchday": 2, "home": "NEW ZEALAND",        "away": "EGYPT",              "venue": "State Farm Stadium, Glendale",    "stage": "GROUP"},
    # Group H
    {"date": "2026-06-21", "group": "H", "matchday": 2, "home": "SPAIN",              "away": "SAUDI ARABIA",       "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    {"date": "2026-06-21", "group": "H", "matchday": 2, "home": "URUGUAY",            "away": "CAPE VERDE ISLANDS", "venue": "SoFi Stadium, Los Angeles",       "stage": "GROUP"},
    # Group I
    {"date": "2026-06-22", "group": "I", "matchday": 2, "home": "FRANCE",             "away": "IRAQ",               "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-23", "group": "I", "matchday": 2, "home": "NORWAY",             "away": "SENEGAL",            "venue": "Estadio Azteca, Mexico City",      "stage": "GROUP"},
    # Group J
    {"date": "2026-06-22", "group": "J", "matchday": 2, "home": "ARGENTINA",          "away": "AUSTRIA",            "venue": "Hard Rock Stadium, Miami",         "stage": "GROUP"},
    {"date": "2026-06-23", "group": "J", "matchday": 2, "home": "JORDAN",             "away": "ALGERIA",            "venue": "NRG Stadium, Houston",            "stage": "GROUP"},
    # Group K
    {"date": "2026-06-23", "group": "K", "matchday": 2, "home": "PORTUGAL",           "away": "UZBEKISTAN",         "venue": "Arrowhead Stadium, Kansas City",   "stage": "GROUP"},
    {"date": "2026-06-24", "group": "K", "matchday": 2, "home": "COLOMBIA",           "away": "CONGO DR",           "venue": "Estadio BBVA, Monterrey",         "stage": "GROUP"},
    # Group L
    {"date": "2026-06-23", "group": "L", "matchday": 2, "home": "ENGLAND",            "away": "GHANA",              "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    {"date": "2026-06-23", "group": "L", "matchday": 2, "home": "PANAMA",             "away": "CROATIA",            "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},

    # ── MATCHDAY 3 (June 24–28) ──────────────────────────────────────────────
    # Group A (simultaneous)
    {"date": "2026-06-25", "group": "A", "matchday": 3, "home": "CZECHIA",            "away": "MEXICO",             "venue": "Levi's Stadium, San Jose",        "stage": "GROUP"},
    {"date": "2026-06-25", "group": "A", "matchday": 3, "home": "SOUTH AFRICA",       "away": "SOUTH KOREA",        "venue": "SoFi Stadium, Los Angeles",       "stage": "GROUP"},
    # Group B (simultaneous)
    {"date": "2026-06-24", "group": "B", "matchday": 3, "home": "SWITZERLAND",        "away": "CANADA",             "venue": "Empower Field, Denver",           "stage": "GROUP"},
    {"date": "2026-06-24", "group": "B", "matchday": 3, "home": "BOSNIA-HERZEGOVINA", "away": "QATAR",              "venue": "BC Place, Vancouver",             "stage": "GROUP"},
    # Group C (simultaneous)
    {"date": "2026-06-24", "group": "C", "matchday": 3, "home": "MOROCCO",            "away": "HAITI",              "venue": "Hard Rock Stadium, Miami",         "stage": "GROUP"},
    {"date": "2026-06-24", "group": "C", "matchday": 3, "home": "SCOTLAND",           "away": "BRAZIL",             "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    # Group D (simultaneous)
    {"date": "2026-06-26", "group": "D", "matchday": 3, "home": "TURKEY",             "away": "USA",               "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-26", "group": "D", "matchday": 3, "home": "PARAGUAY",           "away": "AUSTRALIA",          "venue": "Arrowhead Stadium, Kansas City",   "stage": "GROUP"},
    # Group E (simultaneous)
    {"date": "2026-06-25", "group": "E", "matchday": 3, "home": "ECUADOR",            "away": "GERMANY",            "venue": "Lincoln Financial, Philadelphia",  "stage": "GROUP"},
    {"date": "2026-06-25", "group": "E", "matchday": 3, "home": "CURACAO",            "away": "IVORY COAST",        "venue": "NRG Stadium, Houston",            "stage": "GROUP"},
    # Group F (simultaneous)
    {"date": "2026-06-25", "group": "F", "matchday": 3, "home": "TUNISIA",            "away": "NETHERLANDS",        "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-25", "group": "F", "matchday": 3, "home": "JAPAN",              "away": "SWEDEN",             "venue": "Empower Field, Denver",           "stage": "GROUP"},
    # Group G (simultaneous)
    {"date": "2026-06-27", "group": "G", "matchday": 3, "home": "NEW ZEALAND",        "away": "BELGIUM",            "venue": "Gillette Stadium, Boston",        "stage": "GROUP"},
    {"date": "2026-06-27", "group": "G", "matchday": 3, "home": "EGYPT",              "away": "IRAN",               "venue": "State Farm Stadium, Glendale",    "stage": "GROUP"},
    # Group H (simultaneous)
    {"date": "2026-06-27", "group": "H", "matchday": 3, "home": "URUGUAY",            "away": "SPAIN",              "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    {"date": "2026-06-27", "group": "H", "matchday": 3, "home": "CAPE VERDE ISLANDS", "away": "SAUDI ARABIA",       "venue": "SoFi Stadium, Los Angeles",       "stage": "GROUP"},
    # Group I (simultaneous)
    {"date": "2026-06-26", "group": "I", "matchday": 3, "home": "NORWAY",             "away": "FRANCE",             "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
    {"date": "2026-06-26", "group": "I", "matchday": 3, "home": "SENEGAL",            "away": "IRAQ",               "venue": "Estadio Azteca, Mexico City",      "stage": "GROUP"},
    # Group J (simultaneous)
    {"date": "2026-06-28", "group": "J", "matchday": 3, "home": "JORDAN",             "away": "ARGENTINA",          "venue": "Hard Rock Stadium, Miami",         "stage": "GROUP"},
    {"date": "2026-06-28", "group": "J", "matchday": 3, "home": "ALGERIA",            "away": "AUSTRIA",            "venue": "NRG Stadium, Houston",            "stage": "GROUP"},
    # Group K (simultaneous)
    {"date": "2026-06-27", "group": "K", "matchday": 3, "home": "COLOMBIA",           "away": "PORTUGAL",           "venue": "Arrowhead Stadium, Kansas City",   "stage": "GROUP"},
    {"date": "2026-06-27", "group": "K", "matchday": 3, "home": "CONGO DR",           "away": "UZBEKISTAN",         "venue": "Estadio BBVA, Monterrey",         "stage": "GROUP"},
    # Group L (simultaneous)
    {"date": "2026-06-27", "group": "L", "matchday": 3, "home": "PANAMA",             "away": "ENGLAND",            "venue": "MetLife Stadium, New York",        "stage": "GROUP"},
    {"date": "2026-06-27", "group": "L", "matchday": 3, "home": "CROATIA",            "away": "GHANA",              "venue": "AT&T Stadium, Dallas",            "stage": "GROUP"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RESULTS SO FAR (Matchday 1, June 11–17)
# Real scores from football-data.org API.
# UPDATE AFTER EACH MATCH DAY!
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_SO_FAR: list[dict] = [
    # Group A — June 11–12
    {"date": "2026-06-11", "home": "MEXICO",      "away": "SOUTH AFRICA",       "home_score": 2, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-12", "home": "SOUTH KOREA", "away": "CZECHIA",            "home_score": 2, "away_score": 1, "stage": "GROUP"},
    # Group B — June 12–13
    {"date": "2026-06-12", "home": "CANADA",      "away": "BOSNIA-HERZEGOVINA", "home_score": 1, "away_score": 1, "stage": "GROUP"},
    {"date": "2026-06-13", "home": "QATAR",       "away": "SWITZERLAND",        "home_score": 1, "away_score": 1, "stage": "GROUP"},
    # Group C — June 13–14
    {"date": "2026-06-13", "home": "BRAZIL",      "away": "MOROCCO",            "home_score": 1, "away_score": 1, "stage": "GROUP"},
    {"date": "2026-06-14", "home": "HAITI",       "away": "SCOTLAND",           "home_score": 0, "away_score": 1, "stage": "GROUP"},
    # Group D — June 13–14
    {"date": "2026-06-13", "home": "USA",         "away": "PARAGUAY",           "home_score": 4, "away_score": 1, "stage": "GROUP"},
    {"date": "2026-06-14", "home": "AUSTRALIA",   "away": "TURKEY",             "home_score": 2, "away_score": 0, "stage": "GROUP"},
    # Group E — June 14
    {"date": "2026-06-14", "home": "GERMANY",     "away": "CURACAO",            "home_score": 7, "away_score": 1, "stage": "GROUP"},
    {"date": "2026-06-14", "home": "IVORY COAST", "away": "ECUADOR",            "home_score": 1, "away_score": 0, "stage": "GROUP"},
    # Group F — June 14–15
    {"date": "2026-06-14", "home": "NETHERLANDS", "away": "JAPAN",              "home_score": 2, "away_score": 2, "stage": "GROUP"},
    {"date": "2026-06-15", "home": "SWEDEN",      "away": "TUNISIA",            "home_score": 5, "away_score": 1, "stage": "GROUP"},
    # Group G — June 15–16
    {"date": "2026-06-15", "home": "BELGIUM",     "away": "EGYPT",              "home_score": 1, "away_score": 1, "stage": "GROUP"},
    {"date": "2026-06-16", "home": "IRAN",        "away": "NEW ZEALAND",        "home_score": 2, "away_score": 2, "stage": "GROUP"},
    # Group H — June 15
    {"date": "2026-06-15", "home": "SPAIN",              "away": "CAPE VERDE ISLANDS", "home_score": 0, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-15", "home": "SAUDI ARABIA",       "away": "URUGUAY",            "home_score": 1, "away_score": 1, "stage": "GROUP"},
    # Group I — June 16
    {"date": "2026-06-16", "home": "FRANCE",      "away": "SENEGAL",            "home_score": 3, "away_score": 1, "stage": "GROUP"},
    {"date": "2026-06-16", "home": "IRAQ",        "away": "NORWAY",             "home_score": 1, "away_score": 4, "stage": "GROUP"},
    # Group J — June 17
    {"date": "2026-06-17", "home": "ARGENTINA",   "away": "ALGERIA",            "home_score": 3, "away_score": 0, "stage": "GROUP"},
    {"date": "2026-06-17", "home": "AUSTRIA",     "away": "JORDAN",             "home_score": 3, "away_score": 1, "stage": "GROUP"},
    # Group K — June 17
    {"date": "2026-06-17", "home": "PORTUGAL",    "away": "CONGO DR",           "home_score": 1, "away_score": 1, "stage": "GROUP"},
    # Group L — June 17
    {"date": "2026-06-17", "home": "ENGLAND",     "away": "CROATIA",            "home_score": 4, "away_score": 2, "stage": "GROUP"},
    # Note: Ghana vs Panama and Uzbekistan vs Colombia still in progress on June 17
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: DEMO ODDS (Matchday 2 upcoming matches — ELO-based estimates)
# Used when no Odds API key is configured.
# ─────────────────────────────────────────────────────────────────────────────
DEMO_ODDS: dict[str, dict[str, float]] = {
    # Group A MD2 (June 18–19)
    "CZECHIA_vs_SOUTH AFRICA":        {"home": 2.00, "draw": 3.20, "away": 3.90, "over_25": 1.95, "btts": 2.05},
    "MEXICO_vs_SOUTH KOREA":          {"home": 2.15, "draw": 3.20, "away": 3.50, "over_25": 1.88, "btts": 1.95},
    # Group B MD2 (June 18)
    "SWITZERLAND_vs_BOSNIA-HERZEGOVINA": {"home": 1.72, "draw": 3.50, "away": 5.50, "over_25": 1.78, "btts": 1.88},
    "CANADA_vs_QATAR":                {"home": 1.90, "draw": 3.40, "away": 4.20, "over_25": 1.88, "btts": 1.98},
    # Group C MD2 (June 19–20)
    "SCOTLAND_vs_MOROCCO":            {"home": 3.20, "draw": 3.10, "away": 2.40, "over_25": 1.92, "btts": 2.00},
    "BRAZIL_vs_HAITI":                {"home": 1.15, "draw": 6.00, "away": 18.0, "over_25": 1.55, "btts": 1.80},
    # Group D MD2 (June 19–20)
    "USA_vs_AUSTRALIA":               {"home": 1.85, "draw": 3.50, "away": 4.50, "over_25": 1.85, "btts": 1.95},
    "TURKEY_vs_PARAGUAY":             {"home": 2.10, "draw": 3.20, "away": 3.80, "over_25": 1.95, "btts": 2.05},
    # Group E MD2 (June 20–21)
    "GERMANY_vs_IVORY COAST":         {"home": 1.38, "draw": 4.80, "away": 9.00, "over_25": 1.60, "btts": 1.85},
    "ECUADOR_vs_CURACAO":             {"home": 1.50, "draw": 4.00, "away": 7.00, "over_25": 1.72, "btts": 1.90},
    # Group F MD2 (June 20–21)
    "NETHERLANDS_vs_SWEDEN":          {"home": 1.62, "draw": 3.80, "away": 5.50, "over_25": 1.72, "btts": 1.88},
    "TUNISIA_vs_JAPAN":               {"home": 3.40, "draw": 3.10, "away": 2.20, "over_25": 1.90, "btts": 2.00},
    # Group G MD2 (June 21–22)
    "BELGIUM_vs_IRAN":                {"home": 1.52, "draw": 4.00, "away": 7.00, "over_25": 1.75, "btts": 1.92},
    "NEW ZEALAND_vs_EGYPT":           {"home": 3.50, "draw": 3.00, "away": 2.30, "over_25": 2.00, "btts": 2.10},
    # Group H MD2 (June 21)
    "SPAIN_vs_SAUDI ARABIA":          {"home": 1.28, "draw": 5.50, "away": 12.0, "over_25": 1.65, "btts": 1.92},
    "URUGUAY_vs_CAPE VERDE ISLANDS":  {"home": 1.35, "draw": 5.00, "away": 9.00, "over_25": 1.78, "btts": 2.05},
    # Group I MD2 (June 22–23)
    "FRANCE_vs_IRAQ":                 {"home": 1.22, "draw": 6.00, "away": 14.0, "over_25": 1.60, "btts": 1.88},
    "NORWAY_vs_SENEGAL":              {"home": 2.60, "draw": 3.20, "away": 2.80, "over_25": 1.92, "btts": 2.00},
    # Group J MD2 (June 22–23)
    "ARGENTINA_vs_AUSTRIA":           {"home": 1.48, "draw": 4.20, "away": 7.50, "over_25": 1.78, "btts": 1.95},
    "JORDAN_vs_ALGERIA":              {"home": 3.20, "draw": 3.00, "away": 2.60, "over_25": 2.00, "btts": 2.10},
    # Group K MD2 (June 23–24)
    "PORTUGAL_vs_UZBEKISTAN":         {"home": 1.30, "draw": 5.20, "away": 11.0, "over_25": 1.68, "btts": 1.90},
    "COLOMBIA_vs_CONGO DR":           {"home": 1.55, "draw": 3.80, "away": 6.50, "over_25": 1.82, "btts": 1.98},
    # Group L MD2 (June 23)
    "ENGLAND_vs_GHANA":               {"home": 1.35, "draw": 5.00, "away": 9.00, "over_25": 1.72, "btts": 1.90},
    "PANAMA_vs_CROATIA":              {"home": 5.00, "draw": 3.40, "away": 1.75, "over_25": 1.85, "btts": 2.00},
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CURRENT_STAGE: str = "GROUP"

STAGE_ORDER: list[str] = ["GROUP", "R32", "R16", "QF", "SF", "FINAL"]
STAGE_LABELS: dict[str, str] = {
    "GROUP": "Group Stage",
    "R32":   "Round of 32",
    "R16":   "Round of 16",
    "QF":    "Quarterfinal",
    "SF":    "Semifinal",
    "FINAL": "Final",
}

ALL_TEAMS: list[str] = sorted({t for teams in GROUPS.values() for t in teams})


def get_team_group(team: str) -> str | None:
    """Return the group letter for a given team name, or None if not found."""
    for group, teams in GROUPS.items():
        if team in teams:
            return group
    return None


def get_demo_odds(home: str, away: str) -> dict[str, float] | None:
    """Look up demo odds for a match by team names (tries both directions)."""
    key = f"{home}_vs_{away}"
    if key in DEMO_ODDS:
        return DEMO_ODDS[key]

    rev_key = f"{away}_vs_{home}"
    if rev_key in DEMO_ODDS:
        odds = DEMO_ODDS[rev_key].copy()
        odds["home"], odds["away"] = odds["away"], odds["home"]
        return odds

    return None


def upcoming_matches(as_of_date: str = "2026-06-17") -> list[dict]:
    """Return unplayed matches on or after as_of_date."""
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
