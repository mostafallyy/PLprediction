"""
Team name normalization shared by both data sources.

football-data.org gives full club names ("Manchester City FC"); the
Kaggle historical CSV gives short/press names ("Man City"). Neither
source shares a numeric team id with the other, so the rolling-form
and head-to-head joins in gold_features.py are keyed on a normalized
`team_key` string derived here instead of on football-data.org's
numeric team_id (which stays around, nullable, purely for serving/UI -
crest lookups, the /team/{id} and /h2h/{id}/{id} endpoints).

normalize() strips punctuation/suffixes and applies a small alias
table for the handful of clubs whose short name doesn't collapse to
the same string as their full name (Man City / Manchester City,
Spurs / Tottenham, Wolves / Wolverhampton, etc). Every club that has
ever played in the Premier League since 2000 is covered.
"""

import re

_ALIASES = {
    "MANCITY": "MANCHESTERCITY",
    "MANUNITED": "MANCHESTERUNITED",
    "MANUTD": "MANCHESTERUNITED",
    "SPURS": "TOTTENHAMHOTSPUR",
    "TOTTENHAM": "TOTTENHAMHOTSPUR",
    "WOLVES": "WOLVERHAMPTONWANDERERS",
    "NOTTMFOREST": "NOTTINGHAMFOREST",
    "NOTTINGHAMFOREST": "NOTTINGHAMFOREST",
    "WESTBROM": "WESTBROMWICHALBION",
    "WESTBROMWICH": "WESTBROMWICHALBION",
    "WESTHAM": "WESTHAMUNITED",
    "NEWCASTLE": "NEWCASTLEUNITED",
    "LEICESTER": "LEICESTERCITY",
    "LEEDS": "LEEDSUNITED",
    "NORWICH": "NORWICHCITY",
    "STOKE": "STOKECITY",
    "SWANSEA": "SWANSEACITY",
    "HULL": "HULLCITY",
    "CARDIFF": "CARDIFFCITY",
    "BIRMINGHAM": "BIRMINGHAMCITY",
    "CHARLTON": "CHARLTONATHLETIC",
    "BOLTON": "BOLTONWANDERERS",
    "BLACKBURN": "BLACKBURNROVERS",
    "BRADFORD": "BRADFORDCITY",
    "COVENTRY": "COVENTRYCITY",
    "DERBY": "DERBYCOUNTY",
    "IPSWICH": "IPSWICHTOWN",
    "LEEDSUNITED": "LEEDSUNITED",
    "HUDDERSFIELD": "HUDDERSFIELDTOWN",
    "BRIGHTON": "BRIGHTONHOVEALBION",
    "SHEFFIELDUNITED": "SHEFFIELDUNITED",
    "SHEFFIELDWEDNESDAY": "SHEFFIELDWEDNESDAY",
    "QPR": "QUEENSPARKRANGERS",
    "WIGAN": "WIGANATHLETIC",
    "PORTSMOUTH": "PORTSMOUTH",
    "MIDDLESBROUGH": "MIDDLESBROUGH",
    "SUNDERLAND": "SUNDERLAND",
    "BOURNEMOUTH": "AFCBOURNEMOUTH",
    "LUTON": "LUTONTOWN",
    "BURNLEY": "BURNLEY",
    "FULHAM": "FULHAM",
    "EVERTON": "EVERTON",
    "LIVERPOOL": "LIVERPOOL",
    "ARSENAL": "ARSENAL",
    "CHELSEA": "CHELSEA",
    "ASTONVILLA": "ASTONVILLA",
    "CRYSTALPALACE": "CRYSTALPALACE",
    "SOUTHAMPTON": "SOUTHAMPTON",
    "WATFORD": "WATFORD",
    "WESTHAMUNITED": "WESTHAMUNITED",
    "BLACKPOOL": "BLACKPOOL",
    "READING": "READING",
    "BRENTFORD": "BRENTFORD",
}

_SUFFIX_RE = re.compile(r"\b(FC|AFC)\b")
_NONALNUM_RE = re.compile(r"[^A-Z0-9]")


def normalize(name: str) -> str:
    """Collapse a club name from either source down to a stable join key."""
    if not name:
        return ""
    key = name.upper()
    key = _SUFFIX_RE.sub("", key)
    key = _NONALNUM_RE.sub("", key)
    return _ALIASES.get(key, key)
