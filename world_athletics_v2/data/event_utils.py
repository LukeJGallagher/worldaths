"""
Event name normalization and utilities.

Canonical source for event name handling across the entire project.
Ported from what_it_takes_to_win.py with improvements.

IMPORTANT: NEVER use str.contains() for event filtering.
ALWAYS use normalize_event_for_match() then exact ==.
"""

import re
from typing import Optional


# ── Bidirectional mapping: DB format <-> Display format ────────────────

EVENT_DB_TO_DISPLAY = {
    # Sprints
    "100-metres": "100m",
    "200-metres": "200m",
    "400-metres": "400m",
    # Middle Distance
    "800-metres": "800m",
    "1500-metres": "1500m",
    "mile": "Mile",
    "3000-metres": "3000m",
    # Long Distance
    "5000-metres": "5000m",
    "10000-metres": "10,000m",
    "marathon": "Marathon",
    "half-marathon": "Half Marathon",
    "3000-metres-steeplechase": "3000m SC",
    # Hurdles
    "100-metres-hurdles": "100m H",
    "110-metres-hurdles": "110m H",
    "400-metres-hurdles": "400m H",
    # Jumps
    "high-jump": "High Jump",
    "pole-vault": "Pole Vault",
    "long-jump": "Long Jump",
    "triple-jump": "Triple Jump",
    # Throws
    "shot-put": "Shot Put",
    "discus-throw": "Discus",
    "hammer-throw": "Hammer",
    "javelin-throw": "Javelin",
    # Combined
    "decathlon": "Decathlon",
    "heptathlon": "Heptathlon",
    # Walks
    "20-kilometres-race-walk": "20km Walk",
    "35-kilometres-race-walk": "35km Walk",
    "race-walk-mixed-relay": "Walk Mixed Relay",
    # Relays
    "4x100-metres-relay": "4x100m",
    "4x400-metres-relay": "4x400m",
    "4x400-metres-relay-mixed": "4x400m Mixed",
    # Short track (indoor variants in master data)
    "400-metres-short-track": "400m (ST)",
    "800-metres-short-track": "800m (ST)",
    "1500-metres-short-track": "1500m (ST)",
    "5000-metres-short-track": "5000m (ST)",
}

EVENT_DISPLAY_TO_DB = {v: k for k, v in EVENT_DB_TO_DISPLAY.items()}

# WA GraphQL API returns discipline names in this format.
# Map them to our display format so format_event_name() works.
_API_DISCIPLINE_TO_DISPLAY = {
    "100 metres": "100m",
    "150 metres": "150m",
    "200 metres": "200m",
    "300 metres": "300m",
    "400 metres": "400m",
    "600 metres": "600m",
    "800 metres": "800m",
    "1500 metres": "1500m",
    "3000 metres": "3000m",
    "5000 metres": "5000m",
    "10000 metres": "10,000m",
    "60 metres": "60m",
    "100 metres hurdles": "100m H",
    "110 metres hurdles": "110m H",
    "300 metres hurdles": "300m H",
    "400 metres hurdles": "400m H",
    "60 metres hurdles": "60m H",
    "3000 metres steeplechase": "3000m SC",
    "high jump": "High Jump",
    "pole vault": "Pole Vault",
    "long jump": "Long Jump",
    "triple jump": "Triple Jump",
    "shot put": "Shot Put",
    "discus throw": "Discus",
    "hammer throw": "Hammer",
    "javelin throw": "Javelin",
    "decathlon": "Decathlon",
    "heptathlon": "Heptathlon",
    "marathon": "Marathon",
    "half marathon": "Half Marathon",
    "4x100 metres relay": "4x100m",
    "4x400 metres relay": "4x400m",
    "4x400 metres relay mixed": "4x400m Mixed",
    "4x400 metres relay short track": "4x400m (ST)",
    "400 metres short track": "400m (ST)",
    "800 metres short track": "800m (ST)",
    "1500 metres short track": "1500m (ST)",
    "5000 metres short track": "5000m (ST)",
    "sprint medley 1000m": "Sprint Medley",
    # Variants with equipment specifications (just map to base event)
    "110 metres hurdles (99.0cm)": "110m H",
    "110 metres hurdles (91.4cm)": "110m H",
    "400m hurdles (84.0cm)": "400m H",
    "shot put (6kg)": "Shot Put",
    # Shorthand formats from WA API primary_event field (e.g. "Men's 110mH")
    "100mh": "100m H",
    "110mh": "110m H",
    "400mh": "400m H",
    "3000msc": "3000m SC",
    "overall ranking": "Overall Ranking",
}

# GraphQL API discipline codes (used by GetTopList, GetWorldRankings, etc.)
DISCIPLINE_CODES = {
    "100m": "100",
    "200m": "200",
    "400m": "400",
    "800m": "800",
    "1500m": "1500",
    "Mile": "MILE",
    "3000m": "3000",
    "5000m": "5000",
    "10,000m": "10K",
    "Marathon": "MAR",
    "Half Marathon": "HMAR",
    "3000m SC": "3KSC",
    "100m H": "100H",
    "110m H": "110H",
    "400m H": "400H",
    "High Jump": "HJ",
    "Pole Vault": "PV",
    "Long Jump": "LJ",
    "Triple Jump": "TJ",
    "Shot Put": "SP",
    "Discus": "DT",
    "Hammer": "HT",
    "Javelin": "JT",
    "Decathlon": "DEC",
    "Heptathlon": "HEP",
    "20km Walk": "20KW",
    "35km Walk": "35KW",
    "4x100m": "4X1",
    "4x400m": "4X4",
}

# Event group codes for GetWorldRankings
# NOTE: API is case-sensitive. All slugs must be lowercase.
# NOTE: getWorldRankings only returns Women's data with our API key.
#       Men's-only events (110m H, Decathlon) will return no data.
EVENT_GROUPS = {
    "100m": "100m",
    "200m": "200m",
    "400m": "400m",
    "800m": "800m",
    "1500m": "1500m",
    "5000m": "5000m",
    "10,000m": "10000m",
    "Marathon": "marathon",
    "3000m SC": "3000msc",
    "100m H": "100mh",
    "110m H": "110mh",       # Men's only - will fail on Women's-only API
    "400m H": "400mh",
    "High Jump": "high-jump",
    "Pole Vault": "pole-vault",
    "Long Jump": "long-jump",
    "Triple Jump": "triple-jump",
    "Shot Put": "shot-put",
    "Discus": "discus-throw",
    "Hammer": "hammer-throw",
    "Javelin": "javelin-throw",
    "Decathlon": "decathlon",  # Men's only - will fail on Women's-only API
    "Heptathlon": "heptathlon",
    "20km Walk": "race-walking",
    "35km Walk": "race-walking",  # Same group as 20km
    # Additional event groups available:
    # "Road Running": "road-running",
    # "Cross Country": "cross-country",
}

# Events grouped by category (for UI event picker)
EVENT_CATEGORIES = {
    "Sprints": ["100m", "200m", "400m"],
    "Middle Distance": ["800m", "1500m", "Mile", "3000m"],
    "Long Distance": ["5000m", "10,000m", "Marathon", "Half Marathon", "3000m SC"],
    "Hurdles": ["100m H", "110m H", "400m H"],
    "Jumps": ["High Jump", "Pole Vault", "Long Jump", "Triple Jump"],
    "Throws": ["Shot Put", "Discus", "Hammer", "Javelin"],
    "Combined": ["Decathlon", "Heptathlon"],
    "Walks": ["20km Walk", "35km Walk"],
    "Relays": ["4x100m", "4x400m", "4x400m Mixed"],
}

# Gender-specific events
MENS_ONLY = {"110m H", "Decathlon"}
WOMENS_ONLY = {"100m H", "Heptathlon"}


def normalize_event_for_match(event: str) -> str:
    """Strip all non-alphanumeric characters and lowercase for exact matching.

    ALWAYS use this + == for event filtering. NEVER use str.contains().

    Examples:
        "100-metres" -> "100metres"
        "100 Metres" -> "100metres"
        "100m" -> "100m"
    """
    return re.sub(r'[^0-9a-z]', '', event.lower())


def _strip_gender_prefix(name: str) -> str:
    """Strip 'Men's ' / 'Women's ' prefix from event names."""
    for prefix in ("Men's ", "Women's ", "men's ", "women's "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def format_event_name(db_name: str) -> str:
    """Convert DB format or API format to display format.

    Examples:
        "100-metres" -> "100m"
        "100 Metres" -> "100m"
        "high-jump" -> "High Jump"
        "Men's 100m" -> "100m"
        "Women's 400m Hurdles" -> "400m H"
    """
    # Strip gender prefix (e.g. "Men's 100m" -> "100m")
    cleaned = _strip_gender_prefix(db_name)
    # Try DB format first (e.g. "100-metres")
    if cleaned in EVENT_DB_TO_DISPLAY:
        return EVENT_DB_TO_DISPLAY[cleaned]
    # Try WA API format (e.g. "100 Metres")
    api_key = cleaned.lower().strip()
    if api_key in _API_DISCIPLINE_TO_DISPLAY:
        return _API_DISCIPLINE_TO_DISPLAY[api_key]
    # Try direct match (e.g. "100m" already in display format)
    if cleaned in EVENT_DISPLAY_TO_DB or cleaned in DISCIPLINE_CODES:
        return cleaned
    # Fallback: title case
    return cleaned.replace("-", " ").title()


def display_to_db(display_name: str) -> str:
    """Convert display format to DB format.

    Examples:
        "100m" -> "100-metres"
        "High Jump" -> "high-jump"
    """
    return EVENT_DISPLAY_TO_DB.get(display_name, display_name.lower().replace(" ", "-"))


def get_discipline_code(display_name: str) -> Optional[str]:
    """Get GraphQL API discipline code for toplist/ranking queries."""
    return DISCIPLINE_CODES.get(display_name)


def get_event_group(display_name: str) -> Optional[str]:
    """Get event group code for GetWorldRankings query."""
    return EVENT_GROUPS.get(display_name)


def get_event_type(event: str) -> str:
    """Determine if lower is better (time) or higher is better (distance/points).

    Returns: 'time', 'distance', or 'points'
    """
    normalized = normalize_event_for_match(event)

    # Points-based (combined events)
    points_events = {"decathlon", "heptathlon"}
    for p in points_events:
        if p in normalized:
            return "points"

    # Distance/height events (higher is better)
    field_events = {"highjump", "polevault", "longjump", "triplejump",
                    "shotput", "discus", "hammer", "javelin"}
    for f in field_events:
        if f in normalized:
            return "distance"

    # Everything else is time-based (lower is better)
    return "time"


def get_events_for_gender(gender: str) -> list:
    """Get all events available for a gender."""
    gender_upper = gender.upper()
    all_events = []
    for category, events in EVENT_CATEGORIES.items():
        for event in events:
            if gender_upper == "M" and event in WOMENS_ONLY:
                continue
            if gender_upper == "F" and event in MENS_ONLY:
                continue
            all_events.append(event)
    return all_events


# Asian country codes (for Asian Games filtering)
ASIAN_COUNTRY_CODES = {
    "AFG", "BRN", "BAN", "BHU", "BRU", "CAM", "CHN", "TPE", "HKG",
    "IND", "INA", "IRI", "IRQ", "JPN", "JOR", "KAZ", "KUW", "KGZ",
    "LAO", "LBN", "MAS", "MDV", "MGL", "MYA", "NEP", "PRK", "OMA",
    "PAK", "PLE", "PHI", "QAT", "KSA", "SGP", "KOR", "SRI", "SYR",
    "TJK", "THA", "TLS", "TKM", "UAE", "UZB", "VIE", "YEM",
}
