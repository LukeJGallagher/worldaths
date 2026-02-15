"""
Common filter widgets for the dashboard.

Provides consistent event picker, gender toggle, championship selector,
age category filter, and country filter across all pages.
"""

from typing import Optional, List, Tuple

import streamlit as st

from data.event_utils import (
    EVENT_CATEGORIES,
    MENS_ONLY,
    WOMENS_ONLY,
    get_events_for_gender,
    ASIAN_COUNTRY_CODES,
)


def event_gender_picker(
    key_prefix: str = "main",
    default_event: str = "100m",
    default_gender: str = "M",
    show_gender: bool = True,
) -> Tuple[str, str]:
    """Event picker grouped by category + gender toggle.

    Returns: (event_display_name, gender_code)
    """
    col1, col2 = st.columns([3, 1]) if show_gender else (st.container(), None)

    with col1:
        # Build grouped options
        gender = default_gender
        if show_gender and col2:
            with col2:
                gender = st.radio(
                    "Gender", ["M", "F"],
                    index=0 if default_gender == "M" else 1,
                    horizontal=True,
                    key=f"{key_prefix}_gender",
                )

        available_events = get_events_for_gender(gender)

        # Group by category
        event_options = []
        for category, events in EVENT_CATEGORIES.items():
            for event in events:
                if event in available_events:
                    event_options.append(event)

        default_idx = event_options.index(default_event) if default_event in event_options else 0
        event = st.selectbox(
            "Event",
            event_options,
            index=default_idx,
            key=f"{key_prefix}_event",
        )

    return event, gender


def championship_selector(key: str = "championship") -> str:
    """Dropdown for major championship selection."""
    championships = [
        "Asian Games 2026",
        "World Championships 2025",
        "Olympics 2028",
        "Asian Games 2023",
        "World Championships 2023",
        "Olympics 2024",
    ]
    return st.selectbox("Championship", championships, key=key)


def age_category_filter(key: str = "age_cat") -> Optional[str]:
    """Age category filter for toplists."""
    options = ["All Ages", "U20", "U23", "Senior"]
    selected = st.selectbox("Age Category", options, key=key)
    return None if selected == "All Ages" else selected


def country_filter(key: str = "country", default: str = "All") -> Optional[str]:
    """Country filter with common athletics nations."""
    common_countries = [
        "All", "KSA", "---",
        "USA", "GBR", "JPN", "CHN", "KEN", "ETH", "JAM",
        "GER", "FRA", "AUS", "ITA", "NED", "CAN", "BRA",
        "---",
        "QAT", "BRN", "UAE", "KUW", "OMA", "JOR", "IRQ",
        "IND", "CHN", "JPN", "KOR",
    ]
    # Remove separator duplicates
    clean = []
    for c in common_countries:
        if c == "---":
            if clean and clean[-1] != "---":
                clean.append(c)
        else:
            clean.append(c)

    selected = st.selectbox("Country", clean, key=key)
    if selected == "All" or selected == "---":
        return None
    return selected


def region_filter(key: str = "region") -> Optional[str]:
    """Region filter for Asian/Global context."""
    options = ["Global", "Asia", "Europe", "Africa", "Americas", "Oceania"]
    selected = st.selectbox("Region", options, key=key)
    return None if selected == "Global" else selected.lower()


def date_range_filter(key: str = "dates") -> Tuple[Optional[str], Optional[str]]:
    """Date range picker."""
    from datetime import datetime, timedelta

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "From",
            value=datetime.now() - timedelta(days=365),
            key=f"{key}_start",
        )
    with col2:
        end = st.date_input(
            "To",
            value=datetime.now() + timedelta(days=365),
            key=f"{key}_end",
        )
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
