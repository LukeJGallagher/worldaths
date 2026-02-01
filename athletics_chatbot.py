"""
Athletics AI Analyst Chatbot
World-class sports analysis powered by OpenRouter free models + RAG

Uses your athletics dataset (2.3M+ records) to provide intelligent analysis
for KSA athletes, competitors, qualification standards, and strategic insights.

Inspired by MCP_Analysis RAG architecture for robust retrieval.
"""

import streamlit as st
import pandas as pd
import os
import json
import requests
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import data connectors
try:
    from data_connector import (
        get_ksa_athletes, get_rankings_data, get_ksa_rankings,
        get_benchmarks_data, get_data_mode, query
    )
    DATA_CONNECTOR_AVAILABLE = True
except ImportError:
    DATA_CONNECTOR_AVAILABLE = False
    logging.warning("Data connector not available")

# Import analytics modules
try:
    from athletics_analytics_agents import (
        AthleticsAnalytics, DISCIPLINE_KNOWLEDGE, MAJOR_GAMES,
        COMPETITION_CATEGORIES
    )
    from what_it_takes_to_win import WhatItTakesToWin
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logging.warning("Analytics modules not available")

# Import elite athletics knowledge base
try:
    from elite_athletics_knowledge import (
        SPRINTS_KNOWLEDGE, MIDDLE_DISTANCE_KNOWLEDGE, LONG_DISTANCE_KNOWLEDGE,
        HURDLES_KNOWLEDGE, JUMPS_KNOWLEDGE, THROWS_KNOWLEDGE,
        COMBINED_EVENTS_KNOWLEDGE, ATHLETICS_TERMINOLOGY, ANALYSIS_FRAMEWORKS,
        get_event_knowledge, get_discipline_category, format_knowledge_for_context
    )
    ELITE_KNOWLEDGE_AVAILABLE = True
except ImportError:
    ELITE_KNOWLEDGE_AVAILABLE = False
    logging.warning("Elite athletics knowledge base not available")

###################################
# Team Saudi Brand Colors
###################################
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'
TEAL_LIGHT = '#009688'
GRAY_BLUE = '#78909C'

###################################
# OpenRouter Configuration
###################################
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models on OpenRouter (updated January 2025)
FREE_MODELS = {
    'deepseek/deepseek-r1-distill-llama-70b:free': 'DeepSeek R1 70B (Best Free)',
    'deepseek/deepseek-chat-v3-0324:free': 'DeepSeek Chat v3 (Fast)',
    'meta-llama/llama-3.1-8b-instruct:free': 'Llama 3.1 8B (Balanced)',
    'meta-llama/llama-3.2-3b-instruct:free': 'Llama 3.2 3B (Fastest)',
    'google/gemma-2-9b-it:free': 'Gemma 2 9B (Google)',
    'mistralai/mistral-7b-instruct:free': 'Mistral 7B (Efficient)',
    'qwen/qwen-2-7b-instruct:free': 'Qwen 2 7B (Multilingual)',
}

DEFAULT_MODEL = 'deepseek/deepseek-r1-distill-llama-70b:free'


def get_openrouter_key() -> Optional[str]:
    """Get OpenRouter API key from environment or Streamlit secrets."""
    # Try environment variable first
    key = os.getenv('OPENROUTER_API_KEY')

    # Try Streamlit secrets
    if not key:
        try:
            if hasattr(st, 'secrets'):
                if 'OPENROUTER_API_KEY' in st.secrets:
                    key = st.secrets['OPENROUTER_API_KEY']
                elif 'openrouter' in st.secrets:
                    key = st.secrets.openrouter.get('OPENROUTER_API_KEY')
        except:
            pass

    return key


###################################
# Athletics Knowledge Base
###################################
ATHLETICS_KNOWLEDGE_BASE = """
## World Athletics Ranking System
- Points based on performance and competition level
- OW (Olympic/World): 1.0x multiplier
- GL (Diamond League/Gold): 0.9x
- A (Continental Champs): 0.8x
- B (National Champs): 0.7x

## Competition Levels
- Tier 1: Olympics, World Championships
- Tier 2: Asian Games, Continental Championships, Diamond League
- Tier 3: Regional Championships (Arab, GCC, West Asian)
- Tier 4: National Championships, World U20

## Wind Rules
- Legal wind limit: ≤2.0 m/s for sprints and horizontal jumps
- Results with wind >2.0 are wind-assisted (w) and don't count for records

## Altitude Effects
- High altitude (>1000m) benefits sprints (less air resistance)
- High altitude disadvantages distance events (less oxygen)

## Terminology
- FAT: Fully Automatic Timing (electronic to 0.01s)
- PB: Personal Best, SB: Season Best, WL: World Lead
- WR: World Record, OR: Olympic Record, NR: National Record
- DNS: Did Not Start, DNF: Did Not Finish, DQ: Disqualified
"""


def get_elite_knowledge_for_event(event: str) -> str:
    """Get detailed elite knowledge for a specific event."""
    if not ELITE_KNOWLEDGE_AVAILABLE:
        return ""

    knowledge_text = format_knowledge_for_context(event, include_benchmarks=True)
    return knowledge_text if knowledge_text else ""


def get_discipline_context(discipline: str) -> str:
    """Get comprehensive context for a discipline category."""
    if not ELITE_KNOWLEDGE_AVAILABLE:
        return ""

    discipline_map = {
        'sprints': SPRINTS_KNOWLEDGE,
        'middle_distance': MIDDLE_DISTANCE_KNOWLEDGE,
        'long_distance': LONG_DISTANCE_KNOWLEDGE,
        'hurdles': HURDLES_KNOWLEDGE,
        'jumps': JUMPS_KNOWLEDGE,
        'throws': THROWS_KNOWLEDGE,
        'combined_events': COMBINED_EVENTS_KNOWLEDGE
    }

    knowledge = discipline_map.get(discipline.lower(), {})
    if not knowledge:
        return ""

    parts = [f"## {discipline.replace('_', ' ').title()} Expertise\n"]

    for event_key, event_data in knowledge.items():
        if isinstance(event_data, dict):
            parts.append(f"### {event_key.upper()}")
            if 'overview' in event_data:
                parts.append(event_data['overview'])
            if 'benchmarks' in event_data:
                parts.append("\n**Performance Standards:**")
                for gender, marks in event_data['benchmarks'].items():
                    if isinstance(marks, dict):
                        for level, mark in list(marks.items())[:4]:
                            parts.append(f"- {gender.title()} {level.replace('_', ' ')}: {mark}")

    return "\n".join(parts)


###################################
# Athletics Data Context Builder
###################################

# Singleton instance for context builder (prevents repeated data loading)
_CONTEXT_BUILDER_INSTANCE = None


def get_context_builder():
    """
    Get singleton instance of AthleticsContextBuilder.

    This avoids repeated data loading when the chatbot is reinitialized.
    The data connector already caches the parquet files, but this prevents
    redundant initialization logic.
    """
    global _CONTEXT_BUILDER_INSTANCE
    if _CONTEXT_BUILDER_INSTANCE is None:
        _CONTEXT_BUILDER_INSTANCE = AthleticsContextBuilder()
    return _CONTEXT_BUILDER_INSTANCE


class AthleticsContextBuilder:
    """Build relevant context from athletics data for the chatbot."""

    def __init__(self):
        self._ksa_athletes = None
        self._ksa_rankings = None
        self._benchmarks = None
        self._wittw = None
        self._data_loaded = False
        # DON'T load data on init - use lazy loading instead

    @property
    def ksa_athletes(self):
        """Lazy load KSA athletes only when needed."""
        if self._ksa_athletes is None and DATA_CONNECTOR_AVAILABLE:
            try:
                self._ksa_athletes = get_ksa_athletes()
                logging.info(f"Lazy loaded {len(self._ksa_athletes) if self._ksa_athletes is not None else 0} KSA athletes")
            except Exception as e:
                logging.error(f"Error loading KSA athletes: {e}")
                self._ksa_athletes = pd.DataFrame()
        return self._ksa_athletes

    @property
    def rankings(self):
        """Lazy load KSA rankings for backward compatibility."""
        if self._ksa_rankings is None and DATA_CONNECTOR_AVAILABLE:
            try:
                self._ksa_rankings = get_ksa_rankings()
                logging.info(f"Lazy loaded {len(self._ksa_rankings) if self._ksa_rankings is not None else 0} KSA ranking records")
            except Exception as e:
                logging.error(f"Error loading KSA rankings: {e}")
                self._ksa_rankings = pd.DataFrame()
        return self._ksa_rankings

    def query_master(self, sql: str) -> pd.DataFrame:
        """Execute SQL query against the full master database (2.3M+ records)."""
        if not DATA_CONNECTOR_AVAILABLE:
            return pd.DataFrame()
        try:
            return query(sql)
        except Exception as e:
            logging.error(f"Error querying master database: {e}")
            return pd.DataFrame()

    def get_event_world_rankings(self, event: str, gender: str = 'men', top_n: int = 20) -> pd.DataFrame:
        """Get world rankings for an event from master database."""
        gender_filter = "Men" if gender.lower() in ['men', 'male', 'm'] else "Women"
        sql = f"""
            SELECT competitor, nat, result, venue, date, rank
            FROM master
            WHERE event LIKE '%{event}%'
              AND gender = '{gender_filter}'
              AND result IS NOT NULL
            ORDER BY
                CASE
                    WHEN result LIKE '%:%' THEN
                        CAST(SPLIT_PART(result, ':', 1) AS DOUBLE) * 60 +
                        CAST(REPLACE(SPLIT_PART(result, ':', 2), 's', '') AS DOUBLE)
                    ELSE CAST(REPLACE(REPLACE(result, 'm', ''), 's', '') AS DOUBLE)
                END ASC
            LIMIT {top_n}
        """
        try:
            return self.query_master(sql)
        except:
            # Fallback for field events (higher is better)
            sql = f"""
                SELECT competitor, nat, result, venue, date, rank
                FROM master
                WHERE event LIKE '%{event}%'
                  AND gender = '{gender_filter}'
                  AND result IS NOT NULL
                ORDER BY result_numeric DESC NULLS LAST
                LIMIT {top_n}
            """
            return self.query_master(sql)

    def get_athlete_from_master(self, athlete_name: str) -> pd.DataFrame:
        """Search for athlete in full master database."""
        sql = f"""
            SELECT competitor, event, result, venue, date, nat, rank, gender
            FROM master
            WHERE LOWER(competitor) LIKE LOWER('%{athlete_name}%')
            ORDER BY date DESC
            LIMIT 50
        """
        return self.query_master(sql)

    def get_competition_results(self, competition: str, limit: int = 100) -> pd.DataFrame:
        """Get results from a specific competition."""
        sql = f"""
            SELECT competitor, event, result, nat, pos, gender
            FROM master
            WHERE LOWER(venue) LIKE LOWER('%{competition}%')
            ORDER BY event, pos
            LIMIT {limit}
        """
        return self.query_master(sql)

    def get_country_rankings(self, country: str, event: str = None, limit: int = 50) -> pd.DataFrame:
        """Get rankings for athletes from a specific country."""
        event_filter = f"AND event LIKE '%{event}%'" if event else ""
        sql = f"""
            SELECT competitor, event, result, venue, date, rank, gender
            FROM master
            WHERE UPPER(nat) = UPPER('{country}')
            {event_filter}
            ORDER BY date DESC
            LIMIT {limit}
        """
        return self.query_master(sql)

    @property
    def benchmarks(self):
        """Lazy load benchmarks only when needed."""
        if self._benchmarks is None and DATA_CONNECTOR_AVAILABLE:
            try:
                self._benchmarks = get_benchmarks_data()
                logging.info("Lazy loaded benchmarks data")
            except Exception as e:
                logging.error(f"Error loading benchmarks: {e}")
                self._benchmarks = pd.DataFrame()
        return self._benchmarks

    @property
    def wittw(self):
        """Lazy load What It Takes To Win analyzer only when needed."""
        if self._wittw is None and ANALYTICS_AVAILABLE:
            try:
                self._wittw = WhatItTakesToWin()
                # DON'T call load_scraped_data() here - it loads 2.3M records
                # The wittw will load data on demand when methods are called
                logging.info("Initialized What It Takes To Win analyzer (data loads on demand)")
            except Exception as e:
                logging.error(f"Error initializing WITTW: {e}")
        return self._wittw

    def get_ksa_athlete_summary(self) -> str:
        """Get summary of KSA athletes for context."""
        if self.ksa_athletes is None or self.ksa_athletes.empty:
            return "No KSA athlete data available."

        summary_parts = ["## KSA Athletes Overview\n"]

        # Count total athletes
        summary_parts.append(f"**Total Athletes:** {len(self.ksa_athletes)}")

        # Count by event type
        if 'primary_event' in self.ksa_athletes.columns:
            event_counts = self.ksa_athletes['primary_event'].value_counts().head(10)
            summary_parts.append("\n**Athletes by Event:**")
            for event, count in event_counts.items():
                summary_parts.append(f"- {event}: {count} athletes")

        # List athletes with their events
        summary_parts.append("\n**KSA Athletes Roster:**")
        for _, row in self.ksa_athletes.head(30).iterrows():
            name = row.get('full_name', row.get('competitor', 'Unknown'))
            event = row.get('primary_event', row.get('event', 'Unknown'))
            summary_parts.append(f"- {name} ({event})")

        return "\n".join(summary_parts)

    def get_athlete_details(self, athlete_name: str) -> str:
        """Get detailed info about a specific athlete from full master database."""
        # First try the master database (2.3M+ records)
        matches = self.get_athlete_from_master(athlete_name)

        if matches.empty:
            # Fall back to KSA rankings
            if self.rankings is not None and not self.rankings.empty:
                name_cols = ['competitor', 'Athlete', 'full_name', 'Name']
                df = self.rankings
                for col in name_cols:
                    if col in df.columns:
                        matches = df[df[col].str.contains(athlete_name, case=False, na=False)]
                        if not matches.empty:
                            break

        if matches.empty:
            return f"No results found for '{athlete_name}'"

        details = [f"## Performance Data: {athlete_name}\n"]

        # Determine columns
        event_col = next((c for c in ['event', 'Event Type', 'Event'] if c in matches.columns), None)
        result_col = next((c for c in ['result', 'Result', 'Mark'] if c in matches.columns), None)
        venue_col = next((c for c in ['venue', 'Competition', 'Venue'] if c in matches.columns), None)
        date_col = next((c for c in ['date', 'Date'] if c in matches.columns), None)
        rank_col = next((c for c in ['rank', 'Rank', 'place'] if c in matches.columns), None)
        nat_col = next((c for c in ['nat', 'Country', 'country'] if c in matches.columns), None)

        # Show country if available
        if nat_col and matches[nat_col].notna().any():
            country = matches[nat_col].dropna().iloc[0] if not matches[nat_col].dropna().empty else 'Unknown'
            details.append(f"**Country:** {country}\n")

        # Group by event
        if event_col:
            for event in matches[event_col].unique():
                event_data = matches[matches[event_col] == event]
                details.append(f"\n**{event}** ({len(event_data)} results):")

                for _, row in event_data.head(10).iterrows():
                    result = row.get(result_col, 'N/A') if result_col else 'N/A'
                    venue = row.get(venue_col, 'Unknown') if venue_col else 'Unknown'
                    date = row.get(date_col, '') if date_col else ''
                    rank = row.get(rank_col, '') if rank_col else ''
                    rank_str = f" (Rank {rank})" if rank else ""
                    details.append(f"  - {result}{rank_str} at {venue} ({date})")

        return "\n".join(details)

    def get_event_rankings(self, event: str, gender: str = 'men', top_n: int = 20) -> str:
        """Get top world rankings for an event from master database."""
        # First try the master database for world rankings
        df = self.get_event_world_rankings(event, gender, top_n)

        if df.empty:
            # Fall back to KSA rankings
            if self.rankings is not None and not self.rankings.empty:
                df = self.rankings.copy()
                event_col = next((c for c in ['event', 'Event Type', 'Event'] if c in df.columns), None)
                if event_col:
                    df = df[df[event_col].str.contains(event, case=False, na=False)]
                gender_col = next((c for c in ['gender', 'Gender'] if c in df.columns), None)
                if gender_col:
                    gender_val = 'M' if gender.lower() in ['men', 'male', 'm'] else 'F'
                    df = df[(df[gender_col].str.upper() == gender_val) |
                            (df[gender_col].str.lower() == gender.lower())]

        if df.empty:
            return f"No results for {event} ({gender})"

        # Get column names
        name_col = next((c for c in ['competitor', 'Name', 'Athlete'] if c in df.columns), None)
        result_col = next((c for c in ['result', 'Result', 'Mark', 'Score'] if c in df.columns), None)
        country_col = next((c for c in ['nat', 'Country', 'country'] if c in df.columns), None)
        venue_col = next((c for c in ['venue', 'Competition', 'Venue'] if c in df.columns), None)

        rankings = [f"## World Top {min(top_n, len(df))} in {event} ({gender.title()})\n"]

        for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
            name = row.get(name_col, 'Unknown') if name_col else 'Unknown'
            result = row.get(result_col, 'N/A') if result_col else 'N/A'
            country = row.get(country_col, '') if country_col else ''
            venue = row.get(venue_col, '') if venue_col else ''
            country_str = f" ({country})" if country else ""
            venue_str = f" @ {venue}" if venue else ""
            rankings.append(f"{i}. {name}{country_str} - {result}{venue_str}")

        return "\n".join(rankings)

    def get_qualification_standards(self, event: str = None) -> str:
        """Get qualification standards for championships."""
        if self.benchmarks is None or self.benchmarks.empty:
            return "No qualification standards available."

        df = self.benchmarks.copy()

        if event:
            event_cols = [c for c in df.columns if 'event' in c.lower()]
            if event_cols:
                df = df[df[event_cols[0]].str.contains(event, case=False, na=False)]

        if df.empty:
            return f"No standards found for {event}" if event else "No standards data"

        standards = ["## Qualification Standards\n"]
        for _, row in df.head(20).iterrows():
            row_dict = row.to_dict()
            # Format nicely
            standards.append(f"- {row_dict}")

        return "\n".join(standards)

    def get_what_it_takes_to_win(self, event: str, gender: str = 'men') -> str:
        """Get medal standards from What It Takes To Win analysis."""
        if self.wittw is None:
            return "What It Takes To Win analysis not available."

        try:
            report = self.wittw.generate_what_it_takes_report(gender=gender, comp_type='World Champs')

            if report.empty:
                return f"No winning marks data for {event}"

            # Filter for specific event
            event_data = report[report['Event'].str.contains(event, case=False, na=False)]

            if event_data.empty:
                return f"No data for {event} in major championships"

            results = [f"## What It Takes to Win: {event} ({gender.title()})\n"]

            for _, row in event_data.iterrows():
                results.append(f"**{row['Event']}**")
                results.append(f"- Gold Standard: {row.get('Gold', 'N/A')}")
                results.append(f"- Silver Standard: {row.get('Silver', 'N/A')}")
                results.append(f"- Bronze Standard: {row.get('Bronze', 'N/A')}")
                results.append(f"- Final Standard (8th): {row.get('Final Standard (8th)', 'N/A')}")
                results.append("")

            return "\n".join(results)

        except Exception as e:
            logging.error(f"Error getting what it takes data: {e}")
            return f"Error analyzing {event}: {str(e)}"

    def search_by_keyword(self, keyword: str) -> str:
        """Search data by keyword in master database (2.3M+ records)."""
        results = []

        # Search in master database first (full 2.3M records)
        try:
            # Search in competitor names
            sql = f"""
                SELECT competitor, event, result, nat, venue, date
                FROM master
                WHERE LOWER(competitor) LIKE LOWER('%{keyword}%')
                   OR LOWER(venue) LIKE LOWER('%{keyword}%')
                   OR LOWER(event) LIKE LOWER('%{keyword}%')
                   OR LOWER(nat) LIKE LOWER('%{keyword}%')
                ORDER BY date DESC
                LIMIT 20
            """
            matches = self.query_master(sql)
            if not matches.empty:
                results.append(f"## Master Database Results ({len(matches)} shown)\n")
                for _, row in matches.iterrows():
                    results.append(f"- {row.get('competitor', 'Unknown')} ({row.get('nat', '')}) - {row.get('event', '')} - {row.get('result', '')} @ {row.get('venue', '')} ({row.get('date', '')})")
        except Exception as e:
            logging.error(f"Error searching master database: {e}")

        # Also search in KSA athletes for context
        if self.ksa_athletes is not None and not self.ksa_athletes.empty:
            for col in self.ksa_athletes.columns:
                if self.ksa_athletes[col].dtype == 'object':
                    matches = self.ksa_athletes[
                        self.ksa_athletes[col].str.contains(keyword, case=False, na=False)
                    ]
                    if not matches.empty:
                        results.append(f"\n## KSA Athletes ({len(matches)} matches)")
                        for _, row in matches.head(5).iterrows():
                            name = row.get('full_name', row.get('competitor', 'Unknown'))
                            event = row.get('primary_event', row.get('event', ''))
                            results.append(f"  - {name} ({event})")
                        break

        if not results:
            return f"No matches found for '{keyword}'"

        return "\n".join(results)

    def build_context_for_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Build relevant context based on the user's query.
        Returns (context_text, sources_list).
        """
        query_lower = query.lower()
        context_parts = []
        sources = []

        # Always include base athletics knowledge
        context_parts.append(ATHLETICS_KNOWLEDGE_BASE)
        sources.append("Athletics Knowledge Base")

        # Check for specific athlete mentions
        if self.ksa_athletes is not None:
            name_cols = ['full_name', 'competitor', 'Name']
            for col in name_cols:
                if col in self.ksa_athletes.columns:
                    for name in self.ksa_athletes[col].dropna().unique():
                        name_str = str(name)
                        # Check if any part of the name is in the query
                        name_parts = name_str.lower().split()
                        if any(part in query_lower for part in name_parts if len(part) > 2):
                            athlete_data = self.get_athlete_details(name_str)
                            if "No results" not in athlete_data:
                                context_parts.append(athlete_data)
                                sources.append(f"Athlete: {name_str}")
                                break
                    break

        # Check for event mentions and add ELITE KNOWLEDGE
        events = ['100m', '200m', '400m', '800m', '1500m', '5000m', '10000m',
                  'marathon', 'hurdles', '110m', '400mh', '100mh', '110mh',
                  'high jump', 'long jump', 'triple jump', 'pole vault',
                  'shot put', 'discus', 'hammer', 'javelin', 'decathlon', 'heptathlon']

        detected_event = None
        for event in events:
            if event.lower() in query_lower:
                detected_event = event
                # Determine gender
                gender = 'women' if any(w in query_lower for w in ['women', 'female', "women's"]) else 'men'

                # ADD ELITE KNOWLEDGE for this event
                elite_knowledge = get_elite_knowledge_for_event(event)
                if elite_knowledge:
                    context_parts.append(elite_knowledge)
                    sources.append(f"Elite {event} Expertise")

                # Get event rankings
                event_data = self.get_event_rankings(event, gender)
                if "No" not in event_data:
                    context_parts.append(event_data)
                    sources.append(f"Event Rankings: {event}")

                # Get what it takes to win
                wittw_data = self.get_what_it_takes_to_win(event, gender)
                if "not available" not in wittw_data and "Error" not in wittw_data:
                    context_parts.append(wittw_data)
                    sources.append(f"Championship Standards: {event}")
                break

        # Check for discipline category mentions and add expertise
        discipline_keywords = {
            'sprints': ['sprint', 'sprinter', 'fast', 'speed', 'acceleration'],
            'middle_distance': ['middle distance', '800', '1500', 'miler'],
            'long_distance': ['distance', 'marathon', 'endurance', '5000', '10000'],
            'hurdles': ['hurdle', 'hurdles', 'hurdler'],
            'jumps': ['jump', 'jumper', 'vault', 'vaulter', 'high jump', 'long jump', 'triple'],
            'throws': ['throw', 'thrower', 'shot', 'discus', 'hammer', 'javelin', 'putting'],
            'combined_events': ['decathlon', 'heptathlon', 'combined', 'multi-event']
        }

        for discipline, keywords in discipline_keywords.items():
            if any(kw in query_lower for kw in keywords):
                discipline_context = get_discipline_context(discipline)
                if discipline_context and discipline_context not in str(context_parts):
                    context_parts.append(discipline_context)
                    sources.append(f"{discipline.replace('_', ' ').title()} Expertise")
                break

        # Check for technique/coaching related queries
        technique_keywords = ['technique', 'form', 'phase', 'biomechanic', 'coaching', 'drill',
                              'training', 'workout', 'improve', 'faster', 'further', 'higher']
        if any(word in query_lower for word in technique_keywords) and detected_event:
            # Already have elite knowledge, but emphasize technique
            if ELITE_KNOWLEDGE_AVAILABLE:
                event_knowledge = get_event_knowledge(detected_event)
                if event_knowledge and 'technique_components' in event_knowledge:
                    technique_text = "\n## Technique Focus\n"
                    for comp, info in event_knowledge['technique_components'].items():
                        if isinstance(info, dict):
                            technique_text += f"- **{comp}**: {info.get('action', info.get('key_cue', ''))}\n"
                            if 'coaching_cues' in info:
                                technique_text += f"  - Cues: {', '.join(info['coaching_cues'])}\n"
                    if technique_text != "\n## Technique Focus\n":
                        context_parts.append(technique_text)
                        sources.append("Technique & Coaching Cues")

        # Check for qualification/standards mentions
        if any(word in query_lower for word in ['qualify', 'standard', 'championship', 'olympic', 'world', 'asian']):
            context_parts.append(self.get_qualification_standards())
            sources.append("Qualification Standards")

        # Include KSA overview for team-related queries
        if any(word in query_lower for word in ['ksa', 'saudi', 'team', 'athletes', 'roster']):
            context_parts.append(self.get_ksa_athlete_summary())
            sources.append("KSA Athletes Database")

        # Check for country/competitor analysis (using master database)
        country_codes = {
            'usa': 'USA', 'united states': 'USA', 'america': 'USA',
            'jamaica': 'JAM', 'kenya': 'KEN', 'ethiopia': 'ETH',
            'japan': 'JPN', 'china': 'CHN', 'india': 'IND',
            'qatar': 'QAT', 'bahrain': 'BRN', 'uae': 'UAE',
            'iran': 'IRI', 'morocco': 'MAR', 'south africa': 'RSA',
            'great britain': 'GBR', 'britain': 'GBR', 'uk': 'GBR',
            'germany': 'GER', 'france': 'FRA', 'italy': 'ITA',
            'spain': 'ESP', 'netherlands': 'NED', 'poland': 'POL',
            'australia': 'AUS', 'canada': 'CAN', 'brazil': 'BRA',
            'nigeria': 'NGR', 'cuba': 'CUB', 'sweden': 'SWE'
        }
        for country_name, code in country_codes.items():
            if country_name in query_lower:
                country_data = self.get_country_rankings(code, detected_event)
                if not country_data.empty:
                    context_parts.append(f"## {code} Athletes Performance\n")
                    for _, row in country_data.head(15).iterrows():
                        context_parts.append(f"- {row.get('competitor', 'Unknown')} - {row.get('event', '')} - {row.get('result', '')} ({row.get('date', '')})")
                    sources.append(f"Master Database: {code} Athletes")
                break

        # If no specific context found, do a keyword search
        if len(context_parts) <= 1:  # Only has base knowledge
            # Extract potential keywords (words > 3 chars that aren't common)
            common_words = {'what', 'how', 'when', 'where', 'which', 'who', 'does', 'can', 'will',
                           'the', 'and', 'for', 'that', 'this', 'with', 'from', 'about', 'their'}
            words = [w for w in query_lower.split() if len(w) > 3 and w not in common_words]
            for word in words[:3]:  # Try first 3 keywords
                search_results = self.search_by_keyword(word)
                if "No matches" not in search_results:
                    context_parts.append(search_results)
                    sources.append(f"Keyword Search: {word}")
                    break

        return "\n\n---\n\n".join(context_parts), sources

    def build_context(self, query: str, knowledge_source: str = "Database Only") -> str:
        """
        Compatibility wrapper for build_context_for_query.
        Returns just the context string (not sources).
        Used by OpenRouterClient in World_Ranking_Deploy_v3.py.

        Args:
            query: User's question
            knowledge_source: Ignored (kept for API compatibility)
        """
        context, _ = self.build_context_for_query(query)
        return context


###################################
# OpenRouter Chat Client
###################################
class OpenRouterClient:
    """Client for OpenRouter API with free models, using OpenAI SDK pattern."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        # Use singleton context builder to avoid repeated data loading
        self.context_builder = get_context_builder()

        # Use OpenAI client with OpenRouter base URL (like in MCP_Analysis)
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the athletics analyst."""
        return """You are an ELITE world-class sports analyst and coach specializing in athletics (track and field).
You work for Team Saudi and provide EXPERT-LEVEL analysis with deep technical knowledge.

## Your Expertise Areas

1. **Athlete Performance Analysis**
   - Analyze times, distances, progressions against world-class benchmarks
   - Identify technical strengths and areas for improvement
   - Compare performances across competition conditions (altitude, wind, indoor/outdoor)

2. **Event-Specific Technical Expertise**
   - **Sprints (100m/200m/400m)**: Race phases, block clearance, drive phase, max velocity, speed maintenance, ground contact time, stride mechanics
   - **Middle Distance (800m/1500m)**: Tactical patterns, lap splits, kick timing, sit-and-kick vs front-running
   - **Long Distance (5000m/10000m/Marathon)**: Pacing strategies, negative splits, surge responses, drafting
   - **Hurdles (110mH/100mH/400mH)**: Hurdle clearance, trail leg, stride patterns, rhythm
   - **Jumps (HJ/LJ/TJ/PV)**: Approach phases, takeoff mechanics, flight positions, landing
   - **Throws (SP/DT/HT/JT)**: Release angles, rotational/linear technique, implement velocity
   - **Combined Events**: Decathlon/Heptathlon scoring, event order strategy, recovery

3. **Competition Strategy**
   - Championship-specific race tactics
   - Rounds progression (heats, semis, finals)
   - Championship pressure factors (~0.5% slower under major pressure)

4. **Qualification & Rankings Analysis**
   - Olympics, World Championships, Asian Games standards
   - World Athletics ranking system and points
   - Entry standards vs world ranking pathway

5. **Competitor Intelligence**
   - Head-to-head analysis, strengths/weaknesses
   - Form trends (improving/stable/declining)
   - Gap analysis to rivals

## Technical Terminology You Use
- FAT (Fully Automatic Timing), RT (Reaction Time), PB (Personal Best), SB (Season Best)
- WL (World Lead), WR (World Record), NR (National Record), AR (Area Record)
- DNS (Did Not Start), DNF (Did Not Finish), DQ (Disqualified)
- Wind legal (≤2.0 m/s), altitude adjusted, age-graded

## Response Guidelines

1. **Use your elite knowledge** to provide technical insights beyond basic data
2. **Be specific** with times, distances, rankings, and biomechanical parameters
3. **Always clarify** wind conditions for sprints and horizontal jumps
4. **Note altitude effects** (>1000m benefits sprints/throws, hurts distance)
5. **Use proper terminology** and explain technical concepts when relevant
6. **Provide actionable insights** - training focus, tactical adjustments, areas to develop
7. **Reference benchmarks** - world class, national class, regional elite standards
8. **Analyze form trends** - improving, stable, or declining based on recent performances

Format responses with clear headers, bullet points, and technical precision.
Be the expert coach and analyst that Team Saudi athletes deserve."""

    def chat(self, user_message: str, chat_history: List[Dict] = None) -> Tuple[str, str]:
        """
        Send a message and get a response.
        Returns (answer, sources_text).
        """
        if not self.api_key:
            return ("Error: OpenRouter API key not configured. "
                    "Please add OPENROUTER_API_KEY to your .env file or get a free key at https://openrouter.ai/keys", "")

        # Build context from athletics data
        data_context, sources = self.context_builder.build_context_for_query(user_message)

        # Build messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]

        # Add data context as a system message
        if data_context:
            messages.append({
                "role": "system",
                "content": f"## Available Data Context\n\n{data_context}"
            })

        # Add chat history (last 6 exchanges for context)
        if chat_history:
            for msg in chat_history[-12:]:
                messages.append(msg)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Make API request using OpenAI SDK
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower for more factual responses
                max_tokens=2000,
                top_p=0.9,
                extra_headers={
                    "HTTP-Referer": "https://team-saudi-athletics.streamlit.app",
                    "X-Title": "Team Saudi Athletics Analyst"
                }
            )

            answer = response.choices[0].message.content.strip()

            # Format sources
            sources_text = ""
            if sources:
                sources_text = "**Sources Used:**\n"
                for source in sources:
                    sources_text += f"- {source}\n"

            return answer, sources_text

        except Exception as e:
            logging.error(f"Error calling OpenRouter API: {str(e)}")
            return f"Error: {str(e)}", ""


###################################
# Streamlit UI
###################################
def apply_chat_theme():
    """Apply Team Saudi theme to chat interface."""
    st.markdown(f"""
    <style>
    /* Dark theme base */
    .stApp {{
        background-color: #0a0a0a !important;
    }}

    /* Main container */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}

    /* Chat messages */
    .stChatMessage {{
        background-color: rgba(0, 113, 103, 0.08) !important;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(0, 113, 103, 0.2);
    }}

    /* User message content */
    [data-testid="stChatMessageContent"] {{
        color: white !important;
    }}

    /* Chat input container */
    .stChatInputContainer {{
        background-color: #1a1a1a !important;
        border: 2px solid {TEAL_PRIMARY} !important;
        border-radius: 12px;
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label {{
        color: white !important;
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {TEAL_PRIMARY} !important;
    }}

    /* Gold accent text */
    .gold-text {{
        color: {GOLD_ACCENT} !important;
    }}

    /* Success/Info/Warning boxes */
    .stSuccess {{
        background-color: rgba(0, 113, 103, 0.2) !important;
        border-color: {TEAL_PRIMARY} !important;
    }}

    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: rgba(0, 113, 103, 0.1) !important;
        border-radius: 8px;
    }}

    /* Button styling */
    .stButton button {{
        background-color: {TEAL_PRIMARY} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }}
    .stButton button:hover {{
        background-color: {TEAL_DARK} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Athletics AI Analyst - Team Saudi",
        page_icon="🏃",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_chat_theme()

    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                display: flex; align-items: center; gap: 1rem;">
        <div style="font-size: 3rem;">🏃</div>
        <div>
            <h1 style="color: white !important; margin: 0; font-size: 1.8rem;">
                Athletics AI Analyst
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.25rem 0 0 0; font-size: 0.95rem;">
                World-class sports analysis for Team Saudi | Powered by OpenRouter Free Models
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Configuration")

        # API Key input (if not in env)
        api_key = get_openrouter_key()
        if not api_key:
            st.warning("API Key Required")
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                help="Get your free API key at https://openrouter.ai/keys"
            )
            st.markdown("[Get Free API Key](https://openrouter.ai/keys)")
        else:
            st.success("API Key Configured")

        # Model selection
        st.markdown("### AI Model")
        selected_model = st.selectbox(
            "Select Model",
            options=list(FREE_MODELS.keys()),
            format_func=lambda x: FREE_MODELS[x],
            index=0,  # Default to DeepSeek R1
            help="All models are free. DeepSeek R1 70B provides best quality."
        )

        st.markdown("---")

        # Data status
        st.markdown("### Data Status")
        if DATA_CONNECTOR_AVAILABLE:
            mode = get_data_mode()
            if mode == "azure":
                st.success(f"Azure Cloud")
            else:
                st.info(f"Local Data")

            try:
                athletes = get_ksa_athletes()
                if athletes is not None and not athletes.empty:
                    st.success(f"{len(athletes)} KSA Athletes")
                else:
                    st.warning("No athletes loaded")
            except Exception as e:
                st.warning(f"Data error: {e}")
        else:
            st.error("Data Connector Unavailable")

        st.markdown("---")

        # Example queries
        st.markdown("### Try These Questions")
        example_queries = [
            "How are KSA sprinters performing?",
            "What time is needed to make the 100m final at World Championships?",
            "Tell me about Yaser Triki",
            "Compare our 400m athletes to Asian competition",
            "What are the Olympic qualification standards for triple jump?",
            "Who should we target for medal contention?"
        ]

        for q in example_queries:
            if st.button(q, key=f"ex_{hash(q)}"):
                st.session_state.pending_query = q

        st.markdown("---")

        # Clear chat button at bottom of sidebar
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "client" not in st.session_state or st.session_state.get("current_model") != selected_model:
        if api_key:
            st.session_state.client = OpenRouterClient(api_key, selected_model)
            st.session_state.current_model = selected_model

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources if available
            if message.get("sources"):
                with st.expander("Sources", expanded=False):
                    st.markdown(message["sources"])

    # Handle pending query from sidebar examples
    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query

        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Get response
        if hasattr(st.session_state, 'client'):
            with st.chat_message("assistant"):
                with st.spinner("Analyzing athletics data..."):
                    response, sources = st.session_state.client.chat(
                        query,
                        [{"role": m["role"], "content": m["content"]}
                         for m in st.session_state.messages[:-1]]
                    )
                st.markdown(response)
                if sources:
                    with st.expander("Sources", expanded=False):
                        st.markdown(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about athletics performance, competitors, or strategy..."):
        if not api_key:
            st.error("Please configure your OpenRouter API key in the sidebar.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing athletics data..."):
                    response, sources = st.session_state.client.chat(
                        prompt,
                        [{"role": m["role"], "content": m["content"]}
                         for m in st.session_state.messages[:-1]]
                    )
                st.markdown(response)
                if sources:
                    with st.expander("Sources", expanded=False):
                        st.markdown(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })


if __name__ == "__main__":
    main()
