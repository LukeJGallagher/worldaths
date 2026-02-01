import streamlit as st
import pandas as pd
import sqlite3
import datetime
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import hashlib
from openai import OpenAI

# Import athletics analytics
from athletics_analytics_agents import AthleticsAnalytics, DISCIPLINE_KNOWLEDGE, MAJOR_GAMES
from what_it_takes_to_win import WhatItTakesToWin

# Import chatbot context builder (singleton)
try:
    from athletics_chatbot import get_context_builder
    CHATBOT_CONTEXT_AVAILABLE = True
except ImportError:
    CHATBOT_CONTEXT_AVAILABLE = False
    get_context_builder = None

# Document RAG replaced with NotebookLM MCP integration
# See docs/NOTEBOOKLM_RAG_GUIDE.md for setup instructions
DOCUMENT_RAG_AVAILABLE = False  # Legacy flag, kept for compatibility

# Import NotebookLM client for AI chat
try:
    from notebooklm_client import query_notebook, check_notebooklm_available, NOTEBOOK_ID
    NOTEBOOKLM_AVAILABLE = check_notebooklm_available()
except ImportError:
    NOTEBOOKLM_AVAILABLE = False
    query_notebook = None
    NOTEBOOK_ID = None

# Import Coach View module
try:
    from coach_view import render_coach_view
    COACH_VIEW_AVAILABLE = True
except ImportError:
    COACH_VIEW_AVAILABLE = False

# Import Azure/Parquet data connector
try:
    from data_connector import (
        get_ksa_athletes, get_data_mode, query as duckdb_query,
        get_rankings_data, get_ksa_rankings, get_benchmarks_data,
        get_road_to_tokyo_data
    )
    DATA_CONNECTOR_AVAILABLE = True
except ImportError:
    DATA_CONNECTOR_AVAILABLE = False

# Import analytics helpers
try:
    from analytics_helpers import (
        calculate_consistency_score, calculate_near_miss,
        head_to_head_comparison, country_comparison,
        parse_result_to_seconds, is_field_event, get_regional_rivals
    )
    ANALYTICS_HELPERS_AVAILABLE = True
except ImportError:
    ANALYTICS_HELPERS_AVAILABLE = False

# Import Race Intelligence module
try:
    from race_intelligence import (
        get_competitor_form_cards, build_race_preview,
        get_career_milestones, get_standards_progression
    )
    RACE_INTELLIGENCE_AVAILABLE = True
except ImportError:
    RACE_INTELLIGENCE_AVAILABLE = False

###################################
# Team Saudi Brand Colors
###################################
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'
TEAL_LIGHT = '#009688'
GRAY_BLUE = '#78909C'

###################################
# Helper Functions
###################################
def is_field_event(event_name: str) -> bool:
    """Check if an event is a field event (throws/jumps) vs track event."""
    if not event_name:
        return False
    event_lower = str(event_name).lower()
    return any(x in event_lower for x in ['shot', 'discus', 'hammer', 'javelin', 'jump', 'vault', 'throw', 'put'])

###################################
# Project East 2026 - Asian Games Strategy
###################################
PROJECT_EAST_ATHLETES = [
    {
        'name': 'Mohammed Al Atafi',
        'event': '200m',
        'pb': 20.68,
        'sb': 20.68,
        'medal_standard': 20.20,
        'asian_games_2023': 20.96,
        'trajectory': 'Improving - PB in 2024',
        'status': 'medal_contender',
        'age': 23,
        'notes': 'Consistent improvements, potential to break 20.50'
    },
    {
        'name': 'Hussain Al Hizam',
        'event': 'Pole Vault',
        'pb': 5.72,
        'sb': 5.72,
        'medal_standard': 5.70,
        'asian_games_2023': 5.60,
        'trajectory': 'Peaked - Defending Asian champion',
        'status': 'medal_favorite',
        'age': 27,
        'notes': 'Reigning Asian Games champion, consistent performer'
    },
    {
        'name': 'Mohamed Tolo',
        'event': 'Shot Put',
        'pb': 20.23,
        'sb': 20.09,
        'medal_standard': 20.50,
        'asian_games_2023': 19.87,
        'trajectory': 'Improving - Room for growth',
        'status': 'medal_contender',
        'age': 26,
        'notes': '4th at 2023 Asian Games, closing gap to medal zone'
    },
    {
        'name': 'Yasser Bakheet',
        'event': 'Triple Jump',
        'pb': 17.08,
        'sb': 17.08,
        'medal_standard': 16.90,
        'asian_games_2023': 16.58,
        'trajectory': 'Improving - New PB in 2024',
        'status': 'medal_contender',
        'age': 24,
        'notes': 'Breakthrough season in 2024, medal potential'
    },
    {
        'name': 'Abdelati Bizimana',
        'event': '800m',
        'pb': 104.50,  # 1:44.50 in seconds
        'sb': 105.20,  # 1:45.20 in seconds
        'medal_standard': 104.00,  # 1:44.00
        'asian_games_2023': 106.80,  # 1:46.80
        'trajectory': 'Development - Building base',
        'status': 'development',
        'age': 22,
        'notes': 'Talented young runner, 2-3 year development plan'
    },
    {
        'name': 'Muaz Al Dubaisi',
        'event': 'Hammer Throw',
        'pb': 73.50,
        'sb': 72.80,
        'medal_standard': 74.00,
        'asian_games_2023': 71.20,
        'trajectory': 'Improving - Consistent gains',
        'status': 'medal_contender',
        'age': 25,
        'notes': 'Strong domestic performer, needs international exposure'
    },
    {
        'name': 'Yaqoub Al Muawi',
        'event': '400m Hurdles',
        'pb': 49.12,
        'sb': 49.45,
        'medal_standard': 48.80,
        'asian_games_2023': 50.02,
        'trajectory': 'Improving - Technical refinement',
        'status': 'medal_contender',
        'age': 24,
        'notes': 'PB at World Championships qualifier'
    },
    {
        'name': 'Nasser Mohammed',
        'event': '100m',
        'pb': 10.18,
        'sb': 10.22,
        'medal_standard': 10.05,
        'asian_games_2023': 10.35,
        'trajectory': 'Improving - Speed development',
        'status': 'development',
        'age': 21,
        'notes': 'Young sprinter with potential, needs race experience'
    },
    {
        'name': '4x100m Relay',
        'event': '4x100m Relay',
        'pb': 38.88,
        'sb': 39.12,
        'medal_standard': 38.50,
        'asian_games_2023': 39.45,
        'trajectory': 'Improving - Team cohesion building',
        'status': 'medal_contender',
        'age': None,
        'notes': 'Best Saudi relay time in 2024, medal potential with clean exchanges'
    }
]

PROJECT_EAST_MEDAL_GOALS = {
    'gold': 1,
    'silver': 1,
    'bronze': 2,
    'total_target': '3-5',
    'high_probability': ['Pole Vault', 'Triple Jump'],
    'medal_contenders': ['200m', 'Shot Put', '400m Hurdles', '4x100m Relay'],
    'development': ['800m', '100m', 'Hammer Throw']
}

PROJECT_EAST_TIMELINE = {
    'phase_1': {'name': 'Foundation', 'dates': 'Jan-Mar 2025', 'focus': 'Base training, technique'},
    'phase_2': {'name': 'Build', 'dates': 'Apr-Jun 2025', 'focus': 'Competition exposure, rankings'},
    'phase_3': {'name': 'Peak', 'dates': 'Jul-Sep 2026', 'focus': 'Asian Games preparation'},
    'asian_games': {'name': 'Asian Games', 'dates': 'Sep 2026', 'location': 'Aichi-Nagoya, Japan'}
}

###################################
# AI Chatbot Configuration
###################################
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

FREE_MODELS = {
    'meta-llama/llama-3.2-3b-instruct:free': 'Llama 3.2 3B (Fastest)',
    'google/gemma-3-27b-it:free': 'Gemma 3 27B (Balanced)',
    'google/gemini-2.0-flash-exp:free': 'Gemini 2.0 Flash (Google)',
    'meta-llama/llama-3.3-70b-instruct:free': 'Llama 3.3 70B (Best Quality)',
    'qwen/qwen2.5-vl-7b-instruct:free': 'Qwen 2.5 VL 7B (Multilingual)',
    'deepseek/deepseek-r1:free': 'DeepSeek R1 (Reasoning)',
}
# Default to fastest model for better UX
DEFAULT_MODEL = 'meta-llama/llama-3.2-3b-instruct:free'

ATHLETICS_KNOWLEDGE = """
You are an elite sports analyst specializing in athletics (track and field) for Team Saudi.

Your expertise includes:
- Sprint events (100m, 200m, 400m) - Reaction times, phase analysis, speed endurance
- Distance events (800m-Marathon) - Pacing, lactate threshold, race tactics
- Field events (Jumps, Throws) - Technical phases, approach velocities, release angles
- Combined events (Decathlon, Heptathlon) - Point scoring, event priorities

Key performance contexts:
- World Athletics rankings and scoring system
- Olympic and World Championship qualification standards
- Asian Games and regional competition levels
- Personal best (PB) vs season best (SB) analysis
- Age-grade performance comparisons

When analyzing data, always consider:
1. Event-specific performance metrics
2. Competition level and conditions
3. Historical trends and progression
4. Comparison to qualification standards
5. Strategic recommendations for improvement

Respond in a professional yet accessible manner, suitable for coaches and athletes.
Use metric units. Reference specific data when available.
"""

def get_openrouter_key():
    """Get OpenRouter API key from environment or Streamlit secrets."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        try:
            if hasattr(st, 'secrets') and 'OPENROUTER_API_KEY' in st.secrets:
                api_key = st.secrets['OPENROUTER_API_KEY']
        except (FileNotFoundError, KeyError):
            pass
    return api_key


# Module-level cached data loaders for AI context (prevents repeated Azure calls)
# Uses 1800 seconds (30 min) TTL - same as main CACHE_TTL defined later
@st.cache_data(ttl=1800, show_spinner=False)
def _cached_ksa_data_for_ai():
    """Module-level cached KSA rankings data for AI context builder."""
    if DATA_CONNECTOR_AVAILABLE:
        try:
            return get_ksa_rankings()
        except Exception:
            pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_benchmarks_for_ai():
    """Module-level cached benchmark data for AI context builder."""
    if DATA_CONNECTOR_AVAILABLE:
        try:
            return get_benchmarks_data()
        except Exception:
            pass
    return None


class AthleticsContextBuilder:
    """
    RAG-style context builder that retrieves relevant data from the database.

    Searches:
    - KSA athlete profiles and rankings
    - Benchmark/qualification standards
    - Historical performance data

    Returns grounded context for LLM responses.
    """

    def __init__(self):
        self.data_available = DATA_CONNECTOR_AVAILABLE

    def _get_ksa_data(self):
        """Load KSA rankings data (uses module-level cache for speed)."""
        return _cached_ksa_data_for_ai()

    def _get_benchmarks(self):
        """Load benchmark standards (uses module-level cache for speed)."""
        return _cached_benchmarks_for_ai()

    def _search_athletes(self, query: str, limit: int = 10) -> str:
        """Search for athletes matching the query."""
        df = self._get_ksa_data()
        if df is None or df.empty:
            return ""

        query_lower = query.lower()
        results = []

        # Determine column names
        athlete_col = next((c for c in ['competitor', 'Competitor', 'full_name'] if c in df.columns), None)
        event_col = next((c for c in ['event', 'Event'] if c in df.columns), None)
        result_col = next((c for c in ['result', 'Result'] if c in df.columns), None)
        date_col = next((c for c in ['date', 'Date'] if c in df.columns), None)
        venue_col = next((c for c in ['venue', 'Venue'] if c in df.columns), None)

        if not athlete_col:
            return ""

        # Extract athlete names from query
        athlete_matches = df[df[athlete_col].str.lower().str.contains(query_lower, na=False)]

        # Also check for event-specific queries
        event_keywords = ['100m', '200m', '400m', '800m', '1500m', '5000m', '10000m',
                         'hurdles', 'jump', 'throw', 'shot', 'discus', 'javelin',
                         'hammer', 'pole vault', 'relay', 'marathon', 'walk']

        event_match = None
        for kw in event_keywords:
            if kw in query_lower:
                event_match = kw
                break

        if event_match and event_col:
            event_data = df[df[event_col].str.lower().str.contains(event_match, na=False)]
            if not event_data.empty:
                # Group by athlete and get best result
                grouped = event_data.groupby(athlete_col)
                for athlete, group in list(grouped)[:limit]:
                    if result_col and athlete_col:
                        best = group[result_col].iloc[0] if not group.empty else 'N/A'
                        recent_date = group[date_col].max() if date_col and date_col in group.columns else 'N/A'
                        results.append(f"- {athlete}: {best} ({event_match}, {recent_date})")

        elif not athlete_matches.empty:
            # Return athlete-specific data
            for athlete in athlete_matches[athlete_col].unique()[:limit]:
                athlete_data = df[df[athlete_col] == athlete]
                if event_col and result_col:
                    events = athlete_data.groupby(event_col).agg({
                        result_col: 'first'
                    }).head(5)
                    for event, row in events.iterrows():
                        results.append(f"- {athlete}: {event} - {row[result_col]}")

        if results:
            return "KSA ATHLETE DATA:\n" + "\n".join(results[:15])
        return ""

    def _search_benchmarks(self, query: str) -> str:
        """Search for relevant benchmark standards."""
        df = self._get_benchmarks()
        if df is None or df.empty:
            return ""

        query_lower = query.lower()
        results = []

        # Check for event mentions
        event_keywords = {
            '100m': ['100m', '100 metres', '100-metres'],
            '200m': ['200m', '200 metres', '200-metres'],
            '400m': ['400m', '400 metres', '400-metres'],
            '800m': ['800m', '800 metres', '800-metres'],
            '1500m': ['1500m', '1500 metres', '1500-metres'],
            'long jump': ['long jump', 'long-jump'],
            'high jump': ['high jump', 'high-jump'],
            'triple jump': ['triple jump', 'triple-jump'],
            'shot put': ['shot put', 'shot-put'],
            'discus': ['discus throw', 'discus-throw'],
            'javelin': ['javelin throw', 'javelin-throw'],
            'pole vault': ['pole vault', 'pole-vault'],
        }

        matched_event = None
        for event, variations in event_keywords.items():
            if any(v in query_lower for v in variations) or event in query_lower:
                matched_event = event
                break

        # Determine column names
        event_col = next((c for c in ['Event', 'event', 'Event Type'] if c in df.columns), None)

        if matched_event and event_col:
            event_data = df[df[event_col].str.lower().str.contains(matched_event, na=False)]
            if not event_data.empty:
                results.append(f"\nBENCHMARK STANDARDS for {matched_event.upper()}:")
                for _, row in event_data.head(5).iterrows():
                    row_info = []
                    for col in ['Gold Standard', 'Silver Standard', 'Bronze Standard', 'Final Standard (8th)', 'Gender']:
                        if col in row.index and pd.notna(row[col]):
                            row_info.append(f"{col}: {row[col]}")
                    if row_info:
                        results.append("  " + ", ".join(row_info))

        # Check for qualification-related queries
        if any(term in query_lower for term in ['qualify', 'qualification', 'standard', 'entry', 'medal']):
            # Return general qualification info
            if df is not None and not df.empty:
                results.append("\nQUALIFICATION STANDARDS AVAILABLE:")
                if event_col:
                    events = df[event_col].unique()[:10]
                    results.append(f"Events with standards: {', '.join(str(e) for e in events)}")

        return "\n".join(results) if results else ""

    def _get_static_context(self) -> str:
        """Return static knowledge about the database."""
        return """
DATABASE SUMMARY:
- Master rankings: 2.3M+ performance records from World Athletics
- KSA athlete profiles: 152 Saudi athletes with PBs and rankings
- Benchmarks: Medal standards for Olympics, World Champs, Asian Games
- Events: All track & field events (sprints, distance, jumps, throws)

ANSWER GUIDELINES:
- Cite specific data when available (e.g., "Mohammed Al Atafi ran 20.68s in 200m")
- If data not found, say so clearly
- Reference qualification standards when discussing targets
- Use metric units
"""

    def build_context(self, query: str, knowledge_source: str = "Both") -> str:
        """
        Build RAG context by searching relevant data sources.

        Args:
            query: User's question
            knowledge_source: "Database Only" (documents now via NotebookLM MCP)

        Returns context string with:
        1. Static database summary
        2. Relevant athlete data (if query mentions athletes/events)
        3. Benchmark standards (if query mentions qualification/standards)

        Note: For document/rulebook queries, use NotebookLM MCP integration.
        """
        context_parts = []

        # Always include database context
        context_parts.append(self._get_static_context())

        # Search for athlete/event data
        athlete_context = self._search_athletes(query)
        if athlete_context:
            context_parts.append(athlete_context)

        # Search for benchmark standards
        benchmark_context = self._search_benchmarks(query)
        if benchmark_context:
            context_parts.append(benchmark_context)

        # Add Project East athletes if relevant
        query_lower = query.lower()
        if any(term in query_lower for term in ['project east', 'asian games', '2026', 'nagoya', 'medal target']):
            pe_context = self._get_project_east_context()
            if pe_context:
                context_parts.append(pe_context)

        if not context_parts:
            return "General athletics query - using base knowledge."

        return "\n\n".join(context_parts)

    def _get_project_east_context(self) -> str:
        """Return Project East 2026 athlete context."""
        return """
PROJECT EAST 2026 - ASIAN GAMES MEDAL TARGETS:
Target: 3-5 medals at Aichi-Nagoya Asian Games

PRIORITY ATHLETES:
1. Hussain Al Hizam (Pole Vault) - PB 5.72m, Medal Standard 5.70m - MEDAL FAVORITE
2. Yasser Bakheet (Triple Jump) - PB 17.08m, Medal Standard 16.90m - MEDAL CONTENDER
3. Mohammed Al Atafi (200m) - PB 20.68s, Medal Standard 20.20s - MEDAL CONTENDER
4. Mohamed Tolo (Shot Put) - PB 20.23m, Medal Standard 20.50m - MEDAL CONTENDER
5. Yaqoub Al Muawi (400m Hurdles) - PB 49.12s, Medal Standard 48.80s - MEDAL CONTENDER
6. Muaz Al Dubaisi (Hammer Throw) - PB 73.50m, Medal Standard 74.00m - DEVELOPMENT
7. Abdelati Bizimana (800m) - PB 1:44.50, Medal Standard 1:44.00 - DEVELOPMENT
8. Nasser Mohammed (100m) - PB 10.18s, Medal Standard 10.05s - DEVELOPMENT
9. 4x100m Relay - PB 38.88s, Medal Standard 38.50s - MEDAL CONTENDER
"""


###################################
# Query Intent Classification
###################################
def classify_query_intent(query: str) -> dict:
    """
    Classify the query intent to optimize context building.

    Returns dict with:
    - intent: 'athlete', 'event', 'standards', 'comparison', 'general'
    - needs_benchmarks: bool
    - needs_rankings: bool
    - needs_full_context: bool
    - detected_athlete: str or None
    - detected_event: str or None
    """
    query_lower = query.lower()

    result = {
        'intent': 'general',
        'needs_benchmarks': False,
        'needs_rankings': False,
        'needs_full_context': False,
        'detected_athlete': None,
        'detected_event': None
    }

    # Detect events
    events = {
        '100m': ['100m', '100 meter', '100 metre', 'hundred'],
        '200m': ['200m', '200 meter', '200 metre'],
        '400m': ['400m', '400 meter', '400 metre'],
        '800m': ['800m', '800 meter', '800 metre'],
        '1500m': ['1500m', '1500 meter', '1500 metre', 'mile'],
        '5000m': ['5000m', '5k', '5km'],
        '10000m': ['10000m', '10k', '10km'],
        'marathon': ['marathon'],
        'long jump': ['long jump', 'longjump'],
        'high jump': ['high jump', 'highjump'],
        'triple jump': ['triple jump', 'triplejump'],
        'pole vault': ['pole vault', 'polevault'],
        'shot put': ['shot put', 'shotput', 'shot'],
        'discus': ['discus'],
        'hammer': ['hammer'],
        'javelin': ['javelin'],
        '110m hurdles': ['110m hurdles', '110mh', '110 hurdles'],
        '400m hurdles': ['400m hurdles', '400mh', '400 hurdles'],
        'relay': ['relay', '4x100', '4x400'],
    }

    for event_name, keywords in events.items():
        if any(kw in query_lower for kw in keywords):
            result['detected_event'] = event_name
            result['intent'] = 'event'
            result['needs_rankings'] = True
            break

    # Detect KSA athlete names (common patterns)
    ksa_athletes = [
        'al atafi', 'atafi', 'mohammed al atafi',
        'al hizam', 'hizam', 'hussain al hizam',
        'tolo', 'mohamed tolo',
        'bakheet', 'yasser bakheet',
        'bizimana', 'abdelati bizimana',
        'al muawi', 'muawi', 'yaqoub',
        'al dubaisi', 'dubaisi', 'muaz',
        'nasser mohammed', 'nasser',
        'triki', 'yaser triki'
    ]

    for athlete in ksa_athletes:
        if athlete in query_lower:
            result['detected_athlete'] = athlete.title()
            result['intent'] = 'athlete'
            result['needs_rankings'] = True
            break

    # Detect standards/qualification queries
    standards_keywords = ['qualify', 'qualification', 'standard', 'entry', 'medal',
                         'olympic', 'world championship', 'asian games', 'target']
    if any(kw in query_lower for kw in standards_keywords):
        result['needs_benchmarks'] = True
        if result['intent'] == 'general':
            result['intent'] = 'standards'

    # Detect comparison queries
    comparison_keywords = ['compare', 'vs', 'versus', 'against', 'rival', 'competitor', 'gap']
    if any(kw in query_lower for kw in comparison_keywords):
        result['intent'] = 'comparison'
        result['needs_rankings'] = True
        result['needs_benchmarks'] = True

    # Project East queries need full context
    if any(term in query_lower for term in ['project east', 'asian games 2026', 'nagoya']):
        result['needs_full_context'] = True

    return result


###################################
# Response Caching with TTL
###################################
class ResponseCache:
    """Simple response cache with TTL for AI responses."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self.cache = {}
        self.ttl = ttl_seconds
        self.max_size = max_size

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent cache keys."""
        # Lowercase, strip whitespace, remove punctuation
        normalized = query.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _make_key(self, query: str, model: str, knowledge_source: str) -> str:
        """Create cache key from query + model + source."""
        normalized = self._normalize_query(query)
        key_str = f"{normalized}|{model}|{knowledge_source}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, model: str, knowledge_source: str) -> dict | None:
        """Get cached response if valid."""
        key = self._make_key(query, model, knowledge_source)
        if key not in self.cache:
            return None

        entry = self.cache[key]
        if time.time() - entry['timestamp'] > self.ttl:
            # Expired
            del self.cache[key]
            return None

        return entry['response']

    def set(self, query: str, model: str, knowledge_source: str, response: dict):
        """Cache a response."""
        # Enforce max size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        key = self._make_key(query, model, knowledge_source)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }

    def clear(self):
        """Clear all cached responses."""
        self.cache = {}

    def stats(self) -> dict:
        """Get cache statistics."""
        now = time.time()
        valid = sum(1 for e in self.cache.values() if now - e['timestamp'] <= self.ttl)
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid,
            'ttl_seconds': self.ttl
        }


# Global response cache instance
_response_cache = ResponseCache(ttl_seconds=300, max_size=100)


class OpenRouterClient:
    """Client for OpenRouter API using OpenAI SDK with streaming support and response caching."""

    # Class-level context cache for faster repeated queries
    _context_cache = {}
    _cache_max_size = 50

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key
        )
        self.model = model
        # Use singleton context builder if available, otherwise use local lightweight version
        if CHATBOT_CONTEXT_AVAILABLE and get_context_builder is not None:
            try:
                self.context_builder = get_context_builder()
            except Exception:
                self.context_builder = AthleticsContextBuilder()
        else:
            self.context_builder = AthleticsContextBuilder()

    def _get_cached_context(self, user_query: str, knowledge_source: str = "Both") -> tuple:
        """
        Get context from cache or build new one using intent classification.

        Returns: (context_string, query_intent_dict)
        """
        # Simple hash for cache key (include knowledge source)
        cache_key = hash(f"{user_query.lower().strip()}_{knowledge_source}")

        if cache_key in self._context_cache:
            cached = self._context_cache[cache_key]
            if isinstance(cached, tuple):
                return cached
            # Legacy cache format - just string
            return cached, {'intent': 'general'}

        # Classify query intent for smarter context building
        intent = classify_query_intent(user_query)

        # Build optimized context based on intent
        context = self._build_intent_aware_context(user_query, knowledge_source, intent)

        # Cache with size limit (store both context and intent)
        if len(self._context_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._context_cache))
            del self._context_cache[oldest_key]

        self._context_cache[cache_key] = (context, intent)
        return context, intent

    def _build_intent_aware_context(self, query: str, knowledge_source: str, intent: dict) -> str:
        """Build context optimized for the query intent."""
        # For simple/fast queries, use minimal context
        if intent['intent'] == 'general' and not intent['needs_full_context']:
            # Quick general query - just base knowledge
            return "General athletics query - using base knowledge."

        # For specific intents, use the context builder but with hints
        # The context builder already handles different cases, but we can be smarter
        if intent['intent'] == 'athlete' and intent['detected_athlete']:
            # Optimize for athlete-specific query
            enhanced_query = f"athlete {intent['detected_athlete']} {query}"
            return self.context_builder.build_context(enhanced_query, knowledge_source)

        if intent['intent'] == 'event' and intent['detected_event']:
            # Optimize for event-specific query
            enhanced_query = f"event {intent['detected_event']} {query}"
            return self.context_builder.build_context(enhanced_query, knowledge_source)

        # Default - full context building
        return self.context_builder.build_context(query, knowledge_source)

    def chat(self, messages: list, user_query: str = None, knowledge_source: str = "Both") -> dict:
        """Send chat completion request with response caching."""
        try:
            # Check response cache first (only for single user queries, not conversations)
            if user_query and len(messages) <= 1:
                cached_response = _response_cache.get(user_query, self.model, knowledge_source)
                if cached_response:
                    cached_response['from_cache'] = True
                    return cached_response

            # Build context if this is a new user message (with caching)
            context = ""
            intent = {'intent': 'general'}
            if user_query:
                context, intent = self._get_cached_context(user_query, knowledge_source)

            # Prepare system message with context
            system_content = ATHLETICS_KNOWLEDGE
            if context and context != "General athletics query - using base knowledge.":
                system_content += f"\n\nRELEVANT DATA:\n{context}"

            # Build messages with system prompt
            full_messages = [{"role": "system", "content": system_content}]
            full_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=2048,
                temperature=0.3  # Lower temperature for factual analytics
            )

            result = {
                "success": True,
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0
                },
                "context_used": context,
                "query_intent": intent['intent'],
                "from_cache": False
            }

            # Cache the response for single queries
            if user_query and len(messages) <= 1:
                _response_cache.set(user_query, self.model, knowledge_source, result)

            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": f"Error: {str(e)}"
            }

    def chat_stream(self, messages: list, user_query: str = None, knowledge_source: str = "Both"):
        """
        Send chat completion request with streaming and response caching.
        Yields content chunks for real-time display.
        Returns context at the end.
        """
        try:
            # Check response cache first (only for single user queries)
            if user_query and len(messages) <= 1:
                cached_response = _response_cache.get(user_query, self.model, knowledge_source)
                if cached_response and cached_response.get('content'):
                    # Yield cached content as a single chunk for instant display
                    yield {"type": "content", "content": cached_response['content']}
                    yield {
                        "type": "done",
                        "full_content": cached_response['content'],
                        "context_used": cached_response.get('context_used', ''),
                        "chunks_received": 1,
                        "from_cache": True,
                        "query_intent": cached_response.get('query_intent', 'general')
                    }
                    return

            # Build context if this is a new user message (with intent classification)
            context = ""
            intent = {'intent': 'general'}
            if user_query:
                context, intent = self._get_cached_context(user_query, knowledge_source)

            # Prepare system message with context
            system_content = ATHLETICS_KNOWLEDGE
            if context and context != "General athletics query - using base knowledge.":
                system_content += f"\n\nRELEVANT DATA:\n{context}"

            # Build messages with system prompt
            full_messages = [{"role": "system", "content": system_content}]
            full_messages.extend(messages)

            # Create streaming response with OpenRouter headers
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=2048,
                temperature=0.3,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://team-saudi-athletics.streamlit.app",
                    "X-Title": "Team Saudi Athletics Dashboard"
                }
            )

            # Yield chunks
            full_content = ""
            chunk_count = 0
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    chunk_count += 1
                    yield {"type": "content", "content": content}

            # Cache the complete response for future queries
            if user_query and len(messages) <= 1 and full_content:
                _response_cache.set(user_query, self.model, knowledge_source, {
                    'success': True,
                    'content': full_content,
                    'context_used': context,
                    'query_intent': intent['intent']
                })

            # Yield final metadata (even if no content chunks received)
            yield {
                "type": "done",
                "full_content": full_content,
                "context_used": context,
                "chunks_received": chunk_count,
                "from_cache": False,
                "query_intent": intent['intent']
            }

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            yield {"type": "error", "error": str(e), "detail": error_detail}

###################################
# 1) Streamlit Setup
###################################
st.set_page_config(
    page_title="Saudi Athletics Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

###################################
# 2) Team Saudi Styling
###################################
def apply_team_saudi_theme():
    css = f"""
    <style>
    .stApp {{
        background-color: #0a0a0a !important;
        color: white !important;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.8) !important;
        padding: 2rem;
        border-radius: 12px;
        color: white !important;
    }}
    /* Only apply teal to main content headings, not inside alert boxes */
    .block-container > div > h1,
    .block-container > div > h2,
    .block-container > div > h3,
    .block-container > div > h4,
    .block-container > div > h5,
    .block-container > div > h6,
    .stTabs h1, .stTabs h2, .stTabs h3 {{
        color: {TEAL_PRIMARY} !important;
    }}
    /* Ensure text inside success/warning/info boxes is readable */
    .stAlert p, .stAlert div, .stAlert span {{
        color: inherit !important;
    }}
    .stSuccess p, .stSuccess div {{
        color: white !important;
    }}
    label, .stTextInput label, .stSelectbox label, .stSlider label {{
        color: #DDD !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #aaa;
        background-color: #111;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {TEAL_PRIMARY};
        border-bottom: 3px solid {TEAL_PRIMARY};
    }}
    .stDataFrame, .stTable {{
        background-color: rgba(255, 255, 255, 0.03) !important;
        color: white !important;
    }}
    .athlete-card {{
        background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
    }}
    .athlete-card h1, .athlete-card h2, .athlete-card h3, .athlete-card h4 {{
        color: white !important;
    }}
    .metric-card {{
        background: rgba(0, 113, 103, 0.15);
        border: 1px solid {TEAL_PRIMARY};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}
    .gold-highlight {{
        color: {GOLD_ACCENT} !important;
        font-weight: bold;
    }}
    .standard-card {{
        background: rgba(160, 142, 102, 0.2);
        border: 1px solid {GOLD_ACCENT};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    /* Ensure subheaders in expanders are visible */
    .streamlit-expanderHeader {{
        color: {TEAL_PRIMARY} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_team_saudi_theme()

###################################
# 3) Database Paths
###################################
SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'world_athletics', 'Data')

DB_MEN_RANK = os.path.join(SQL_DIR, 'rankings_men_all_events.db')
DB_WOMEN_RANK = os.path.join(SQL_DIR, 'rankings_women_all_events.db')
DB_KSA_MEN = os.path.join(SQL_DIR, 'ksa_modal_results_men.db')
DB_KSA_WOMEN = os.path.join(SQL_DIR, 'ksa_modal_results_women.db')
DB_PROFILES = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')

###################################
# 4) Data Loading Functions
###################################

# Cache TTL: 30 minutes for development, 1 hour for production
# Reduced from 3600 to 1800 for faster iteration during development
CACHE_TTL = 1800

@st.cache_data
def load_sqlite_table(db_path, table_name):
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading athlete profiles...")
def load_athlete_profiles():
    """Load all athlete profile data from Azure Parquet or local SQLite."""
    # Try Azure/Parquet first (for Streamlit Cloud)
    if DATA_CONNECTOR_AVAILABLE:
        try:
            # Debug: show data mode
            mode = get_data_mode()
            if mode == "local":
                # Check if secrets exist
                has_secret = 'AZURE_STORAGE_CONNECTION_STRING' in st.secrets if hasattr(st, 'secrets') else False
                st.info(f"Data mode: {mode} | Secret found: {has_secret}")

            athletes = get_ksa_athletes()
            if athletes is not None and not athletes.empty:
                # Return athletes with empty dataframes for other tables
                # (rankings, breakdown, pbs, progression not in Parquet yet)
                return athletes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            st.warning(f"Azure data connector error: {e}")

    # Fall back to local SQLite
    if not os.path.exists(DB_PROFILES):
        return None, None, None, None, None

    try:
        conn = sqlite3.connect(DB_PROFILES)

        athletes = pd.read_sql('SELECT * FROM ksa_athletes', conn)
        rankings = pd.read_sql('SELECT * FROM athlete_rankings', conn)
        breakdown = pd.read_sql('SELECT * FROM ranking_breakdown', conn)
        pbs = pd.read_sql('SELECT * FROM athlete_pbs', conn)
        progression = pd.read_sql('SELECT * FROM athlete_progression', conn)

        conn.close()

        return athletes, rankings, breakdown, pbs, progression
    except Exception as e:
        return None, None, None, None, None

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading qualification data...")
def load_road_to_data():
    """Load Road to Tokyo qualification data from Azure or local CSV."""
    # Try Azure parquet first
    if DATA_CONNECTOR_AVAILABLE:
        try:
            df = get_road_to_tokyo_data()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            st.warning(f"Azure road to tokyo error: {e}")

    # Fall back to local CSV files
    try:
        road_to_path = os.path.join(DATA_DIR, 'road_to')
        if os.path.exists(road_to_path):
            csv_files = glob.glob(os.path.join(road_to_path, 'road_to_tokyo_batch_*.csv'))
            if csv_files:
                all_data = []
                for f in csv_files:
                    try:
                        df = pd.read_csv(f)
                        all_data.append(df)
                    except:
                        pass
                if all_data:
                    return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading standards...")
def load_qualification_standards():
    """Load qualification standards from Azure or local CSV."""
    # Try Azure benchmarks first
    if DATA_CONNECTOR_AVAILABLE:
        try:
            df = get_benchmarks_data()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            st.warning(f"Azure benchmarks error: {e}")

    # Fall back to local CSV
    try:
        qual_path = os.path.join(DATA_DIR, 'qualification_processes', 'qualification_processes_summary.csv')
        if os.path.exists(qual_path):
            return pd.read_csv(qual_path)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

###################################
# 5) Historical Standards Data
# (What it takes to win/make finals)
###################################
# Historical World Championship winning marks (approximate)
HISTORICAL_WINNING_MARKS = {
    '100m': {
        'men': {
            2023: 9.83, 2022: 9.86, 2019: 9.76, 2017: 9.92, 2015: 9.79,
            2013: 9.77, 2011: 9.92, 2009: 9.58, 2007: 9.85, 2005: 9.88
        },
        'women': {
            2023: 10.65, 2022: 10.67, 2019: 10.71, 2017: 10.71, 2015: 10.76,
            2013: 10.71, 2011: 10.90, 2009: 10.73, 2007: 11.01, 2005: 10.93
        }
    },
    '200m': {
        'men': {
            2023: 19.52, 2022: 19.73, 2019: 19.83, 2017: 20.09, 2015: 19.55,
            2013: 19.66, 2011: 19.40, 2009: 19.19, 2007: 19.76, 2005: 20.04
        },
        'women': {
            2023: 21.41, 2022: 21.81, 2019: 21.88, 2017: 22.05, 2015: 21.63,
            2013: 22.22, 2011: 22.04, 2009: 22.02, 2007: 22.34, 2005: 22.16
        }
    },
    '400m': {
        'men': {
            2023: 44.29, 2022: 43.71, 2019: 43.48, 2017: 43.98, 2015: 43.48,
            2013: 43.74, 2011: 44.60, 2009: 44.06, 2007: 43.45, 2005: 43.96
        },
        'women': {
            2023: 48.53, 2022: 50.07, 2019: 48.97, 2017: 49.46, 2015: 49.26,
            2013: 49.41, 2011: 49.89, 2009: 49.64, 2007: 49.89, 2005: 49.55
        }
    },
    '800m': {
        'men': {
            2023: '1:41.56', 2022: '1:43.71', 2019: '1:42.17', 2017: '1:44.67', 2015: '1:45.84',
            2013: '1:43.31', 2011: '1:43.91', 2009: '1:44.31', 2007: '1:47.09', 2005: '1:44.24'
        },
        'women': {
            2023: '1:56.00', 2022: '1:56.30', 2019: '1:58.04', 2017: '1:55.16', 2015: '1:58.90',
            2013: '1:57.38', 2011: '1:55.87', 2009: '1:55.45', 2007: '1:56.04', 2005: '1:58.82'
        }
    },
    '1500m': {
        'men': {
            2023: '3:29.38', 2022: '3:30.69', 2019: '3:31.70', 2017: '3:33.61', 2015: '3:34.40',
            2013: '3:36.28', 2011: '3:35.69', 2009: '3:35.93', 2007: '3:34.77', 2005: '3:37.88'
        },
        'women': {
            2023: '3:54.87', 2022: '3:52.96', 2019: '3:54.22', 2017: '4:02.90', 2015: '4:08.09',
            2013: '4:02.67', 2011: '4:05.40', 2009: '4:03.74', 2007: '3:58.75', 2005: '4:00.24'
        }
    },
    'long-jump': {
        'men': {
            2023: 8.52, 2022: 8.08, 2019: 8.69, 2017: 8.48, 2015: 8.41,
            2013: 8.56, 2011: 8.45, 2009: 8.54, 2007: 8.57, 2005: 8.60
        },
        'women': {
            2023: 7.12, 2022: 7.09, 2019: 7.30, 2017: 7.02, 2015: 7.14,
            2013: 7.01, 2011: 6.82, 2009: 7.10, 2007: 7.03, 2005: 7.01
        }
    },
    'high-jump': {
        'men': {
            2023: 2.36, 2022: 2.33, 2019: 2.37, 2017: 2.35, 2015: 2.34,
            2013: 2.41, 2011: 2.35, 2009: 2.32, 2007: 2.35, 2005: 2.32
        },
        'women': {
            2023: 2.01, 2022: 2.02, 2019: 2.04, 2017: 2.03, 2015: 2.01,
            2013: 2.03, 2011: 2.05, 2009: 2.01, 2007: 2.03, 2005: 2.00
        }
    },
    '110mh': {
        'men': {
            2023: 12.98, 2022: 13.03, 2019: 12.98, 2017: 13.04, 2015: 12.98,
            2013: 12.92, 2011: 13.16, 2009: 13.14, 2007: 12.95, 2005: 13.07
        }
    },
    '400mh': {
        'men': {
            2023: 45.00, 2022: 46.29, 2019: 47.42, 2017: 48.35, 2015: 47.79,
            2013: 47.69, 2011: 48.26, 2009: 47.91, 2007: 47.61, 2005: 47.30
        },
        'women': {
            2023: 50.58, 2022: 50.68, 2019: 52.16, 2017: 52.64, 2015: 53.50,
            2013: 52.83, 2011: 52.47, 2009: 52.42, 2007: 53.31, 2005: 52.90
        }
    }
}

# Finals qualifying marks (8th place marks) - approximate
FINALS_QUALIFYING_MARKS = {
    '100m': {'men': {2023: 10.02, 2022: 10.05, 2019: 10.01}, 'women': {2023: 10.98, 2022: 11.02, 2019: 11.05}},
    '200m': {'men': {2023: 20.12, 2022: 20.25, 2019: 20.18}, 'women': {2023: 22.35, 2022: 22.45, 2019: 22.40}},
    '400m': {'men': {2023: 44.95, 2022: 45.10, 2019: 44.88}, 'women': {2023: 50.25, 2022: 50.50, 2019: 50.30}},
}


def convert_time_to_seconds(time_val):
    """Convert time string to seconds for comparison."""
    if pd.isna(time_val):
        return None
    if isinstance(time_val, (int, float)):
        return float(time_val)
    try:
        time_str = str(time_val)
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(time_str)
    except:
        return None


###################################
# 5) Header with Logo
###################################
# Load logo for header
import base64

def get_logo_base64():
    """Load Saudi logo and convert to base64 for HTML embedding."""
    logo_path = os.path.join(os.path.dirname(__file__), 'Saudilogo.png')
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_base64()

# Build header HTML - logo is optional
if logo_b64:
    header_html = f'''<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; display: flex; align-items: center;"><img src="data:image/png;base64,{logo_b64}" style="height: 60px; margin-right: 1rem;"><div><h1 style="color: white; margin: 0; font-size: 2rem;">Saudi Athletics Dashboard</h1><p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">World Rankings, Performance Analysis &amp; Road to Tokyo 2025</p></div></div>'''
else:
    header_html = f'''<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;"><h1 style="color: white; margin: 0; font-size: 2rem;">Saudi Athletics Dashboard</h1><p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">World Rankings, Performance Analysis &amp; Road to Tokyo 2025</p></div>'''

st.markdown(header_html, unsafe_allow_html=True)

# Database last updated info
if os.path.exists(DB_MEN_RANK):
    try:
        modified_time = os.path.getmtime(DB_MEN_RANK)
        modified_dt = datetime.datetime.fromtimestamp(modified_time)
        st.markdown(f"""
            <p style='text-align: center; color: #888; margin-top: -1rem; font-size: 0.85rem;'>
                Rankings last updated: {modified_dt.strftime('%d %b %Y, %H:%M')}
            </p>
        """, unsafe_allow_html=True)
    except:
        pass

###################################
# 6) Load Data
###################################

# Try Azure parquet first, fall back to SQLite
@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading men's rankings...")
def load_men_rankings():
    """Load men's rankings from Azure or SQLite."""
    if DATA_CONNECTOR_AVAILABLE:
        try:
            df = get_rankings_data(gender='Men')
            if df is not None and not df.empty:
                # Rename columns to match expected format
                col_map = {
                    'event': 'Event Type',
                    'rank': 'Rank',
                    'competitor': 'Name',
                    'nat': 'Country',
                    'result': 'Score'
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                return df
        except Exception as e:
            st.warning(f"Azure rankings error: {e}")
    return load_sqlite_table(DB_MEN_RANK, 'rankings_men_all_events')

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading women's rankings...")
def load_women_rankings():
    """Load women's rankings from Azure or SQLite."""
    if DATA_CONNECTOR_AVAILABLE:
        try:
            df = get_rankings_data(gender='Women')
            if df is not None and not df.empty:
                col_map = {
                    'event': 'Event Type',
                    'rank': 'Rank',
                    'competitor': 'Name',
                    'nat': 'Country',
                    'result': 'Score'
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                return df
        except Exception as e:
            st.warning(f"Azure rankings error: {e}")
    return load_sqlite_table(DB_WOMEN_RANK, 'rankings_women_all_events')

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading KSA rankings...")
def load_ksa_combined_rankings():
    """Load KSA rankings from Azure or SQLite."""
    if DATA_CONNECTOR_AVAILABLE:
        try:
            df = get_ksa_rankings()
            if df is not None and not df.empty:
                col_map = {
                    'event': 'Event Type',
                    'rank': 'Rank',
                    'competitor': 'Name',
                    'nat': 'Country',
                    'result': 'Score'
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                return df
        except Exception as e:
            st.warning(f"Azure KSA rankings error: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading full rankings (2.3M records)...")
def load_full_rankings_cached():
    """
    Cached wrapper for full rankings data (2.3M rows).
    Use this instead of calling get_rankings_data() directly to avoid duplicate Azure downloads.
    """
    if DATA_CONNECTOR_AVAILABLE:
        try:
            df = get_rankings_data()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            st.warning(f"Azure full rankings error: {e}")
    return pd.DataFrame()

def load_ksa_rankings_raw_cached():
    """
    Wrapper for KSA rankings WITHOUT column renaming.
    Use this for internal code that expects original column names (competitor, event, result, etc).
    Note: No caching here - get_ksa_rankings() is already cached.
    """
    if DATA_CONNECTOR_AVAILABLE:
        try:
            df = get_ksa_rankings()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            pass  # Silent fail - caller checks for empty
    return pd.DataFrame()

# NOTE: Large rankings data is loaded lazily in tabs that need it
# This improves startup performance significantly

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ai_filter_options():
    """
    Cached function to get athlete and event lists for AI chatbot filters.
    Prevents repeated data loading on every interaction.
    """
    athletes = ["(All Athletes)"]
    events = ["(All Events)"]

    if DATA_CONNECTOR_AVAILABLE:
        try:
            ksa_df = get_ksa_athletes()
            if ksa_df is not None and not ksa_df.empty:
                # Get athlete names
                name_col = 'competitor' if 'competitor' in ksa_df.columns else 'full_name' if 'full_name' in ksa_df.columns else None
                if name_col:
                    athlete_names = sorted(ksa_df[name_col].dropna().unique().tolist())
                    athletes.extend(athlete_names)

                # Get events
                if 'event' in ksa_df.columns:
                    event_list = sorted(ksa_df['event'].dropna().unique().tolist())
                    events.extend(event_list)
                elif 'primary_event' in ksa_df.columns:
                    event_list = sorted(ksa_df['primary_event'].dropna().unique().tolist())
                    events.extend(event_list)
        except Exception:
            pass

    return athletes, events

try:
    ksa_men_results = load_sqlite_table(DB_KSA_MEN, 'ksa_modal_results_men')
except:
    ksa_men_results = None

try:
    ksa_women_results = load_sqlite_table(DB_KSA_WOMEN, 'ksa_modal_results_women')
except:
    ksa_women_results = None

# Load athlete profiles (smaller dataset - OK to load upfront)
athletes_df, rankings_df, breakdown_df, pbs_df, progression_df = load_athlete_profiles()

def paginate_dataframe(df: pd.DataFrame, page_size: int = 100, page_num: int = 0) -> pd.DataFrame:
    """Return a page of the dataframe for display."""
    start_idx = page_num * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]

# Load Road to Tokyo and qualification data (lazy load)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_road_to_df():
    return load_road_to_data()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_qual_standards_df():
    return load_qualification_standards()

# Global cached WhatItTakesToWin analyzer (used by Tab 1 and Tab 7)
@st.cache_resource
def get_wittw_analyzer():
    """Load and cache the WhatItTakesToWin analyzer with scraped championship data."""
    analyzer = WhatItTakesToWin()
    analyzer.load_scraped_data()
    return analyzer

###################################
# 7) Tabs
###################################
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    'Event Standards & Progression',
    'Athlete Profiles',
    'Combined Rankings',
    'Saudi Athletes Rankings',
    'World Champs Qualification',
    'Major Games Analytics',
    'What It Takes to Win (Live)',
    'AI Analyst',
    'Coach View',
    'Project East 2026',
    'Race Intelligence'
])

###################################
# Tab 1: Event Standards & Progression (NEW)
###################################
with tab1:
    st.header('What It Takes to Win')
    st.markdown(f"""
    <p style='color: #ccc; font-size: 0.95em;'>
    Year-on-year progression of winning performances and finals qualifying marks at
    <strong style="color: #FFD700;">Olympics</strong>,
    <strong style="color: {GOLD_ACCENT};">World Championships</strong>,
    <strong style="color: {TEAL_PRIMARY};">Asian Games</strong>, and more.
    Compare with KSA athlete performances.
    </p>
    """, unsafe_allow_html=True)

    # Load WhatItTakesToWin analyzer for live data
    wittw_tab1 = get_wittw_analyzer()

    # Selection filters - Apply championship filter FIRST to get available events
    col1, col2, col3 = st.columns(3)

    # Pre-filter data by championship type for event dropdown
    tab1_filtered_data = None
    tab1_available_events = list(HISTORICAL_WINNING_MARKS.keys())  # Default fallback

    with col1:
        # Championship type dropdown - use keys matching MAJOR_COMP_KEYWORDS
        comp_types = ['All', 'World Champs', 'Olympic', 'Asian Games', 'Diamond League']
        if wittw_tab1.data is not None:
            # Get actual competition types from data
            actual_types = wittw_tab1.get_competition_types()
            if actual_types:
                comp_types = actual_types
        selected_comp_tab1 = st.selectbox("Championship Type", comp_types, key="tab1_comp_type")

        # Apply championship filter immediately for event dropdown
        if wittw_tab1.data is not None and len(wittw_tab1.data) > 0:
            tab1_filtered_data = wittw_tab1.filter_by_competition(selected_comp_tab1)
            if tab1_filtered_data is not None and len(tab1_filtered_data) > 0:
                # Get events from filtered data
                event_col = 'event' if 'event' in tab1_filtered_data.columns else 'Event'
                if event_col in tab1_filtered_data.columns:
                    live_events = sorted(tab1_filtered_data[event_col].dropna().unique().tolist())
                    if live_events:
                        tab1_available_events = live_events

    with col2:
        selected_event = st.selectbox("Select Event", tab1_available_events, key="standards_event")

    with col3:
        gender_options = ['men', 'women']
        # Get genders from filtered data if available
        if tab1_filtered_data is not None:
            gender_col = 'gender' if 'gender' in tab1_filtered_data.columns else 'Gender'
            if gender_col in tab1_filtered_data.columns:
                genders = tab1_filtered_data[gender_col].dropna().unique()
                genders_normalized = sorted(set([g.lower() if isinstance(g, str) else 'men' for g in genders]))
                if genders_normalized:
                    gender_options = genders_normalized
        selected_gender = st.selectbox("Select Gender", gender_options, key="standards_gender")

    # Get LIVE data from WhatItTakesToWin analyzer (using pre-filtered data from above)
    live_data_available = False
    df_progression = None

    # Show filter diagnostic
    if tab1_filtered_data is not None:
        champ_type_col = 'championship_type' if 'championship_type' in tab1_filtered_data.columns else None
        if champ_type_col:
            unique_types = tab1_filtered_data[champ_type_col].dropna().unique().tolist()
            st.caption(f"Filter: {selected_comp_tab1} | Data contains: {', '.join(unique_types[:5])} | Records: {len(tab1_filtered_data):,}")
        else:
            st.caption(f"Filter: {selected_comp_tab1} (no championship_type column) | Records: {len(tab1_filtered_data):,}")

    if tab1_filtered_data is not None and len(tab1_filtered_data) > 0:
        # Use pre-filtered data (already filtered by championship type above)
        filtered_data = tab1_filtered_data

        if filtered_data is not None and len(filtered_data) > 0:
            # Filter by event and gender - handle column name case variations
            event_col = 'event' if 'event' in filtered_data.columns else 'Event'
            gender_col = 'gender' if 'gender' in filtered_data.columns else 'Gender'

            if event_col not in filtered_data.columns or gender_col not in filtered_data.columns:
                st.warning(f"Data missing required columns. Available: {list(filtered_data.columns)[:10]}")
            else:
                # Normalize event matching: '100m' should match '100-metres', '100 metres', '100m', etc.
                # Extract the base event (e.g., '100' from '100m' or '100-metres')
                import re
                event_base = re.sub(r'[^0-9a-z]', '', selected_event.lower())  # '100m' -> '100m', '100-metres' -> '100metres'
                # Also try just the number for distance events
                event_num = re.sub(r'[^0-9]', '', selected_event)  # '100m' -> '100'

                # Build flexible mask: match if event contains the base OR the number followed by word boundary
                event_lower = filtered_data[event_col].str.lower().str.replace('-', '', regex=False).str.replace(' ', '', regex=False)
                event_mask = event_lower.str.contains(event_base, na=False) | filtered_data[event_col].str.contains(f'{event_num}', na=False)
                # Flexible gender matching: 'men'/'M'/'male' or 'women'/'W'/'F'/'female'
                gender_lower = filtered_data[gender_col].astype(str).str.lower().str.strip()
                if selected_gender.lower() == 'men':
                    gender_mask = gender_lower.isin(['men', 'm', 'male', 'man'])
                else:
                    gender_mask = gender_lower.isin(['women', 'w', 'f', 'female', 'woman'])

                event_data = filtered_data[event_mask & gender_mask].copy()

                if len(event_data) > 0:
                    # Extract year from date - handle column name variations
                    date_col = 'date' if 'date' in event_data.columns else 'Date'
                    # Use 'pos' column for finishing position (1st, 2nd, etc), NOT 'rank' which is world ranking
                    pos_col = 'pos' if 'pos' in event_data.columns else ('Pos' if 'Pos' in event_data.columns else None)

                    if date_col in event_data.columns:
                        event_data['Year'] = pd.to_datetime(event_data[date_col], errors='coerce').dt.year
                        event_data = event_data.dropna(subset=['Year'])
                        event_data['Year'] = event_data['Year'].astype(int)

                        # Get winning marks (position 1) by year using 'pos' column
                        winners = pd.DataFrame()
                        if pos_col and pos_col in event_data.columns:
                            # Convert pos to numeric for reliable comparison (handles '1' strings)
                            event_data['_pos_num'] = pd.to_numeric(event_data[pos_col], errors='coerce')
                            winners = event_data[event_data['_pos_num'] == 1].copy()

                        # Fallback: if no pos=1 found, find best result per year
                        if len(winners) == 0:
                            result_num_col = 'result_numeric' if 'result_numeric' in event_data.columns else 'Result_Numeric'
                            if result_num_col in event_data.columns:
                                is_field = any(x in selected_event.lower() for x in ['shot', 'discus', 'hammer', 'javelin', 'jump', 'vault', 'throw'])
                                # For track: min time wins. For field: max distance wins
                                if is_field:
                                    idx = event_data.groupby('Year')[result_num_col].idxmax()
                                else:
                                    idx = event_data.groupby('Year')[result_num_col].idxmin()
                                winners = event_data.loc[idx].copy()

                        if len(winners) > 0:
                                # Group by year, get best result (for track: min, for field: max)
                                _is_field = is_field_event(selected_event)
                                result_col = 'result' if 'result' in winners.columns else 'Result'
                                result_num_col = 'result_numeric' if 'result_numeric' in winners.columns else 'Result_Numeric'

                                if result_col in winners.columns and result_num_col in winners.columns:
                                    winning_by_year = winners.groupby('Year').agg({
                                        result_col: 'first',
                                        result_num_col: 'max' if _is_field else 'min'
                                    }).reset_index()

                                    years = sorted(winning_by_year['Year'].tolist())
                                    marks = [winning_by_year[winning_by_year['Year'] == y][result_col].iloc[0] for y in years]
                                    marks_seconds = [winning_by_year[winning_by_year['Year'] == y][result_num_col].iloc[0] for y in years]

                                    df_progression = pd.DataFrame({
                                        'Year': years,
                                        'Winning Mark': marks,
                                        'Mark (seconds/meters)': marks_seconds
                                    })
                                    live_data_available = True

    # Fallback to hardcoded data ONLY for World Champs or All (HISTORICAL_WINNING_MARKS is World Champs data)
    # Do NOT show static World Champs data when user selects Olympic, Asian Games, etc.
    static_data_valid = selected_comp_tab1 in ['All', 'World Champs']
    if not live_data_available and static_data_valid and selected_event in HISTORICAL_WINNING_MARKS and selected_gender in HISTORICAL_WINNING_MARKS[selected_event]:
        event_data_static = HISTORICAL_WINNING_MARKS[selected_event][selected_gender]
        years = sorted(event_data_static.keys())
        marks = [event_data_static[y] for y in years]
        marks_seconds = [convert_time_to_seconds(m) for m in marks]
        df_progression = pd.DataFrame({
            'Year': years,
            'Winning Mark': marks,
            'Mark (seconds/meters)': marks_seconds
        })

    if df_progression is None or len(df_progression) == 0:
        # No data available for this filter combination
        st.warning(f"No data available for **{selected_event}** ({selected_gender}) at **{selected_comp_tab1}** championships. "
                   f"Try selecting 'All' or a different championship type.")
        st.info("The event list updates based on available data for the selected championship type. "
                "If an event doesn't appear, it may not have data for that championship.")

    if df_progression is not None and len(df_progression) > 0:
        # Display metrics
        champ_label = selected_comp_tab1 if selected_comp_tab1 != 'All' else 'Major Championships'
        data_source = "Live Data" if live_data_available else "Historical Archive (World Champs only)"
        st.subheader(f"{selected_event.upper()} - {selected_gender.capitalize()} Winning Progression ({champ_label})")
        st.caption(f"Data source: {data_source} | {len(df_progression)} years of data")

        col1, col2, col3, col4 = st.columns(4)

        marks = df_progression['Winning Mark'].tolist()
        marks_seconds = df_progression['Mark (seconds/meters)'].tolist()
        years = df_progression['Year'].tolist()

        with col1:
            latest_year = max(years) if years else 'N/A'
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: {GOLD_ACCENT}; margin: 0; font-size: 0.85rem;">Latest ({latest_year})</p>
                <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0;">{marks[-1] if marks else 'N/A'}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Determine best mark based on event type
            is_field = any(x in selected_event.lower() for x in ['shot', 'discus', 'hammer', 'javelin', 'jump', 'vault', 'throw'])
            valid_seconds = [m for m in marks_seconds if m is not None and m > 0]
            best_idx = marks_seconds.index(max(valid_seconds)) if is_field and valid_seconds else marks_seconds.index(min(valid_seconds)) if valid_seconds else 0
            best_mark = marks[best_idx] if marks else 'N/A'
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: {TEAL_LIGHT}; margin: 0; font-size: 0.85rem;">Best Ever</p>
                <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0;">{best_mark}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            improvement = marks_seconds[-1] - marks_seconds[0] if len(marks_seconds) >= 2 and all(m for m in [marks_seconds[0], marks_seconds[-1]]) else 0
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #aaa; margin: 0; font-size: 0.85rem;">Change</p>
                <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0;">{improvement:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # Show qualification standard if available
            qual_mark = "N/A"
            qual_standards_df = get_qual_standards_df()
            if qual_standards_df is not None and not qual_standards_df.empty:
                event_search = selected_event.replace('-', ' ')
                if 'Display_Name' in qual_standards_df.columns:
                    event_qual = qual_standards_df[qual_standards_df['Display_Name'].str.contains(event_search, case=False, na=False)]
                    if not event_qual.empty:
                        qual_mark = event_qual.iloc[0].get('entry_standard', 'N/A')
                elif 'Event' in qual_standards_df.columns:
                    event_qual = qual_standards_df[qual_standards_df['Event'].str.contains(event_search, case=False, na=False)]
                    if not event_qual.empty:
                        qual_mark = event_qual.iloc[0].get('Gold Standard', 'N/A')

            st.markdown(f"""
            <div class="standard-card">
                <p style="color: {GOLD_ACCENT}; margin: 0; font-size: 0.85rem;">Gold Standard</p>
                <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{qual_mark}</p>
            </div>
            """, unsafe_allow_html=True)

        # Progression Chart
        fig = go.Figure()

        # Winning marks line - with dynamic championship name
        fig.add_trace(go.Scatter(
            x=df_progression['Year'],
            y=df_progression['Mark (seconds/meters)'],
            mode='lines+markers',
            name=f'{champ_label} Winner',
            line=dict(color=GOLD_ACCENT, width=3),
            marker=dict(size=10, symbol='star'),
            text=df_progression['Winning Mark'],
            hovertemplate="<b>%{x}</b><br>Winning Mark: %{text}<extra></extra>"
        ))

        # Add 8th place line from live data if available
        if wittw_tab1.data is not None and len(wittw_tab1.data) > 0:
            filtered_data = wittw_tab1.filter_by_competition(selected_comp_tab1)
            if filtered_data is not None and len(filtered_data) > 0:
                # Handle column name case variations
                event_col = 'event' if 'event' in filtered_data.columns else 'Event'
                gender_col = 'gender' if 'gender' in filtered_data.columns else 'Gender'
                date_col = 'date' if 'date' in filtered_data.columns else 'Date'
                # Use 'pos' column for finishing position, NOT 'rank' which is world ranking
                pos_col = 'pos' if 'pos' in filtered_data.columns else ('Pos' if 'Pos' in filtered_data.columns else None)
                result_num_col = 'result_numeric' if 'result_numeric' in filtered_data.columns else 'Result_Numeric'

                if event_col in filtered_data.columns and gender_col in filtered_data.columns:
                    # Normalize event matching (same as above)
                    event_base = re.sub(r'[^0-9a-z]', '', selected_event.lower())
                    event_num = re.sub(r'[^0-9]', '', selected_event)
                    event_lower = filtered_data[event_col].str.lower().str.replace('-', '', regex=False).str.replace(' ', '', regex=False)
                    event_mask = event_lower.str.contains(event_base, na=False) | filtered_data[event_col].str.contains(f'{event_num}', na=False)
                    gender_mask = filtered_data[gender_col].str.lower() == selected_gender.lower()
                    event_data = filtered_data[event_mask & gender_mask].copy()

                    if len(event_data) > 0 and date_col in event_data.columns:
                        event_data['Year'] = pd.to_datetime(event_data[date_col], errors='coerce').dt.year
                        event_data = event_data.dropna(subset=['Year'])
                        event_data['Year'] = event_data['Year'].astype(int)

                        # Get 8th place marks using 'pos' column for finishing position
                        if pos_col and pos_col in event_data.columns and result_num_col in event_data.columns:
                            event_data['_pos_num'] = pd.to_numeric(event_data[pos_col], errors='coerce')
                            eighth = event_data[event_data['_pos_num'] == 8].copy()
                            if len(eighth) > 0:
                                is_field = any(x in selected_event.lower() for x in ['shot', 'discus', 'hammer', 'javelin', 'jump', 'vault', 'throw'])
                                eighth_by_year = eighth.groupby('Year').agg({
                                    result_num_col: 'max' if is_field else 'min'
                                }).reset_index()

                                fig.add_trace(go.Scatter(
                                    x=eighth_by_year['Year'],
                                    y=eighth_by_year[result_num_col],
                                    mode='lines+markers',
                                    name='Finals Qualifying (8th place)',
                                    line=dict(color=TEAL_LIGHT, width=2, dash='dash'),
                                    marker=dict(size=8),
                                    hovertemplate="<b>%{x}</b><br>8th Place: %{y}<extra></extra>"
                                ))

        # Fallback to hardcoded 8th place data
        elif selected_event in FINALS_QUALIFYING_MARKS and selected_gender in FINALS_QUALIFYING_MARKS[selected_event]:
            finals_data = FINALS_QUALIFYING_MARKS[selected_event][selected_gender]
            finals_years = sorted(finals_data.keys())
            finals_marks = [finals_data[y] for y in finals_years]

            fig.add_trace(go.Scatter(
                x=finals_years,
                y=finals_marks,
                mode='lines+markers',
                name='Finals Qualifying (8th place)',
                line=dict(color=TEAL_LIGHT, width=2, dash='dash'),
                marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>Finals Mark: %{y}<extra></extra>"
            ))

        # Add KSA athlete PBs if available
        if pbs_df is not None and not pbs_df.empty:
            event_pattern = selected_event.replace('-', ' ').replace('m', ' m')
            ksa_pbs = pbs_df[pbs_df['event_name'].str.lower().str.contains(selected_event.replace('-', '').lower(), na=False)]

            if not ksa_pbs.empty:
                for _, pb in ksa_pbs.iterrows():
                    pb_val = convert_time_to_seconds(pb['pb_result'])
                    if pb_val:
                        athlete_name = "KSA Athlete"
                        if athletes_df is not None and not athletes_df.empty:
                            athlete_info = athletes_df[athletes_df['athlete_id'] == pb['athlete_id']]
                            if not athlete_info.empty:
                                athlete_name = athlete_info.iloc[0]['full_name']

                        fig.add_hline(
                            y=pb_val,
                            line_dash="dot",
                            line_color=TEAL_PRIMARY,
                            annotation_text=f"{athlete_name}: {pb['pb_result']}",
                            annotation_position="right"
                        )

        fig.update_layout(
            title=f"{selected_event.upper()} - {champ_label} Winning Progression ({selected_gender.capitalize()})",
            xaxis_title="Year",
            yaxis_title="Mark (seconds/meters)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )

        st.plotly_chart(fig, width='stretch')

        # Show data table
        st.subheader("Historical Data")
        st.dataframe(df_progression, width='stretch', hide_index=True)
    else:
        st.warning(f"No winning progression data available for {selected_event} ({selected_gender}) in {selected_comp_tab1}")
        # Debug info for troubleshooting
        if wittw_tab1.data is not None:
            filtered = wittw_tab1.filter_by_competition(selected_comp_tab1)
            st.caption(f"Debug: {len(wittw_tab1.data):,} total records, {len(filtered) if filtered is not None else 0:,} after {selected_comp_tab1} filter")

    # === FINALS PERFORMANCE PROGRESSION CHART (1st-8th over time) ===
    st.markdown("---")
    st.subheader("Finals Performance Progression (1st-8th)")
    st.markdown(f"<p style='color: #aaa;'>How performances have evolved across championships for positions 1-8</p>", unsafe_allow_html=True)

    # Position filter controls
    all_positions = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
    pos_col1, pos_col2 = st.columns([3, 1])
    with pos_col1:
        selected_positions = st.multiselect(
            "Show positions:",
            options=all_positions,
            default=['1st', '2nd', '3rd', '8th'],  # Default: medals + 8th for gap analysis
            key="tab1_positions"
        )
    with pos_col2:
        if st.button("Select All", key="tab1_select_all"):
            st.session_state['tab1_positions'] = all_positions
            st.rerun()

    # Use live data from WhatItTakesToWin analyzer
    if wittw_tab1.data is not None and len(wittw_tab1.data) > 0:
        # Get column names
        event_col = 'Event' if 'Event' in wittw_tab1.data.columns else 'event'
        gender_col = 'Gender' if 'Gender' in wittw_tab1.data.columns else 'gender'
        rank_col = 'Rank' if 'Rank' in wittw_tab1.data.columns else 'rank'
        mark_col = 'Mark' if 'Mark' in wittw_tab1.data.columns else ('result' if 'result' in wittw_tab1.data.columns else 'Result')

        # Filter data for selected event, gender, and championship type
        prog_data = wittw_tab1.data.copy()
        prog_data = prog_data[prog_data[event_col].astype(str).str.contains(selected_event, case=False, na=False)]
        prog_data = prog_data[prog_data[gender_col].astype(str).str.lower().str.contains(selected_gender.lower(), na=False)]

        # Filter by championship type if not "All"
        if selected_comp_tab1 != 'All':
            prog_data = wittw_tab1.filter_by_competition(selected_comp_tab1)
            prog_data = prog_data[prog_data[event_col].astype(str).str.contains(selected_event, case=False, na=False)]
            prog_data = prog_data[prog_data[gender_col].astype(str).str.lower().str.contains(selected_gender.lower(), na=False)]

        # Parse marks to numeric
        is_field = wittw_tab1.is_field_event(selected_event)
        if is_field:
            prog_data['ParsedMark'] = prog_data[mark_col].apply(wittw_tab1.parse_distance_to_meters)
        else:
            prog_data['ParsedMark'] = prog_data[mark_col].apply(wittw_tab1.parse_time_to_seconds)

        prog_data = prog_data.dropna(subset=['ParsedMark'])

        if rank_col in prog_data.columns and 'year' in prog_data.columns:
            # Convert rank to numeric (handles string values like "1", "2" etc.)
            prog_data[rank_col] = pd.to_numeric(prog_data[rank_col], errors='coerce')
            prog_data = prog_data.dropna(subset=[rank_col])

            # Filter to top 8 positions
            prog_data = prog_data[prog_data[rank_col] <= 8]

            if not prog_data.empty:
                # Place labels
                place_labels = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th', 5: '5th', 6: '6th', 7: '7th', 8: '8th'}
                prog_data['Place'] = prog_data[rank_col].map(place_labels)

                # Color scale - medal colors for 1-3, distinct colors for 4-8
                place_colors = {
                    '1st': '#FFD700', '2nd': '#C0C0C0', '3rd': '#CD7F32',
                    '4th': '#4CAF50', '5th': '#2196F3', '6th': '#9C27B0',
                    '7th': '#FF5722', '8th': '#607D8B'
                }

                # Create progression chart
                fig_prog = go.Figure()

                # Only show selected positions
                positions_to_show = selected_positions if selected_positions else ['1st', '2nd', '3rd', '8th']
                for place in positions_to_show:
                    place_data = prog_data[prog_data['Place'] == place].sort_values('year')
                    if not place_data.empty:
                        fig_prog.add_trace(go.Scatter(
                            x=place_data['year'],
                            y=place_data['ParsedMark'],
                            mode='lines+markers',
                            name=place,
                            line=dict(color=place_colors.get(place, '#888'), width=2, shape='spline'),
                            marker=dict(size=8),
                            hovertemplate=f"<b>{place}</b><br>Year: %{{x}}<br>Mark: %{{y:.2f}}<extra></extra>"
                        ))

                # Update layout
                fig_prog.update_layout(
                    title=f"{selected_event.upper()} - Finals Performance Over Time ({selected_comp_tab1})",
                    xaxis_title="Year",
                    yaxis_title="Performance",
                    yaxis=dict(autorange='reversed') if not is_field else dict(),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=450,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    hovermode='x unified'
                )
                fig_prog.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                fig_prog.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

                st.plotly_chart(fig_prog, width='stretch')

                # Key insight - show gap between Gold and 8th
                gold_data = prog_data[prog_data['Place'] == '1st']
                eighth_data = prog_data[prog_data['Place'] == '8th']
                if not gold_data.empty and not eighth_data.empty:
                    latest_gold = gold_data.sort_values('year').iloc[-1]['ParsedMark']
                    latest_eighth = eighth_data.sort_values('year').iloc[-1]['ParsedMark']
                    gap = abs(latest_gold - latest_eighth)
                    st.info(f"**Gap Analysis:** Latest gap between Gold and 8th place: **{gap:.2f}** {'seconds' if not is_field else 'm'}")
            else:
                st.info(f"No position data available for {selected_event} in {selected_comp_tab1}")
        else:
            st.info("Progression chart requires rank and year columns in the data")
    else:
        st.info("Live championship data not available. Showing historical winning marks only.")

    # === ROUND STANDARDS SUMMARY (Finals/Semis/Heats) ===
    st.markdown("---")
    st.subheader("Championship Round Standards")
    st.markdown(f"<p style='color: #aaa;'>Final Summary by Place showing Average, Slowest, and Fastest performances</p>", unsafe_allow_html=True)

    if wittw_tab1.data is not None and len(wittw_tab1.data) > 0:
        # Get filtered data
        filtered_data = wittw_tab1.filter_by_competition(selected_comp_tab1)
        if filtered_data is not None and len(filtered_data) > 0:
            event_col = 'Event' if 'Event' in filtered_data.columns else 'event'
            gender_col = 'Gender' if 'Gender' in filtered_data.columns else 'gender'
            rank_col = 'Rank' if 'Rank' in filtered_data.columns else 'rank'

            # Normalize event matching (same as above)
            event_base = re.sub(r'[^0-9a-z]', '', selected_event.lower())
            event_num = re.sub(r'[^0-9]', '', selected_event)
            event_lower = filtered_data[event_col].astype(str).str.lower().str.replace('-', '', regex=False).str.replace(' ', '', regex=False)
            event_mask = event_lower.str.contains(event_base, na=False) | filtered_data[event_col].astype(str).str.contains(f'{event_num}', na=False)
            gender_mask = filtered_data[gender_col].astype(str).str.lower() == selected_gender.lower()

            round_data = filtered_data[event_mask & gender_mask].copy()

            if len(round_data) > 0 and rank_col in round_data.columns:
                # Parse marks to numeric
                mark_col = 'result' if 'result' in round_data.columns else 'Mark'
                is_field = any(x in selected_event.lower() for x in ['shot', 'discus', 'hammer', 'javelin', 'jump', 'vault', 'throw'])

                if 'result_numeric' in round_data.columns:
                    round_data['ParsedMark'] = round_data['result_numeric']
                elif is_field:
                    round_data['ParsedMark'] = pd.to_numeric(round_data[mark_col], errors='coerce')
                else:
                    round_data['ParsedMark'] = round_data[mark_col].apply(lambda x: convert_time_to_seconds(str(x)) if pd.notna(x) else None)

                round_data = round_data.dropna(subset=['ParsedMark'])

                # Convert rank to numeric before comparison
                round_data[rank_col] = pd.to_numeric(round_data[rank_col], errors='coerce')
                round_data = round_data.dropna(subset=[rank_col])

                # Filter to top 12 positions for comprehensive view
                round_data = round_data[round_data[rank_col] <= 12]

                if len(round_data) > 0:
                    # Create summary table by rank
                    summary_rows = []
                    for rank in range(1, 13):
                        rank_results = round_data[round_data[rank_col] == rank]['ParsedMark']
                        if len(rank_results) > 0:
                            avg_mark = rank_results.mean()
                            # For track: best=min, worst=max. For field: best=max, worst=min
                            if is_field:
                                best_mark = rank_results.max()
                                worst_mark = rank_results.min()
                            else:
                                best_mark = rank_results.min()
                                worst_mark = rank_results.max()

                            # Format marks appropriately
                            def format_mark(val, is_field_event):
                                if val is None or pd.isna(val):
                                    return 'N/A'
                                if is_field_event:
                                    return f"{val:.2f}m"
                                else:
                                    if val >= 60:
                                        mins = int(val // 60)
                                        secs = val % 60
                                        return f"{mins}:{secs:05.2f}"
                                    return f"{val:.2f}"

                            summary_rows.append({
                                'Rank': rank,
                                'Average': format_mark(avg_mark, is_field),
                                'Slowest': format_mark(worst_mark, is_field),
                                'Fastest': format_mark(best_mark, is_field),
                                'Count': len(rank_results)
                            })

                    if summary_rows:
                        col_table, col_estimate = st.columns([1, 1])

                        with col_table:
                            st.markdown(f"**Final Summary by Place ({selected_comp_tab1})**")
                            summary_df = pd.DataFrame(summary_rows)
                            st.dataframe(summary_df, width='stretch', hide_index=True)

                        with col_estimate:
                            # Estimate Heats/Semis/Finals standards
                            st.markdown("**Round Qualification Estimates**")

                            # Get 8th place average (finals cutoff)
                            eighth_place = round_data[round_data[rank_col] == 8]['ParsedMark']
                            if len(eighth_place) > 0:
                                final_standard = eighth_place.mean()

                                # Estimate semi/heat standards based on typical margins
                                if is_field:
                                    # Field events: semis ~2-3% lower, heats ~5% lower
                                    semi_standard = final_standard * 0.97
                                    heat_standard = final_standard * 0.95
                                else:
                                    # Track events: semis ~0.5-1% slower, heats ~1-2% slower
                                    semi_standard = final_standard * 1.005
                                    heat_standard = final_standard * 1.015

                                def format_mark_simple(val, is_field_event):
                                    if is_field_event:
                                        return f"{val:.2f}m"
                                    else:
                                        if val >= 60:
                                            mins = int(val // 60)
                                            secs = val % 60
                                            return f"{mins}:{secs:05.2f}"
                                        return f"{val:.2f}"

                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                                    <p style="color: {GOLD_ACCENT}; margin: 0; font-size: 0.9rem;">Finals (8th place)</p>
                                    <p style="color: white; font-size: 1.3rem; font-weight: bold; margin: 0;">{format_mark_simple(final_standard, is_field)}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                <div style="background: #444; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                                    <p style="color: #C0C0C0; margin: 0; font-size: 0.9rem;">Semi-Finals (est.)</p>
                                    <p style="color: white; font-size: 1.3rem; font-weight: bold; margin: 0;">{format_mark_simple(semi_standard, is_field)}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                <div style="background: #333; padding: 1rem; border-radius: 8px;">
                                    <p style="color: #888; margin: 0; font-size: 0.9rem;">Heats (est.)</p>
                                    <p style="color: white; font-size: 1.3rem; font-weight: bold; margin: 0;">{format_mark_simple(heat_standard, is_field)}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                st.caption("*Estimates based on typical championship progression margins")
                else:
                    st.info(f"No round data available for {selected_event}")
            else:
                st.info(f"No position data available for round analysis")
    else:
        st.info("Championship data not available for round analysis")

###################################
# Tab 2: Athlete Profiles
###################################
with tab2:
    st.header('KSA Athlete Profiles')

    if athletes_df is None or athletes_df.empty:
        st.warning("No athlete profiles found. Run the scraper first: `python scrape_ksa_athlete_profiles_v2.py`")
    else:
        # Initialize session state for athlete profile selection to prevent navigation issues
        if 'selected_profile_athlete' not in st.session_state:
            st.session_state.selected_profile_athlete = "All Athletes"
        if 'selected_profile_gender' not in st.session_state:
            st.session_state.selected_profile_gender = "All"
        # Initialize widget keys to prevent AttributeError in callbacks
        if 'profile_athlete' not in st.session_state:
            st.session_state.profile_athlete = "All Athletes"
        if 'profile_gender' not in st.session_state:
            st.session_state.profile_gender = "All"

        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            athlete_names = ["All Athletes"] + sorted(athletes_df['full_name'].dropna().unique().tolist())
            # Use on_change callback to update session state
            def on_athlete_change():
                st.session_state.selected_profile_athlete = st.session_state.profile_athlete

            selected_athlete = st.selectbox(
                "Select Athlete",
                athlete_names,
                index=athlete_names.index(st.session_state.selected_profile_athlete) if st.session_state.selected_profile_athlete in athlete_names else 0,
                key="profile_athlete",
                on_change=on_athlete_change
            )

        with col_filter2:
            gender_options = ["All", "Men", "Women"]
            def on_gender_change():
                st.session_state.selected_profile_gender = st.session_state.profile_gender

            selected_gender_profile = st.selectbox(
                "Gender",
                gender_options,
                index=gender_options.index(st.session_state.selected_profile_gender) if st.session_state.selected_profile_gender in gender_options else 0,
                key="profile_gender",
                on_change=on_gender_change
            )

        filtered_athletes = athletes_df.copy()
        if selected_gender_profile != "All":
            filtered_athletes = filtered_athletes[filtered_athletes['gender'] == selected_gender_profile.lower()]

        if selected_athlete != "All Athletes":
            filtered_athletes = filtered_athletes[filtered_athletes['full_name'] == selected_athlete]

            if len(filtered_athletes) == 1:
                athlete = filtered_athletes.iloc[0]
                athlete_id = athlete['athlete_id']

                col_img, col_info = st.columns([1, 3])

                with col_img:
                    if athlete.get('profile_image_url'):
                        st.image(athlete['profile_image_url'], width=150)
                    else:
                        st.markdown(f"""
                        <div style="width: 150px; height: 180px; background: {TEAL_DARK};
                                    border-radius: 8px; display: flex; align-items: center;
                                    justify-content: center; color: white; font-size: 3rem;">
                            🏃
                        </div>
                        """, unsafe_allow_html=True)

                with col_info:
                    st.markdown(f"""
                    <div class="athlete-card">
                        <h2 style="color: white !important; margin: 0;">{athlete['full_name']}</h2>
                        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0;">
                            <strong>Primary Event:</strong> {athlete.get('primary_event', 'N/A')} |
                            <strong>DOB:</strong> {athlete.get('date_of_birth', 'N/A')} |
                            <strong>Status:</strong> {athlete.get('status', 'active').capitalize()}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show key metrics from profile
                col1, col2, col3 = st.columns(3)
                with col1:
                    best_rank = athlete.get('best_world_rank', None)
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: {GOLD_ACCENT}; font-size: 0.9rem; margin: 0;">Best World Rank</p>
                        <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.25rem 0;">
                            #{int(best_rank) if pd.notna(best_rank) else 'N/A'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    best_score = athlete.get('best_score', None)
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: {TEAL_LIGHT}; font-size: 0.9rem; margin: 0;">Best Score</p>
                        <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.25rem 0;">
                            {int(best_score) if pd.notna(best_score) else 'N/A'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #aaa; font-size: 0.9rem; margin: 0;">Primary Event</p>
                        <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0.25rem 0;">
                            {athlete.get('primary_event', 'N/A')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Get athlete results from master parquet (Azure)
                if DATA_CONNECTOR_AVAILABLE:
                    athlete_name = athlete['full_name']
                    try:
                        # Get all results for this athlete from master data
                        ksa_data = load_ksa_rankings_raw_cached()
                        if ksa_data is not None and not ksa_data.empty:
                            # Match by full name (case-insensitive)
                            athlete_results = ksa_data[ksa_data['competitor'].str.upper() == athlete_name.upper()]
                            # If no exact match, try partial match on last name
                            if athlete_results.empty:
                                last_name = athlete_name.upper().split()[-1]
                                athlete_results = ksa_data[ksa_data['competitor'].str.upper().str.contains(last_name, na=False)]

                            if not athlete_results.empty:
                                # Determine actual primary event from results (most competitions)
                                primary_event = None
                                if 'event' in athlete_results.columns:
                                    event_counts = athlete_results['event'].value_counts()
                                    if not event_counts.empty:
                                        primary_event = event_counts.index[0]
                                        st.markdown(f"<p style='color: {GOLD_ACCENT}; font-size: 0.9rem;'><strong>Main Event:</strong> {primary_event} ({event_counts.iloc[0]} results)</p>", unsafe_allow_html=True)

                                # === PERFORMANCE TREND ANALYSIS ===
                                st.markdown("---")
                                st.subheader("Performance Trend Analysis")

                                # Parse numeric results
                                def parse_result_to_numeric(result, event_name=''):
                                    if pd.isna(result):
                                        return None
                                    result_str = str(result).strip()
                                    try:
                                        # Handle time formats (e.g., "10.45", "1:59.00")
                                        if ':' in result_str:
                                            parts = result_str.split(':')
                                            if len(parts) == 2:
                                                return float(parts[0]) * 60 + float(parts[1])
                                        return float(result_str)
                                    except:
                                        return None

                                athlete_results_copy = athlete_results.copy()
                                athlete_results_copy['result_numeric'] = athlete_results_copy['result'].apply(
                                    lambda x: parse_result_to_numeric(x, primary_event or '')
                                )
                                athlete_results_copy = athlete_results_copy.dropna(subset=['result_numeric'])

                                if not athlete_results_copy.empty and 'date' in athlete_results_copy.columns:
                                    # Convert date and sort
                                    athlete_results_copy['date'] = pd.to_datetime(athlete_results_copy['date'], errors='coerce')
                                    athlete_results_copy = athlete_results_copy.dropna(subset=['date'])
                                    athlete_results_copy = athlete_results_copy.sort_values('date')

                                    # Determine if track (lower=better) or field (higher=better)
                                    is_field = any(kw in (primary_event or '').lower() for kw in ['jump', 'vault', 'put', 'throw', 'discus', 'javelin', 'hammer'])

                                    # Calculate trend metrics
                                    if len(athlete_results_copy) >= 3:
                                        results_list = athlete_results_copy['result_numeric'].tolist()
                                        first_half = np.mean(results_list[:len(results_list)//2])
                                        second_half = np.mean(results_list[len(results_list)//2:])

                                        if is_field:
                                            trend_value = second_half - first_half
                                            trend_direction = "📈 Improving" if trend_value > 0 else ("📉 Declining" if trend_value < 0 else "➡️ Stable")
                                        else:
                                            trend_value = first_half - second_half
                                            trend_direction = "📈 Improving" if trend_value > 0 else ("📉 Declining" if trend_value < 0 else "➡️ Stable")

                                        # Season best
                                        current_year = pd.Timestamp.now().year
                                        season_data = athlete_results_copy[athlete_results_copy['date'].dt.year == current_year]
                                        if not season_data.empty:
                                            if is_field:
                                                season_best = season_data['result_numeric'].max()
                                            else:
                                                season_best = season_data['result_numeric'].min()
                                        else:
                                            season_best = None

                                        # Personal best
                                        if is_field:
                                            pb = athlete_results_copy['result_numeric'].max()
                                            pb_row = athlete_results_copy[athlete_results_copy['result_numeric'] == pb].iloc[0]
                                        else:
                                            pb = athlete_results_copy['result_numeric'].min()
                                            pb_row = athlete_results_copy[athlete_results_copy['result_numeric'] == pb].iloc[0]

                                        # Display metrics
                                        trend_cols = st.columns(5)
                                        with trend_cols[0]:
                                            st.markdown(f"""
                                            <div style="background: {TEAL_PRIMARY}; padding: 0.8rem; border-radius: 8px; text-align: center;">
                                                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.75rem;">Form Trend</p>
                                                <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{trend_direction}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        with trend_cols[1]:
                                            pb_venue = pb_row.get('venue', 'Unknown')[:20] if pb_row.get('venue') else 'Unknown'
                                            pb_date = pb_row['date'].strftime('%Y-%m-%d') if pd.notna(pb_row.get('date')) else 'N/A'
                                            st.markdown(f"""
                                            <div style="background: {GOLD_ACCENT}; padding: 0.8rem; border-radius: 8px; text-align: center;">
                                                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.75rem;">Personal Best</p>
                                                <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{pb_row['result']}</p>
                                                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.6rem;">{pb_venue} | {pb_date}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        with trend_cols[2]:
                                            sb_display = 'N/A'
                                            if season_best is not None and not season_data.empty:
                                                try:
                                                    if is_field:
                                                        sb_idx = season_data['result_numeric'].idxmax()
                                                    else:
                                                        sb_idx = season_data['result_numeric'].idxmin()
                                                    sb_display = season_data.loc[sb_idx, 'result']
                                                except:
                                                    sb_display = f"{season_best:.2f}"
                                            st.markdown(f"""
                                            <div style="background: #4CAF50; padding: 0.8rem; border-radius: 8px; text-align: center;">
                                                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.75rem;">Season Best {current_year}</p>
                                                <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{sb_display}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        with trend_cols[3]:
                                            st.markdown(f"""
                                            <div style="background: {GRAY_BLUE}; padding: 0.8rem; border-radius: 8px; text-align: center;">
                                                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.75rem;">Results Count</p>
                                                <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{len(athlete_results_copy)}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        with trend_cols[4]:
                                            avg_result = athlete_results_copy['result_numeric'].mean()
                                            st.markdown(f"""
                                            <div style="background: #2196F3; padding: 0.8rem; border-radius: 8px; text-align: center;">
                                                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.75rem;">Average</p>
                                                <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{avg_result:.2f}</p>
                                            </div>
                                            """, unsafe_allow_html=True)

                                        # === CONSISTENCY SCORE ===
                                        if ANALYTICS_HELPERS_AVAILABLE and len(athlete_results_copy) >= 5:
                                            recent_perfs = athlete_results_copy.tail(10)['result_numeric'].tolist()
                                            consistency = calculate_consistency_score(recent_perfs, is_field)

                                            if consistency and consistency.get('score') is not None:
                                                st.markdown("---")
                                                cons_cols = st.columns([2, 1, 1])
                                                with cons_cols[0]:
                                                    st.markdown(f"""
                                                    <div style="background: {consistency['color']}; padding: 1rem; border-radius: 8px;">
                                                        <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Consistency Score</p>
                                                        <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0;">{consistency['score']}/100</p>
                                                        <p style="color: rgba(255,255,255,0.9); margin: 0.25rem 0 0 0; font-size: 0.9rem;">{consistency.get('interpretation', consistency.get('rating', ''))}</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                with cons_cols[1]:
                                                    st.markdown(f"""
                                                    <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; text-align: center;">
                                                        <p style="color: #aaa; margin: 0; font-size: 0.75rem;">Std Deviation</p>
                                                        <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{consistency.get('std_dev', consistency.get('std', 'N/A'))}</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                with cons_cols[2]:
                                                    st.markdown(f"""
                                                    <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; text-align: center;">
                                                        <p style="color: #aaa; margin: 0; font-size: 0.75rem;">CV %</p>
                                                        <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{consistency['cv']}%</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)

                                        # === BEST 5 WPA RANKING POINTS ===
                                        st.markdown("---")
                                        st.subheader("Best 5 WPA Ranking Points")
                                        st.markdown(f"<p style='color: #aaa;'>Top performances contributing to World Athletics ranking score</p>", unsafe_allow_html=True)

                                        if 'resultscore' in athlete_results_copy.columns:
                                            # Get top 5 by WPA ranking points (resultscore)
                                            top_5_wpa = athlete_results_copy.dropna(subset=['resultscore']).nlargest(5, 'resultscore')

                                            if not top_5_wpa.empty:
                                                wpa_cols = st.columns(5)
                                                for idx, (_, row) in enumerate(top_5_wpa.iterrows()):
                                                    with wpa_cols[idx]:
                                                        venue_display = row.get('venue', 'Unknown')[:20] if row.get('venue') else 'Unknown'
                                                        date_display = row['date'].strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'N/A'
                                                        st.markdown(f"""
                                                        <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 0.8rem; border-radius: 8px; text-align: center; min-height: 140px;">
                                                            <p style="color: {GOLD_ACCENT}; margin: 0; font-size: 0.75rem; font-weight: bold;">#{idx + 1}</p>
                                                            <p style="color: white; font-size: 1.3rem; font-weight: bold; margin: 0.25rem 0;">{int(row['resultscore'])}</p>
                                                            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.8rem;">{row['result']}</p>
                                                            <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.65rem;">{venue_display}</p>
                                                            <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.6rem;">{date_display}</p>
                                                        </div>
                                                        """, unsafe_allow_html=True)

                                                # Show WPA points distribution chart
                                                if len(top_5_wpa) >= 3:
                                                    fig_wpa = go.Figure(data=[
                                                        go.Bar(
                                                            x=[f"#{i+1}" for i in range(len(top_5_wpa))],
                                                            y=top_5_wpa['resultscore'].tolist(),
                                                            marker_color=[GOLD_ACCENT if i == 0 else TEAL_PRIMARY for i in range(len(top_5_wpa))],
                                                            text=[f"{int(s)}" for s in top_5_wpa['resultscore'].tolist()],
                                                            textposition='outside'
                                                        )
                                                    ])
                                                    fig_wpa.update_layout(
                                                        title="WPA Ranking Points Distribution",
                                                        yaxis_title="Points",
                                                        plot_bgcolor='rgba(0,0,0,0)',
                                                        paper_bgcolor='rgba(0,0,0,0)',
                                                        font=dict(color='white'),
                                                        height=250,
                                                        margin=dict(l=20, r=20, t=40, b=20)
                                                    )
                                                    st.plotly_chart(fig_wpa, width='stretch')
                                            else:
                                                st.info("No WPA ranking points data available for this athlete")
                                        else:
                                            st.info("WPA ranking points (resultscore) not available in data")

                                        # === MULTI-EVENT SUMMARY ===
                                        st.markdown("---")
                                        st.subheader("Multi-Event Summary")
                                        st.markdown(f"<p style='color: #aaa;'>Performance breakdown across all events competed</p>", unsafe_allow_html=True)

                                        if 'event' in athlete_results_copy.columns:
                                            event_summary = []
                                            for event_name, event_group in athlete_results_copy.groupby('event'):
                                                event_is_field = any(kw in event_name.lower() for kw in ['jump', 'vault', 'put', 'throw', 'discus', 'javelin', 'hammer'])

                                                if event_is_field:
                                                    best_result = event_group['result_numeric'].max()
                                                    best_row = event_group[event_group['result_numeric'] == best_result].iloc[0]
                                                else:
                                                    best_result = event_group['result_numeric'].min()
                                                    best_row = event_group[event_group['result_numeric'] == best_result].iloc[0]

                                                # Get last season (current year) best
                                                current_year = pd.Timestamp.now().year
                                                last_season_data = event_group[event_group['date'].dt.year >= current_year - 1]
                                                if not last_season_data.empty:
                                                    if event_is_field:
                                                        season_best = last_season_data['result_numeric'].max()
                                                    else:
                                                        season_best = last_season_data['result_numeric'].min()
                                                    season_best_row = last_season_data[last_season_data['result_numeric'] == season_best].iloc[0] if not last_season_data.empty else None
                                                else:
                                                    season_best = None
                                                    season_best_row = None

                                                event_summary.append({
                                                    'Event': event_name,
                                                    'Competitions': len(event_group),
                                                    'PB': best_row['result'],
                                                    'PB Date': best_row['date'].strftime('%Y-%m-%d') if pd.notna(best_row.get('date')) else 'N/A',
                                                    'PB Venue': (best_row.get('venue', 'Unknown')[:30] if best_row.get('venue') else 'Unknown'),
                                                    'Season Best': season_best_row['result'] if season_best_row is not None else 'N/A',
                                                    'Avg': f"{event_group['result_numeric'].mean():.2f}"
                                                })

                                            if event_summary:
                                                event_summary_df = pd.DataFrame(event_summary)
                                                event_summary_df = event_summary_df.sort_values('Competitions', ascending=False)
                                                st.dataframe(event_summary_df, width='stretch', hide_index=True)
                                        else:
                                            st.info("Event data not available for multi-event summary")

                                        # === PERFORMANCE PROGRESSION CHART ===
                                        st.markdown("---")
                                        st.subheader("Performance Progression")

                                        # Filter to primary event for cleaner chart
                                        if primary_event and 'event' in athlete_results_copy.columns:
                                            chart_data = athlete_results_copy[athlete_results_copy['event'] == primary_event].copy()
                                        else:
                                            chart_data = athlete_results_copy.copy()

                                        if not chart_data.empty:
                                            # Create progression chart
                                            fig_athlete = go.Figure()

                                            # Add performance line
                                            fig_athlete.add_trace(go.Scatter(
                                                x=chart_data['date'],
                                                y=chart_data['result_numeric'],
                                                mode='lines+markers',
                                                name='Performance',
                                                line=dict(color=TEAL_PRIMARY, width=2),
                                                marker=dict(size=8, color=TEAL_PRIMARY),
                                                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Result: %{y:.2f}<extra></extra>"
                                            ))

                                            # Add trend line
                                            if len(chart_data) >= 3:
                                                x_numeric = np.arange(len(chart_data))
                                                z = np.polyfit(x_numeric, chart_data['result_numeric'].values, 1)
                                                p = np.poly1d(z)
                                                fig_athlete.add_trace(go.Scatter(
                                                    x=chart_data['date'],
                                                    y=p(x_numeric),
                                                    mode='lines',
                                                    name='Trend',
                                                    line=dict(color=GOLD_ACCENT, width=2, dash='dash'),
                                                ))

                                            # Add PB line
                                            fig_athlete.add_hline(
                                                y=pb,
                                                line_dash="dot",
                                                line_color="#FFD700",
                                                annotation_text=f"PB: {pb_row['result']}",
                                                annotation_position="right"
                                            )

                                            # Update layout
                                            fig_athlete.update_layout(
                                                title=f"{athlete_name} - {primary_event or 'All Events'} Performance",
                                                xaxis_title="Date",
                                                yaxis_title="Performance",
                                                yaxis=dict(autorange='reversed') if not is_field else dict(),
                                                plot_bgcolor='rgba(0,0,0,0)',
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                font=dict(color='white'),
                                                height=400,
                                                showlegend=True,
                                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                                            )
                                            fig_athlete.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                                            fig_athlete.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

                                            st.plotly_chart(fig_athlete, width='stretch')

                                # === HEAD-TO-HEAD COMPARISON ===
                                st.markdown("---")
                                with st.expander("Head-to-Head Comparison", expanded=False):
                                    st.markdown(f"""
                                    <p style='color: #ccc;'>Compare <strong style='color: {GOLD_ACCENT};'>{athlete_name}</strong> against another athlete in {primary_event or 'their primary event'}</p>
                                    """, unsafe_allow_html=True)

                                    if DATA_CONNECTOR_AVAILABLE and primary_event:
                                        try:
                                            # Get all athletes in same event for comparison
                                            all_rankings = load_full_rankings_cached()

                                            if all_rankings is not None and not all_rankings.empty:
                                                event_col = 'event' if 'event' in all_rankings.columns else 'Event Type'
                                                competitor_col = 'competitor' if 'competitor' in all_rankings.columns else 'Name'

                                                event_athletes = all_rankings[all_rankings[event_col] == primary_event][competitor_col].unique()
                                                event_athletes = [a for a in event_athletes if str(a).upper() != athlete_name.upper()]
                                                event_athletes = sorted(event_athletes)[:100]  # Limit for performance

                                                if event_athletes:
                                                    h2h_opponent = st.selectbox(
                                                        "Select opponent to compare",
                                                        event_athletes,
                                                        key="h2h_opponent_selector"
                                                    )

                                                    if st.button("Compare", key="h2h_compare_btn"):
                                                        # Get opponent data
                                                        opponent_results = all_rankings[all_rankings[competitor_col].str.upper() == h2h_opponent.upper()]

                                                        if not opponent_results.empty:
                                                            # Parse results for both athletes
                                                            result_col = 'result' if 'result' in all_rankings.columns else 'Score'

                                                            def parse_h2h(val):
                                                                if pd.isna(val):
                                                                    return None
                                                                val_str = str(val).strip()
                                                                try:
                                                                    if ':' in val_str:
                                                                        parts = val_str.split(':')
                                                                        return float(parts[0]) * 60 + float(parts[1])
                                                                    return float(val_str)
                                                                except:
                                                                    return None

                                                            # Determine if field event locally (in case is_field wasn't set in trend section)
                                                            is_field_local = any(kw in (primary_event or '').lower() for kw in ['jump', 'vault', 'put', 'throw', 'discus', 'javelin', 'hammer', 'decathlon', 'heptathlon'])

                                                            # Filter to event
                                                            a1_event = athlete_results[athlete_results[event_col] == primary_event].copy() if event_col in athlete_results.columns else athlete_results.copy()
                                                            a2_event = opponent_results[opponent_results[event_col] == primary_event].copy() if event_col in opponent_results.columns else opponent_results.copy()

                                                            a1_event['result_num'] = a1_event[result_col].apply(parse_h2h)
                                                            a2_event['result_num'] = a2_event[result_col].apply(parse_h2h)

                                                            a1_event = a1_event.dropna(subset=['result_num'])
                                                            a2_event = a2_event.dropna(subset=['result_num'])

                                                            if not a1_event.empty and not a2_event.empty:
                                                                # Calculate stats
                                                                if is_field_local:
                                                                    a1_pb = a1_event['result_num'].max()
                                                                    a2_pb = a2_event['result_num'].max()
                                                                    a1_avg = a1_event['result_num'].mean()
                                                                    a2_avg = a2_event['result_num'].mean()
                                                                    pb_winner = athlete_name if a1_pb > a2_pb else h2h_opponent
                                                                else:
                                                                    a1_pb = a1_event['result_num'].min()
                                                                    a2_pb = a2_event['result_num'].min()
                                                                    a1_avg = a1_event['result_num'].mean()
                                                                    a2_avg = a2_event['result_num'].mean()
                                                                    pb_winner = athlete_name if a1_pb < a2_pb else h2h_opponent

                                                                # Display comparison
                                                                h2h_cols = st.columns(3)

                                                                with h2h_cols[0]:
                                                                    st.markdown(f"""
                                                                    <div style="background: {TEAL_PRIMARY}; padding: 1rem; border-radius: 8px; text-align: center;">
                                                                        <p style="color: white; font-weight: bold; margin: 0; font-size: 0.9rem;">{athlete_name}</p>
                                                                        <p style="color: rgba(255,255,255,0.9); margin: 0.3rem 0; font-size: 1.3rem; font-weight: bold;">PB: {a1_pb:.2f}</p>
                                                                        <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.85rem;">Avg: {a1_avg:.2f}</p>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)

                                                                with h2h_cols[1]:
                                                                    st.markdown(f"""
                                                                    <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; text-align: center;">
                                                                        <p style="color: #aaa; margin: 0; font-size: 0.85rem;">Comparison</p>
                                                                        <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0.3rem 0;">VS</p>
                                                                        <p style="color: {GOLD_ACCENT}; margin: 0; font-size: 0.85rem;">PB: {pb_winner}</p>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)

                                                                with h2h_cols[2]:
                                                                    st.markdown(f"""
                                                                    <div style="background: {GOLD_ACCENT}; padding: 1rem; border-radius: 8px; text-align: center;">
                                                                        <p style="color: white; font-weight: bold; margin: 0; font-size: 0.9rem;">{h2h_opponent}</p>
                                                                        <p style="color: rgba(255,255,255,0.9); margin: 0.3rem 0; font-size: 1.3rem; font-weight: bold;">PB: {a2_pb:.2f}</p>
                                                                        <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.85rem;">Avg: {a2_avg:.2f}</p>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)
                                                            else:
                                                                st.warning("Insufficient data for comparison")
                                                        else:
                                                            st.warning("Could not find opponent data")
                                                else:
                                                    st.info("No other athletes found in this event for comparison")
                                        except Exception as e:
                                            st.warning(f"H2H comparison error: {str(e)[:100]}")
                                    else:
                                        st.info("Select an athlete with competition data to enable H2H comparison")

                                # === TOP COMPETITORS FORM ANALYSIS ===
                                if RACE_INTELLIGENCE_AVAILABLE and primary_event:
                                    st.markdown("---")
                                    with st.expander("🎯 Top Competitors", expanded=False):
                                        st.markdown(f"<p style='color: #888;'>Top competitors for <b style='color: {GOLD_ACCENT};'>{primary_event}</b> ranked by current form</p>", unsafe_allow_html=True)

                                        competitors = get_competitor_form_cards(primary_event, gender='Men', limit=5)

                                        if competitors:
                                            for i, comp in enumerate(competitors, 1):
                                                is_ksa = comp['country_code'] == 'KSA'
                                                bg_color = 'rgba(0, 84, 48, 0.15)' if is_ksa else 'rgba(0,0,0,0.05)'
                                                border_color = TEAL_PRIMARY if is_ksa else comp['form_color']

                                                best_races_html = ""
                                                for race in comp.get('best_2_races', [])[:2]:
                                                    best_races_html += f"<div>⭐ {race['result']}  {race['venue'][:20] if race['venue'] else ''}  {str(race['date'])[:10]}</div>"

                                                last = comp.get('last_comp', {})
                                                last_html = f"{last.get('result', '')} @ {last.get('venue', '')[:15] if last.get('venue') else ''} ({last.get('days_ago', 0)}d ago)" if last else ""

                                                st.markdown(f"""
                                                <div style="background: {bg_color}; border-left: 4px solid {border_color};
                                                            padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
                                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                                        <div>
                                                            <b style="font-size: 1.1rem;">{i}. {comp['athlete_name']}</b>
                                                            <span style="color: #888;">({comp['country_code']})</span>
                                                        </div>
                                                        <div style="text-align: right;">
                                                            <span style="color: {comp['form_color']}; font-size: 1.2rem; font-weight: bold;">
                                                                {comp['form_icon']} {comp['form_score']:.0f}
                                                            </span>
                                                        </div>
                                                    </div>
                                                    <div style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
                                                        PB: {comp['pb']:.2f if comp['pb'] else 'N/A'} | Avg: {comp['avg_last_5']:.2f} | {comp['form_status']}
                                                    </div>
                                                    <div style="color: #888; font-size: 0.85rem; margin-top: 0.5rem;">
                                                        <b>Best 2:</b><br>{best_races_html if best_races_html else 'No data'}
                                                    </div>
                                                    <div style="color: #aaa; font-size: 0.8rem; margin-top: 0.25rem;">
                                                        Last: {last_html if last_html else 'N/A'}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        else:
                                            st.info("No competitor data available for this event")

                                # === CAREER MILESTONES TIMELINE ===
                                if RACE_INTELLIGENCE_AVAILABLE:
                                    st.markdown("---")
                                    with st.expander("📅 Career Milestones", expanded=False):
                                        milestones = get_career_milestones(athlete_name, primary_event if primary_event else None)

                                        if milestones:
                                            # Group by year
                                            years = {}
                                            for m in milestones:
                                                year = m.get('year', 'Unknown')
                                                if year not in years:
                                                    years[year] = []
                                                years[year].append(m)

                                            # Display timeline
                                            for year in sorted(years.keys(), reverse=True):
                                                st.markdown(f"### {year}")
                                                for m in years[year]:
                                                    icon = m.get('icon', '●')
                                                    desc = m.get('description', '')
                                                    date_str = str(m.get('date', ''))[:10]

                                                    st.markdown(f"""
                                                    <div style="display: flex; align-items: center; margin: 0.5rem 0; padding-left: 1rem; border-left: 2px solid {TEAL_PRIMARY};">
                                                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                                                        <div>
                                                            <b>{desc}</b>
                                                            <span style="color: #888; font-size: 0.85rem; margin-left: 0.5rem;">{date_str}</span>
                                                        </div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                        else:
                                            st.info("No career milestones detected")

                                # === COMPETITION RESULTS TABLE ===
                                st.markdown("---")
                                st.subheader("Competition Results")

                                # Show best results per event - remove duplicates
                                if 'event' in athlete_results.columns and 'result' in athlete_results.columns:
                                    display_cols = ['event', 'result', 'date', 'venue', 'rank']
                                    display_cols = [c for c in display_cols if c in athlete_results.columns]
                                    results_display = athlete_results[display_cols].copy()

                                    # Remove duplicate rows
                                    results_display = results_display.drop_duplicates()

                                    results_display.columns = [c.title() for c in results_display.columns]

                                    # Sort by date descending
                                    if 'Date' in results_display.columns:
                                        results_display = results_display.sort_values('Date', ascending=False)

                                    st.dataframe(results_display.head(20), width='stretch', hide_index=True)

                                    # Show events competed in
                                    if 'Event' in results_display.columns:
                                        events = results_display['Event'].unique()
                                        st.caption(f"Events: {', '.join(events)}")
                    except Exception as e:
                        st.caption(f"Could not load results: {e}")

                # Rankings and PBs from SQLite (legacy support)
                if rankings_df is not None and not rankings_df.empty:
                    athlete_rankings = rankings_df[rankings_df['athlete_id'] == athlete_id]
                    if not athlete_rankings.empty:
                        st.subheader("Current WPA Rankings")
                        cols = st.columns(min(len(athlete_rankings), 4))
                        for i, (_, rank) in enumerate(athlete_rankings.iterrows()):
                            with cols[i % 4]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <p style="color: {GOLD_ACCENT}; font-size: 0.9rem; margin: 0;">{rank['event_name']}</p>
                                    <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.25rem 0;">
                                        #{int(rank['world_rank']) if pd.notna(rank['world_rank']) else 'N/A'}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                if pbs_df is not None and not pbs_df.empty:
                    athlete_pbs = pbs_df[pbs_df['athlete_id'] == athlete_id]
                    if not athlete_pbs.empty:
                        st.subheader("Personal Bests")
                        display_pbs = athlete_pbs[['event_name', 'pb_result', 'pb_date', 'pb_venue']].copy()
                        display_pbs.columns = ['Event', 'PB', 'Date', 'Venue']
                        st.dataframe(display_pbs, width='stretch', hide_index=True)
        else:
            st.markdown(f"**Showing {len(filtered_athletes)} athletes**")
            cols_per_row = 3
            for i in range(0, len(filtered_athletes), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(filtered_athletes):
                        athlete = filtered_athletes.iloc[idx]
                        with col:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {TEAL_PRIMARY}22 0%, {TEAL_DARK}22 100%);
                                        border: 1px solid {TEAL_PRIMARY}; border-radius: 10px;
                                        padding: 1rem; margin-bottom: 1rem; min-height: 100px;">
                                <h4 style="margin: 0; color: {TEAL_PRIMARY} !important;">{athlete['full_name']}</h4>
                                <p style="color: #aaa; margin: 0.25rem 0; font-size: 0.9rem;">
                                    {athlete.get('primary_event', '')} | {athlete.get('gender', '').capitalize()}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

###################################
# Tab 3: Combined Rankings
###################################
with tab3:
    st.header("Combined Global Rankings")

    # Lazy load the large rankings data
    with st.spinner("Loading rankings data..."):
        men_df = load_men_rankings()
        women_df = load_women_rankings()

    # Gender filter
    gender_filter = st.radio("Select Gender", ['Men', 'Women'], horizontal=True, key="combined_gender")

    display_df = men_df if gender_filter == 'Men' else women_df

    if display_df is not None and not display_df.empty:
        # Event filter
        event_col = 'Event Type' if 'Event Type' in display_df.columns else 'event'
        events = sorted(display_df[event_col].dropna().unique()) if event_col in display_df.columns else []
        selected_event = st.selectbox("Filter by Event", ['All Events'] + list(events), key="combined_event")

        if selected_event != 'All Events':
            display_df = display_df[display_df[event_col] == selected_event]

        # Sort by rank (ascending order)
        rank_col = 'Rank' if 'Rank' in display_df.columns else 'rank' if 'rank' in display_df.columns else None
        if rank_col:
            display_df = display_df.sort_values(rank_col, ascending=True, na_position='last')

        # Pagination
        total_rows = len(display_df)
        page_size = 100
        total_pages = max(1, (total_rows // page_size) + (1 if total_rows % page_size > 0 else 0))

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page_num = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="rankings_page"
            ) - 1  # 0-indexed

        st.caption(f"Showing {page_num * page_size + 1}-{min((page_num + 1) * page_size, total_rows)} of {total_rows:,} records")

        # Display paginated data
        page_df = paginate_dataframe(display_df, page_size, page_num)
        st.dataframe(page_df, width='stretch', hide_index=True)
    else:
        st.warning("No rankings data available")

###################################
# Tab 4: Saudi Athletes Rankings
###################################
with tab4:
    st.header('Saudi Athletes Rankings')

    # Load KSA data directly - much faster than filtering full rankings
    saudi_combined = load_ksa_combined_rankings()

    if not saudi_combined.empty:
        # Show data mode indicator
        mode = get_data_mode() if DATA_CONNECTOR_AVAILABLE else 'sqlite'
        st.success(f"Loaded {len(saudi_combined):,} KSA records from {mode} data source")

        # Get event column name (could be 'Event Type' or 'event')
        event_col = 'Event Type' if 'Event Type' in saudi_combined.columns else 'event' if 'event' in saudi_combined.columns else None

        if event_col:
            saudi_events = sorted(saudi_combined[event_col].dropna().unique())
            selected_event_saudi = st.selectbox("Select Event", options=["All"] + list(saudi_events), key="ksa_event_key")
            if selected_event_saudi != "All":
                saudi_combined = saudi_combined[saudi_combined[event_col] == selected_event_saudi]

        saudi_combined = saudi_combined.drop_duplicates()
        st.dataframe(saudi_combined.reset_index(drop=True), width='stretch')

        # === REGIONAL COMPARISON ===
        st.markdown("---")
        st.subheader("Regional Comparison - KSA vs Gulf & Middle East")

        if DATA_CONNECTOR_AVAILABLE:
            try:
                # Get full rankings data
                full_rankings = load_full_rankings_cached()

                if full_rankings is not None and not full_rankings.empty:
                    rival_countries = ['KSA', 'BRN', 'QAT', 'UAE', 'KUW', 'OMA', 'IRN', 'JOR']

                    # Event selector
                    event_col_comp = 'event' if 'event' in full_rankings.columns else 'Event Type'
                    events_available = sorted(full_rankings[event_col_comp].dropna().unique())

                    comp_col1, comp_col2 = st.columns([2, 1])
                    with comp_col1:
                        selected_event_comp = st.selectbox(
                            "Select Event for Comparison",
                            events_available,
                            key="country_comp_event"
                        )
                    with comp_col2:
                        selected_gender_comp = st.selectbox(
                            "Gender",
                            ['Men', 'Women'],
                            key="country_comp_gender"
                        )

                    # Filter by gender if column exists
                    filtered_df = full_rankings.copy()
                    if 'gender' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['gender'].str.lower() == selected_gender_comp.lower()]

                    # Filter to event and rival countries
                    nat_col = 'nat' if 'nat' in filtered_df.columns else 'Country'
                    event_df = filtered_df[(filtered_df[event_col_comp] == selected_event_comp) &
                                          (filtered_df[nat_col].isin(rival_countries))]

                    if not event_df.empty:
                        # Parse results
                        result_col = 'result' if 'result' in event_df.columns else 'Score'

                        def parse_for_comparison(val):
                            if pd.isna(val):
                                return None
                            val_str = str(val).strip()
                            try:
                                if ':' in val_str:
                                    parts = val_str.split(':')
                                    return float(parts[0]) * 60 + float(parts[1])
                                return float(val_str)
                            except:
                                return None

                        event_df = event_df.copy()
                        event_df['result_numeric'] = event_df[result_col].apply(parse_for_comparison)
                        event_df = event_df.dropna(subset=['result_numeric'])

                        # Determine if field event
                        is_field = any(kw in selected_event_comp.lower() for kw in ['jump', 'vault', 'put', 'throw', 'discus', 'javelin', 'hammer'])

                        # Calculate country stats
                        competitor_col = 'competitor' if 'competitor' in event_df.columns else 'Name'
                        country_stats = []
                        for country in rival_countries:
                            country_df = event_df[event_df[nat_col] == country]
                            if country_df.empty:
                                continue

                            # Get best per athlete
                            if is_field:
                                best_by_athlete = country_df.groupby(competitor_col)['result_numeric'].max()
                                top_perf = best_by_athlete.max()
                                top_3_avg = best_by_athlete.nlargest(3).mean() if len(best_by_athlete) >= 3 else best_by_athlete.mean()
                            else:
                                best_by_athlete = country_df.groupby(competitor_col)['result_numeric'].min()
                                top_perf = best_by_athlete.min()
                                top_3_avg = best_by_athlete.nsmallest(3).mean() if len(best_by_athlete) >= 3 else best_by_athlete.mean()

                            country_stats.append({
                                'Country': country,
                                'Athletes': len(best_by_athlete),
                                'Top 1': round(top_perf, 2),
                                'Top 3 Avg': round(top_3_avg, 2)
                            })

                        if country_stats:
                            comparison_df = pd.DataFrame(country_stats)
                            comparison_df = comparison_df.sort_values('Top 1', ascending=not is_field)

                            # Display table with KSA highlighted
                            st.dataframe(comparison_df, width='stretch', hide_index=True)

                            # Visual comparison chart
                            fig_comp = go.Figure()

                            colors = [TEAL_PRIMARY if c == 'KSA' else GRAY_BLUE for c in comparison_df['Country']]

                            fig_comp.add_trace(go.Bar(
                                x=comparison_df['Country'],
                                y=comparison_df['Top 1'],
                                name='Best Performance',
                                marker_color=colors,
                                text=comparison_df['Top 1'],
                                textposition='outside'
                            ))

                            fig_comp.update_layout(
                                title=f"{selected_event_comp} - Regional Best Performances ({selected_gender_comp})",
                                xaxis_title="Country",
                                yaxis_title="Performance",
                                yaxis=dict(autorange='reversed') if not is_field else dict(),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                height=400
                            )
                            fig_comp.update_xaxes(showgrid=False)
                            fig_comp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

                            st.plotly_chart(fig_comp, width='stretch')

                            # KSA position insight
                            ksa_row = comparison_df[comparison_df['Country'] == 'KSA']
                            if not ksa_row.empty:
                                ksa_rank = comparison_df.index.get_loc(ksa_row.index[0]) + 1
                                total_countries = len(comparison_df)
                                if ksa_rank == 1:
                                    st.success(f"KSA leads the region in {selected_event_comp}!")
                                elif ksa_rank <= 3:
                                    st.info(f"KSA ranks #{ksa_rank} of {total_countries} regional rivals in {selected_event_comp}")
                                else:
                                    st.warning(f"KSA ranks #{ksa_rank} of {total_countries} - opportunity for improvement in {selected_event_comp}")
                        else:
                            st.info(f"No data available for regional comparison in {selected_event_comp}")
                    else:
                        st.info(f"No data for {selected_event_comp} among regional rivals")
            except Exception as e:
                st.warning(f"Country comparison unavailable: {str(e)[:100]}")

    else:
        st.warning("No Saudi athletes found in rankings. Data may still be loading from Azure.")

###################################
# Tab 5: World Championships Qualification
###################################
with tab5:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Championship Qualification Standards</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Entry standards (points & times) for <strong style="color: #FFD700;">Olympics</strong>,
            <strong style="color: {GOLD_ACCENT};">World Championships</strong>,
            <strong style="color: {TEAL_LIGHT};">Asian Games</strong>, and
            <strong style="color: #C0C0C0;">GCC Championships</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # === MULTI-CHAMPIONSHIP STANDARDS SECTION ===
    st.subheader("Qualification Standards by Championship")

    # Use WhatItTakesToWin analyzer for live championship data
    wittw_tab5 = get_wittw_analyzer()

    # Championship selector
    col_champ1, col_champ2, col_champ3 = st.columns(3)

    with col_champ1:
        # Use dictionary keys that match MAJOR_COMP_KEYWORDS in what_it_takes_to_win.py
        champ_types_tab5 = ['World Champs', 'Olympic', 'Asian Games', 'Diamond League', 'Arab Championships']
        if wittw_tab5.data is not None:
            actual_types = wittw_tab5.get_competition_types()
            if actual_types:
                champ_types_tab5 = actual_types
        selected_champ_tab5 = st.selectbox("Championship", champ_types_tab5, key="tab5_champ_type")

    with col_champ2:
        gender_tab5 = st.selectbox("Gender", ['men', 'women'], key="tab5_gender")

    with col_champ3:
        year_options_tab5 = ['All Years'] + [str(y) for y in wittw_tab5.get_available_years()] if wittw_tab5.data is not None else ['All Years']
        selected_year_tab5 = st.selectbox("Year", year_options_tab5, key="tab5_year")

    # Generate standards report for selected championship
    if wittw_tab5.data is not None and len(wittw_tab5.data) > 0:
        year_filter = None if selected_year_tab5 == 'All Years' else int(selected_year_tab5)
        standards_report = wittw_tab5.generate_what_it_takes_report(gender_tab5, year_filter, selected_champ_tab5)

        if len(standards_report) > 0:
            st.success(f"**{len(standards_report)} events** with standards data for {selected_champ_tab5}")

            # Show key columns
            display_cols_tab5 = ['Event', 'Gold Standard', 'Silver Standard', 'Bronze Standard', 'Final Standard (8th)', 'Sample Size']
            display_cols_tab5 = [c for c in display_cols_tab5 if c in standards_report.columns]

            st.dataframe(
                standards_report[display_cols_tab5],
                width='stretch',
                hide_index=True
            )

            # Event category summary
            st.markdown("---")
            st.caption("**Quick Reference - Entry Standards by Event Category**")

            # Categorize events
            sprints = standards_report[standards_report['Event'].str.contains('100|200|400', case=False, na=False) & ~standards_report['Event'].str.contains('hurdle', case=False, na=False)]
            mid_dist = standards_report[standards_report['Event'].str.contains('800|1500', case=False, na=False)]
            long_dist = standards_report[standards_report['Event'].str.contains('3000|5000|10000|Marathon', case=False, na=False)]

            cat_col1, cat_col2, cat_col3 = st.columns(3)
            with cat_col1:
                if not sprints.empty:
                    st.markdown(f"**Sprints ({len(sprints)} events)**")
                    for _, row in sprints.head(5).iterrows():
                        gold = row.get('Gold Standard', 'N/A')
                        st.caption(f"• {row['Event']}: {gold}")
            with cat_col2:
                if not mid_dist.empty:
                    st.markdown(f"**Middle Distance ({len(mid_dist)} events)**")
                    for _, row in mid_dist.head(5).iterrows():
                        gold = row.get('Gold Standard', 'N/A')
                        st.caption(f"• {row['Event']}: {gold}")
            with cat_col3:
                if not long_dist.empty:
                    st.markdown(f"**Long Distance ({len(long_dist)} events)**")
                    for _, row in long_dist.head(5).iterrows():
                        gold = row.get('Gold Standard', 'N/A')
                        st.caption(f"• {row['Event']}: {gold}")
        else:
            st.warning(f"No standards data available for {selected_champ_tab5}")
            # Debug info
            if wittw_tab5.data is not None:
                filtered = wittw_tab5.filter_by_competition(selected_champ_tab5)
                events = wittw_tab5.get_available_events()
                st.caption(f"Debug: Total records: {len(wittw_tab5.data):,} | Filtered: {len(filtered) if filtered is not None else 0} | Events: {len(events)}")
                if filtered is not None and len(filtered) > 0:
                    cols = filtered.columns.tolist()
                    st.caption(f"Columns: {', '.join(cols[:10])}")
                    # Check if rank column exists
                    rank_col = 'Rank' if 'Rank' in filtered.columns else ('rank' if 'rank' in filtered.columns else None)
                    if rank_col:
                        st.caption(f"Rank values: {sorted(filtered[rank_col].dropna().unique()[:10])}")
                    else:
                        st.caption("No rank column found in data")
    else:
        st.info("Championship data loading...")

    st.markdown("---")

    # === STANDARDS PROGRESSION CHART ===
    if RACE_INTELLIGENCE_AVAILABLE:
        with st.expander("📈 Standards Progression Over Time", expanded=False):
            st.caption("See how qualification standards have changed across championship cycles")

            prog_col1, prog_col2 = st.columns(2)
            with prog_col1:
                prog_event = st.selectbox("Event", ['100m', '200m', '400m', '800m', '1500m'], key='prog_event_tab5')
            with prog_col2:
                prog_gender = st.selectbox("Gender", ['Men', 'Women'], key='prog_gender_tab5')

            # Get historical standards data
            standards_df = get_standards_progression(prog_event, prog_gender)

            if not standards_df.empty:
                # Create line chart
                fig_prog = go.Figure()

                # Add lines for each championship type
                for champ, color, dash in [('Olympic', '#FFD700', 'solid'), ('World Champs', GOLD_ACCENT, 'dash'), ('Asian Games', TEAL_LIGHT, 'dot')]:
                    if champ in standards_df.columns:
                        valid_data = standards_df[standards_df[champ].notna()]
                        if not valid_data.empty:
                            fig_prog.add_trace(go.Scatter(
                                x=valid_data['Year'],
                                y=valid_data[champ],
                                mode='lines+markers',
                                name=champ,
                                line=dict(color=color, width=2, dash=dash),
                                marker=dict(size=8),
                                hovertemplate=f"<b>{champ}</b><br>Year: %{{x}}<br>Standard: %{{y:.2f}}<extra></extra>"
                            ))

                # Update layout
                is_time_event = prog_event not in ['High Jump', 'Pole Vault', 'Long Jump', 'Triple Jump', 'Shot Put', 'Discus Throw', 'Hammer Throw', 'Javelin Throw']

                fig_prog.update_layout(
                    title=f"{prog_event} {prog_gender} - Entry Standards by Year",
                    xaxis_title="Year",
                    yaxis_title="Standard (seconds)" if is_time_event else "Standard",
                    yaxis=dict(autorange='reversed') if is_time_event else dict(),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=12),
                    height=350,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig_prog.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                fig_prog.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

                st.plotly_chart(fig_prog, width='stretch')

                # Show data table
                st.markdown("**Historical Standards Table**")
                st.dataframe(standards_df, width='stretch', hide_index=True)

                # Trend insight
                if 'Olympic' in standards_df.columns:
                    olympic_vals = standards_df['Olympic'].dropna()
                    if len(olympic_vals) >= 2:
                        first_val = olympic_vals.iloc[-1]  # Oldest
                        last_val = olympic_vals.iloc[0]    # Most recent
                        change = last_val - first_val
                        if is_time_event:
                            direction = "tightening" if change < 0 else "loosening"
                            st.info(f"**Trend Insight:** Olympic standards are {direction} by ~{abs(change):.2f}s over this period")
                        else:
                            direction = "increasing" if change > 0 else "decreasing"
                            st.info(f"**Trend Insight:** Olympic standards are {direction} by ~{abs(change):.2f} over this period")
            else:
                st.info("No historical standards data available for this event")

    st.markdown("---")

    # === WORLD CHAMPIONSHIPS QUALIFICATION TRACKING ===
    st.subheader("World Championships Tokyo 2025 - Live Qualification Tracking")

    # Show data source indicator
    mode = get_data_mode() if DATA_CONNECTOR_AVAILABLE else 'local'

    # Lazy load qualification data
    road_to_df = get_road_to_df()
    qual_standards_df = get_qual_standards_df()

    if road_to_df is not None and not road_to_df.empty:
        # Filter to only qualified athletes (exclude "All_Status" which is summary rows)
        qualified_df = road_to_df[road_to_df['Qualification_Status'] != 'All_Status'].copy()

        # === QUALIFICATION OVERVIEW ===
        st.subheader("Qualification Routes Overview")

        # Count by qualification method
        qual_counts = qualified_df['Qualification_Status'].value_counts()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            entry_std = qual_counts.get('Qualified_by_Entry_Standard', 0)
            st.markdown(f"""
            <div style="background: {TEAL_PRIMARY}; padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">By Entry Standard</p>
                <p style="color: white; font-size: 1.8rem; font-weight: bold; margin: 0.25rem 0;">{entry_std:,}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            by_ranking = qual_counts.get('In_World_Rankings_quota', 0)
            st.markdown(f"""
            <div style="background: {GOLD_ACCENT}; padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">By World Ranking</p>
                <p style="color: white; font-size: 1.8rem; font-weight: bold; margin: 0.25rem 0;">{by_ranking:,}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            wild_card = qual_counts.get('Qualified_by_Wild_Card', 0)
            st.markdown(f"""
            <div style="background: #C0C0C0; padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: #333; margin: 0; font-size: 0.85rem;">Wild Cards</p>
                <p style="color: #333; font-size: 1.8rem; font-weight: bold; margin: 0.25rem 0;">{wild_card:,}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            next_best = qual_counts.get('Next_best_by_World_Rankings', 0)
            st.markdown(f"""
            <div style="background: {GRAY_BLUE}; padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Next Best (Reserve)</p>
                <p style="color: white; font-size: 1.8rem; font-weight: bold; margin: 0.25rem 0;">{next_best:,}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === KSA ATHLETES SECTION ===
        st.subheader("KSA Athletes Qualification Status")

        # Find KSA athletes (Athlete column has country code)
        ksa_in_data = qualified_df[qualified_df['Athlete'] == 'KSA'].copy()

        if not ksa_in_data.empty:
            # Group by qualification status
            ksa_qualified_athletes = ksa_in_data[ksa_in_data['Qualification_Status'].isin(['Qualified_by_Entry_Standard', 'In_World_Rankings_quota', 'Qualified_by_Wild_Card'])]
            ksa_reserve = ksa_in_data[ksa_in_data['Qualification_Status'] == 'Next_best_by_World_Rankings']

            if not ksa_qualified_athletes.empty:
                st.success(f"**{len(ksa_qualified_athletes)} KSA athletes qualified for Tokyo 2025!**")

                ksa_display = ksa_qualified_athletes[['Actual_Event_Name', 'Status', 'QP', 'Qualification_Status', 'Details']].copy()
                ksa_display.columns = ['Event', 'Athlete', 'Qual Position', 'Route', 'Performance']
                ksa_display['Route'] = ksa_display['Route'].str.replace('_', ' ').str.replace('Qualified by ', '').str.replace('In World Rankings quota', 'World Ranking')

                st.dataframe(ksa_display.drop_duplicates(), width='stretch', hide_index=True)
            else:
                st.warning("No KSA athletes currently in qualified positions")

            if not ksa_reserve.empty:
                st.info(f"**{len(ksa_reserve)} KSA athletes in reserve positions** (may qualify if others withdraw)")

                reserve_display = ksa_reserve[['Actual_Event_Name', 'Status', 'QP', 'Details']].copy()
                reserve_display.columns = ['Event', 'Athlete', 'Reserve Position', 'Current Mark']
                st.dataframe(reserve_display.drop_duplicates().head(20), width='stretch', hide_index=True)
        else:
            st.info("Search for KSA athletes in the event analysis below")

        st.markdown("---")

        # === EVENT-SPECIFIC ANALYSIS ===
        st.subheader("Event-by-Event Qualification Analysis")

        col1, col2 = st.columns(2)

        with col1:
            events_list = sorted(qualified_df['Actual_Event_Name'].dropna().unique())
            selected_event_qual = st.selectbox("Select Event", events_list, key="qual_event_select")

        with col2:
            qual_routes = ['All Routes', 'Qualified_by_Entry_Standard', 'In_World_Rankings_quota', 'Qualified_by_Wild_Card', 'Next_best_by_World_Rankings']
            selected_route = st.selectbox("Qualification Route", qual_routes, key="qual_route_select")

        # Filter to selected event
        event_data = qualified_df[qualified_df['Actual_Event_Name'] == selected_event_qual].copy()

        if selected_route != 'All Routes':
            event_data = event_data[event_data['Qualification_Status'] == selected_route]

        if not event_data.empty:
            # Show key insights for this event
            st.markdown(f"### {selected_event_qual}")

            # Count by route for this event
            event_routes = event_data['Qualification_Status'].value_counts()

            insight_cols = st.columns(4)
            with insight_cols[0]:
                std_count = event_routes.get('Qualified_by_Entry_Standard', 0)
                st.metric("By Standard", std_count)
            with insight_cols[1]:
                rank_count = event_routes.get('In_World_Rankings_quota', 0)
                st.metric("By Ranking", rank_count)
            with insight_cols[2]:
                wc_count = event_routes.get('Qualified_by_Wild_Card', 0)
                st.metric("Wild Cards", wc_count)
            with insight_cols[3]:
                total_qual = std_count + rank_count + wc_count
                st.metric("Total Qualified", total_qual)

            # Extract performance marks from Details column
            def extract_mark(detail):
                if pd.isna(detail):
                    return None
                # Look for time pattern (e.g., "10.72") or distance (e.g., "8.50")
                import re
                match = re.search(r'^([\d.:]+)', str(detail))
                if match:
                    return match.group(1)
                return None

            event_data['Mark'] = event_data['Details'].apply(extract_mark)

            # Show the qualified athletes for this event
            if selected_route == 'All Routes':
                # Show by category
                for route in ['Qualified_by_Entry_Standard', 'In_World_Rankings_quota', 'Qualified_by_Wild_Card']:
                    route_data = event_data[event_data['Qualification_Status'] == route]
                    if not route_data.empty:
                        route_name = route.replace('_', ' ').replace('Qualified by ', '').replace('In World Rankings quota', 'By World Ranking')
                        with st.expander(f"{route_name} ({len(route_data)} athletes)", expanded=(route == 'Qualified_by_Entry_Standard')):
                            display = route_data[['QP', 'Status', 'Athlete', 'Mark', 'Details']].copy()
                            display.columns = ['Pos', 'Athlete Name', 'Country', 'Mark', 'Details']
                            display = display.sort_values('Pos')
                            st.dataframe(display.drop_duplicates().head(50), width='stretch', hide_index=True)
            else:
                # Show filtered data
                display = event_data[['QP', 'Status', 'Athlete', 'Mark', 'Details']].copy()
                display.columns = ['Pos', 'Athlete Name', 'Country', 'Mark', 'Details']
                display = display.sort_values('Pos')
                st.dataframe(display.drop_duplicates().head(100), width='stretch', hide_index=True)

            # Show "last qualifier" analysis
            entry_std_athletes = event_data[event_data['Qualification_Status'] == 'Qualified_by_Entry_Standard']
            if not entry_std_athletes.empty and 'QP' in entry_std_athletes.columns:
                last_by_std = entry_std_athletes.nlargest(1, 'QP')
                if not last_by_std.empty:
                    last_mark = last_by_std.iloc[0]['Mark'] if 'Mark' in last_by_std.columns else 'N/A'
                    last_name = last_by_std.iloc[0]['Status']
                    st.info(f"**Last qualifier by entry standard:** {last_name} - {last_mark}")

            # Show first non-qualifier (what was needed but missed)
            reserve = event_data[event_data['Qualification_Status'] == 'Next_best_by_World_Rankings']
            if not reserve.empty and 'QP' in reserve.columns:
                first_reserve = reserve.nsmallest(1, 'QP')
                if not first_reserve.empty:
                    reserve_mark = first_reserve.iloc[0]['Mark'] if 'Mark' in first_reserve.columns else 'N/A'
                    reserve_name = first_reserve.iloc[0]['Status']
                    st.caption(f"First reserve: {reserve_name} ({first_reserve.iloc[0]['Athlete']}) - {reserve_mark}")

            # === WPA POINTS & PERFORMANCE ANALYSIS ===
            st.markdown("---")
            st.subheader("Performance Distribution Analysis")

            # Parse WPA points from Column_6 (e.g., "1312p")
            def parse_wpa_points(value):
                if pd.isna(value):
                    return None
                match = re.search(r'(\d+)p', str(value), re.IGNORECASE)
                return int(match.group(1)) if match else None

            # Parse numeric performance from Details
            def parse_numeric_performance(detail):
                if pd.isna(detail):
                    return None
                # Look for time (e.g., "10.72", "1:59.00") or distance
                match = re.search(r'^([\d:.]+)', str(detail))
                if match:
                    time_str = match.group(1)
                    try:
                        if ':' in time_str:
                            parts = time_str.split(':')
                            if len(parts) == 2:
                                return float(parts[0]) * 60 + float(parts[1])
                        return float(time_str)
                    except:
                        return None
                return None

            # Add parsed columns
            event_data_analysis = event_data.copy()
            if 'Column_6' in event_data_analysis.columns:
                event_data_analysis['WPA_Points'] = event_data_analysis['Column_6'].apply(parse_wpa_points)
            event_data_analysis['Numeric_Mark'] = event_data_analysis['Details'].apply(parse_numeric_performance)

            # Only show for qualified athletes
            qual_for_analysis = event_data_analysis[event_data_analysis['Qualification_Status'].isin(
                ['Qualified_by_Entry_Standard', 'In_World_Rankings_quota', 'Qualified_by_Wild_Card']
            )]

            # Performance statistics
            if not qual_for_analysis.empty and 'Numeric_Mark' in qual_for_analysis.columns:
                marks_valid = qual_for_analysis.dropna(subset=['Numeric_Mark'])

                if not marks_valid.empty:
                    # Determine if it's a track event (lower is better) or field event (higher is better)
                    is_track = any(x in selected_event_qual.lower() for x in ['metres', 'meter', 'hurdles', 'steeplechase', 'marathon', 'walk'])

                    stat_cols = st.columns(5)
                    with stat_cols[0]:
                        best = marks_valid['Numeric_Mark'].min() if is_track else marks_valid['Numeric_Mark'].max()
                        st.metric("Best Qualifier", f"{best:.2f}")
                    with stat_cols[1]:
                        worst = marks_valid['Numeric_Mark'].max() if is_track else marks_valid['Numeric_Mark'].min()
                        st.metric("Last Qualifier", f"{worst:.2f}")
                    with stat_cols[2]:
                        st.metric("Average", f"{marks_valid['Numeric_Mark'].mean():.2f}")
                    with stat_cols[3]:
                        st.metric("Median", f"{marks_valid['Numeric_Mark'].median():.2f}")
                    with stat_cols[4]:
                        st.metric("Std Dev", f"{marks_valid['Numeric_Mark'].std():.2f}")

                    # Box plot of qualifying marks by route
                    fig_box = go.Figure()

                    route_colors = {
                        'Qualified_by_Entry_Standard': TEAL_PRIMARY,
                        'In_World_Rankings_quota': GOLD_ACCENT,
                        'Qualified_by_Wild_Card': '#C0C0C0'
                    }

                    for route in ['Qualified_by_Entry_Standard', 'In_World_Rankings_quota', 'Qualified_by_Wild_Card']:
                        route_data = marks_valid[marks_valid['Qualification_Status'] == route]
                        if not route_data.empty:
                            route_name = route.replace('_', ' ').replace('Qualified by ', '').replace('In World Rankings quota', 'World Ranking')
                            fig_box.add_trace(go.Box(
                                y=route_data['Numeric_Mark'],
                                name=route_name,
                                boxpoints='all',
                                jitter=0.3,
                                pointpos=0,
                                marker_color=route_colors.get(route, GRAY_BLUE),
                                text=route_data['Status'],
                                hovertemplate="<b>%{text}</b><br>Mark: %{y:.2f}<extra></extra>"
                            ))

                    fig_box.update_layout(
                        title=f"{selected_event_qual} - Qualifying Marks Distribution",
                        yaxis_title="Mark (seconds or meters)",
                        yaxis=dict(autorange='reversed') if is_track else dict(),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig_box, width='stretch')

            # WPA Points distribution (if available)
            if 'WPA_Points' in qual_for_analysis.columns:
                points_valid = qual_for_analysis.dropna(subset=['WPA_Points'])
                if not points_valid.empty and len(points_valid) > 5:
                    st.subheader("World Athletics Ranking Points")

                    pts_cols = st.columns(4)
                    with pts_cols[0]:
                        st.metric("Highest Points", f"{points_valid['WPA_Points'].max():,.0f}")
                    with pts_cols[1]:
                        st.metric("Lowest Qualified", f"{points_valid['WPA_Points'].min():,.0f}")
                    with pts_cols[2]:
                        st.metric("Average", f"{points_valid['WPA_Points'].mean():,.0f}")
                    with pts_cols[3]:
                        st.metric("Median", f"{points_valid['WPA_Points'].median():,.0f}")

                    # Histogram of WPA points
                    fig_hist = go.Figure(data=[
                        go.Histogram(
                            x=points_valid['WPA_Points'],
                            nbinsx=20,
                            marker_color=TEAL_PRIMARY,
                            opacity=0.75
                        )
                    ])
                    fig_hist.update_layout(
                        title="Distribution of WPA Ranking Points (Qualified Athletes)",
                        xaxis_title="WPA Points",
                        yaxis_title="Number of Athletes",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=350
                    )
                    st.plotly_chart(fig_hist, width='stretch')

        st.markdown("---")

        # === COUNTRY ANALYSIS ===
        st.subheader("Qualification by Country")

        # Count qualified athletes by country
        actual_qualified = qualified_df[qualified_df['Qualification_Status'].isin(['Qualified_by_Entry_Standard', 'In_World_Rankings_quota', 'Qualified_by_Wild_Card'])]
        country_counts = actual_qualified.groupby('Athlete').size().sort_values(ascending=False).head(20)

        if not country_counts.empty:
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=country_counts.index,
                    y=country_counts.values,
                    marker_color=TEAL_PRIMARY,
                    text=country_counts.values,
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Top 20 Countries by Qualified Athletes",
                xaxis_title="Country",
                yaxis_title="Qualified Athletes",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, width='stretch')

            # Show KSA position
            if 'KSA' in country_counts.index:
                ksa_rank = list(country_counts.index).index('KSA') + 1
                ksa_count = country_counts['KSA']
                st.success(f"**KSA: Ranked #{ksa_rank} with {ksa_count} qualified athletes**")
            else:
                ksa_total = len(actual_qualified[actual_qualified['Athlete'] == 'KSA'])
                if ksa_total > 0:
                    st.info(f"KSA has {ksa_total} qualified athletes")

        # === NEAR MISS ANALYSIS ===
        st.markdown("---")
        st.subheader("Near Miss Analysis - Active Athletes Close to Standards")
        st.markdown(f"""
        <p style='color: #ccc; font-size: 0.9em;'>
        <strong>Active KSA athletes (2023+)</strong> within <strong style="color: {GOLD_ACCENT};">5%</strong> of qualification standards who could break through with focused training.
        </p>
        """, unsafe_allow_html=True)

        # Get KSA athletes from master data (ACTIVE athletes only - 2023+)
        if DATA_CONNECTOR_AVAILABLE:
            try:
                ksa_rankings = load_ksa_rankings_raw_cached()
                if ksa_rankings is not None and not ksa_rankings.empty:
                    # Filter to active athletes (results in 2023 or later)
                    if 'year' in ksa_rankings.columns:
                        active_rankings = ksa_rankings[ksa_rankings['year'] >= 2023]
                    elif 'date' in ksa_rankings.columns:
                        ksa_rankings['_temp_year'] = pd.to_datetime(ksa_rankings['date'], errors='coerce').dt.year
                        active_rankings = ksa_rankings[ksa_rankings['_temp_year'] >= 2023].copy()
                        active_rankings = active_rankings.drop(columns=['_temp_year'], errors='ignore')
                    else:
                        active_rankings = ksa_rankings  # Fall back to all if no date column

                    near_miss_data = []

                    # Tokyo 2025 entry standards (in seconds for track, meters for field)
                    tokyo_standards = {
                        '100m': 10.00, '100-metres': 10.00, '100 metres': 10.00,
                        '200m': 20.24, '200-metres': 20.24, '200 metres': 20.24,
                        '400m': 44.90, '400-metres': 44.90, '400 metres': 44.90,
                        '800m': 103.50, '800-metres': 103.50, '800 metres': 103.50,  # 1:43.50
                        '1500m': 213.50, '1500-metres': 213.50, '1500 metres': 213.50,  # 3:33.50
                        'long-jump': 8.27, 'long jump': 8.27,
                        'high-jump': 2.33, 'high jump': 2.33,
                        'triple-jump': 17.22, 'triple jump': 17.22,
                        'shot-put': 21.10, 'shot put': 21.10,
                        'discus-throw': 66.00, 'discus throw': 66.00,
                        'javelin-throw': 85.20, 'javelin throw': 85.20
                    }

                    # Helper functions for near miss calculation
                    def parse_result_to_numeric(result_str, event_name):
                        """Parse result string to numeric value (seconds for track, meters for field)."""
                        if pd.isna(result_str):
                            return None
                        result_str = str(result_str).strip()
                        try:
                            # Handle time formats like "1:43.50" or "10.72"
                            if ':' in result_str:
                                parts = result_str.split(':')
                                if len(parts) == 2:
                                    return float(parts[0]) * 60 + float(parts[1])
                                elif len(parts) == 3:
                                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                            # Handle simple numeric values
                            return float(result_str.replace('m', '').replace('s', '').strip())
                        except (ValueError, AttributeError):
                            return None

                    def is_field_event_check(event_name):
                        """Check if event is a field event (higher is better)."""
                        field_events = ['jump', 'put', 'throw', 'shot', 'discus', 'javelin', 'hammer', 'pole vault']
                        return any(fe in str(event_name).lower() for fe in field_events)

                    def calculate_near_miss_pct(pb, standard, is_field):
                        """Calculate percentage gap to standard."""
                        if pb is None or standard is None or standard == 0:
                            return None, None
                        if is_field:
                            # Field events: higher is better
                            gap = standard - pb
                            pct = (gap / standard) * 100
                        else:
                            # Track events: lower is better
                            gap = pb - standard
                            pct = (gap / standard) * 100
                        return gap, pct

                    # Determine event column name
                    event_col = 'event' if 'event' in active_rankings.columns else 'Event Type' if 'Event Type' in active_rankings.columns else None
                    result_col = 'result' if 'result' in active_rankings.columns else 'Result' if 'Result' in active_rankings.columns else None
                    athlete_col = 'competitor' if 'competitor' in active_rankings.columns else 'Competitor' if 'Competitor' in active_rankings.columns else None

                    if event_col and result_col and athlete_col:
                        # Group by athlete and event (active athletes 2023+ only)
                        for (athlete, event), group in active_rankings.groupby([athlete_col, event_col]):
                            # Parse results and get PB
                            results = group[result_col].apply(lambda x: parse_result_to_numeric(x, event)).dropna()
                            if results.empty:
                                continue

                            is_field = is_field_event_check(event)
                            pb = results.max() if is_field else results.min()

                            # Normalize event name for lookup
                            event_lower = str(event).lower()
                            event_normalized = event_lower.replace(' ', '-').replace('metres', 'm')

                            # Try different variations to find the standard
                            standard = tokyo_standards.get(event_normalized) or tokyo_standards.get(event_lower) or tokyo_standards.get(event_lower.replace('-', ' '))

                            if standard:
                                gap, pct = calculate_near_miss_pct(pb, standard, is_field)

                                if pct is not None and pct > 0 and pct <= 5.0:
                                    # Determine status and color based on percentage
                                    if pct <= 1.0:
                                        status = "Very Close (<1%)"
                                        color = TEAL_PRIMARY
                                    elif pct <= 2.5:
                                        status = f"Close ({pct:.1f}%)"
                                        color = TEAL_LIGHT
                                    else:
                                        status = f"Within Reach ({pct:.1f}%)"
                                        color = GOLD_ACCENT

                                    near_miss_data.append({
                                        'Athlete': athlete,
                                        'Event': event,
                                        'PB': pb,
                                        'Standard': standard,
                                        'Gap': gap,
                                        'Percentage': pct,
                                        'Status': status,
                                        'Color': color
                                    })

                    if near_miss_data:
                        # Sort by closest to standard (lowest percentage)
                        near_miss_df = pd.DataFrame(near_miss_data).sort_values('Percentage')

                        st.success(f"Found **{len(near_miss_df)} KSA athletes** within 5% of major championship standards!")

                        # Display as cards
                        for _, row in near_miss_df.head(10).iterrows():
                            # Format display values
                            if row['PB'] >= 60:  # Time in seconds, convert to mm:ss
                                mins = int(row['PB'] // 60)
                                secs = row['PB'] % 60
                                pb_display = f"{mins}:{secs:05.2f}"
                            else:
                                pb_display = f"{row['PB']:.2f}"

                            if row['Standard'] >= 60:
                                mins = int(row['Standard'] // 60)
                                secs = row['Standard'] % 60
                                std_display = f"{mins}:{secs:05.2f}"
                            else:
                                std_display = f"{row['Standard']:.2f}"

                            gap_display = f"{abs(row['Gap']):.2f}"

                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.3) 100%);
                                        border-left: 4px solid {row['Color']}; padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="color: white; font-size: 1.1rem;">{row['Athlete']}</strong>
                                        <p style="color: #aaa; margin: 0.25rem 0 0 0;">{row['Event']}</p>
                                    </div>
                                    <div style="text-align: right;">
                                        <p style="color: {row['Color']}; font-weight: bold; margin: 0; font-size: 1.2rem;">{row['Status']}</p>
                                        <p style="color: #aaa; margin: 0; font-size: 0.85rem;">PB: {pb_display} | Std: {std_display} | Gap: {gap_display}</p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No KSA athletes currently within 5% of major championship standards.")
                else:
                    st.info("No KSA rankings data available for near miss analysis.")
            except Exception as e:
                st.warning(f"Near miss analysis unavailable: {str(e)[:100]}")
        else:
            st.info("Data connector not available for near miss analysis.")

    else:
        st.warning("No qualification data found. Data may still be loading from Azure.")

    # Show qualification standards (from benchmarks parquet)
    if qual_standards_df is not None and not qual_standards_df.empty:
        st.markdown("---")
        st.subheader("Championship Performance Standards")
        st.caption("Historical medal and final standards based on championship data")

        # Check which columns exist (benchmarks parquet has different structure)
        if 'Event' in qual_standards_df.columns and 'Gold Standard' in qual_standards_df.columns:
            # Using benchmarks parquet format
            display_cols = [col for col in ['Event', 'Gender', 'Gold Standard', 'Silver Standard', 'Bronze Standard', 'Final Standard (8th)'] if col in qual_standards_df.columns]
            st.dataframe(qual_standards_df[display_cols], width='stretch', hide_index=True)
        elif 'Display_Name' in qual_standards_df.columns:
            # Using old CSV format
            display_qual = qual_standards_df[['Display_Name', 'entry_number', 'entry_standard', 'maximum_quota', 'athletes_by_entry_standard']].copy()
            display_qual.columns = ['Event', 'Entry Quota', 'Entry Standard', 'Max per Country', 'Qualified by Standard']
            st.dataframe(display_qual, width='stretch', hide_index=True)
        else:
            st.dataframe(qual_standards_df, width='stretch', hide_index=True)

    # === STANDARDS PROGRESSION HISTORY ===
    st.markdown("---")
    with st.expander("📈 Qualification Standards History", expanded=False):
        st.markdown(f"""
        <p style='color: #888;'>Track how qualification standards have changed over the years</p>
        """, unsafe_allow_html=True)

        prog_col1, prog_col2 = st.columns(2)
        with prog_col1:
            prog_event = st.selectbox("Event", ['100m', '200m', '400m', '800m', '1500m'], key='prog_event')
        with prog_col2:
            prog_gender = st.selectbox("Gender", ['Men', 'Women'], key='prog_gender')

        if RACE_INTELLIGENCE_AVAILABLE:
            standards_df = get_standards_progression(prog_event, prog_gender)

            if not standards_df.empty:
                # Line chart
                fig = go.Figure()

                for col in ['Olympic', 'World Champs', 'Asian Games']:
                    if col in standards_df.columns:
                        valid_data = standards_df[standards_df[col].notna()]
                        if not valid_data.empty:
                            fig.add_trace(go.Scatter(
                                x=valid_data['Year'],
                                y=valid_data[col],
                                mode='lines+markers',
                                name=col,
                                line={'width': 3},
                                marker={'size': 10}
                            ))

                fig.update_layout(
                    title=f'{prog_event} {prog_gender} Qualification Standards',
                    xaxis_title='Year',
                    yaxis_title='Standard',
                    yaxis={'autorange': 'reversed'},  # Lower time = better
                    height=400,
                    plot_bgcolor='white',
                    font={'family': 'Inter, sans-serif'}
                )

                st.plotly_chart(fig, width='stretch')

                # Table
                st.dataframe(standards_df, width='stretch', hide_index=True)

                # Trend insight
                if len(standards_df) >= 2:
                    latest = standards_df.iloc[0]
                    oldest = standards_df.iloc[-1]
                    for col in ['Olympic', 'World Champs']:
                        if col in standards_df.columns and pd.notna(latest[col]) and pd.notna(oldest[col]):
                            change = latest[col] - oldest[col]
                            years = latest['Year'] - oldest['Year']
                            if years > 0:
                                per_year = change / years
                                direction = "tightening" if change < 0 else "loosening"
                                st.info(f"💡 {col} standards {direction} by ~{abs(per_year):.2f}s per year")
            else:
                st.info(f"No historical data available for {prog_event}")
        else:
            st.warning("Race intelligence module not available")

###################################
# Tab 6: Major Games Analytics
###################################
with tab6:
    st.header("Major Games Analytics")
    st.markdown(f"""
    <p style='color: #ccc; font-size: 0.95em;'>
    Performance analysis of KSA athletes at major international games including
    <strong style="color: {GOLD_ACCENT};">Olympics</strong>,
    <strong style="color: {TEAL_LIGHT};">Asian Games</strong>, and
    <strong style="color: white;">World Championships</strong>.
    </p>
    """, unsafe_allow_html=True)

    # Initialize analytics
    @st.cache_resource
    def get_analytics():
        return AthleticsAnalytics()

    analytics = get_analytics()

    # Major Games Summary
    st.subheader("KSA Major Games Performance Summary")

    major_summary = analytics.major_games.get_major_games_summary()

    if 'error' not in major_summary and len(major_summary) > 0:
        # Create metrics row
        num_cols = min(len(major_summary), 6)
        cols = st.columns(num_cols)

        game_colors = {
            'Olympic': '#FFD700',
            'World Championships': '#C0C0C0',
            'Asian Championships': TEAL_PRIMARY,
            'Asian Games': TEAL_LIGHT,
            'West Asian': '#4682B4',
            'Arab Championships': GOLD_ACCENT,
            'GCC': '#8FBC8F',
            'World U20': '#DDA0DD'
        }

        for i, (game_type, data) in enumerate(major_summary.items()):
            if i < 6:
                with cols[i]:
                    color = game_colors.get(game_type, GRAY_BLUE)
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.5); border: 2px solid {color}; border-radius: 8px; padding: 1rem; text-align: center;">
                        <p style="color: {color}; margin: 0; font-size: 0.75rem; font-weight: bold;">{game_type}</p>
                        <p style="color: white; font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{data['total_performances']}</p>
                        <p style="color: #aaa; margin: 0; font-size: 0.7rem;">{data['unique_athletes']} athletes</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Show detailed breakdown
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Olympic Performers")
            olympic_data = analytics.major_games.get_olympic_performers()

            if 'athletes' in olympic_data:
                for athlete, data in olympic_data.get('athletes', {}).items():
                    events = list(set(data.get('event_name', [])))
                    best_place = data.get('place', 'N/A')
                    st.markdown(f"""
                    <div class="athlete-card" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border-left: 4px solid #FFD700;">
                        <h4 style="color: white; margin: 0;">{athlete}</h4>
                        <p style="color: #aaa; margin: 0.3rem 0;">Events: {', '.join(events)}</p>
                        <p style="color: #FFD700; margin: 0;">Best Placement: {int(best_place) if best_place and pd.notna(best_place) else 'N/A'}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No Olympic data available yet")

        with col2:
            st.subheader("Finals at Major Games")
            try:
                finals_data = analytics.major_games.get_finals_appearances()

                if 'by_athlete' in finals_data:
                    st.metric("Total Finals Appearances", finals_data.get('total_finals', 0))

                    for athlete, data in list(finals_data.get('by_athlete', {}).items())[:5]:
                        games = list(set(data.get('game_category', [])))
                        events = list(set(data.get('event_name', [])))
                        st.markdown(f"""
                        <div style="background: rgba(0, 113, 103, 0.2); border-radius: 8px; padding: 0.8rem; margin: 0.5rem 0;">
                            <strong style="color: white;">{athlete}</strong><br>
                            <span style="color: {TEAL_LIGHT}; font-size: 0.85rem;">{', '.join(games)}</span><br>
                            <span style="color: #aaa; font-size: 0.8rem;">{', '.join(events)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(finals_data.get('message', 'No finals data available'))
            except Exception as e:
                st.info(f"Finals data unavailable: {str(e)[:100]}")

        # Discipline Knowledge Section
        st.markdown("---")
        st.subheader("Discipline Knowledge Base")

        discipline_options = list(DISCIPLINE_KNOWLEDGE.keys())
        selected_discipline = st.selectbox("Select Discipline", discipline_options, key="discipline_select")

        if selected_discipline:
            knowledge = DISCIPLINE_KNOWLEDGE.get(selected_discipline, {})

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**Events:** {', '.join(knowledge.get('events', []))}")

            with col2:
                if 'wind_legal_limit' in knowledge:
                    st.markdown(f"**Wind Legal Limit:** {knowledge['wind_legal_limit']} m/s")

            with col3:
                if 'altitude_effect' in knowledge:
                    st.markdown(f"**Altitude Effect:** {knowledge['altitude_effect']}")

            # Key Factors
            if 'key_factors' in knowledge:
                st.markdown("**Key Performance Factors:**")
                factors = knowledge['key_factors']
                if isinstance(factors, list):
                    for factor in factors:
                        st.markdown(f"- {factor}")
                elif isinstance(factors, dict):
                    for event, event_factors in factors.items():
                        st.markdown(f"*{event}:* {', '.join(event_factors)}")

        # Athlete Analysis Section
        st.markdown("---")
        st.subheader("Individual Athlete Analysis")

        # Get list of athletes from Azure or local database
        athletes_for_analysis = None

        # Try Azure first
        if DATA_CONNECTOR_AVAILABLE:
            try:
                ksa_profiles = get_ksa_athletes()
                if ksa_profiles is not None and not ksa_profiles.empty:
                    if 'full_name' in ksa_profiles.columns:
                        athletes_for_analysis = ksa_profiles[['full_name']].dropna()
                        if 'primary_event' in ksa_profiles.columns:
                            athletes_for_analysis = ksa_profiles[ksa_profiles['primary_event'].notna()][['full_name', 'primary_event']]
            except Exception as e:
                st.caption(f"Azure load: {str(e)[:50]}")

        # Fall back to local SQLite
        if athletes_for_analysis is None or athletes_for_analysis.empty:
            profiles_db = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')
            if os.path.exists(profiles_db):
                conn = sqlite3.connect(profiles_db)
                athletes_for_analysis = pd.read_sql("SELECT full_name, primary_event FROM ksa_athletes WHERE primary_event IS NOT NULL ORDER BY full_name", conn)
                conn.close()

        if athletes_for_analysis is not None and not athletes_for_analysis.empty:
            athlete_options = sorted(athletes_for_analysis['full_name'].unique().tolist())
            selected_athlete = st.selectbox("Select Athlete", athlete_options, key="athlete_analysis_select")

            if selected_athlete:
                try:
                    athlete_analysis = analytics.major_games.analyze_athlete_major_games(selected_athlete)

                    if 'error' not in athlete_analysis:
                        st.markdown(f"### {selected_athlete}")
                        st.metric("Total Major Games Performances", athlete_analysis.get('total_major_performances', 0))

                        # Highlights
                        if athlete_analysis.get('highlights'):
                            st.markdown("**Highlights (Top 8 Finishes):**")
                            for highlight in athlete_analysis['highlights']:
                                st.markdown(f"- **{highlight['place']}th place** at {highlight['game']} ({highlight['event']})")

                        # Breakdown by game type
                        if athlete_analysis.get('by_game_type'):
                            st.markdown("**Performance by Competition:**")
                            for game_type, game_data in athlete_analysis['by_game_type'].items():
                                with st.expander(f"{game_type} ({game_data['count']} performances)"):
                                    for result in game_data.get('results', [])[:10]:
                                        st.markdown(f"- {result.get('event_name', 'N/A')}: {result.get('result_value', 'N/A')} (Place: {result.get('place', 'N/A')})")
                    else:
                        st.info(athlete_analysis.get('error', 'No major games data for this athlete'))
                except Exception as e:
                    st.info(f"Could not analyze athlete: {str(e)[:100]}")
        else:
            st.info("No athlete profiles available. Data may still be loading from Azure.")

    else:
        if 'error' in major_summary:
            st.error("Could not load major games data.")
            error_msg = major_summary.get('error', 'Unknown error')
            st.info(f"Details: {error_msg}")
        else:
            st.warning("No major games data available.")
        if DATA_CONNECTOR_AVAILABLE:
            st.caption(f"Data mode: {get_data_mode()} | Try refreshing the page - Azure may be waking up.")

###################################
# Tab 7: What It Takes to Win (Live Data)
###################################
with tab7:
    st.header("What It Takes to Win")
    st.markdown(f"""
    <p style='color: #ccc; font-size: 0.95em;'>
    Performance standards from <strong style="color: #FFD700;">Olympics</strong>,
    <strong style="color: {GOLD_ACCENT};">World Championships</strong>,
    <strong style="color: {TEAL_PRIMARY};">Asian Games</strong>, and
    <strong style="color: {TEAL_LIGHT};">Arab Championships</strong>.
    Shows what marks are needed to medal across all levels and seasons.
    </p>
    """, unsafe_allow_html=True)

    # Use globally cached analyzer (defined at top of file)
    wittw = get_wittw_analyzer()

    # Show data source
    if DATA_CONNECTOR_AVAILABLE:
        st.caption(f"Data source: Azure Parquet ({len(wittw.data):,} records)" if wittw.data is not None else "Loading...")
    else:
        st.caption("Data source: Local files")

    if wittw.data is not None and len(wittw.data) > 0:
        # Show available columns for debugging
        available_cols = wittw.data.columns.tolist()
        has_venue = any(c in available_cols for c in ['venue', 'Venue', 'Competition', 'competition'])

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(wittw.data):,}")
        with col2:
            st.metric("Events Covered", len(wittw.get_available_events()))
        with col3:
            years = wittw.get_available_years()
            st.metric("Years", f"{min(years) if years else 'N/A'}-{max(years) if years else 'N/A'}")

        # Data quality indicator
        with st.expander("Data Status", expanded=False):
            st.caption(f"Columns: {available_cols}")
            if has_venue:
                st.success("Competition/venue data available - filtering enabled")
            else:
                st.warning("No venue column - competition filtering will show all data. The scraped data may be ranking-only format.")

        st.markdown("---")

        # Competition Level, Gender and Year selection
        col1, col2, col3 = st.columns(3)
        with col1:
            comp_types = wittw.get_competition_types()
            wittw_comp = st.selectbox("Competition Level", comp_types, index=0, key="wittw_comp")
        with col2:
            wittw_gender = st.selectbox("Select Gender", ['men', 'women'], index=0, key="wittw_gender")
        with col3:
            year_options = ['All Years'] + [str(y) for y in wittw.get_available_years()]
            wittw_year = st.selectbox("Select Year", year_options, index=0, key="wittw_year")

        selected_year = None if wittw_year == 'All Years' else int(wittw_year)

        # Filter by competition type
        if wittw_comp != 'All':
            filtered_data = wittw.filter_by_competition(wittw_comp)
            st.caption(f"Filtered to {wittw_comp}: {len(filtered_data):,} records")

        # Generate What It Takes to Win Report
        st.subheader(f"Medal Standards - {wittw_gender.title()} ({wittw_year}) - {wittw_comp}")

        report = wittw.generate_what_it_takes_report(wittw_gender, selected_year, wittw_comp)

        if len(report) > 0:
            # Event category filter
            events = sorted(report['Event'].unique())
            categories = {
                'All': events,
                'Sprints': [e for e in events if any(x in e.lower() for x in ['100', '200', '400']) and 'hurdle' not in e.lower()],
                'Middle/Long Distance': [e for e in events if any(x in e.lower() for x in ['800', '1500', '3000', '5000', '10000', 'marathon'])],
                'Hurdles': [e for e in events if 'hurdle' in e.lower()],
                'Jumps': [e for e in events if any(x in e.lower() for x in ['jump', 'vault'])],
                'Throws': [e for e in events if any(x in e.lower() for x in ['shot', 'discus', 'javelin', 'hammer'])],
                'Combined': [e for e in events if any(x in e.lower() for x in ['decathlon', 'heptathlon'])]
            }

            category_select = st.selectbox("Event Category", list(categories.keys()), key="wittw_category")
            filtered_events = categories[category_select]

            filtered_report = report[report['Event'].isin(filtered_events)]

            # Display styled table with Top 20 Average
            display_cols = ['Event', 'Gold Standard', 'Silver Standard', 'Bronze Standard',
                           'Final Standard (8th)', 'Top 8 Average', 'Top 20 Average', 'Sample Size']
            # Only include columns that exist
            display_cols = [c for c in display_cols if c in filtered_report.columns]

            st.dataframe(
                filtered_report[display_cols],
                width='stretch',
                hide_index=True
            )

            # Event-specific analysis
            st.markdown("---")
            st.subheader("Event Deep Dive")

            selected_event = st.selectbox("Select Event for Detailed Analysis",
                                          filtered_events, key="wittw_event_detail")

            if selected_event:
                standards = wittw.get_medal_standards(selected_event, wittw_gender, selected_year)

                if standards['gold']:
                    # Medal standards cards
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #FFD700 0%, #B8860B 100%);
                                    border-radius: 10px; padding: 1rem; text-align: center;">
                            <p style="color: #333; margin: 0; font-size: 0.8rem; font-weight: bold;">GOLD</p>
                            <p style="color: #333; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                                {wittw.format_mark(standards['gold'], selected_event)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #C0C0C0 0%, #A8A8A8 100%);
                                    border-radius: 10px; padding: 1rem; text-align: center;">
                            <p style="color: #333; margin: 0; font-size: 0.8rem; font-weight: bold;">SILVER</p>
                            <p style="color: #333; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                                {wittw.format_mark(standards['silver'], selected_event)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #CD7F32 0%, #A0522D 100%);
                                    border-radius: 10px; padding: 1rem; text-align: center;">
                            <p style="color: #fff; margin: 0; font-size: 0.8rem; font-weight: bold;">BRONZE</p>
                            <p style="color: #fff; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                                {wittw.format_mark(standards['bronze'], selected_event)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
                                    border-radius: 10px; padding: 1rem; text-align: center;">
                            <p style="color: #fff; margin: 0; font-size: 0.8rem; font-weight: bold;">FINAL (8th)</p>
                            <p style="color: #fff; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                                {wittw.format_mark(standards['final_standard'], selected_event)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # === FINALS PERFORMANCE PROGRESSION CHART (1st-8th over time) ===
                    st.markdown("---")
                    st.subheader("Finals Performance Progression (1st-8th)")
                    st.markdown(f"<p style='color: #aaa;'>How performances have evolved across championships for positions 1-8</p>", unsafe_allow_html=True)

                    # Get year-by-year progression data for positions 1-8
                    event_col = 'Event' if 'Event' in wittw.data.columns else 'event'
                    gender_col = 'Gender' if 'Gender' in wittw.data.columns else 'gender'
                    rank_col = 'Rank' if 'Rank' in wittw.data.columns else 'rank'
                    mark_col = 'Mark' if 'Mark' in wittw.data.columns else ('result' if 'result' in wittw.data.columns else 'Result')

                    prog_data = wittw.data.copy()
                    prog_data = prog_data[prog_data[event_col].astype(str).str.contains(selected_event, case=False, na=False)]
                    prog_data = prog_data[prog_data[gender_col].astype(str).str.lower().str.contains(wittw_gender.lower(), na=False)]

                    # Parse marks
                    is_field = wittw.is_field_event(selected_event)
                    if is_field:
                        prog_data['ParsedMark'] = prog_data[mark_col].apply(wittw.parse_distance_to_meters)
                    else:
                        prog_data['ParsedMark'] = prog_data[mark_col].apply(wittw.parse_time_to_seconds)

                    prog_data = prog_data.dropna(subset=['ParsedMark'])

                    if rank_col in prog_data.columns and 'year' in prog_data.columns:
                        # Filter to top 8 positions
                        prog_data = prog_data[prog_data[rank_col] <= 8]

                        if not prog_data.empty:
                            # Place labels
                            place_labels = {1: '🥇 1st', 2: '🥈 2nd', 3: '🥉 3rd',
                                           4: '4th', 5: '5th', 6: '6th', 7: '7th', 8: '8th'}
                            prog_data['Place'] = prog_data[rank_col].map(place_labels)

                            # Color scale
                            place_colors = {
                                '🥇 1st': '#FFD700', '🥈 2nd': '#C0C0C0', '🥉 3rd': '#CD7F32',
                                '4th': '#4CAF50', '5th': '#2196F3', '6th': '#9C27B0',
                                '7th': '#FF5722', '8th': '#607D8B'
                            }

                            # Create progression chart
                            fig_prog = go.Figure()

                            for place in ['🥇 1st', '🥈 2nd', '🥉 3rd', '4th', '5th', '6th', '7th', '8th']:
                                place_data = prog_data[prog_data['Place'] == place].sort_values('year')
                                if not place_data.empty:
                                    fig_prog.add_trace(go.Scatter(
                                        x=place_data['year'],
                                        y=place_data['ParsedMark'],
                                        mode='lines+markers',
                                        name=place,
                                        line=dict(color=place_colors.get(place, '#888'), width=2),
                                        marker=dict(size=8),
                                        hovertemplate=f"<b>{place}</b><br>Year: %{{x}}<br>Mark: %{{y:.2f}}<extra></extra>"
                                    ))

                            # Update layout
                            fig_prog.update_layout(
                                title=f"{selected_event} - Finals Performance Over Time",
                                xaxis_title="Year",
                                yaxis_title="Performance",
                                yaxis=dict(autorange='reversed') if not is_field else dict(),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                height=450,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="center",
                                    x=0.5
                                ),
                                hovermode='x unified'
                            )
                            fig_prog.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                            fig_prog.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

                            st.plotly_chart(fig_prog, width='stretch')

                            # Key insight - Gold trend
                            gold_data = prog_data[prog_data['Place'] == '🥇 1st'].sort_values('year')
                            if len(gold_data) >= 2:
                                first_gold = gold_data.iloc[0]['ParsedMark']
                                last_gold = gold_data.iloc[-1]['ParsedMark']
                                first_year = gold_data.iloc[0]['year']
                                last_year = gold_data.iloc[-1]['year']

                                if is_field:
                                    improvement = last_gold - first_gold
                                    direction = "further/higher" if improvement > 0 else "shorter/lower"
                                else:
                                    improvement = first_gold - last_gold
                                    direction = "faster" if improvement > 0 else "slower"

                                st.info(f"**Gold Trend:** From **{wittw.format_mark(first_gold, selected_event)}** ({int(first_year)}) to **{wittw.format_mark(last_gold, selected_event)}** ({int(last_year)}) — **{abs(improvement):.2f} {direction}**")
                        else:
                            st.info("No progression data available - position data may not be available for all years")
                    else:
                        st.info("Progression chart requires rank and year columns in the data")

                    # Top Athletes Progression - Shows what it really takes to win
                    st.markdown("---")
                    st.subheader("Top Athletes Progression")
                    st.markdown(f"<p style='color: #aaa;'>Year-by-year progression of top athletes in {selected_event}</p>", unsafe_allow_html=True)

                    progression_df = wittw.get_top_athletes_progression(selected_event, wittw_gender, top_n=10)
                    if not progression_df.empty:
                        # Show core columns first
                        display_cols = ['Athlete', 'Country', 'Best', 'Performances', 'Years_Active']
                        # Add year columns
                        year_cols = [c for c in progression_df.columns if c.startswith('Y20')]
                        year_cols = sorted(year_cols, reverse=True)[:5]  # Last 5 years
                        display_cols.extend(year_cols)

                        st.dataframe(
                            progression_df[[c for c in display_cols if c in progression_df.columns]],
                            width='stretch',
                            hide_index=True
                        )
                    else:
                        st.info("No progression data available for this event.")

                    # KSA Athlete Comparison
                    st.markdown("---")
                    st.subheader("Compare KSA Athlete")

                    # Get KSA athletes from Azure or local
                    ksa_athletes = pd.DataFrame()
                    if DATA_CONNECTOR_AVAILABLE:
                        try:
                            ksa_data = load_ksa_rankings_raw_cached()
                            if ksa_data is not None and not ksa_data.empty:
                                # Get best result per athlete per event
                                comp_col = 'competitor' if 'competitor' in ksa_data.columns else 'Athlete'
                                evt_col = 'event' if 'event' in ksa_data.columns else 'Event Type'
                                res_col = 'result' if 'result' in ksa_data.columns else 'Result'

                                if comp_col in ksa_data.columns:
                                    ksa_athletes = ksa_data.groupby([comp_col, evt_col])[res_col].first().reset_index()
                                    ksa_athletes.columns = ['full_name', 'event_name', 'pb_result']
                        except Exception as e:
                            st.warning(f"Could not load KSA data: {e}")

                    # Fall back to SQLite if Azure not available
                    if ksa_athletes.empty:
                        profiles_db = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')
                        if os.path.exists(profiles_db):
                            conn = sqlite3.connect(profiles_db)
                            ksa_athletes = pd.read_sql("""
                                SELECT DISTINCT a.full_name, p.pb_result, p.event_name
                                FROM ksa_athletes a
                                LEFT JOIN athlete_pbs p ON a.athlete_id = p.athlete_id
                                WHERE p.pb_result IS NOT NULL
                            """, conn)
                            conn.close()

                    if not ksa_athletes.empty:
                        # Filter to only athletes who have data for the selected event
                        # Create multiple search patterns for flexible matching
                        event_base = selected_event.replace('-', '').replace('metres', 'm')  # "200m"
                        event_spaced = selected_event.replace('-', ' ')  # "200 metres"
                        event_hyphen = selected_event  # "200-metres"

                        # Match any of these patterns
                        athletes_with_event_data = ksa_athletes[
                            ksa_athletes['event_name'].str.contains(event_base, case=False, na=False) |
                            ksa_athletes['event_name'].str.contains(event_spaced, case=False, na=False) |
                            ksa_athletes['event_name'].str.contains(event_hyphen, case=False, na=False)
                        ]['full_name'].unique().tolist()

                        if athletes_with_event_data:
                            ksa_options = ['Select athlete...'] + sorted(athletes_with_event_data)
                            compare_athlete = st.selectbox(
                                f"Select KSA Athlete to Compare ({len(athletes_with_event_data)} with {selected_event} data)",
                                ksa_options, key="wittw_compare"
                            )

                            if compare_athlete != 'Select athlete...':
                                # Get athlete's PB for event
                                athlete_pbs = ksa_athletes[ksa_athletes['full_name'] == compare_athlete]
                                event_match = athlete_pbs[
                                    athlete_pbs['event_name'].str.contains(event_base, case=False, na=False) |
                                    athlete_pbs['event_name'].str.contains(event_spaced, case=False, na=False) |
                                    athlete_pbs['event_name'].str.contains(event_hyphen, case=False, na=False)
                                ]

                                if not event_match.empty:
                                    pb_str = event_match.iloc[0]['pb_result']
                                    is_field = wittw.is_field_event(selected_event)

                                    if is_field:
                                        athlete_mark = wittw.parse_distance_to_meters(pb_str)
                                    else:
                                        athlete_mark = wittw.parse_time_to_seconds(pb_str)

                                    if athlete_mark:
                                        comparison = wittw.compare_athlete_to_standards(
                                            athlete_mark, selected_event, wittw_gender, selected_year
                                        )

                                        # Position indicator
                                        position_colors = {
                                            'Gold Medal': '#FFD700',
                                            'Silver Medal': '#C0C0C0',
                                            'Bronze Medal': '#CD7F32',
                                            'Finals': TEAL_PRIMARY,
                                            'Outside Finals': GRAY_BLUE
                                        }

                                        pos_color = position_colors.get(comparison['projected_position'], GRAY_BLUE)

                                        st.markdown(f"""
                                        <div style="background: {pos_color}22; border: 2px solid {pos_color};
                                                    border-radius: 10px; padding: 1.5rem; text-align: center;">
                                            <p style="color: #aaa; margin: 0;">Personal Best</p>
                                            <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
                                                {comparison['athlete_mark_formatted']}
                                            </p>
                                            <p style="color: {pos_color}; font-size: 1.2rem; font-weight: bold;">
                                                Projected: {comparison['projected_position']}
                                            </p>
                                            <p style="color: #aaa; margin-top: 0.5rem;">
                                                Gap to Gold: {comparison['gap_to_gold_formatted']}
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                        else:
                            st.info(f"No KSA athletes have data for {selected_event}. Check if the event name matches the data format.")

                    # Year over Year Trends
                    st.markdown("---")
                    st.subheader("Year over Year Trends")

                    trends = wittw.get_year_over_year_trends(selected_event, wittw_gender)

                    if len(trends) > 1:
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=trends['Year'],
                            y=trends['Gold'],
                            mode='lines+markers',
                            name='Gold Standard',
                            line=dict(color='#FFD700', width=3),
                            marker=dict(size=10)
                        ))

                        if 'Top 8 Avg' in trends.columns:
                            fig.add_trace(go.Scatter(
                                x=trends['Year'],
                                y=trends['Top 8 Avg'],
                                mode='lines+markers',
                                name='Top 8 Average',
                                line=dict(color=TEAL_PRIMARY, width=2, dash='dash'),
                                marker=dict(size=8)
                            ))

                        fig.update_layout(
                            title=f"{selected_event} - Performance Trends",
                            xaxis_title="Year",
                            yaxis_title="Mark" if wittw.is_field_event(selected_event) else "Time (seconds)",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            yaxis=dict(autorange='reversed') if not wittw.is_field_event(selected_event) else dict()
                        )

                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Not enough data for trend analysis. Need multiple years of data.")

        else:
            st.warning("No report data available for the selected filters.")
            st.info(f"Data loaded: {len(wittw.data):,} records | Available events: {len(wittw.get_available_events())}")

            # Help diagnose the issue
            venue_col = None
            for col in ['venue', 'Venue', 'Competition', 'competition']:
                if col in wittw.data.columns:
                    venue_col = col
                    break

            if venue_col and wittw_comp != 'All':
                st.caption(f"Tried to filter by '{wittw_comp}' but no matching venues found.")
                st.caption(f"Available columns: {list(wittw.data.columns)}")
            elif not venue_col and wittw_comp != 'All':
                st.caption(f"No venue column available - try selecting 'All' as Competition Level to see all data.")
            else:
                st.caption("Try selecting different filters (Gender, Year) or 'All' for Competition Level.")
    else:
        st.warning("No performance data loaded.")
        if DATA_CONNECTOR_AVAILABLE:
            mode = get_data_mode()
            st.info(f"Data mode: {mode}")
            if mode == "azure":
                st.caption("Azure connection may be slow on first load. Try refreshing the page.")
            else:
                st.caption("Azure not configured. Using local data mode.")
        else:
            st.caption("Data connector not available. Check data_connector.py")

###################################
# Tab 8: AI Analyst (Chatbot)
###################################
with tab8:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">AI Athletics Analyst</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Ask questions about athlete performance, competition standards, rulebooks, and strategic insights.
            RAG-powered search across Azure Blob data and PDF documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for chatbot
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = []
    if 'chatbot_model' not in st.session_state:
        st.session_state.chatbot_model = DEFAULT_MODEL
    # Initialize AI context filter state to prevent selection reset
    if 'ai_athlete_filter' not in st.session_state:
        st.session_state.ai_athlete_filter = "(All Athletes)"
    if 'ai_event_filter' not in st.session_state:
        st.session_state.ai_event_filter = "(All Events)"
    if 'ai_context_prefix' not in st.session_state:
        st.session_state.ai_context_prefix = ""

    # Check for API key
    api_key = get_openrouter_key()

    if not api_key:
        st.warning("OpenRouter API key not configured.")
        st.info("""
        **To enable the AI Analyst:**
        1. Get a free API key from [OpenRouter](https://openrouter.ai/keys)
        2. Add to your `.env` file: `OPENROUTER_API_KEY=sk-or-v1-your-key-here`
        3. Or add to Streamlit secrets for cloud deployment
        """)
    else:
        # Controls row
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            # Model selection
            model_names = list(FREE_MODELS.values())
            model_keys = list(FREE_MODELS.keys())
            current_idx = model_keys.index(st.session_state.chatbot_model) if st.session_state.chatbot_model in model_keys else 0

            selected_model_name = st.selectbox(
                "AI Model",
                model_names,
                index=current_idx,
                key="chatbot_model_select"
            )
            selected_model_key = model_keys[model_names.index(selected_model_name)]
            st.session_state.chatbot_model = selected_model_key

        with col2:
            # Data status
            if DATA_CONNECTOR_AVAILABLE:
                data_mode = get_data_mode()
                st.markdown(f"""
                <div style="background: rgba(0, 113, 103, 0.2); padding: 0.5rem 1rem; border-radius: 4px; margin-top: 1.7rem;">
                    <span style="color: {TEAL_LIGHT};">Data: {data_mode.upper()}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(160, 142, 102, 0.2); padding: 0.5rem 1rem; border-radius: 4px; margin-top: 1.7rem;">
                    <span style="color: {GOLD_ACCENT};">Data: Limited</span>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            # Clear chat and cache buttons
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                if st.button("Clear Chat", key="chatbot_clear"):
                    st.session_state.chatbot_messages = []
                    st.rerun()
            with c_col2:
                cache_stats = _response_cache.stats()
                if cache_stats['valid_entries'] > 0:
                    if st.button(f"Cache ({cache_stats['valid_entries']})", key="clear_cache", help="Clear response cache for fresh answers"):
                        _response_cache.clear()
                        st.rerun()

        # === AI BACKEND SELECTOR ===
        st.markdown("#### AI Backend")

        if 'ai_backend' not in st.session_state:
            st.session_state.ai_backend = "NotebookLM" if NOTEBOOKLM_AVAILABLE else "OpenRouter"

        backend_col1, backend_col2 = st.columns([1, 2])

        with backend_col1:
            backend_options = []
            if NOTEBOOKLM_AVAILABLE:
                backend_options.append("Hybrid")
                backend_options.append("NotebookLM")
            backend_options.append("OpenRouter")

            selected_backend = st.radio(
                "Select AI",
                backend_options,
                index=0,
                key="ai_backend_radio",
                label_visibility="collapsed",
                horizontal=True
            )
            st.session_state.ai_backend = selected_backend

        with backend_col2:
            if selected_backend == "Hybrid":
                st.success("**Hybrid**: Documents + Live data combined (best)")
            elif selected_backend == "NotebookLM":
                st.info("**NotebookLM**: Fast, citation-backed from documents")
            else:
                st.info("**OpenRouter**: LLM with live database context")

        # Set knowledge source for context builder
        selected_knowledge = "Database Only"

        # === QUICK ACTION BUTTONS ===
        st.markdown("#### Quick Analysis")
        qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)

        with qa_col1:
            if st.button("Medal Gap Analysis", key="qa_medal_gap", help="Analyze gaps to Asian Games medal standards"):
                st.session_state['pending_quick_action'] = "Analyze the gap to medal standards for our top KSA athletes at Asian Games 2026. Show who is closest to medals and what improvements are needed."

        with qa_col2:
            if st.button("Top Rivals", key="qa_rivals", help="Identify key competitors in priority events"):
                st.session_state['pending_quick_action'] = "Who are the main Asian rivals for our top KSA athletes in their events? Focus on athletes likely to compete at Asian Games 2026."

        with qa_col3:
            if st.button("Form Trends", key="qa_form", help="Recent performance trends for KSA athletes"):
                st.session_state['pending_quick_action'] = "Analyze recent form trends for our top 5 KSA athletes. Who is improving, stable, or declining based on their last few competitions?"

        with qa_col4:
            if st.button("Qualification Status", key="qa_qualify", help="Track progress toward entry standards"):
                st.session_state['pending_quick_action'] = "What is the qualification status for KSA athletes targeting Olympics and World Championships? Who has achieved entry standards and who is close?"

        st.markdown("---")

        # === CONTEXT FILTER: Select Athlete and Event ===
        with st.expander("Filter Context (Optional)", expanded=False):
            st.caption("Select an athlete and/or event to focus your question. This helps ensure accurate name matching.")

            filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

            # Get available athletes from data (CACHED for speed)
            available_athletes, available_events = get_ai_filter_options()

            with filter_col1:
                # Compute index from session state (preserves selection across reruns)
                athlete_idx = 0
                if st.session_state.ai_athlete_filter in available_athletes:
                    athlete_idx = available_athletes.index(st.session_state.ai_athlete_filter)

                selected_ai_athlete = st.selectbox(
                    "Athlete",
                    available_athletes,
                    index=athlete_idx,
                    key="ai_athlete_filter"
                )

            with filter_col2:
                # Compute index from session state (preserves selection across reruns)
                event_idx = 0
                if st.session_state.ai_event_filter in available_events:
                    event_idx = available_events.index(st.session_state.ai_event_filter)

                selected_ai_event = st.selectbox(
                    "Event",
                    available_events,
                    index=event_idx,
                    key="ai_event_filter"
                )

            # AUTO-APPLY context when athlete or event is selected (no button needed)
            context_prefix = ""
            if selected_ai_athlete != "(All Athletes)":
                context_prefix += f"About {selected_ai_athlete}"
            if selected_ai_event != "(All Events)":
                if context_prefix:
                    context_prefix += f" in {selected_ai_event}"
                else:
                    context_prefix += f"About {selected_ai_event}"

            # Store the context prefix
            st.session_state['ai_context_prefix'] = context_prefix

            with filter_col3:
                if context_prefix:
                    st.success(f"**{context_prefix}**")
                    if st.button("Clear", key="clear_ai_context", width='stretch'):
                        # Reset selections to "(All)"
                        st.session_state['ai_athlete_filter'] = "(All Athletes)"
                        st.session_state['ai_event_filter'] = "(All Events)"
                        st.session_state['ai_context_prefix'] = ""
                        st.rerun()
                else:
                    st.caption("Select athlete/event to focus your question")

        st.markdown("---")

        # Chat messages display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chatbot_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # Show sources if available
                    if msg["role"] == "assistant" and msg.get("context"):
                        with st.expander("View Data Sources"):
                            st.text(msg["context"])

        # Input area using form to prevent tab reset on submit
        st.markdown("<br>", unsafe_allow_html=True)

        # Build dynamic placeholder based on context
        current_context = st.session_state.get('ai_context_prefix', '')
        if current_context:
            placeholder_text = f"Ask about {current_context.replace('About ', '')}..."
        else:
            placeholder_text = "Ask about athletics performance, KSA athletes, qualification standards..."

        # Use form to prevent full page rerun
        with st.form(key="chat_form", clear_on_submit=True):
            input_col, btn_col = st.columns([5, 1])
            with input_col:
                user_input = st.text_input(
                    "Your question",
                    placeholder=placeholder_text,
                    key="chatbot_input",
                    label_visibility="collapsed"
                )
            with btn_col:
                send_clicked = st.form_submit_button("Send", type="primary", use_container_width=True)

        # Handle pending quick action (from quick analysis buttons)
        if 'pending_quick_action' in st.session_state:
            quick_query = st.session_state.pop('pending_quick_action')
            st.session_state.chatbot_messages.append({
                "role": "user",
                "content": quick_query
            })
            try:
                selected_backend = st.session_state.get('ai_backend', 'OpenRouter')

                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    context_used = ""

                    if selected_backend == "Hybrid" and NOTEBOOKLM_AVAILABLE and query_notebook:
                        # Hybrid: NotebookLM + OpenRouter
                        response_placeholder.markdown("_Querying NotebookLM + Live Data..._")
                        nlm_response = query_notebook(quick_query)
                        hybrid_prompt = f"Research context:\n{nlm_response}\n\nQuestion: {quick_query}"
                        client = OpenRouterClient(api_key, st.session_state.chatbot_model)
                        for chunk in client.chat_stream([{"role": "user", "content": hybrid_prompt}], user_query=quick_query, knowledge_source="Database Only"):
                            if chunk["type"] == "content":
                                full_response += chunk["content"]
                                response_placeholder.markdown(full_response + "▌")
                            elif chunk["type"] == "done":
                                full_response = chunk["full_content"]
                                context_used = "Hybrid"
                                response_placeholder.markdown(full_response if full_response else nlm_response)
                            elif chunk["type"] == "error":
                                full_response = nlm_response
                                context_used = "NotebookLM"
                                response_placeholder.markdown(full_response)
                    elif selected_backend == "NotebookLM" and NOTEBOOKLM_AVAILABLE and query_notebook:
                        response_placeholder.markdown("_Querying NotebookLM..._")
                        full_response = query_notebook(quick_query)
                        context_used = "NotebookLM"
                        response_placeholder.markdown(full_response if full_response else "_No response_")
                    else:
                        client = OpenRouterClient(api_key, st.session_state.chatbot_model)
                        api_messages = [{"role": m["role"], "content": m["content"]}
                                       for m in st.session_state.chatbot_messages]

                        ks = st.session_state.get('ai_knowledge_source', 'Database Only')
                        for chunk in client.chat_stream(api_messages, user_query=quick_query, knowledge_source=ks):
                            if chunk["type"] == "content":
                                full_response += chunk["content"]
                                response_placeholder.markdown(full_response + "▌")
                            elif chunk["type"] == "done":
                                full_response = chunk["full_content"]
                                context_used = chunk.get("context_used", "")
                                response_placeholder.markdown(full_response if full_response else "_No response_")
                            elif chunk["type"] == "error":
                                response_placeholder.error(f"Error: {chunk['error']}")
                                full_response = f"Error: {chunk['error']}"

                if full_response:
                    st.session_state.chatbot_messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "context": context_used
                    })
            except Exception as e:
                st.error(f"Quick action error: {str(e)}")

        # Process input
        if send_clicked and user_input:
            # Check for context prefix (from athlete/event filter)
            context_prefix = st.session_state.get('ai_context_prefix', '')
            if context_prefix:
                # Prepend context to query for better matching
                enhanced_query = f"{context_prefix}: {user_input}"
                # DON'T clear the prefix - keep it for follow-up questions
                # st.session_state['ai_context_prefix'] = ""
            else:
                enhanced_query = user_input

            # Add user message (show original to user)
            st.session_state.chatbot_messages.append({
                "role": "user",
                "content": user_input
            })

            # Get response based on selected backend
            try:
                selected_backend = st.session_state.get('ai_backend', 'OpenRouter')

                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    context_used = ""

                    if selected_backend == "Hybrid" and NOTEBOOKLM_AVAILABLE and query_notebook:
                        # HYBRID: Query NotebookLM first, then enhance with OpenRouter
                        response_placeholder.markdown("_Querying NotebookLM + Live Data..._")

                        # Get NotebookLM response (documents + briefings)
                        nlm_response = query_notebook(enhanced_query)

                        # Build hybrid prompt with NotebookLM context
                        hybrid_prompt = f"""Based on this research from our knowledge base:

---
{nlm_response}
---

Now answer the user's question using the above context plus any additional analysis.
Focus on KSA athletes and provide specific, actionable insights.

Question: {enhanced_query}"""

                        # Get enhanced response from OpenRouter
                        client = OpenRouterClient(api_key, st.session_state.chatbot_model)
                        hybrid_messages = [{"role": "user", "content": hybrid_prompt}]

                        for chunk in client.chat_stream(hybrid_messages, user_query=enhanced_query, knowledge_source="Database Only"):
                            if chunk["type"] == "content":
                                full_response += chunk["content"]
                                response_placeholder.markdown(full_response + "▌")
                            elif chunk["type"] == "done":
                                full_response = chunk["full_content"]
                                context_used = "Hybrid (NotebookLM + Database)"
                                response_placeholder.markdown(full_response if full_response else "_No response_")
                            elif chunk["type"] == "error":
                                # Fallback to just NotebookLM response
                                full_response = nlm_response
                                context_used = "NotebookLM (fallback)"
                                response_placeholder.markdown(full_response)

                    elif selected_backend == "NotebookLM" and NOTEBOOKLM_AVAILABLE and query_notebook:
                        # Use NotebookLM only
                        response_placeholder.markdown("_Querying NotebookLM..._")
                        full_response = query_notebook(enhanced_query)
                        context_used = "NotebookLM"
                        if full_response:
                            response_placeholder.markdown(full_response)
                        else:
                            response_placeholder.warning("No response from NotebookLM.")
                            full_response = "_No response from NotebookLM_"
                    else:
                        # Use OpenRouter only
                        client = OpenRouterClient(api_key, st.session_state.chatbot_model)
                        api_messages = [{"role": m["role"], "content": m["content"]}
                                       for m in st.session_state.chatbot_messages]

                        got_response = False
                        ks = st.session_state.get('ai_knowledge_source', 'Database Only')
                        for chunk in client.chat_stream(api_messages, user_query=enhanced_query, knowledge_source=ks):
                            if chunk["type"] == "content":
                                full_response += chunk["content"]
                                response_placeholder.markdown(full_response + "▌")
                                got_response = True
                            elif chunk["type"] == "done":
                                full_response = chunk["full_content"]
                                context_used = chunk.get("context_used", "")
                                if full_response:
                                    response_placeholder.markdown(full_response)
                                else:
                                    response_placeholder.warning("No response generated.")
                                    full_response = "_No response generated_"
                                got_response = True
                            elif chunk["type"] == "error":
                                response_placeholder.error(f"Error: {chunk['error']}")
                                full_response = f"Error: {chunk['error']}"
                                got_response = True

                        if not got_response:
                            response_placeholder.warning("No response received.")
                            full_response = "_No response received_"

                # Add assistant response to history
                if full_response:
                    st.session_state.chatbot_messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "context": context_used
                    })

            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                if st.session_state.chatbot_messages and st.session_state.chatbot_messages[-1]["role"] == "user":
                    st.session_state.chatbot_messages.pop()

        # Example questions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #888; font-size: 0.85rem;'>Example questions:</p>", unsafe_allow_html=True)

        example_cols = st.columns(3)
        example_questions = [
            "How are KSA sprinters performing in 2024?",
            "What time is needed to qualify for World Championships in 100m?",
            "Compare our 400m athletes to qualification standards"
        ]

        for i, (col, question) in enumerate(zip(example_cols, example_questions)):
            with col:
                if st.button(question, key=f"example_{i}", width='stretch'):
                    st.session_state.chatbot_messages.append({
                        "role": "user",
                        "content": question
                    })
                    # Trigger streaming response
                    try:
                        client = OpenRouterClient(api_key, st.session_state.chatbot_model)
                        api_messages = [{"role": m["role"], "content": m["content"]}
                                       for m in st.session_state.chatbot_messages]

                        # Stream the response
                        with st.chat_message("assistant"):
                            response_placeholder = st.empty()
                            full_response = ""
                            context_used = ""
                            got_response = False

                            # Get knowledge source from session state
                            ks = st.session_state.get('ai_knowledge_source', 'Database Only')
                            for chunk in client.chat_stream(api_messages, user_query=question, knowledge_source=ks):
                                if chunk["type"] == "content":
                                    full_response += chunk["content"]
                                    response_placeholder.markdown(full_response + "▌")
                                    got_response = True
                                elif chunk["type"] == "done":
                                    full_response = chunk["full_content"]
                                    context_used = chunk.get("context_used", "")
                                    if full_response:
                                        response_placeholder.markdown(full_response)
                                    else:
                                        response_placeholder.warning("No response generated. Please try rephrasing your question.")
                                        full_response = "_No response generated_"
                                    got_response = True
                                elif chunk["type"] == "error":
                                    response_placeholder.error(f"Error: {chunk['error']}")
                                    full_response = f"Error: {chunk['error']}"
                                    got_response = True

                            if not got_response:
                                response_placeholder.warning("No response received from AI. Please check your connection and try again.")
                                full_response = "_No response received_"

                        if full_response:
                            st.session_state.chatbot_messages.append({
                                "role": "assistant",
                                "content": full_response,
                                "context": context_used
                            })
                        # Response is already displayed, no need to rerun
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        if st.session_state.chatbot_messages and st.session_state.chatbot_messages[-1]["role"] == "user":
                            st.session_state.chatbot_messages.pop()

###################################
# Tab 9: Coach View
###################################
with tab9:
    if COACH_VIEW_AVAILABLE:
        # Load master rankings data for Coach View
        try:
            if DATA_CONNECTOR_AVAILABLE:
                coach_df = load_full_rankings_cached()
                if coach_df is not None and not coach_df.empty:
                    render_coach_view(coach_df)
                else:
                    st.warning("No data available for Coach View. Please ensure data is loaded.")
            else:
                st.error("Data connector not available. Coach View requires the data_connector module.")
        except Exception as e:
            st.error(f"Error loading Coach View: {str(e)}")
            st.info("Coach View requires the master rankings data to be available.")
    else:
        st.warning("Coach View module not available.")
        st.info("""
        The Coach View module provides:
        - **Competition Prep Hub** - Prepare athletes for upcoming championships
        - **Athlete Report Cards** - Comprehensive performance analysis with projections
        - **Competitor Watch** - Monitor rivals and competitive landscape

        To enable, ensure `coach_view.py` and its dependencies are installed.
        """)

###################################
# Tab 10: Project East 2026
###################################

# Helper functions for Project East - defined at module level to avoid Streamlit tokenization issues
def _parse_project_east_result(result_str, event_name):
    """Parse result to numeric (seconds for track, meters for field)."""
    if pd.isna(result_str):
        return None
    result_str = str(result_str).strip()
    try:
        if ':' in result_str:
            parts = result_str.split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        return float(result_str.replace('m', '').replace('s', '').strip())
    except:
        return None

# Name aliases for World Athletics database variations
_PROJECT_EAST_NAME_ALIASES = {
    'Mohammed Al Atafi': ['atafi', 'al-atafi', 'al atafi', 'mohammed atafi', 'abdulaziz atafi'],
    'Hussain Al Hizam': ['hizam', 'al-hizam', 'al hizam', 'hussain hizam'],
    'Mohamed Tolo': ['tolo', 'mohamed tolo', 'mohammed tolo'],
    'Yasser Bakheet': ['bakheet', 'yasser bakheet', 'bakhit'],
    'Abdelati Bizimana': ['bizimana', 'abdelati bizimana'],
    'Muaz Al Dubaisi': ['dubaisi', 'al-dubaisi', 'al dubaisi', 'muaz dubaisi'],
    'Yaqoub Al Muawi': ['muawi', 'al-muawi', 'al muawi', 'yaqoub muawi', 'mowawi'],
    'Nasser Mohammed': ['nasser', 'nasser mohammed', 'nasser mohamed']
}

def _match_project_east_athlete(x, name_parts, aliases):
    """Check if database name matches athlete name parts or aliases."""
    if pd.isna(x):
        return False
    x_lower = str(x).lower()
    # Check if all name parts are in the database name
    if all(part in x_lower for part in name_parts):
        return True
    # Check aliases
    for alias in aliases:
        if alias in x_lower:
            return True
    return False

def get_project_east_live_data():
    """
    Fetch live performance data for Project East athletes from master.parquet.
    Updates PB/SB dynamically while preserving strategic metadata.
    """
    if not DATA_CONNECTOR_AVAILABLE:
        return PROJECT_EAST_ATHLETES

    try:
        # Get KSA rankings data
        ksa_data = load_ksa_rankings_raw_cached()
        if ksa_data is None or ksa_data.empty:
            return PROJECT_EAST_ATHLETES

        # Determine column names
        athlete_col = next((c for c in ['competitor', 'Competitor', 'full_name', 'name'] if c in ksa_data.columns), None)
        event_col = next((c for c in ['event', 'Event', 'Event Type'] if c in ksa_data.columns), None)
        result_col = next((c for c in ['result', 'Result', 'result_numeric'] if c in ksa_data.columns), None)
        date_col = next((c for c in ['date', 'Date', 'competition_date'] if c in ksa_data.columns), None)
        venue_col = next((c for c in ['venue', 'Venue', 'competition'] if c in ksa_data.columns), None)

        if not all([athlete_col, event_col, result_col]):
            return PROJECT_EAST_ATHLETES

        # Event name mapping for matching
        event_map = {
            '200m': ['200m', '200-metres', '200 metres', '200 meters'],
            'Pole Vault': ['pole-vault', 'pole vault'],
            'Shot Put': ['shot-put', 'shot put'],
            'Triple Jump': ['triple-jump', 'triple jump'],
            '800m': ['800m', '800-metres', '800 metres'],
            'Hammer Throw': ['hammer-throw', 'hammer throw'],
            '400m Hurdles': ['400m-hurdles', '400-metres-hurdles', '400 metres hurdles', '400m hurdles'],
            '100m': ['100m', '100-metres', '100 metres'],
            '4x100m Relay': ['4x100m', '4x100-metres-relay', '4x100m relay', '4 x 100 metres relay']
        }

        updated_athletes = []
        current_year = 2026  # Season year for SB

        for athlete_template in PROJECT_EAST_ATHLETES:
            athlete = athlete_template.copy()

            # Skip relay for now (needs separate handling)
            if 'Relay' in athlete['event']:
                # Try to find relay results
                relay_variants = event_map.get(athlete['event'], [athlete['event'].lower()])
                relay_mask = ksa_data[event_col].str.lower().isin([v.lower() for v in relay_variants])
                relay_data = ksa_data[relay_mask]

                if not relay_data.empty:
                    results = relay_data[result_col].apply(lambda x: _parse_project_east_result(x, athlete['event'])).dropna()
                    if not results.empty:
                        athlete['pb'] = results.min()
                        # Get SB (current year results)
                        if date_col:
                            try:
                                relay_data_copy = relay_data.copy()
                                relay_data_copy['_parsed'] = relay_data_copy[result_col].apply(lambda x: _parse_project_east_result(x, athlete['event']))
                                relay_data_copy['_year'] = pd.to_datetime(relay_data_copy[date_col], errors='coerce').dt.year
                                current_season = relay_data_copy[relay_data_copy['_year'] >= current_year]
                                if not current_season.empty:
                                    athlete['sb'] = current_season['_parsed'].dropna().min()
                            except:
                                athlete['sb'] = athlete['pb']

                updated_athletes.append(athlete)
                continue

            # Find athlete in data using module-level helpers (avoids tokenization issues)
            name_parts = athlete['name'].lower().split()
            aliases = _PROJECT_EAST_NAME_ALIASES.get(athlete['name'], [])
            athlete_mask = ksa_data[athlete_col].apply(lambda x: _match_project_east_athlete(x, name_parts, aliases))

            # Filter by event
            event_variants = event_map.get(athlete['event'], [athlete['event'].lower()])
            event_mask = ksa_data[event_col].str.lower().isin([v.lower() for v in event_variants])

            athlete_event_data = ksa_data[athlete_mask & event_mask]

            if not athlete_event_data.empty:
                # Parse all results
                is_field = is_field_event(athlete['event'])
                results = athlete_event_data[result_col].apply(lambda x: _parse_project_east_result(x, athlete['event'])).dropna()

                if not results.empty:
                    # PB: best result (max for field, min for track)
                    athlete['pb'] = results.max() if is_field else results.min()

                    # SB: best result from current season
                    if date_col:
                        try:
                            athlete_event_data = athlete_event_data.copy()
                            athlete_event_data['_parsed'] = athlete_event_data[result_col].apply(lambda x: _parse_project_east_result(x, athlete['event']))
                            athlete_event_data['_year'] = pd.to_datetime(athlete_event_data[date_col], errors='coerce').dt.year
                            current_season = athlete_event_data[athlete_event_data['_year'] >= current_year]
                            if not current_season.empty and not current_season['_parsed'].dropna().empty:
                                athlete['sb'] = current_season['_parsed'].max() if is_field else current_season['_parsed'].min()
                            else:
                                athlete['sb'] = athlete['pb']
                        except:
                            athlete['sb'] = athlete['pb']
                    else:
                        athlete['sb'] = athlete['pb']

                    # Get recent competition results for profile
                    if date_col and venue_col:
                        try:
                            recent = athlete_event_data.nlargest(5, date_col) if date_col else athlete_event_data.head(5)
                            athlete['recent_results'] = recent[[date_col, venue_col, result_col]].to_dict('records')
                        except:
                            pass

            updated_athletes.append(athlete)

        return updated_athletes

    except Exception as e:
        st.sidebar.caption(f"Project East data: using cached ({str(e)[:30]})")
        return PROJECT_EAST_ATHLETES

with tab10:
    # Header with Asian Games branding
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="color: white; margin: 0; font-size: 2rem;">Project East 2026</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">Saudi Arabia Athletics Strategy - Asian Games Aichi-Nagoya</p>
            </div>
            <div style="text-align: right;">
                <p style="color: {GOLD_ACCENT}; font-size: 2.5rem; font-weight: bold; margin: 0;">3-5</p>
                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">Medal Target</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Executive Summary - Medal Goals
    st.subheader("Executive Summary")

    medal_cols = st.columns(4)
    with medal_cols[0]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="color: rgba(0,0,0,0.6); margin: 0; font-size: 0.85rem;">Gold Target</p>
            <p style="color: #000; font-size: 2.5rem; font-weight: bold; margin: 0;">{PROJECT_EAST_MEDAL_GOALS['gold']}</p>
        </div>
        """, unsafe_allow_html=True)
    with medal_cols[1]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #C0C0C0 0%, #A8A8A8 100%); padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="color: rgba(0,0,0,0.6); margin: 0; font-size: 0.85rem;">Silver Target</p>
            <p style="color: #000; font-size: 2.5rem; font-weight: bold; margin: 0;">{PROJECT_EAST_MEDAL_GOALS['silver']}</p>
        </div>
        """, unsafe_allow_html=True)
    with medal_cols[2]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #CD7F32 0%, #B8860B 100%); padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Bronze Target</p>
            <p style="color: white; font-size: 2.5rem; font-weight: bold; margin: 0;">{PROJECT_EAST_MEDAL_GOALS['bronze']}</p>
        </div>
        """, unsafe_allow_html=True)
    with medal_cols[3]:
        st.markdown(f"""
        <div style="background: {TEAL_PRIMARY}; padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Total Range</p>
            <p style="color: white; font-size: 2.5rem; font-weight: bold; margin: 0;">{PROJECT_EAST_MEDAL_GOALS['total_target']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Strategic Categories
    st.markdown("---")
    st.subheader("Strategic Framework")

    strat_cols = st.columns(3)
    with strat_cols[0]:
        st.markdown(f"""
        <div style="background: rgba(0,113,103,0.2); border-left: 4px solid {TEAL_PRIMARY}; padding: 1rem; border-radius: 0 8px 8px 0;">
            <h4 style="color: {TEAL_PRIMARY}; margin: 0;">High Probability</h4>
            <p style="color: #ccc; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Medal favorites with proven results</p>
            <ul style="color: white; margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
                {''.join([f'<li>{e}</li>' for e in PROJECT_EAST_MEDAL_GOALS['high_probability']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with strat_cols[1]:
        st.markdown(f"""
        <div style="background: rgba(160,142,102,0.2); border-left: 4px solid {GOLD_ACCENT}; padding: 1rem; border-radius: 0 8px 8px 0;">
            <h4 style="color: {GOLD_ACCENT}; margin: 0;">Medal Contenders</h4>
            <p style="color: #ccc; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Within striking distance of podium</p>
            <ul style="color: white; margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
                {''.join([f'<li>{e}</li>' for e in PROJECT_EAST_MEDAL_GOALS['medal_contenders']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with strat_cols[2]:
        st.markdown(f"""
        <div style="background: rgba(120,144,156,0.2); border-left: 4px solid {GRAY_BLUE}; padding: 1rem; border-radius: 0 8px 8px 0;">
            <h4 style="color: {GRAY_BLUE}; margin: 0;">Development Track</h4>
            <p style="color: #ccc; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Building towards 2030</p>
            <ul style="color: white; margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
                {''.join([f'<li>{e}</li>' for e in PROJECT_EAST_MEDAL_GOALS['development']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Medal Portfolio Table
    st.markdown("---")
    st.subheader("Medal Portfolio - 9 Elite Athletes")

    # Fetch live data for athletes (already cached at function level)
    project_east_live = get_project_east_live_data()

    # Show data status
    data_status_col1, data_status_col2 = st.columns([3, 1])
    with data_status_col2:
        if DATA_CONNECTOR_AVAILABLE:
            st.caption(f"Data: Live from rankings | Last refresh: {pd.Timestamp.now().strftime('%H:%M')}")
        else:
            st.caption("Data: Static (database unavailable)")

    # Create portfolio dataframe
    portfolio_data = []
    for athlete in project_east_live:
        is_800m = athlete['event'] == '800m'

        # Format times for 800m
        if is_800m:
            pb_display = f"{int(athlete['pb']//60)}:{athlete['pb']%60:05.2f}"
            sb_display = f"{int(athlete['sb']//60)}:{athlete['sb']%60:05.2f}"
            medal_std_display = f"{int(athlete['medal_standard']//60)}:{athlete['medal_standard']%60:05.2f}"
        else:
            pb_display = f"{athlete['pb']:.2f}"
            sb_display = f"{athlete['sb']:.2f}"
            medal_std_display = f"{athlete['medal_standard']:.2f}"

        # Calculate gap to medal standard
        if is_field_event(athlete['event']):
            gap = athlete['medal_standard'] - athlete['pb']
            gap_pct = (gap / athlete['medal_standard']) * 100
            qualified = athlete['pb'] >= athlete['medal_standard']
        else:
            gap = athlete['pb'] - athlete['medal_standard']
            gap_pct = (gap / athlete['medal_standard']) * 100
            qualified = athlete['pb'] <= athlete['medal_standard']

        # Status badge
        if athlete['status'] == 'medal_favorite':
            status_badge = '🥇 Favorite'
            status_color = TEAL_PRIMARY
        elif athlete['status'] == 'medal_contender':
            status_badge = '🎯 Contender'
            status_color = GOLD_ACCENT
        else:
            status_badge = '📈 Development'
            status_color = GRAY_BLUE

        portfolio_data.append({
            'Athlete': athlete['name'],
            'Event': athlete['event'],
            'PB': pb_display,
            'SB': sb_display,
            'Medal Std': medal_std_display,
            'Gap %': f"{abs(gap_pct):.1f}%",
            'Status': status_badge,
            'Trajectory': athlete['trajectory'].split(' - ')[0]
        })

    portfolio_df = pd.DataFrame(portfolio_data)

    # Style the dataframe
    def highlight_status(row):
        if '🥇' in row['Status']:
            return [f'background-color: rgba(0,113,103,0.3)'] * len(row)
        elif '🎯' in row['Status']:
            return [f'background-color: rgba(160,142,102,0.2)'] * len(row)
        return [''] * len(row)

    st.dataframe(
        portfolio_df.style.apply(highlight_status, axis=1),
        width='stretch',
        hide_index=True,
        height=400
    )

    # Individual Athlete Profiles
    st.markdown("---")
    st.subheader("Individual Athlete Profiles")

    # Athlete selector
    athlete_names = [a['name'] for a in project_east_live]
    selected_athlete = st.selectbox("Select Athlete", athlete_names, key="project_east_athlete")

    # Get selected athlete data (using live data)
    athlete_data = next((a for a in project_east_live if a['name'] == selected_athlete), None)

    if athlete_data:
        prof_cols = st.columns([2, 1])

        with prof_cols[0]:
            # Profile card
            status_color = TEAL_PRIMARY if athlete_data['status'] == 'medal_favorite' else (GOLD_ACCENT if athlete_data['status'] == 'medal_contender' else GRAY_BLUE)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.3) 100%);
                        border-left: 5px solid {status_color}; padding: 1.5rem; border-radius: 0 12px 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <h2 style="color: white; margin: 0;">{athlete_data['name']}</h2>
                        <p style="color: {GOLD_ACCENT}; font-size: 1.2rem; margin: 0.25rem 0;">{athlete_data['event']}</p>
                        <p style="color: #aaa; margin: 0.5rem 0 0 0;">Age: {athlete_data['age'] if athlete_data['age'] else 'N/A'}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.85rem;">Personal Best</p>
                        <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0;">
                            {f"{int(athlete_data['pb']//60)}:{athlete_data['pb']%60:05.2f}" if athlete_data['event'] == '800m' else f"{athlete_data['pb']:.2f}"}
                        </p>
                    </div>
                </div>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                    <p style="color: #ccc; margin: 0;"><strong>Trajectory:</strong> {athlete_data['trajectory']}</p>
                    <p style="color: #aaa; margin: 0.5rem 0 0 0; font-style: italic;">{athlete_data['notes']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with prof_cols[1]:
            # Performance metrics
            is_800m = athlete_data['event'] == '800m'
            is_field = is_field_event(athlete_data['event'])

            if is_field:
                gap = athlete_data['medal_standard'] - athlete_data['pb']
                gap_pct = (gap / athlete_data['medal_standard']) * 100
            else:
                gap = athlete_data['pb'] - athlete_data['medal_standard']
                gap_pct = (gap / athlete_data['medal_standard']) * 100

            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                <p style="color: #aaa; margin: 0; font-size: 0.8rem;">Season Best</p>
                <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0;">
                    {f"{int(athlete_data['sb']//60)}:{athlete_data['sb']%60:05.2f}" if is_800m else f"{athlete_data['sb']:.2f}"}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                <p style="color: #aaa; margin: 0; font-size: 0.8rem;">Medal Standard</p>
                <p style="color: {GOLD_ACCENT}; font-size: 1.5rem; font-weight: bold; margin: 0;">
                    {f"{int(athlete_data['medal_standard']//60)}:{athlete_data['medal_standard']%60:05.2f}" if is_800m else f"{athlete_data['medal_standard']:.2f}"}
                </p>
            </div>
            """, unsafe_allow_html=True)

            gap_color = TEAL_PRIMARY if gap_pct <= 0 else (GOLD_ACCENT if gap_pct <= 3 else '#dc3545')
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px;">
                <p style="color: #aaa; margin: 0; font-size: 0.8rem;">Gap to Medal</p>
                <p style="color: {gap_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">
                    {abs(gap_pct):.1f}% {'✓' if gap_pct <= 0 else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Performance progression chart
        st.markdown("#### Performance Progression")

        # Create mock progression data (in a real implementation, pull from database)
        # Using the data we have: Asian Games 2023 result, current SB, current PB
        is_800m = athlete_data['event'] == '800m'

        progression_data = {
            'Period': ['Asian Games 2023', '2024 SB', '2024 PB', 'Medal Standard'],
            'Performance': [
                athlete_data['asian_games_2023'],
                athlete_data['sb'],
                athlete_data['pb'],
                athlete_data['medal_standard']
            ]
        }

        fig_prog = go.Figure()

        # Actual performances
        fig_prog.add_trace(go.Scatter(
            x=progression_data['Period'][:3],
            y=progression_data['Performance'][:3],
            mode='lines+markers',
            name='Actual',
            line=dict(color=TEAL_PRIMARY, width=3),
            marker=dict(size=12, color=TEAL_PRIMARY)
        ))

        # Medal standard line
        fig_prog.add_hline(
            y=athlete_data['medal_standard'],
            line_dash="dash",
            line_color=GOLD_ACCENT,
            annotation_text="Medal Standard",
            annotation_position="right"
        )

        # PB marker
        fig_prog.add_trace(go.Scatter(
            x=['2024 PB'],
            y=[athlete_data['pb']],
            mode='markers',
            name='Personal Best',
            marker=dict(symbol='star', size=18, color=GOLD_ACCENT)
        ))

        # Invert y-axis for track events (lower is better)
        if not is_field:
            fig_prog.update_yaxes(autorange='reversed')

        fig_prog.update_layout(
            title=f"{athlete_data['name']} - {athlete_data['event']} Progression",
            xaxis_title="",
            yaxis_title="Performance" + (" (seconds)" if is_800m or athlete_data['event'] in ['100m', '200m', '400m', '400m Hurdles'] else " (m)"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350,
            showlegend=True,
            legend=dict(orientation='h', y=-0.15)
        )
        fig_prog.update_xaxes(showgrid=False)
        fig_prog.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

        st.plotly_chart(fig_prog, width='stretch')

        # Show recent competition results if available
        if 'recent_results' in athlete_data and athlete_data['recent_results']:
            st.markdown("#### Recent Competition Results")
            recent_df = pd.DataFrame(athlete_data['recent_results'])
            # Rename columns for display
            col_rename = {'date': 'Date', 'venue': 'Competition', 'result': 'Result'}
            recent_df = recent_df.rename(columns={k: v for k, v in col_rename.items() if k in recent_df.columns})
            st.dataframe(recent_df, width='stretch', hide_index=True)

    # 4x100m Relay Speed Benchmark Section
    st.markdown("---")
    st.subheader("4x100m Relay Speed Benchmark")

    # Relay athlete speeds (mock data based on 100m times)
    relay_data = {
        'Leg': ['Leg 1 (Start)', 'Leg 2', 'Leg 3', 'Leg 4 (Anchor)'],
        'Athlete': ['Nasser Mohammed', 'Mohammed Al Atafi', 'Athlete C', 'Athlete D'],
        '100m PB': [10.18, 10.45, 10.52, 10.35],
        'Flying Speed': [9.80, 10.05, 10.12, 9.95],  # Estimated flying 100m
        'Split Target': [10.20, 9.50, 9.50, 9.30]  # Target relay splits
    }

    relay_df = pd.DataFrame(relay_data)

    relay_cols = st.columns([1, 1])

    with relay_cols[0]:
        st.markdown("**Leg Analysis**")
        st.dataframe(relay_df, width='stretch', hide_index=True)

        st.markdown(f"""
        <div style="background: rgba(0,113,103,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <p style="color: white; margin: 0;"><strong>Current Best:</strong> 38.88</p>
            <p style="color: {GOLD_ACCENT}; margin: 0.25rem 0 0 0;"><strong>Target:</strong> 38.50 (Medal Standard)</p>
            <p style="color: #aaa; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Gap: 0.38s - Achievable with clean exchanges
            </p>
        </div>
        """, unsafe_allow_html=True)

    with relay_cols[1]:
        # Relay speed comparison chart
        fig_relay = go.Figure()

        fig_relay.add_trace(go.Bar(
            name='100m PB',
            x=relay_data['Leg'],
            y=relay_data['100m PB'],
            marker_color=GRAY_BLUE,
            text=relay_data['100m PB'],
            textposition='outside'
        ))

        fig_relay.add_trace(go.Bar(
            name='Target Split',
            x=relay_data['Leg'],
            y=relay_data['Split Target'],
            marker_color=TEAL_PRIMARY,
            text=relay_data['Split Target'],
            textposition='outside'
        ))

        fig_relay.update_layout(
            title="100m PB vs Target Relay Split",
            barmode='group',
            xaxis_title="",
            yaxis_title="Time (seconds)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350,
            yaxis=dict(range=[8.5, 11]),
            legend=dict(orientation='h', y=-0.15)
        )
        fig_relay.update_xaxes(showgrid=False)
        fig_relay.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

        st.plotly_chart(fig_relay, width='stretch')

    # Timeline
    st.markdown("---")
    st.subheader("Road to Aichi-Nagoya 2026")

    timeline_cols = st.columns(4)

    for i, (phase_key, phase_data) in enumerate(PROJECT_EAST_TIMELINE.items()):
        with timeline_cols[i]:
            is_current = phase_key == 'phase_1'  # Assume we're in phase 1
            bg_color = TEAL_PRIMARY if is_current else 'rgba(0,0,0,0.3)'

            st.markdown(f"""
            <div style="background: {bg_color}; padding: 1rem; border-radius: 8px; text-align: center; height: 150px;">
                <p style="color: {GOLD_ACCENT if is_current else '#aaa'}; margin: 0; font-size: 0.8rem;">{phase_data['dates']}</p>
                <h4 style="color: white; margin: 0.5rem 0;">{phase_data['name']}</h4>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.85rem;">
                    {phase_data.get('focus', phase_data.get('location', ''))}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Key Performance Indicators
    st.markdown("---")
    st.subheader("Success Metrics & KPIs")

    kpi_cols = st.columns(4)

    with kpi_cols[0]:
        # Count athletes at/above medal standard (using live data)
        qualified_count = sum(1 for a in project_east_live
                            if (is_field_event(a['event']) and a['pb'] >= a['medal_standard']) or
                               (not is_field_event(a['event']) and a['pb'] <= a['medal_standard']))
        st.metric("Athletes at Medal Standard", f"{qualified_count}/9", delta=None)

    with kpi_cols[1]:
        # Average gap to medal standard (using live data)
        gaps = []
        for a in project_east_live:
            if is_field_event(a['event']):
                gap_pct = ((a['medal_standard'] - a['pb']) / a['medal_standard']) * 100
            else:
                gap_pct = ((a['pb'] - a['medal_standard']) / a['medal_standard']) * 100
            gaps.append(gap_pct)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        st.metric("Avg Gap to Medal Std", f"{avg_gap:.1f}%", delta=None)

    with kpi_cols[2]:
        # Medal favorites count (using live data)
        favorites = sum(1 for a in project_east_live if a['status'] == 'medal_favorite')
        st.metric("Medal Favorites", favorites, delta=None)

    with kpi_cols[3]:
        # Contenders count (using live data)
        contenders = sum(1 for a in project_east_live if a['status'] == 'medal_contender')
        st.metric("Medal Contenders", contenders, delta=None)

###################################
# Tab 11: Race Intelligence
###################################
with tab11:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
         padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid {GOLD_ACCENT};">
        <h2 style="color: white; margin: 0;">🏁 Race Intelligence</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Build race previews, analyze competitor form, and calculate advancement probabilities
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not RACE_INTELLIGENCE_AVAILABLE:
        st.warning("Race Intelligence module not available. Check race_intelligence.py")
    else:
        # Event and Championship selection
        ri_col1, ri_col2, ri_col3 = st.columns(3)

        with ri_col1:
            ri_event = st.selectbox("Event", ['100m', '200m', '400m', '800m', '1500m'], key='ri_event')
        with ri_col2:
            ri_gender = st.selectbox("Gender", ['Men', 'Women'], key='ri_gender')
        with ri_col3:
            ri_championship = st.selectbox("Championship", ['Asian Games', 'World Championships', 'Olympics'], key='ri_champ')

        st.markdown("---")

        # Two sections
        form_tab, preview_tab = st.tabs(["📊 Form Rankings", "🎯 Race Preview"])

        with form_tab:
            st.subheader("Current Form Rankings")
            st.caption(f"Athletes ranked by current form, not just PB - {ri_event} {ri_gender}")

            competitors = get_competitor_form_cards(ri_event, ri_gender, limit=20)

            if competitors:
                # Create DataFrame for display
                form_data = []
                for i, c in enumerate(competitors, 1):
                    form_data.append({
                        'Rank': i,
                        'Athlete': c['athlete_name'],
                        'NAT': c['country_code'],
                        'Form': f"{c['form_score']:.0f} {c['form_icon']}",
                        'Avg (5)': f"{c['avg_last_5']:.2f}",
                        'PB': f"{c['pb']:.2f}" if c['pb'] else '-',
                        'Last': f"{str(c['last_comp'].get('date', ''))[:10]} ({c['last_comp']['days_ago']}d)" if c.get('last_comp') else '-'
                    })

                form_df = pd.DataFrame(form_data)

                # Highlight KSA athletes
                def highlight_ksa(row):
                    if row['NAT'] == 'KSA':
                        return [f'background-color: rgba(0, 84, 48, 0.2)'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    form_df.style.apply(highlight_ksa, axis=1),
                    width='stretch',
                    hide_index=True,
                    height=500
                )
            else:
                st.info("No competitor data available for this event")

        with preview_tab:
            st.subheader("Race Preview Builder")
            st.caption("Select athletes and see round-by-round advancement probabilities")

            # Get available athletes
            if competitors:
                athlete_options = [f"{c['athlete_name']} ({c['country_code']})" for c in competitors[:15]]

                selected_athletes = st.multiselect(
                    "Select Athletes (max 8)",
                    athlete_options,
                    default=athlete_options[:4] if len(athlete_options) >= 4 else athlete_options,
                    max_selections=8,
                    key='ri_athletes'
                )

                # Use session state to persist race preview
                if 'ri_show_preview' not in st.session_state:
                    st.session_state.ri_show_preview = False
                if 'ri_preview_data' not in st.session_state:
                    st.session_state.ri_preview_data = None

                if selected_athletes and st.button("Build Race Preview", type="primary", key='ri_build_btn'):
                    st.session_state.ri_show_preview = True

                if st.session_state.ri_show_preview and selected_athletes:
                    # Build athlete data for preview
                    preview_athletes = []
                    for sel in selected_athletes:
                        name = sel.split(' (')[0]
                        country = sel.split('(')[1].replace(')', '') if '(' in sel else ''

                        # Find in competitors
                        for c in competitors:
                            if c['athlete_name'] == name:
                                preview_athletes.append({
                                    'name': name,
                                    'country': country,
                                    'form_avg': c['avg_last_5']
                                })
                                break

                    if preview_athletes:
                        preview = build_race_preview(preview_athletes, ri_championship, ri_event)

                        # Display results
                        st.markdown(f"### {ri_championship} - {ri_event}")

                        # Round standards
                        standards = preview.get('standards', {})
                        std_cols = st.columns(6)
                        std_items = list(standards.items())
                        for i, (round_name, std) in enumerate(std_items[:6]):
                            with std_cols[i]:
                                st.metric(round_name.title(), f"{std:.2f}")

                        st.markdown("---")

                        # Per-athlete probabilities
                        for athlete in preview['athletes']:
                            is_ksa = athlete['country'] == 'KSA'
                            bg = 'rgba(0, 84, 48, 0.15)' if is_ksa else 'rgba(0,0,0,0.05)'

                            probs = athlete['probabilities']

                            st.markdown(f"""
                            <div style="background: {bg}; padding: 1rem; margin: 0.5rem 0; border-radius: 8px;">
                                <b>{athlete['name']}</b> ({athlete['country']}) - Form: {athlete['form_avg']:.2f}
                            </div>
                            """, unsafe_allow_html=True)

                            prob_cols = st.columns(6)
                            prob_items = list(probs.items())
                            for i, (round_name, prob_data) in enumerate(prob_items[:6]):
                                with prob_cols[i]:
                                    prob = prob_data['probability']
                                    need = prob_data['need']
                                    color = TEAL_PRIMARY if prob >= 70 else (GOLD_ACCENT if prob >= 40 else '#dc3545')
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 0.5rem; background: {color}; color: white; border-radius: 4px;">
                                        <div style="font-size: 1.2rem; font-weight: bold;">{prob:.0f}%</div>
                                        <div style="font-size: 0.8rem;">{round_name.title()}</div>
                                        <div style="font-size: 0.7rem;">Need: {need:.2f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Gap to bronze
                            gap = athlete['gap_to_bronze']
                            gap_color = TEAL_PRIMARY if gap <= 0 else '#dc3545'
                            st.markdown(f"<p style='color: {gap_color}; font-size: 0.9rem;'>Gap to Bronze: {gap:+.2f}s</p>", unsafe_allow_html=True)
            else:
                st.info("No competitor data available. Try selecting a different event.")

###################################
# Footer
###################################
st.markdown(f"""
    <hr style='margin-top: 30px; border: 1px solid #333;'>
    <div style='text-align: center; color: #666; font-size: 0.85rem; padding: 1rem 0;'>
        Saudi Athletics Dashboard — Created by <strong style="color: {TEAL_PRIMARY};">Luke Gallagher</strong> | Team Saudi
    </div>
""", unsafe_allow_html=True)
