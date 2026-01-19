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

###################################
# Team Saudi Brand Colors
###################################
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'
TEAL_LIGHT = '#009688'
GRAY_BLUE = '#78909C'

###################################
# AI Chatbot Configuration
###################################
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

FREE_MODELS = {
    'google/gemma-3-27b-it:free': 'Gemma 3 27B (Recommended)',
    'google/gemini-2.0-flash-exp:free': 'Gemini 2.0 Flash (Google)',
    'meta-llama/llama-3.3-70b-instruct:free': 'Llama 3.3 70B (Best)',
    'meta-llama/llama-3.2-3b-instruct:free': 'Llama 3.2 3B (Fast)',
    'qwen/qwen2.5-vl-7b-instruct:free': 'Qwen 2.5 VL 7B (Multilingual)',
    'nousresearch/hermes-3-llama-3.1-405b:free': 'Hermes 3 405B (Advanced)',
}
DEFAULT_MODEL = 'google/gemma-3-27b-it:free'

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


class AthleticsContextBuilder:
    """Builds lightweight context for RAG without heavy data loading."""

    # Pre-computed static context about the database
    STATIC_CONTEXT = """
DATABASE SUMMARY:
- Master rankings: 2.3M+ performance records from World Athletics
- KSA athlete profiles: 152 Saudi athletes with PBs and rankings
- Benchmarks: Medal standards for Olympics, World Champs, Asian Games, Arab Champs
- Events covered: All track & field events (sprints, distance, jumps, throws, combined)
- Years: 2015-2025

KEY SAUDI EVENTS (by athlete count):
- 100m, 200m, 400m (sprints)
- Long Jump, Triple Jump, High Jump
- Shot Put, Discus, Javelin
- 800m, 1500m (middle distance)

QUALIFICATION STANDARDS (2024 World Champs examples):
- Men's 100m: 10.00s (entry), sub-10.00 for finals
- Men's 400m: 44.50s (entry)
- Men's Long Jump: 8.15m (entry)
- Women's 100m: 11.07s (entry)

The dashboard contains detailed athlete profiles, head-to-head comparisons,
gap analysis to qualification standards, and historical performance trends.
"""

    def __init__(self):
        self.data_available = DATA_CONNECTOR_AVAILABLE
        self._ksa_cache = None

    def get_ksa_summary(self) -> str:
        """Get cached summary of KSA athletes (lightweight)."""
        if self._ksa_cache:
            return self._ksa_cache

        if not self.data_available:
            return "KSA athlete data available in dashboard tabs."

        try:
            # Use cached function from Streamlit
            df = get_ksa_athletes()
            if df is None or df.empty:
                return "KSA athlete profiles available in Athlete Profiles tab."

            summary_parts = [f"KSA Athletes: {len(df)} profiles in database"]

            if 'discipline' in df.columns:
                events = df['discipline'].value_counts().head(5)
                summary_parts.append(f"Top events: {', '.join([f'{e} ({c})' for e, c in events.items()])}")

            self._ksa_cache = "\n".join(summary_parts)
            return self._ksa_cache
        except Exception:
            return "KSA athlete data available in dashboard tabs."

    def build_context(self, query: str) -> str:
        """Build lightweight context without loading large datasets."""
        context_parts = [self.STATIC_CONTEXT]

        query_lower = query.lower()

        # Add KSA-specific context only for KSA queries
        if any(term in query_lower for term in ['ksa', 'saudi', 'our', 'team']):
            context_parts.append(self.get_ksa_summary())

        # Add event-specific hints
        event_hints = {
            '100m': "100m: World class ~9.80-10.00s, Asian level ~10.00-10.20s",
            '200m': "200m: World class ~19.70-20.00s, Asian level ~20.00-20.50s",
            '400m': "400m: World class ~43.50-44.50s, Asian level ~44.50-45.50s",
            '800m': "800m: World class ~1:43-1:45, Asian level ~1:45-1:47",
            '1500m': "1500m: World class ~3:28-3:32, Asian level ~3:32-3:38",
            'long jump': "Long Jump: World class 8.20-8.50m, Asian level 8.00-8.20m",
            'high jump': "High Jump: World class 2.30-2.40m, Asian level 2.25-2.32m",
            'shot put': "Shot Put: World class 21-23m, Asian level 19-21m",
        }

        for event, hint in event_hints.items():
            if event in query_lower:
                context_parts.append(hint)
                break

        return "\n\n".join(context_parts)


class OpenRouterClient:
    """Client for OpenRouter API using OpenAI SDK."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key
        )
        self.model = model
        # Use singleton context builder - avoids reloading data on each message
        self.context_builder = get_context_builder()

    def chat(self, messages: list, user_query: str = None) -> dict:
        """Send chat completion request."""
        try:
            # Build context if this is a new user message
            context = ""
            if user_query:
                context = self.context_builder.build_context(user_query)

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
                temperature=0.7
            )

            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0
                },
                "context_used": context
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": f"Error: {str(e)}"
            }

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

# Cache TTL: 1 hour (3600 seconds) for Azure data, no TTL for local SQLite
CACHE_TTL = 3600

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
logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height: 60px; margin-right: 1rem;">' if logo_b64 else ''

st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; display: flex; align-items: center;">
    {logo_html}
    <div>
        <h1 style="color: white !important; margin: 0; font-size: 2rem;">Saudi Athletics Dashboard</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">World Rankings, Performance Analysis & Road to Tokyo 2025</p>
    </div>
</div>
""", unsafe_allow_html=True)

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

# NOTE: Large rankings data is loaded lazily in tabs that need it
# This improves startup performance significantly

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

# Load Road to Tokyo and qualification data (lazy load)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_road_to_df():
    return load_road_to_data()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_qual_standards_df():
    return load_qualification_standards()

###################################
# 7) Tabs
###################################
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    'Event Standards & Progression',
    'Athlete Profiles',
    'Combined Rankings',
    'Saudi Athletes Rankings',
    'World Champs Qualification',
    'Major Games Analytics',
    'What It Takes to Win (Live)',
    'AI Analyst',
    'Coach View'
])

###################################
# Tab 1: Event Standards & Progression (NEW)
###################################
with tab1:
    st.header('What It Takes to Win')
    st.markdown(f"""
    <p style='color: #ccc; font-size: 0.95em;'>
    Year-on-year progression of winning performances and finals qualifying marks at <strong style="color: {GOLD_ACCENT};">World Championships</strong>.
    Compare with KSA athlete performances.
    </p>
    """, unsafe_allow_html=True)

    # Event selection
    col1, col2 = st.columns(2)

    with col1:
        available_events = list(HISTORICAL_WINNING_MARKS.keys())
        selected_event = st.selectbox("Select Event", available_events, key="standards_event")

    with col2:
        gender_options = ['men', 'women']
        if selected_event in HISTORICAL_WINNING_MARKS:
            gender_options = list(HISTORICAL_WINNING_MARKS[selected_event].keys())
        selected_gender = st.selectbox("Select Gender", gender_options, key="standards_gender")

    # Get data for selected event
    if selected_event in HISTORICAL_WINNING_MARKS and selected_gender in HISTORICAL_WINNING_MARKS[selected_event]:
        event_data = HISTORICAL_WINNING_MARKS[selected_event][selected_gender]

        # Convert to DataFrame
        years = sorted(event_data.keys())
        marks = [event_data[y] for y in years]
        marks_seconds = [convert_time_to_seconds(m) for m in marks]

        df_progression = pd.DataFrame({
            'Year': years,
            'Winning Mark': marks,
            'Mark (seconds/meters)': marks_seconds
        })

        # Display metrics
        st.subheader(f"{selected_event.upper()} - {selected_gender.capitalize()} Winning Progression")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: {GOLD_ACCENT}; margin: 0; font-size: 0.85rem;">Latest (2023)</p>
                <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0;">{marks[-1]}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: {TEAL_LIGHT}; margin: 0; font-size: 0.85rem;">Best Ever</p>
                <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0;">{min(marks) if marks_seconds[0] else max(marks)}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            improvement = marks_seconds[-1] - marks_seconds[0] if all(marks_seconds) else 0
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #aaa; margin: 0; font-size: 0.85rem;">10yr Change</p>
                <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0;">{improvement:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # Show qualification standard if available
            qual_mark = "N/A"
            qual_standards_df = get_qual_standards_df()
            if qual_standards_df is not None and not qual_standards_df.empty:
                # Handle both old CSV format (Display_Name) and new parquet format (Event)
                event_search = selected_event.replace('-', ' ')
                if 'Display_Name' in qual_standards_df.columns:
                    event_qual = qual_standards_df[qual_standards_df['Display_Name'].str.contains(event_search, case=False, na=False)]
                    if not event_qual.empty:
                        qual_mark = event_qual.iloc[0].get('entry_standard', 'N/A')
                elif 'Event' in qual_standards_df.columns:
                    # Use benchmarks parquet format - match event name
                    event_qual = qual_standards_df[qual_standards_df['Event'].str.contains(event_search, case=False, na=False)]
                    if not event_qual.empty:
                        # Get Gold Standard from benchmarks
                        qual_mark = event_qual.iloc[0].get('Gold Standard', 'N/A')

            st.markdown(f"""
            <div class="standard-card">
                <p style="color: {GOLD_ACCENT}; margin: 0; font-size: 0.85rem;">Gold Standard</p>
                <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{qual_mark}</p>
            </div>
            """, unsafe_allow_html=True)

        # Progression Chart
        fig = go.Figure()

        # Winning marks line
        fig.add_trace(go.Scatter(
            x=df_progression['Year'],
            y=df_progression['Mark (seconds/meters)'],
            mode='lines+markers',
            name='World Championship Winner',
            line=dict(color=GOLD_ACCENT, width=3),
            marker=dict(size=10, symbol='star'),
            text=df_progression['Winning Mark'],
            hovertemplate="<b>%{x}</b><br>Winning Mark: %{text}<extra></extra>"
        ))

        # Add finals qualifying line if available
        if selected_event in FINALS_QUALIFYING_MARKS and selected_gender in FINALS_QUALIFYING_MARKS[selected_event]:
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
            # Match event name
            event_pattern = selected_event.replace('-', ' ').replace('m', ' m')
            ksa_pbs = pbs_df[pbs_df['event_name'].str.lower().str.contains(selected_event.replace('-', '').lower(), na=False)]

            if not ksa_pbs.empty:
                for _, pb in ksa_pbs.iterrows():
                    pb_val = convert_time_to_seconds(pb['pb_result'])
                    if pb_val:
                        # Get athlete name
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
            title=f"{selected_event.upper()} - Historical Winning Progression ({selected_gender.capitalize()})",
            xaxis_title="Year",
            yaxis_title="Mark (seconds/meters)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        st.subheader("Historical Data")
        st.dataframe(df_progression, use_container_width=True, hide_index=True)

    else:
        st.info(f"No historical data available for {selected_event} ({selected_gender})")

###################################
# Tab 2: Athlete Profiles
###################################
with tab2:
    st.header('KSA Athlete Profiles')

    if athletes_df is None or athletes_df.empty:
        st.warning("No athlete profiles found. Run the scraper first: `python scrape_ksa_athlete_profiles_v2.py`")
    else:
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            athlete_names = ["All Athletes"] + sorted(athletes_df['full_name'].dropna().unique().tolist())
            selected_athlete = st.selectbox("Select Athlete", athlete_names, key="profile_athlete")

        with col_filter2:
            gender_options = ["All", "Men", "Women"]
            selected_gender_profile = st.selectbox("Gender", gender_options, key="profile_gender")

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
                        ksa_data = get_ksa_rankings()
                        if ksa_data is not None and not ksa_data.empty:
                            # Match by full name (case-insensitive)
                            athlete_results = ksa_data[ksa_data['competitor'].str.upper() == athlete_name.upper()]
                            # If no exact match, try partial match on last name
                            if athlete_results.empty:
                                last_name = athlete_name.upper().split()[-1]
                                athlete_results = ksa_data[ksa_data['competitor'].str.upper().str.contains(last_name, na=False)]

                            if not athlete_results.empty:
                                # Determine actual primary event from results (most competitions)
                                if 'event' in athlete_results.columns:
                                    event_counts = athlete_results['event'].value_counts()
                                    if not event_counts.empty:
                                        actual_primary_event = event_counts.index[0]
                                        st.markdown(f"<p style='color: {GOLD_ACCENT}; font-size: 0.9rem;'><strong>Main Event:</strong> {actual_primary_event} ({event_counts.iloc[0]} results)</p>", unsafe_allow_html=True)

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

                                    st.dataframe(results_display.head(20), use_container_width=True, hide_index=True)

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
                        st.dataframe(display_pbs, use_container_width=True, hide_index=True)
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
    st.header('Combined Rankings')
    gender = st.selectbox('Select Gender', ['All', 'Men', 'Women'], index=1, key="combined_gender")

    # Lazy load rankings only when this tab is accessed
    with st.spinner("Loading rankings..."):
        if gender == 'Men':
            data = load_men_rankings()
        elif gender == 'Women':
            data = load_women_rankings()
        else:
            men_rankings = load_men_rankings()
            women_rankings = load_women_rankings()
            data = pd.concat([men_rankings, women_rankings])

    if not data.empty and 'Event Type' in data.columns:
        events = sorted(data['Event Type'].dropna().unique())
        selected_event_rank = st.selectbox("Select Event", options=events, key="combined_event_select")
        data = data[data['Event Type'] == selected_event_rank]

        if not data.empty and 'Rank' in data.columns and 'Score' in data.columns:
            min_rank, max_rank = int(data['Rank'].min()), int(data['Rank'].max())
            min_score, max_score = int(data['Score'].min()), int(data['Score'].max())

            selected_rank = st.slider("Select Rank Range", min_rank, max_rank, (min_rank, min(50, max_rank)))

            filtered = data[data['Rank'].between(*selected_rank)]
            filtered = filtered.drop_duplicates()
            st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

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
        st.dataframe(saudi_combined.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("No Saudi athletes found in rankings. Data may still be loading from Azure.")

###################################
# Tab 5: World Championships Qualification
###################################
with tab5:
    st.header('World Championships Qualification')

    # Show data source indicator
    mode = get_data_mode() if DATA_CONNECTOR_AVAILABLE else 'local'

    # Lazy load qualification data
    road_to_df = get_road_to_df()
    qual_standards_df = get_qual_standards_df()

    if road_to_df is not None and not road_to_df.empty:
        # Get KSA athlete names for matching
        ksa_names = []
        if athletes_df is not None and not athletes_df.empty:
            if 'full_name' in athletes_df.columns:
                ksa_names = athletes_df['full_name'].tolist()

        # Find KSA athletes in qualification data (Status column contains athlete names)
        ksa_qualified = pd.DataFrame()
        if ksa_names and 'Status' in road_to_df.columns:
            ksa_qualified = road_to_df[road_to_df['Status'].isin(ksa_names)]

        # KSA Athletes Section
        st.subheader(f"KSA Athletes ({len(ksa_qualified)} entries)")

        if not ksa_qualified.empty:
            st.success(f"Found {len(ksa_qualified)} KSA athlete entries in qualification data")

            # Status column contains the actual athlete names
            # Create display dataframe with only needed columns
            ksa_display = ksa_qualified[['Actual_Event_Name', 'Status', 'Qualification_Status', 'Details']].copy()
            ksa_display.columns = ['Event', 'Athlete', 'Status', 'Details']

            st.dataframe(ksa_display.drop_duplicates(), use_container_width=True, hide_index=True)
        else:
            st.info("No KSA athletes found in current qualification data")

        # Global Data Section
        st.markdown("---")
        st.subheader("All Athletes Qualification Status")
        st.caption(f"Total: {len(road_to_df):,} records from {mode} data source")

        # Filters
        col1, col2 = st.columns(2)

        selected_event_rt = "All Events"
        selected_status = "All"

        with col1:
            if 'Actual_Event_Name' in road_to_df.columns:
                events_rt = sorted(road_to_df['Actual_Event_Name'].dropna().unique())
                selected_event_rt = st.selectbox("Select Event", ["All Events"] + list(events_rt), key="road_to_event")

        with col2:
            if 'Qualification_Status' in road_to_df.columns:
                statuses = sorted(road_to_df['Qualification_Status'].dropna().unique())
                selected_status = st.selectbox("Qualification Status", ["All"] + list(statuses), key="qual_status")

        # Filter data
        filtered_rt = road_to_df.copy()
        if selected_event_rt != "All Events":
            filtered_rt = filtered_rt[filtered_rt['Actual_Event_Name'] == selected_event_rt]
        if selected_status != "All":
            filtered_rt = filtered_rt[filtered_rt['Qualification_Status'] == selected_status]

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", len(filtered_rt))
        with col2:
            qualified = len(filtered_rt[filtered_rt['Qualification_Status'].str.contains('Qualified', na=False)]) if 'Qualification_Status' in filtered_rt.columns else 0
            st.metric("Qualified", qualified)
        with col3:
            events_count = filtered_rt['Actual_Event_Name'].nunique() if 'Actual_Event_Name' in filtered_rt.columns else 0
            st.metric("Events", events_count)

        # Create display dataframe with only needed columns (Status contains athlete names)
        display_cols_src = ['Actual_Event_Name', 'Status', 'Qualification_Status', 'Details']
        display_cols_src = [c for c in display_cols_src if c in filtered_rt.columns]
        filtered_display = filtered_rt[display_cols_src].copy()
        # Rename columns for clarity
        col_rename = {'Actual_Event_Name': 'Event', 'Status': 'Athlete', 'Qualification_Status': 'Status'}
        filtered_display = filtered_display.rename(columns=col_rename)

        st.dataframe(filtered_display.drop_duplicates().head(500), use_container_width=True, hide_index=True)

    else:
        st.warning("No qualification data found. Data may still be loading from Azure.")

    # Show qualification standards (from benchmarks parquet)
    if qual_standards_df is not None and not qual_standards_df.empty:
        st.subheader("Performance Standards")

        # Check which columns exist (benchmarks parquet has different structure)
        if 'Event' in qual_standards_df.columns and 'Gold Standard' in qual_standards_df.columns:
            # Using benchmarks parquet format
            display_cols = [col for col in ['Event', 'Gender', 'Gold Standard', 'Silver Standard', 'Bronze Standard', 'Final Standard (8th)'] if col in qual_standards_df.columns]
            st.dataframe(qual_standards_df[display_cols], use_container_width=True, hide_index=True)
        elif 'Display_Name' in qual_standards_df.columns:
            # Using old CSV format
            display_qual = qual_standards_df[['Display_Name', 'entry_number', 'entry_standard', 'maximum_quota', 'athletes_by_entry_standard']].copy()
            display_qual.columns = ['Event', 'Entry Quota', 'Entry Standard', 'Max per Country', 'Qualified by Standard']
            st.dataframe(display_qual, use_container_width=True, hide_index=True)
        else:
            st.dataframe(qual_standards_df, use_container_width=True, hide_index=True)

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

    # Initialize What It Takes to Win analyzer and load data
    @st.cache_resource
    def get_wittw_analyzer():
        analyzer = WhatItTakesToWin()
        analyzer.load_scraped_data()
        return analyzer

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

            # Display styled table
            display_cols = ['Event', 'Gold Standard', 'Silver Standard', 'Bronze Standard',
                           'Final Standard (8th)', 'Top 8 Average', 'Sample Size']

            st.dataframe(
                filtered_report[display_cols].style.background_gradient(
                    subset=['Sample Size'], cmap='Greens'
                ),
                use_container_width=True,
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
                            use_container_width=True,
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
                            ksa_data = get_ksa_rankings()
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
                        ksa_options = ['Select athlete...'] + sorted(ksa_athletes['full_name'].unique().tolist())
                        compare_athlete = st.selectbox("Select KSA Athlete to Compare",
                                                      ksa_options, key="wittw_compare")

                        if compare_athlete != 'Select athlete...':
                            # Get athlete's PB for event
                            athlete_pbs = ksa_athletes[ksa_athletes['full_name'] == compare_athlete]
                            event_match = athlete_pbs[athlete_pbs['event_name'].str.contains(
                                selected_event.replace('-', ' ').replace('metres', 'm'),
                                case=False, na=False
                            )]

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
                                st.info(f"No PB found for {compare_athlete} in {selected_event}")

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

                        st.plotly_chart(fig, use_container_width=True)
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
            Ask questions about athlete performance, competition standards, and strategic insights.
            Powered by free AI models via OpenRouter.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for chatbot
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = []
    if 'chatbot_model' not in st.session_state:
        st.session_state.chatbot_model = DEFAULT_MODEL

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
            # Clear chat button
            if st.button("Clear Chat", key="chatbot_clear", use_container_width=True):
                st.session_state.chatbot_messages = []
                st.rerun()

        st.markdown("---")

        # === CONTEXT FILTER: Select Athlete and Event ===
        with st.expander("Filter Context (Optional)", expanded=False):
            st.caption("Select an athlete and/or event to focus your question. This helps ensure accurate name matching.")

            filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

            # Get available athletes from data
            available_athletes = ["(All Athletes)"]
            available_events = ["(All Events)"]

            if DATA_CONNECTOR_AVAILABLE:
                try:
                    ksa_df = get_ksa_athletes()
                    if ksa_df is not None and not ksa_df.empty:
                        # Get athlete names
                        name_col = 'competitor' if 'competitor' in ksa_df.columns else 'full_name' if 'full_name' in ksa_df.columns else None
                        if name_col:
                            athlete_names = sorted(ksa_df[name_col].dropna().unique().tolist())
                            available_athletes.extend(athlete_names)

                        # Get events
                        if 'event' in ksa_df.columns:
                            events = sorted(ksa_df['event'].dropna().unique().tolist())
                            available_events.extend(events)
                        elif 'primary_event' in ksa_df.columns:
                            events = sorted(ksa_df['primary_event'].dropna().unique().tolist())
                            available_events.extend(events)
                except Exception:
                    pass

            with filter_col1:
                selected_ai_athlete = st.selectbox(
                    "Athlete",
                    available_athletes,
                    index=0,
                    key="ai_athlete_filter"
                )

            with filter_col2:
                selected_ai_event = st.selectbox(
                    "Event",
                    available_events,
                    index=0,
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
                    if st.button("Clear", key="clear_ai_context", use_container_width=True):
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

        # Input area (using text_input + button since chat_input doesn't work in tabs)
        st.markdown("<br>", unsafe_allow_html=True)
        input_col, btn_col = st.columns([5, 1])

        # Build dynamic placeholder based on context
        current_context = st.session_state.get('ai_context_prefix', '')
        if current_context:
            placeholder_text = f"Ask about {current_context.replace('About ', '')}..."
        else:
            placeholder_text = "Ask about athletics performance, KSA athletes, qualification standards..."

        with input_col:
            user_input = st.text_input(
                "Your question",
                placeholder=placeholder_text,
                key="chatbot_input",
                label_visibility="collapsed"
            )

        with btn_col:
            send_clicked = st.button("Send", key="chatbot_send", type="primary", use_container_width=True)

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

            # Create client and get response
            with st.spinner("Analyzing..."):
                client = OpenRouterClient(api_key, st.session_state.chatbot_model)

                # Prepare messages for API (without context field)
                api_messages = [{"role": m["role"], "content": m["content"]}
                               for m in st.session_state.chatbot_messages]

                # Use enhanced query for context building
                response = client.chat(api_messages, user_query=enhanced_query)

            # Add assistant response
            st.session_state.chatbot_messages.append({
                "role": "assistant",
                "content": response["content"],
                "context": response.get("context_used", "")
            })

            # Clear input and rerun to show new messages
            st.rerun()

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
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    st.session_state.chatbot_messages.append({
                        "role": "user",
                        "content": question
                    })
                    # Trigger processing
                    with st.spinner("Analyzing..."):
                        client = OpenRouterClient(api_key, st.session_state.chatbot_model)
                        api_messages = [{"role": m["role"], "content": m["content"]}
                                       for m in st.session_state.chatbot_messages]
                        response = client.chat(api_messages, user_query=question)

                    st.session_state.chatbot_messages.append({
                        "role": "assistant",
                        "content": response["content"],
                        "context": response.get("context_used", "")
                    })
                    st.rerun()

###################################
# Tab 9: Coach View
###################################
with tab9:
    if COACH_VIEW_AVAILABLE:
        # Load master rankings data for Coach View
        try:
            if DATA_CONNECTOR_AVAILABLE:
                coach_df = get_rankings_data()
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
# Footer
###################################
st.markdown(f"""
    <hr style='margin-top: 30px; border: 1px solid #333;'>
    <div style='text-align: center; color: #666; font-size: 0.85rem; padding: 1rem 0;'>
        Saudi Athletics Dashboard — Created by <strong style="color: {TEAL_PRIMARY};">Luke Gallagher</strong> | Team Saudi
    </div>
""", unsafe_allow_html=True)
