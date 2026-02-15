"""
AI Analyst - Elite athletics chatbot with live data + NotebookLM.

Hybrid approach: NotebookLM documents + OpenRouter LLM + live database.
Quick-analysis tabs for instant pre-built queries (no AI roundtrip).
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import pandas as pd
import streamlit as st

from components.theme import (
    get_theme_css, render_page_header, render_sidebar, render_section_header,
    render_metric_card, TEAL_PRIMARY, TEAL_DARK, GOLD_ACCENT, GRAY_BLUE,
)
from data.connector import get_connector
from data.event_utils import get_event_type, format_event_name

logger = logging.getLogger(__name__)

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Analyst", page_icon="🏃", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header("AI Analyst", "World-class sports analysis for Team Saudi")

dc = get_connector()

# ── NotebookLM Integration ────────────────────────────────────────────

# Lazy import from project root
_NLM_AVAILABLE = None


def _check_nlm():
    """Check if NotebookLM CLI is available (cached in session state)."""
    global _NLM_AVAILABLE
    if _NLM_AVAILABLE is not None:
        return _NLM_AVAILABLE
    if "nlm_available" in st.session_state:
        _NLM_AVAILABLE = st.session_state.nlm_available
        return _NLM_AVAILABLE

    try:
        # Import from project root
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from notebooklm_client import check_notebooklm_available
        _NLM_AVAILABLE = check_notebooklm_available()
    except Exception:
        _NLM_AVAILABLE = False

    st.session_state.nlm_available = _NLM_AVAILABLE
    return _NLM_AVAILABLE


def _query_nlm(question: str) -> Optional[str]:
    """Query NotebookLM and return the response."""
    try:
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from notebooklm_client import query_notebook
        return query_notebook(question)
    except Exception as e:
        return f"NotebookLM error: {e}"


# ── OpenRouter Configuration ─────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

FREE_MODELS = {
    'meta-llama/llama-3.3-70b-instruct:free': 'Llama 3.3 70B (Recommended)',
    'nousresearch/hermes-3-llama-3.1-405b:free': 'Hermes 3 405B (Best Quality)',
    'google/gemma-3-27b-it:free': 'Gemma 3 27B (Google)',
    'mistralai/mistral-small-3.1-24b-instruct:free': 'Mistral Small 3.1 24B',
    'google/gemma-3-4b-it:free': 'Gemma 3 4B (Fastest)',
    'deepseek/deepseek-r1:free': 'DeepSeek R1 (Reasoning)',
}

DEFAULT_MODEL = 'meta-llama/llama-3.3-70b-instruct:free'


def _get_api_key() -> Optional[str]:
    """Get OpenRouter API key from env or Streamlit secrets."""
    key = os.getenv('OPENROUTER_API_KEY')
    if not key:
        try:
            if hasattr(st, 'secrets'):
                key = st.secrets.get('OPENROUTER_API_KEY')
                if not key and 'openrouter' in st.secrets:
                    key = st.secrets.openrouter.get('OPENROUTER_API_KEY')
        except Exception:
            pass
    return key


# ── Athletics Knowledge Base ─────────────────────────────────────────

ATHLETICS_KNOWLEDGE = """## World Athletics Ranking System
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

## Terminology
- FAT: Fully Automatic Timing (electronic to 0.01s)
- PB: Personal Best, SB: Season Best, WL: World Lead
- WR: World Record, OR: Olympic Record, NR: National Record
- DNS: Did Not Start, DNF: Did Not Finish, DQ: Disqualified
"""

SYSTEM_PROMPT = """You are an ELITE world-class sports analyst and coach specializing in athletics (track and field).
You work for Team Saudi and provide EXPERT-LEVEL analysis with deep technical knowledge.

## Your Expertise
1. **Performance Analysis** - Times, distances, progressions against world-class benchmarks
2. **Event-Specific Technical Knowledge** - Sprints, middle distance, hurdles, jumps, throws, combined
3. **Competition Strategy** - Championship tactics, rounds progression, pressure factors
4. **Qualification & Rankings** - Olympics, World Championships, Asian Games standards
5. **Competitor Intelligence** - H2H analysis, form trends, gap analysis

## Response Guidelines
- Use your elite knowledge to provide technical insights beyond basic data
- Be specific with times, distances, rankings
- Note wind conditions for sprints and horizontal jumps
- Provide actionable insights - training focus, tactical adjustments
- Reference benchmarks - world class, regional elite standards
- Format responses with clear headers and bullet points
"""


# ── Context Builder ──────────────────────────────────────────────────

class ContextBuilder:
    """Build relevant data context for AI queries using v2 connector."""

    def __init__(self):
        self._dc = get_connector()

    def _query(self, sql: str):
        """Safe DuckDB query."""
        try:
            return self._dc.query(sql)
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return pd.DataFrame()

    def get_ksa_summary(self) -> str:
        """Summary of KSA athletes."""
        athletes = self._dc.get_ksa_athletes()
        if athletes.empty:
            return "No KSA athlete data available."

        parts = ["## KSA Athletes Overview\n"]
        parts.append(f"**Total Athletes:** {len(athletes)}")

        name_col = next((c for c in ['full_name', 'competitor'] if c in athletes.columns), None)
        event_col = next((c for c in ['primary_event', 'event'] if c in athletes.columns), None)
        rank_col = next((c for c in ['best_world_rank', 'rank'] if c in athletes.columns), None)

        if name_col:
            parts.append("\n**Athletes:**")
            for _, row in athletes.head(30).iterrows():
                name = row.get(name_col, 'Unknown')
                event = row.get(event_col, '') if event_col else ''
                rank = row.get(rank_col, '') if rank_col else ''
                rank_str = f" (WR#{rank})" if rank and str(rank) != 'nan' else ""
                parts.append(f"- {name} - {event}{rank_str}")

        return "\n".join(parts)

    def get_athlete_details(self, athlete_name: str) -> str:
        """Get detailed data for a specific athlete using v2 + legacy data."""
        parts = [f"## Performance Data: {athlete_name}\n"]
        safe_name = athlete_name.replace("'", "''")

        # Try v2 PBs first
        pbs = self._dc.get_ksa_athlete_pbs(athlete_name)
        if not pbs.empty:
            parts.append("**Personal Bests:**")
            disc_col = "discipline" if "discipline" in pbs.columns else "event"
            mark_col = "mark" if "mark" in pbs.columns else "result"
            for _, row in pbs.iterrows():
                disc = row.get(disc_col, "")
                mark = row.get(mark_col, "")
                venue = row.get("venue", "")
                date = row.get("date", "")
                parts.append(f"  - {disc}: {mark} ({venue}, {date})")

        # Try v2 recent results
        results = self._dc.get_ksa_results(athlete_name=athlete_name, limit=15)
        if not results.empty:
            parts.append("\n**Recent Competition Results:**")
            mark_col = "mark" if "mark" in results.columns else "result"
            disc_col = "discipline" if "discipline" in results.columns else "event"
            for _, row in results.iterrows():
                disc = row.get(disc_col, "")
                mark = row.get(mark_col, "")
                comp = row.get("competition", row.get("venue", ""))
                date = row.get("date", "")
                place = row.get("place", row.get("pos", ""))
                parts.append(f"  - {disc}: {mark} (P{place}) at {comp} ({date})")
        else:
            # Fall back to legacy master
            df = self._query(f"""
                SELECT competitor, event, result, venue, date, nat, rank, gender
                FROM master
                WHERE LOWER(competitor) LIKE LOWER('%{safe_name}%')
                ORDER BY date DESC
                LIMIT 50
            """)
            if not df.empty:
                country = df['nat'].dropna().iloc[0] if 'nat' in df.columns and not df['nat'].dropna().empty else 'Unknown'
                parts.append(f"**Country:** {country}\n")
                if 'event' in df.columns:
                    for event in df['event'].unique():
                        event_data = df[df['event'] == event]
                        parts.append(f"\n**{event}** ({len(event_data)} results):")
                        for _, row in event_data.head(8).iterrows():
                            parts.append(f"  - {row.get('result', 'N/A')} at {row.get('venue', '')} ({row.get('date', '')})")

        if len(parts) == 1:
            return f"No results found for '{athlete_name}'"

        return "\n".join(parts)

    def get_event_rankings(self, event: str, gender: str = 'men', top_n: int = 20) -> str:
        """Get top performers for an event."""
        evt_type = get_event_type(event)

        df = self._dc.get_world_rankings(event=event, gender=gender[0].upper(), limit=top_n)

        if df.empty:
            return f"No rankings for {event} ({gender})"

        name_col = next((c for c in ['athlete', 'competitor'] if c in df.columns), None)
        result_col = next((c for c in ['mark', 'result'] if c in df.columns), None)
        country_col = next((c for c in ['country', 'nat'] if c in df.columns), None)

        parts = [f"## Top {min(top_n, len(df))} in {event} ({gender.title()})\n"]
        for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
            name = row.get(name_col, 'Unknown') if name_col else 'Unknown'
            result = row.get(result_col, 'N/A') if result_col else 'N/A'
            country = row.get(country_col, '') if country_col else ''
            parts.append(f"{i}. {name} ({country}) - {result}")

        return "\n".join(parts)

    def get_rivals_context(self, event: str) -> str:
        """Get rival data for an event from v2 scraped data."""
        rivals = self._dc.get_rivals(event=event, limit=15)
        if rivals.empty:
            return f"No rival data for {event}."

        parts = [f"## Top Rivals: {event}\n"]
        name_col = "full_name" if "full_name" in rivals.columns else "athlete"
        for _, row in rivals.iterrows():
            name = row.get(name_col, "Unknown")
            country = row.get("country_code", row.get("country", ""))
            rank = row.get("world_rank", "")
            score = row.get("ranking_score", "")
            region = row.get("region", "")
            parts.append(f"- {name} ({country}) - WR#{rank}, Score: {score} [{region}]")

        return "\n".join(parts)

    def get_qualification_standards(self) -> str:
        """Get championship qualification standards."""
        benchmarks = self._dc.get_benchmarks()
        if benchmarks.empty:
            return "No qualification standards available."

        parts = ["## Championship Standards\n"]
        for _, row in benchmarks.head(20).iterrows():
            parts.append(f"- {row.to_dict()}")

        return "\n".join(parts)

    def build_context(self, query: str) -> Tuple[str, List[str]]:
        """Build relevant context based on the user's query."""
        query_lower = query.lower()
        context_parts = [ATHLETICS_KNOWLEDGE]
        sources = ["Athletics Knowledge Base"]

        # Check for athlete name mentions
        athletes = self._dc.get_ksa_athletes()
        if not athletes.empty:
            name_col = next((c for c in ['full_name', 'competitor'] if c in athletes.columns), None)
            if name_col:
                for name in athletes[name_col].dropna().unique():
                    name_parts = str(name).lower().split()
                    if any(part in query_lower for part in name_parts if len(part) > 2):
                        data = self.get_athlete_details(str(name))
                        if "No results" not in data:
                            context_parts.append(data)
                            sources.append(f"Athlete: {name}")
                            break

        # Check for event mentions
        events = ['100m', '200m', '400m', '800m', '1500m', '5000m', '10000m',
                  'marathon', '110m hurdles', '400m hurdles', '100m hurdles',
                  'high jump', 'long jump', 'triple jump', 'pole vault',
                  'shot put', 'discus', 'hammer', 'javelin', 'decathlon', 'heptathlon']

        for event in events:
            if event in query_lower:
                gender = 'women' if any(w in query_lower for w in ['women', 'female', "women's"]) else 'men'
                rankings = self.get_event_rankings(event, gender)
                if "No rankings" not in rankings:
                    context_parts.append(rankings)
                    sources.append(f"Rankings: {event}")
                rivals_ctx = self.get_rivals_context(event)
                if "No rival" not in rivals_ctx:
                    context_parts.append(rivals_ctx)
                    sources.append(f"Rivals: {event}")
                break

        # Qualification/standards queries
        if any(word in query_lower for word in ['qualify', 'standard', 'championship', 'olympic', 'asian']):
            context_parts.append(self.get_qualification_standards())
            sources.append("Qualification Standards")

        # KSA team queries
        if any(word in query_lower for word in ['ksa', 'saudi', 'team', 'athletes', 'roster']):
            context_parts.append(self.get_ksa_summary())
            sources.append("KSA Athletes Database")

        # Country queries - search master database
        country_codes = {
            'usa': 'USA', 'jamaica': 'JAM', 'kenya': 'KEN', 'ethiopia': 'ETH',
            'japan': 'JPN', 'china': 'CHN', 'india': 'IND', 'qatar': 'QAT',
            'bahrain': 'BRN', 'great britain': 'GBR', 'germany': 'GER',
        }
        for country_name, code in country_codes.items():
            if country_name in query_lower:
                df = self._query(f"""
                    SELECT competitor, event, result, venue, date
                    FROM master WHERE UPPER(nat) = '{code}'
                    ORDER BY date DESC LIMIT 20
                """)
                if not df.empty:
                    context_parts.append(f"## {code} Athletes\n")
                    for _, row in df.head(15).iterrows():
                        context_parts.append(f"- {row.get('competitor', '')} - {row.get('event', '')} - {row.get('result', '')} ({row.get('date', '')})")
                    sources.append(f"Database: {code}")
                break

        # Fallback: keyword search
        if len(context_parts) <= 1:
            common_words = {'what', 'how', 'when', 'where', 'which', 'who', 'does', 'can',
                           'the', 'and', 'for', 'that', 'this', 'with', 'from', 'about'}
            keywords = [w for w in query_lower.split() if len(w) > 3 and w not in common_words]
            for kw in keywords[:2]:
                safe_kw = kw.replace("'", "''")
                df = self._query(f"""
                    SELECT competitor, event, result, nat, venue, date
                    FROM master
                    WHERE LOWER(competitor) LIKE '%{safe_kw}%'
                       OR LOWER(venue) LIKE '%{safe_kw}%'
                       OR LOWER(event) LIKE '%{safe_kw}%'
                    ORDER BY date DESC LIMIT 15
                """)
                if not df.empty:
                    context_parts.append(f"## Search: {kw}\n")
                    for _, row in df.iterrows():
                        context_parts.append(f"- {row.get('competitor', '')} ({row.get('nat', '')}) - {row.get('event', '')} - {row.get('result', '')} @ {row.get('venue', '')} ({row.get('date', '')})")
                    sources.append(f"Search: {kw}")
                    break

        return "\n\n---\n\n".join(context_parts), sources


# ── Chat Client ──────────────────────────────────────────────────────

class ChatClient:
    """OpenRouter chat client with athletics context."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from openai import OpenAI
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self.model = model
        self.context_builder = ContextBuilder()

    def chat(self, message: str, history: List[Dict] = None) -> Tuple[str, str]:
        """Send a message and get a response with sources."""
        data_context, sources = self.context_builder.build_context(message)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if data_context:
            messages.append({
                "role": "system",
                "content": f"## Available Data Context\n\n{data_context}"
            })
        if history:
            messages.extend(history[-12:])
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                top_p=0.9,
                extra_headers={
                    "HTTP-Referer": "https://team-saudi-athletics.streamlit.app",
                    "X-Title": "Team Saudi Athletics Analyst"
                }
            )
            answer = response.choices[0].message.content.strip()
            sources_text = ""
            if sources:
                sources_text = "**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
            return answer, sources_text

        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return f"Error: {str(e)}", ""

    def chat_hybrid(self, message: str, nlm_context: str, history: List[Dict] = None) -> Tuple[str, str]:
        """Hybrid: combine NotebookLM research with OpenRouter analysis."""
        data_context, sources = self.context_builder.build_context(message)
        sources.insert(0, "NotebookLM Documents")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        # Inject NotebookLM research as high-priority context
        if nlm_context:
            messages.append({
                "role": "system",
                "content": (
                    "## Research from NotebookLM Documents (citation-backed)\n\n"
                    f"{nlm_context}\n\n"
                    "Use this research as your primary source. Supplement with "
                    "the live database context below where relevant."
                ),
            })
        if data_context:
            messages.append({
                "role": "system",
                "content": f"## Live Database Context\n\n{data_context}"
            })
        if history:
            messages.extend(history[-12:])
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                top_p=0.9,
                extra_headers={
                    "HTTP-Referer": "https://team-saudi-athletics.streamlit.app",
                    "X-Title": "Team Saudi Athletics Analyst"
                }
            )
            answer = response.choices[0].message.content.strip()
            sources_text = "**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
            return answer, sources_text

        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return f"Error: {str(e)}", ""


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### AI Configuration")

    # Backend selector
    nlm_available = _check_nlm()

    backend_options = []
    if nlm_available:
        backend_options.extend(["Hybrid", "NotebookLM"])
    backend_options.append("OpenRouter")

    selected_backend = st.radio(
        "AI Backend",
        backend_options,
        index=0,
        horizontal=True,
        key="ai_backend",
    )

    # Backend status badges
    if selected_backend == "Hybrid":
        st.success("Documents + Live data (best)")
    elif selected_backend == "NotebookLM":
        st.info("Fast, citation-backed from documents")
    else:
        st.info("LLM with live database context")

    if not nlm_available:
        st.caption("NotebookLM not available. Run `notebooklm login` to enable.")

    st.markdown("---")

    api_key = _get_api_key()
    if selected_backend != "NotebookLM":
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

    selected_model = st.selectbox(
        "AI Model",
        options=list(FREE_MODELS.keys()),
        format_func=lambda x: FREE_MODELS[x],
        index=0,
        key="ai_model",
    )

    st.markdown("---")

    # Data status
    st.markdown("### Data Status")
    status = dc.get_status()
    counts = status.get("counts", {})
    ksa_count = counts.get("ksa_athletes", 0) or counts.get("ksa_profiles", 0)
    master_count = counts.get("master", 0)

    if ksa_count > 0:
        st.success(f"{ksa_count} KSA Athletes")
    if master_count > 0:
        st.success(f"{master_count:,} Master Records")
    if nlm_available:
        st.success("NotebookLM Connected")
    if ksa_count == 0 and master_count == 0:
        st.warning("No data loaded")

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()


# ── Quick Analysis Helpers ───────────────────────────────────────────

def _render_ksa_overview():
    """Quick view: KSA athlete roster with PBs and rankings."""
    athletes = dc.get_ksa_athletes()
    if athletes.empty:
        st.info("No KSA athlete data available.")
        return

    st.dataframe(
        athletes,
        hide_index=True,
        height=400,
        use_container_width=True,
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total Athletes", str(len(athletes)), "neutral")
    with col2:
        if "primary_event" in athletes.columns:
            n_events = athletes["primary_event"].nunique()
            render_metric_card("Events", str(n_events), "good")
    with col3:
        if "best_world_rank" in athletes.columns:
            ranked = athletes[pd.to_numeric(athletes["best_world_rank"], errors="coerce").notna()]
            render_metric_card("Ranked Athletes", str(len(ranked)), "excellent")


def _render_standards_gap():
    """Quick view: KSA athletes vs championship standards."""
    benchmarks = dc.get_benchmarks()
    if benchmarks.empty:
        st.info("No benchmark data available.")
        return

    athletes = dc.get_ksa_athletes()
    if athletes.empty:
        st.info("No KSA athlete data.")
        return

    # Build gap table: for each KSA athlete, show PB vs gold/bronze standards
    pbs = dc.get_ksa_athlete_pbs()
    if pbs.empty:
        st.info("No PB data available.")
        return

    gap_rows = []
    disc_col = "discipline" if "discipline" in pbs.columns else "event"
    mark_col = "mark" if "mark" in pbs.columns else "result"

    for _, pb_row in pbs.iterrows():
        discipline = str(pb_row.get(disc_col, ""))
        mark_str = str(pb_row.get(mark_col, ""))
        athlete = str(pb_row.get("full_name", pb_row.get("athlete_name", "")))

        # Find matching benchmark
        from data.event_utils import normalize_event_for_match
        pb_norm = normalize_event_for_match(discipline)

        for _, bench in benchmarks.iterrows():
            bench_norm = normalize_event_for_match(str(bench.get("Event", "")))
            if bench_norm == pb_norm:
                gold = bench.get("Gold Standard", "")
                bronze = bench.get("Bronze Standard", "")
                gap_rows.append({
                    "Athlete": athlete,
                    "Event": discipline,
                    "PB": mark_str,
                    "Gold Standard": str(gold),
                    "Bronze Standard": str(bronze),
                })
                break

    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        st.dataframe(gap_df, hide_index=True, use_container_width=True, height=400)
    else:
        st.info("No matching standards found for KSA events.")


def _render_rival_watch():
    """Quick view: Top rivals by event from scraped data."""
    rivals = dc.get_rivals(limit=50)
    if rivals.empty:
        st.info("No rival data available. Run the v2 scraper to populate.")
        return

    # Event filter
    event_col = "event" if "event" in rivals.columns else "discipline"
    if event_col in rivals.columns:
        events = sorted(rivals[event_col].dropna().unique())
        selected_event = st.selectbox("Filter Event", ["All Events"] + list(events), key="rw_event")
        if selected_event != "All Events":
            rivals = rivals[rivals[event_col] == selected_event]

    display_cols = []
    for c in ["full_name", "country_code", "event", "world_rank", "ranking_score", "region"]:
        if c in rivals.columns:
            display_cols.append(c)

    rename = {
        "full_name": "Athlete", "country_code": "Country",
        "event": "Event", "world_rank": "World Rank",
        "ranking_score": "Score", "region": "Region",
    }

    if display_cols:
        display_df = rivals[display_cols].rename(columns={c: rename.get(c, c) for c in display_cols})
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)


def _render_championship_results():
    """Quick view: KSA results at major championships."""
    results = dc.get_ksa_results(limit=500)
    if results.empty:
        st.info("No KSA results available.")
        return

    # Championship filter
    if "competition" in results.columns:
        comps = sorted(results["competition"].dropna().unique())
        selected_comp = st.selectbox("Filter Competition", ["All"] + list(comps), key="cr_comp")
        if selected_comp != "All":
            results = results[results["competition"] == selected_comp]

    display_cols = []
    for c in ["full_name", "discipline", "mark", "place", "competition", "date", "venue"]:
        if c in results.columns:
            display_cols.append(c)

    rename = {
        "full_name": "Athlete", "discipline": "Event", "mark": "Mark",
        "place": "Place", "competition": "Competition", "date": "Date",
        "venue": "Venue",
    }

    if display_cols:
        display_df = results[display_cols].rename(columns={c: rename.get(c, c) for c in display_cols})

        # Sort by date
        if "Date" in display_df.columns:
            display_df["_d"] = pd.to_datetime(display_df["Date"], format="mixed", errors="coerce")
            display_df = display_df.sort_values("_d", ascending=False).drop(columns=["_d"])

        st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)

    # Medal count
    if "place" in results.columns:
        places = pd.to_numeric(results["place"].astype(str).str.strip(), errors="coerce").dropna()
        gold = int((places == 1).sum())
        silver = int((places == 2).sum())
        bronze = int((places == 3).sum())
        if gold + silver + bronze > 0:
            st.markdown("")
            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card("Gold", str(gold), "gold")
            with c2:
                render_metric_card("Silver", str(silver), "neutral")
            with c3:
                render_metric_card("Bronze", str(bronze), "warning")


# ── Main Layout: Tabs ────────────────────────────────────────────────

tab_chat, tab_standards, tab_rivals, tab_champs = st.tabs([
    "AI Chat",
    "Standards Gap",
    "Rival Watch",
    "Championship Results",
])

# ── Tab 1: AI Chat ───────────────────────────────────────────────────

with tab_chat:
    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "chat_client" not in st.session_state or st.session_state.get("chat_model") != selected_model:
        if api_key:
            st.session_state.chat_client = ChatClient(api_key, selected_model)
            st.session_state.chat_model = selected_model

    # Quick action buttons
    st.markdown("**Quick Actions:**")
    qa_cols = st.columns(4)
    quick_queries = [
        ("Medal Gap Analysis", "What are the gaps between our top KSA athletes and medal standards at Asian Games 2026? Focus on events where we're closest to medals."),
        ("Top Rivals", "Who are the key Asian rivals for our best KSA athletes? Compare recent form and rankings."),
        ("Form Trends", "Analyze recent performance trends for our KSA athletes. Who is improving and who is declining?"),
        ("Qualification Status", "What is the qualification status for KSA athletes for World Championships Tokyo 2025 and Asian Games Nagoya 2026?"),
    ]
    for i, (label, query) in enumerate(quick_queries):
        with qa_cols[i]:
            if st.button(label, key=f"qa_{i}", use_container_width=True):
                st.session_state.pending_query = query

    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources", expanded=False):
                    st.markdown(msg["sources"])

    # Handle pending query from sidebar or quick actions
    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query

        st.session_state.chat_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Route based on backend
        if selected_backend == "NotebookLM":
            with st.chat_message("assistant"):
                with st.spinner("Querying NotebookLM..."):
                    response = _query_nlm(query)
                    sources = "**Sources:**\n- NotebookLM Documents"
                st.markdown(response)
                with st.expander("Sources", expanded=False):
                    st.markdown(sources)
            st.session_state.chat_messages.append({
                "role": "assistant", "content": response, "sources": sources
            })

        elif selected_backend == "Hybrid" and nlm_available:
            if hasattr(st.session_state, 'chat_client'):
                with st.chat_message("assistant"):
                    with st.spinner("Querying NotebookLM + Live Data..."):
                        nlm_response = _query_nlm(query)
                        response, sources = st.session_state.chat_client.chat_hybrid(
                            query,
                            nlm_response or "",
                            [{"role": m["role"], "content": m["content"]}
                             for m in st.session_state.chat_messages[:-1]]
                        )
                    st.markdown(response)
                    if sources:
                        with st.expander("Sources", expanded=False):
                            st.markdown(sources)
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": response, "sources": sources
                })

        else:  # OpenRouter
            if hasattr(st.session_state, 'chat_client'):
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing athletics data..."):
                        response, sources = st.session_state.chat_client.chat(
                            query,
                            [{"role": m["role"], "content": m["content"]}
                             for m in st.session_state.chat_messages[:-1]]
                        )
                    st.markdown(response)
                    if sources:
                        with st.expander("Sources", expanded=False):
                            st.markdown(sources)
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": response, "sources": sources
                })

        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about athletics performance, competitors, or strategy..."):
        needs_api = selected_backend != "NotebookLM"
        if needs_api and not api_key:
            st.error("Please configure your OpenRouter API key in the sidebar.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Route based on backend
            if selected_backend == "NotebookLM":
                with st.chat_message("assistant"):
                    with st.spinner("Querying NotebookLM..."):
                        response = _query_nlm(prompt)
                        sources = "**Sources:**\n- NotebookLM Documents"
                    st.markdown(response)
                    with st.expander("Sources", expanded=False):
                        st.markdown(sources)
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": response, "sources": sources
                })

            elif selected_backend == "Hybrid" and nlm_available:
                with st.chat_message("assistant"):
                    with st.spinner("Querying NotebookLM + Live Data..."):
                        nlm_response = _query_nlm(prompt)
                        response, sources = st.session_state.chat_client.chat_hybrid(
                            prompt,
                            nlm_response or "",
                            [{"role": m["role"], "content": m["content"]}
                             for m in st.session_state.chat_messages[:-1]]
                        )
                    st.markdown(response)
                    if sources:
                        with st.expander("Sources", expanded=False):
                            st.markdown(sources)
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": response, "sources": sources
                })

            else:  # OpenRouter
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing athletics data..."):
                        response, sources = st.session_state.chat_client.chat(
                            prompt,
                            [{"role": m["role"], "content": m["content"]}
                             for m in st.session_state.chat_messages[:-1]]
                        )
                    st.markdown(response)
                    if sources:
                        with st.expander("Sources", expanded=False):
                            st.markdown(sources)
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": response, "sources": sources
                })

    # Example queries (only when chat is empty)
    if not st.session_state.chat_messages:
        st.markdown("---")
        st.markdown("**Try asking:**")
        ex_cols = st.columns(2)
        examples = [
            "How are KSA sprinters performing?",
            "What time makes the 100m final at Worlds?",
            "Tell me about Abdullah Abkar Mohammed",
            "Compare our 400m athletes to Asian rivals",
            "Olympic qualification standards for triple jump?",
            "Who should target medal contention at Asian Games?",
        ]
        for i, q in enumerate(examples):
            with ex_cols[i % 2]:
                if st.button(q, key=f"ex_{i}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()


# ── Tab 2: Standards Gap ─────────────────────────────────────────────

with tab_standards:
    render_section_header(
        "KSA Standards Gap Analysis",
        "How far are Saudi athletes from championship medal standards?",
    )
    _render_standards_gap()

    # AI follow-up
    st.markdown("---")
    if st.button("Ask AI about gaps", key="ask_gaps", use_container_width=True):
        st.session_state.pending_query = (
            "Analyze the gap between our KSA athletes and championship medal "
            "standards. Which athletes are closest to breakthrough? What events "
            "have the most realistic medal chances at Asian Games 2026?"
        )
        st.rerun()


# ── Tab 3: Rival Watch ───────────────────────────────────────────────

with tab_rivals:
    render_section_header(
        "Rival Watch",
        "Monitor KSA athletes vs key Asian and regional competitors",
    )
    _render_rival_watch()

    st.markdown("---")
    if st.button("Ask AI about rivals", key="ask_rivals", use_container_width=True):
        st.session_state.pending_query = (
            "Who are the biggest threats to KSA athletes at Asian Games 2026? "
            "Compare our top athletes to key Asian rivals by event."
        )
        st.rerun()


# ── Tab 4: Championship Results ──────────────────────────────────────

with tab_champs:
    render_section_header(
        "Championship History",
        "KSA results at major international championships",
    )
    _render_championship_results()

    st.markdown("---")
    if st.button("Ask AI about championships", key="ask_champs", use_container_width=True):
        st.session_state.pending_query = (
            "Summarize KSA's championship history. What events have we performed "
            "best in? Where are the opportunities for improvement?"
        )
        st.rerun()
