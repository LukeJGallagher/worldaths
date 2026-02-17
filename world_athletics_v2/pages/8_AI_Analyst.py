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
    'meta-llama/llama-3.3-70b-instruct': 'Llama 3.3 70B (Recommended)',
    'google/gemini-2.0-flash-001': 'Gemini 2.0 Flash (Fast)',
    'deepseek/deepseek-chat-v3-0324': 'DeepSeek V3 (Best Value)',
    'mistralai/mistral-small-3.1-24b-instruct': 'Mistral Small 3.1 24B',
    'google/gemma-3-27b-it': 'Gemma 3 27B (Google)',
    'deepseek/deepseek-r1': 'DeepSeek R1 (Reasoning)',
}

DEFAULT_MODEL = 'google/gemini-2.0-flash-001'


def _get_api_key() -> Optional[str]:
    """Get OpenRouter API key from env or Streamlit secrets.

    Tries primary key first, then falls back to OPENROUTER_API_KEY_2.
    """
    # Try primary key
    key = os.getenv('OPENROUTER_API_KEY')
    if not key:
        try:
            if hasattr(st, 'secrets'):
                key = st.secrets.get('OPENROUTER_API_KEY')
                if not key and 'openrouter' in st.secrets:
                    key = st.secrets.openrouter.get('OPENROUTER_API_KEY')
        except Exception:
            pass
    # Fallback to secondary key
    if not key:
        key = os.getenv('OPENROUTER_API_KEY_2')
        if not key:
            try:
                if hasattr(st, 'secrets'):
                    key = st.secrets.get('OPENROUTER_API_KEY_2')
            except Exception:
                pass
    return key


# ── Athletics Knowledge Base ─────────────────────────────────────────

ATHLETICS_KNOWLEDGE = """
## World Athletics Ranking System
- Points = Result Score × Competition Category Weight × Placing Score
- Result Score: Converted from performance (IAAF scoring tables, event-specific)
- Competition Categories & Weights:
  - OW (Olympics/World Champs): 1.00x — highest prestige, deepest fields
  - DF (Diamond Final): 0.95x — season-ending Diamond League final
  - GW (Gold Continental/Diamond League): 0.90x — e.g. Asian Games, DL meetings
  - GL (Gold Label/A-level): 0.85x — Continental Championships, top permits
  - B (B-level): 0.80x — National Championships, area permits
  - C-F: 0.70x-0.50x — lower-tier meetings
- Placing Score bonus: Top 8 finishers get 1-8 bonus points at OW/DF/GW events
- Rankings window: Best 5 results from rolling 12-month window (minimum 3 results needed)
- Rankings determine Olympic/Worlds qualification via quota places and entry standards
- Athletes need minimum ranking position AND/OR entry standard to qualify

## Competition Calendar & Championship Structure
- **Olympic Games** (every 4 years): Next = LA 2028. Qualification via entry standards + world rankings quota
- **World Athletics Championships** (annual since 2023): Next = Tokyo 2025. Same qualification system as Olympics
- **Asian Games** (every 4 years): Next = Nagoya-Aichi 2026. Qualification through NOC (national federation) nomination
- **Diamond League** (May-Sep): 14 meetings + final. Points-based series for elite athletes
- **Continental Tour**: Gold, Silver, Bronze tiers. Important for ranking points accumulation
- **Asian Championships** (biennial): Regional championship, important for Asian ranking
- **Arab Championships**: Regional event for GCC and Arab nations
- **Islamic Solidarity Games**: Multi-sport event with athletics program

## Championship Rounds Structure
- **Sprints (100m, 200m)**: Heats → Semi-finals → Final (3 rounds at major champs)
  - Heats: Top 3 per heat (Q) + next fastest (q) advance. ~24 athletes → 24 advance
  - Semi-finals: Top 2 per semi (Q) + next 2 fastest (q) advance. 24 → 8 for final
  - Automatic qualification (Q) is safest; time qualification (q) is risky
- **400m**: Heats → Semi-finals → Final (same structure as sprints)
- **800m/1500m**: Heats → Semi-finals → Final. More tactical, positioning critical
- **5000m/10000m**: Usually straight final or heats → final (no semis)
- **110m/400m Hurdles**: Heats → Semi-finals → Final
- **Field Events**: Qualification round (standard or top 12) → Final (top 8 get 3 extra attempts)
- **Advancement probability**: Heat time needed is ~0.3-0.5s slower than final time for sprints

## Event-Specific Performance Analysis

### Sprints (100m, 200m, 400m)
- **100m**: Reaction time (legal ≥0.100s), drive phase (0-30m), max velocity (60-80m), speed maintenance (80-100m)
  - Elite: Sub-10.00 (world class), 10.00-10.20 (Olympic level), 10.20-10.40 (Asian elite)
  - Key factors: Block start quality, anthropometrics, max velocity, speed endurance
  - Wind: Legal ≤2.0 m/s. +2.0 tailwind worth ~0.10-0.12s. Headwind costs more
- **200m**: Bend running technique, speed endurance, lane draw matters (lane 3-6 optimal)
  - Elite: Sub-20.00 (world class), 20.00-20.50 (major final), 20.50-21.00 (Asian elite)
  - Altitude advantage: ~0.10-0.15s at 1000m+ elevation (Mexico City effect)
- **400m**: Speed endurance event. Race model: fast first 200m, controlled 200-300m, resist deceleration 300-400m
  - Elite: Sub-44.00 (world class), 44.00-45.00 (Olympic level), 45.00-46.00 (Asian elite)
  - Lactate threshold and anaerobic capacity are primary limiters
  - Differential (first 200m vs second 200m): Elite ~1.0-1.5s, poor race model >2.0s

### Middle Distance (800m, 1500m)
- **800m**: Anaerobic + aerobic hybrid. First lap positioning critical (~49-52s for elite first 400m)
  - Elite: Sub-1:43 (world class), 1:43-1:45 (Olympic level), 1:45-1:47 (Asian elite)
  - Tactical vs pace race: Slow pace favours kickers, fast pace favours strong runners
- **1500m**: Tactical masterclass at championships. Last 400m determines medals
  - Elite: Sub-3:30 (world class), 3:30-3:35 (Olympic level), 3:35-3:40 (Asian elite)
  - Championship racing: Often 3:40+ pace through 800m then 50-52s last 400m

### Hurdles
- **110m Hurdles (Men)**: 3-step pattern between hurdles (9.14m spacing), trail leg technique
  - Elite: Sub-13.00 (world class), 13.00-13.40 (Olympic level), 13.40-13.80 (Asian elite)
  - 8-step to first hurdle standard, step frequency critical
- **400m Hurdles**: Stride pattern management (13-17 steps between barriers), fatigue management
  - Elite: Sub-47.00 (world class), 47.00-48.50 (major final), 48.50-49.50 (Asian elite)
  - Alternating lead leg is modern elite technique

### Jumps
- **High Jump**: Approach speed, takeoff angle, bar clearance technique (Fosbury Flop)
  - Elite: 2.30m+ (world class), 2.25-2.30m (major final), 2.20-2.25m (Asian elite)
  - Countback rules: Fewer misses at height wins on ties
- **Long Jump**: Approach speed, board accuracy, takeoff angle (18-22°), flight technique
  - Elite: 8.20m+ (world class), 8.00-8.20m (major final), 7.80-8.00m (Asian elite)
  - Wind effect: +2.0 m/s adds ~12-15cm to distance
- **Triple Jump**: Hop-step-jump ratios (35:30:35 ideal), speed maintenance through phases
  - Elite: 17.00m+ (world class), 16.70-17.00m (major final), 16.40-16.70m (Asian elite)
- **Pole Vault**: Approach speed, grip height, pole stiffness selection
  - Elite: 5.80m+ (world class), 5.60-5.80m (major final), 5.40-5.60m (Asian elite)

### Throws
- **Shot Put**: Rotational vs glide technique. Release angle ~38-42°
  - Elite: 21.50m+ (world class), 20.50-21.50m (major final), 19.50-20.50m (Asian elite)
- **Discus**: Release speed, angle, spin rate. Wind direction impacts flight
  - Elite: 67.00m+ (world class), 64.00-67.00m (major final), 60.00-64.00m (Asian elite)
- **Javelin**: Approach speed, release angle ~34-36°, crosswind technique
  - Elite: 87.00m+ (world class), 83.00-87.00m (major final), 78.00-83.00m (Asian elite)
- **Hammer**: Release speed paramount. 3 or 4 turns. Centripetal force management
  - Elite: 78.00m+ (world class), 75.00-78.00m (major final), 70.00-75.00m (Asian elite)

### Race Walks
- **20km Race Walk**: Contact rule (visible to naked eye), bent knee rule (straighten at vertical)
  - Red cards: 3 from different judges = disqualification. Warning paddle system
  - Elite: Sub-1:18 (world class), 1:18-1:22 (Asian elite)
- **35km Race Walk** (replaced 50km from 2023): Endurance + technique maintenance under fatigue

## Asian Athletics Landscape
- **Dominant nations**: Japan (distance), China (throws/walks), India (javelin/relays), Qatar/Bahrain (sprints/middle distance, often naturalised athletes)
- **KSA strengths historically**: Sprints (100m, 400m), middle distance (800m, 1500m), triple jump, high jump
- **Asian Games medal depth**: Generally 0.5-1.5% slower/shorter than World Championship level
- **Key Asian rivals by event**:
  - 100m: Su Bingtian (CHN retired), Hakim Sani Brown (JPN), Letsile Tebogo trains with Asian-eligible athletes
  - 400m: Kirani James (not Asian), but Qatar/Bahrain have strong 400m programmes
  - 800m/1500m: India, Bahrain have historically strong middle distance
  - Javelin: Neeraj Chopra (IND) - Olympic champion, dominant in Asia
  - Triple Jump: China, Japan have consistent Asian-level jumpers
  - High Jump: Asian record holders from Qatar, Syria, India

## Performance Progression & Periodisation
- **Annual planning**: General Preparation → Specific Preparation → Pre-Competition → Competition → Transition
- **Championship peaking**: Athletes aim to peak for target competition. Form cycle ~2-4 weeks
- **Detraining/retraining**: Performance drops 5-10% after 4-week break, recovers in 6-8 weeks
- **Age curves**: Sprinters peak 24-28, distance 26-32, throwers 27-33, jumpers 24-30
- **Improvement rates**: Year-over-year PB improvement typically 0.5-2% for developing athletes, <0.5% for elite

## Terminology & Abbreviations
- **FAT**: Fully Automatic Timing (electronic to 0.01s, standard at all major competitions)
- **PB/PR**: Personal Best/Personal Record, **SB**: Season Best, **WL**: World Lead (best mark globally this year)
- **WR**: World Record, **OR**: Olympic Record, **CR**: Championship Record, **AR**: Area Record, **NR**: National Record
- **DNS**: Did Not Start, **DNF**: Did Not Finish, **DQ**: Disqualified, **NM**: No Mark (field events)
- **Q**: Qualified by position (automatic), **q**: Qualified by time/distance (fastest losers)
- **w**: Wind-assisted (>2.0 m/s), **A**: Altitude-assisted (>1000m elevation)
- **MR**: Meeting Record, **PB**: Personal Best, **=PB**: Equalled Personal Best
- **WPA Score**: World Athletics points score (performance converted to standardised scale 0-1400)
- **Ranking Score**: WPA points × competition weight × placing bonus (used for world rankings)

## Wind & Conditions
- Legal wind limit: ≤2.0 m/s for 100m, 200m, 100mH, 110mH, long jump, triple jump
- Wind gauge measured for 10s (100m) or 13s (100mH) from gun, at hurdle/runway level
- Wind reading of 0.0 is ideal; slight tailwind (0.5-1.5) often produces best legal performances
- Temperature: Hot conditions (>30°C) affect distance events more. Cold (<15°C) affects sprints
- Altitude: Thinner air reduces drag (helps sprints/jumps), reduces O2 (hurts 800m+)
  - Mexico City (2,240m): 100m ~0.07s faster, 1500m ~3s slower

## Team Saudi / KSA Athletes Roster
Key KSA athletes by event group (use these exact names for data lookup):

**Sprints & Hurdles:**
- Abdullah Abkar MOHAMMED — 100m (WR#284)
- Abdulaziz Abdui ATAFI — 200m (WR#506)
- Yaqoob Salem AL-SAADI — 200m
- Mohammed Salem AL YAMI — 400m
- Mazen Al Yassin MOUMEN — 400m
- Baqer AL JUMAH — 110mH (WR#301)
- Naif Rashid ALSUBAIE — 400mH (WR#406)
- Azzam Ibrahim ABU BAKR — 400mH (WR#424)

**Middle Distance & Steeplechase:**
- Faisal MAGHRABI — 800m (WR#511)
- Sultan Abubaker ALATAWI — 1500m
- Khalid Mohhamed HAZAZI — 3000mSC (WR#329)

**Jumps:**
- Sami BAKHEET — Triple Jump (WR#39) — KSA's highest-ranked athlete
- Hassan NASSER DAROUICHE — Triple Jump (WR#117)
- Hussain Asim AL HIZAM — Pole Vault (WR#48) — 2nd highest ranked KSA athlete
- Essa Saud ALDOSSERI — High Jump
- Saleh Essa Saud ALDOSSERI — High Jump

**Throws & Walks:**
- Saleh AL HADDAD — Shot Put
- Ahmed Yahya ALQARNI — Javelin
- Mohamed Yousef ALROMAIHI — 20km Walk
- Hasan ALHAWSAWI — 20km Walk

## Asian Games Nagoya-Aichi 2026 (PRIMARY TARGET)
- Dates: September 2026 (exact TBC)
- Host: Nagoya & Aichi Prefecture, Japan
- Qualification: NOC nomination (no entry standards — federation selects athletes)
- **Medal depth**: Generally 0.5-1.5% behind World Championship level. Realistic target for KSA top athletes
- **KSA medal contenders**: Sami BAKHEET (Triple Jump), Hussain Asim AL HIZAM (Pole Vault), Abdullah Abkar MOHAMMED (100m)
- **Asian sprint landscape**: Japan (Sani Brown, Saito), China, Thailand, India all competitive in sprints
- **Key Asian rival nations**: Japan (depth across events), China (throws/walks/sprints), India (javelin/relays), Qatar & Bahrain (naturalised sprinters), Iran (throws)
- **Historical KSA Asian Games**: Medals in sprints, middle distance, jumps
- **Per-country limits**: Typically max 3 athletes per country per event at Asian Games

## LA 2028 Olympics (LONG-TERM TARGET)
- Dates: July 14 - August 30, 2028
- Venue: Los Angeles, California, USA
- **Qualification pathway**: Entry Standards OR World Rankings quota
  - Entry standards are very high (e.g., 100m: 10.00, 400m: 44.90, TJ: 17.14m)
  - Rankings quota: Top ~32-48 per event depending on programme (after standard qualifiers removed)
  - Qualification window: Approximately July 2027 - June 2028
- **For KSA**: Most realistic route is world rankings quota, not entry standards
- **Time zones**: Pacific Time (UTC-7), late sessions for Middle East viewers
- **Conditions**: July/August LA heat (30-40°C), low humidity, near sea level
- **Universality places**: Each NOC guaranteed minimum 1 male + 1 female athlete (any event)

## World Championships Tokyo 2025 (IMMEDIATE TARGET)
- Dates: September 13-21, 2025
- Venue: National Stadium, Tokyo, Japan
- **Entry standards** apply (e.g., 100m: 10.00, TJ: 17.00m, PV: 5.72m)
- **World Rankings route**: After standard qualifiers, remaining quota filled by ranking
- **For KSA**: Opportunity to gain ranking points and championship experience before Asian Games
- **Same timezone as Asian Games**: Useful for acclimatisation planning
"""

SYSTEM_PROMPT = """You are an ELITE world-class sports performance analyst specializing in athletics (track and field).
You work exclusively for **Team Saudi** and provide data-driven analysis for coaches and performance directors.

## Your Role
You are the AI Performance Analyst embedded in Team Saudi's analytics dashboard. Present facts, data, comparisons, and trends. Do NOT give training recommendations, coaching advice, or suggest what athletes should do differently — that is the coach's job, not yours.

## Your Expertise
1. **Performance Analysis** — Interpret times, distances, progressions against event-specific benchmarks
2. **Championship Context** — Rounds structure, advancement rates, competition depth by championship level
3. **Competitor Intelligence** — H2H comparisons, form trends, nationality-based competition depth
4. **Rankings & Qualification** — World Athletics ranking system, entry standards, quota places
5. **Asian Games & Olympic Focus** — KSA targets are Asian Games Nagoya 2026 and LA 2028 Olympics

## Response Guidelines
- **Be specific**: Always reference exact times, distances, rankings, dates. Never be vague
- **Data only, no recommendations**: Present analysis and comparisons. Do NOT suggest training changes, tactical adjustments, or coaching actions
- **Use context**: Reference the championship level (Asian Games vs Worlds vs Olympics — very different depth)
- **Flag wind conditions** for sprints and horizontal jumps (legal ≤2.0 m/s)
- **Compare to relevant benchmarks**: Asian Games medal standard, not just world records
- **Note progression trends**: Is the athlete improving? Plateaued? Declining?
- **Format with headers, bullets, and bold** for scanability
- **Never hallucinate data** — if you don't have specific data, say so
- **Know the KSA roster**: Use exact athlete names from the knowledge base for data accuracy
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

def _get_backup_api_key() -> Optional[str]:
    """Get backup OpenRouter API key."""
    key = os.getenv('OPENROUTER_API_KEY_2')
    if not key:
        try:
            if hasattr(st, 'secrets'):
                key = st.secrets.get('OPENROUTER_API_KEY_2')
        except Exception:
            pass
    return key


class ChatClient:
    """OpenRouter chat client with athletics context."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from openai import OpenAI
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self._OpenAI = OpenAI
        self._api_key = api_key
        self.model = model
        self.context_builder = ContextBuilder()

    def _try_backup_key(self):
        """Switch to backup API key on auth failure."""
        backup = _get_backup_api_key()
        if backup and backup != self._api_key:
            self.client = self._OpenAI(base_url=OPENROUTER_BASE_URL, api_key=backup)
            self._api_key = backup
            logger.info("Switched to backup OpenRouter API key")
            return True
        return False

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
            # Retry with backup key on auth failure (401)
            if "401" in str(e) and self._try_backup_key():
                try:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages,
                        temperature=0.3, max_tokens=2000, top_p=0.9,
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
                except Exception as e2:
                    logger.error(f"Backup key also failed: {e2}")
                    return f"Error: {str(e2)}", ""
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
            # Retry with backup key on auth failure (401)
            if "401" in str(e) and self._try_backup_key():
                try:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages,
                        temperature=0.3, max_tokens=2000, top_p=0.9,
                        extra_headers={
                            "HTTP-Referer": "https://team-saudi-athletics.streamlit.app",
                            "X-Title": "Team Saudi Athletics Analyst"
                        }
                    )
                    answer = response.choices[0].message.content.strip()
                    sources_text = "**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
                    return answer, sources_text
                except Exception as e2:
                    logger.error(f"Backup key also failed: {e2}")
                    return f"Error: {str(e2)}", ""
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
    """Quick view: KSA athletes vs championship standards with championship filter."""
    from data.event_utils import normalize_event_for_match, get_event_type, format_event_name

    athletes = dc.get_ksa_athletes()
    if athletes.empty:
        st.info("No KSA athlete data.")
        return

    pbs = dc.get_ksa_athlete_pbs()
    if pbs.empty:
        st.info("No PB data available.")
        return

    # Championship filter
    champ_options = [
        "All Major Championships",
        "Asian Games",
        "World Championships",
        "Olympic Games",
    ]
    selected_champ = st.selectbox(
        "Championship Level",
        champ_options,
        index=1,  # Default to Asian Games
        key="sg_champ",
        help="Standards are computed from actual championship final results",
    )

    CHAMP_TYPE_MAP = {
        "All Major Championships": None,
        "Asian Games": "Asian Games",
        "World Championships": "World Championships",
        "Olympic Games": "Olympic Games",
    }
    champ_type = CHAMP_TYPE_MAP[selected_champ]

    # Get KSA events from PBs
    disc_col = "discipline" if "discipline" in pbs.columns else "event"
    mark_col = "mark" if "mark" in pbs.columns else "result"

    # For each unique KSA event, compute championship standards from master data
    event_standards = {}
    unique_events = pbs[disc_col].dropna().unique()

    for disc in unique_events:
        display_name = format_event_name(str(disc))
        evt_type = get_event_type(str(disc))

        # Query championship finals for this event
        champ_results = dc.get_championship_results(
            event=display_name,
            gender="M",
            championship_type=champ_type,
            finals_only=True,
            limit=2000,
        )

        if champ_results.empty:
            continue

        pos_numeric = pd.to_numeric(champ_results["pos"], errors="coerce")
        champ_results = champ_results.copy()
        champ_results["_pos_num"] = pos_numeric
        finals_8 = champ_results[champ_results["_pos_num"].between(1, 8)]

        if finals_8.empty:
            continue

        gold_marks = finals_8[finals_8["_pos_num"] == 1]["result_numeric"]
        bronze_marks = finals_8[finals_8["_pos_num"].between(1, 3)]["result_numeric"]
        eighth_marks = finals_8[finals_8["_pos_num"] == 8]["result_numeric"]

        if evt_type == "time":
            gold_std = gold_marks.mean() if len(gold_marks) > 0 else None
            bronze_std = bronze_marks.max() if len(bronze_marks) > 0 else None
            eighth_std = eighth_marks.mean() if len(eighth_marks) > 0 else None
        else:
            gold_std = gold_marks.mean() if len(gold_marks) > 0 else None
            bronze_std = bronze_marks.min() if len(bronze_marks) > 0 else None
            eighth_std = eighth_marks.mean() if len(eighth_marks) > 0 else None

        norm_key = normalize_event_for_match(str(disc))
        event_standards[norm_key] = {
            "gold": gold_std,
            "bronze": bronze_std,
            "eighth": eighth_std,
            "type": evt_type,
            "n_results": len(finals_8),
        }

    # Build gap table
    gap_rows = []
    for _, pb_row in pbs.iterrows():
        discipline = str(pb_row.get(disc_col, ""))
        mark_str = str(pb_row.get(mark_col, ""))
        athlete = str(pb_row.get("full_name", pb_row.get("athlete_name", "")))
        pb_norm = normalize_event_for_match(discipline)

        stds = event_standards.get(pb_norm)
        if not stds:
            continue

        # Parse PB to numeric
        try:
            if ":" in mark_str:
                parts = mark_str.split(":")
                pb_numeric = float(parts[0]) * 60 + float(parts[1])
            else:
                pb_numeric = float(mark_str)
        except (ValueError, IndexError):
            continue

        evt_type = stds["type"]

        def _fmt(val):
            if val is None:
                return "-"
            if evt_type == "time" and val >= 60:
                mins = int(val // 60)
                secs = val - mins * 60
                return f"{mins}:{secs:05.2f}"
            return f"{val:.2f}"

        def _gap(pb, std):
            if std is None:
                return "-"
            diff = pb - std if evt_type == "time" else std - pb
            sign = "+" if diff > 0 else ""
            if evt_type == "time" and abs(diff) >= 60:
                mins = int(abs(diff) // 60)
                secs = abs(diff) - mins * 60
                return f"{sign}{mins}:{secs:05.2f}"
            return f"{sign}{diff:.2f}"

        def _status(pb, std):
            if std is None:
                return ""
            diff = pb - std if evt_type == "time" else std - pb
            if diff <= 0:
                return "Achieved"
            elif diff < (0.5 if evt_type == "time" else 0.3):
                return "Close"
            else:
                return "Gap"

        gap_rows.append({
            "Athlete": athlete,
            "Event": format_event_name(discipline),
            "PB": mark_str,
            "Gold": _fmt(stds["gold"]),
            "Gap to Gold": _gap(pb_numeric, stds["gold"]),
            "Bronze": _fmt(stds["bronze"]),
            "Gap to Bronze": _gap(pb_numeric, stds["bronze"]),
            "Final (8th)": _fmt(stds["eighth"]),
            "Status": _status(pb_numeric, stds["bronze"]),
        })

    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)

        # Color-code status
        def _style_status(val):
            if val == "Achieved":
                return f"color: {TEAL_PRIMARY}; font-weight: bold"
            elif val == "Close":
                return f"color: {GOLD_ACCENT}; font-weight: bold"
            elif val == "Gap":
                return "color: #dc3545"
            return ""

        _map_fn = gap_df.style.map if hasattr(gap_df.style, "map") else gap_df.style.applymap
        styled = _map_fn(_style_status, subset=["Status"])
        st.dataframe(styled, hide_index=True, use_container_width=True, height=450)
        st.caption(
            f"Standards computed from **{selected_champ}** final results "
            f"(positions 1st-8th). Gold = avg winner, Bronze = worst 3rd place, "
            f"Final = avg 8th place."
        )
    else:
        st.info(f"No championship final data found for KSA events at {selected_champ}.")


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

    # ── Athlete Selector (ensures correct spelling) ──
    _ksa_athletes = dc.get_ksa_athletes()
    _athlete_names = []
    if not _ksa_athletes.empty:
        _name_col = next((c for c in ['full_name', 'competitor'] if c in _ksa_athletes.columns), None)
        if _name_col:
            _athlete_names = sorted(_ksa_athletes[_name_col].dropna().unique().tolist())

    sel_col1, sel_col2 = st.columns([2, 3])
    with sel_col1:
        _selected_athlete = st.selectbox(
            "Athlete Focus",
            ["None (general query)"] + _athlete_names,
            index=0,
            key="ai_athlete_focus",
            help="Select an athlete to auto-include their name in your query for accurate data lookup",
        )
    with sel_col2:
        if _selected_athlete != "None (general query)":
            # Show quick info for selected athlete
            _ath_row = _ksa_athletes[_ksa_athletes[_name_col] == _selected_athlete].iloc[0]
            _event = _ath_row.get("primary_event", "")
            _rank = _ath_row.get("best_world_rank", "")
            _rank_str = f"World Rank #{int(_rank)}" if pd.notna(_rank) and str(_rank) != 'nan' else "Unranked"
            st.info(f"**{_selected_athlete}** | {_event} | {_rank_str}")
        else:
            st.caption("Select an athlete above to focus questions on a specific KSA athlete.")

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

    # Helper: prepend athlete name to query if selected and not already mentioned
    def _enrich_query(q: str) -> str:
        if _selected_athlete == "None (general query)":
            return q
        # Check if athlete name (or parts of it) is already in the query
        name_parts = _selected_athlete.lower().split()
        q_lower = q.lower()
        if any(part in q_lower for part in name_parts if len(part) > 2):
            return q
        return f"[About {_selected_athlete}] {q}"

    # Handle pending query from sidebar or quick actions
    if "pending_query" in st.session_state:
        query = _enrich_query(st.session_state.pending_query)
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
    _chat_placeholder = (
        f"Ask about {_selected_athlete}..."
        if _selected_athlete != "None (general query)"
        else "Ask about athletics performance, competitors, or strategy..."
    )
    if prompt := st.chat_input(_chat_placeholder):
        prompt = _enrich_query(prompt)
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
