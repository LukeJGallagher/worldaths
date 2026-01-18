"""
Championship Progression Dashboard
Year-on-year athlete progression, finals analysis, round-by-round data,
and probability modeling for making each round at major competitions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Team Saudi Brand Colors
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'
TEAL_LIGHT = '#009688'
GRAY_BLUE = '#78909C'

# Competition category hierarchy (highest to lowest)
COMPETITION_TIERS = {
    'OW': {'name': 'Olympic/World Champs', 'level': 1, 'color': '#FFD700', 'points_multiplier': 1.0},
    'GL': {'name': 'Gold Label/Diamond League', 'level': 2, 'color': '#C0C0C0', 'points_multiplier': 0.9},
    'A': {'name': 'Continental Champs', 'level': 3, 'color': '#CD7F32', 'points_multiplier': 0.8},
    'B': {'name': 'National Champs/B-Level', 'level': 4, 'color': TEAL_PRIMARY, 'points_multiplier': 0.7},
    'C': {'name': 'C-Level International', 'level': 5, 'color': TEAL_LIGHT, 'points_multiplier': 0.6},
    'D': {'name': 'Regional Champs', 'level': 6, 'color': GRAY_BLUE, 'points_multiplier': 0.5},
    'E': {'name': 'E-Level', 'level': 7, 'color': '#999', 'points_multiplier': 0.4},
    'F': {'name': 'F-Level/Local', 'level': 8, 'color': '#666', 'points_multiplier': 0.3},
}

# Round types
ROUND_TYPES = {
    'H': {'name': 'Heat', 'order': 1, 'color': GRAY_BLUE},
    'Q': {'name': 'Qualification', 'order': 1, 'color': GRAY_BLUE},
    'SF': {'name': 'Semifinal', 'order': 2, 'color': TEAL_LIGHT},
    'F': {'name': 'Final', 'order': 3, 'color': GOLD_ACCENT},
}

# Historical championship standards (actual medal marks from World/Olympic)
CHAMPIONSHIP_STANDARDS = {
    '100m': {
        'men': {
            'gold': {'2024': 9.79, '2023': 9.83, '2022': 9.86, '2019': 9.76},
            'finals': {'2024': 10.00, '2023': 10.02, '2022': 10.05, '2019': 10.01},
            'semis': {'2024': 10.08, '2023': 10.10, '2022': 10.12, '2019': 10.08},
        }
    },
    '200m': {
        'men': {
            'gold': {'2024': 19.70, '2023': 19.52, '2022': 19.73, '2019': 19.83},
            'finals': {'2024': 20.10, '2023': 20.12, '2022': 20.25, '2019': 20.18},
            'semis': {'2024': 20.30, '2023': 20.35, '2022': 20.45, '2019': 20.40},
        }
    },
    '400m': {
        'men': {
            'gold': {'2024': 43.65, '2023': 44.29, '2022': 43.71, '2019': 43.48},
            'finals': {'2024': 44.80, '2023': 44.95, '2022': 45.10, '2019': 44.88},
            'semis': {'2024': 45.30, '2023': 45.50, '2022': 45.60, '2019': 45.40},
        }
    },
    '800m': {
        'men': {
            'gold': {'2024': 102.5, '2023': 101.56, '2022': 103.71, '2019': 102.17},
            'finals': {'2024': 104.5, '2023': 104.80, '2022': 105.20, '2019': 104.50},
            'semis': {'2024': 106.0, '2023': 106.20, '2022': 106.50, '2019': 106.00},
        }
    },
    '1500m': {
        'men': {
            'gold': {'2024': 206.0, '2023': 209.38, '2022': 210.69, '2019': 211.70},
            'finals': {'2024': 213.0, '2023': 214.50, '2022': 215.00, '2019': 215.50},
            'semis': {'2024': 218.0, '2023': 219.00, '2022': 220.00, '2019': 220.00},
        }
    },
    'pole-vault': {
        'men': {
            'gold': {'2024': 6.00, '2023': 6.00, '2022': 6.00, '2019': 5.97},
            'finals': {'2024': 5.70, '2023': 5.65, '2022': 5.60, '2019': 5.60},
            'qualification': {'2024': 5.60, '2023': 5.55, '2022': 5.50, '2019': 5.50},
        }
    },
    'shot-put': {
        'men': {
            'gold': {'2024': 22.90, '2023': 23.23, '2022': 22.94, '2019': 22.91},
            'finals': {'2024': 20.80, '2023': 20.70, '2022': 20.50, '2019': 20.40},
            'qualification': {'2024': 20.50, '2023': 20.40, '2022': 20.20, '2019': 20.20},
        }
    },
    'triple-jump': {
        'men': {
            'gold': {'2024': 17.86, '2023': 17.85, '2022': 17.98, '2019': 17.92},
            'finals': {'2024': 16.80, '2023': 16.75, '2022': 16.70, '2019': 16.70},
            'qualification': {'2024': 16.60, '2023': 16.55, '2022': 16.50, '2019': 16.50},
        }
    },
    '400mh': {
        'men': {
            'gold': {'2024': 46.00, '2023': 45.00, '2022': 46.29, '2019': 47.42},
            'finals': {'2024': 48.50, '2023': 48.60, '2022': 48.80, '2019': 49.20},
            'semis': {'2024': 49.50, '2023': 49.60, '2022': 49.80, '2019': 50.20},
        }
    },
    'javelin-throw': {
        'men': {
            'gold': {'2024': 88.00, '2023': 88.88, '2022': 90.54, '2019': 86.89},
            'finals': {'2024': 80.00, '2023': 79.00, '2022': 78.00, '2019': 78.00},
            'qualification': {'2024': 78.00, '2023': 77.00, '2022': 76.00, '2019': 76.00},
        }
    },
    'hammer-throw': {
        'men': {
            'gold': {'2024': 84.12, '2023': 81.25, '2022': 81.98, '2019': 80.54},
            'finals': {'2024': 75.00, '2023': 74.00, '2022': 73.00, '2019': 73.00},
            'qualification': {'2024': 74.00, '2023': 73.00, '2022': 72.00, '2019': 72.00},
        }
    },
}

SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')


class ChampionshipAnalyzer:
    """Analyze championship progression and round-by-round performance."""

    def __init__(self):
        self.modal_results_path = os.path.join(DATA_DIR, 'ksa_modal_results_men.csv')
        self.profiles_db = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')
        self.wittw_db = os.path.join(SQL_DIR, 'what_it_takes_to_win.db')
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Load championship results data."""
        if os.path.exists(self.modal_results_path):
            self.data = pd.read_csv(self.modal_results_path)
            # Parse date
            self.data['ParsedDate'] = pd.to_datetime(self.data['Date'], format='%d %b %Y', errors='coerce')
            self.data['Year'] = self.data['ParsedDate'].dt.year
            return self.data
        return pd.DataFrame()

    def get_athletes(self) -> List[str]:
        """Get list of unique athletes."""
        if self.data is None:
            self.load_data()
        if self.data is not None and 'Athlete' in self.data.columns:
            return sorted(self.data['Athlete'].unique().tolist())
        return []

    def get_events(self) -> List[str]:
        """Get list of unique events."""
        if self.data is None:
            self.load_data()
        if self.data is not None and 'Event Type' in self.data.columns:
            return sorted(self.data['Event Type'].unique().tolist())
        return []

    def get_athlete_progression(self, athlete_name: str) -> pd.DataFrame:
        """Get year-on-year progression for an athlete."""
        if self.data is None:
            self.load_data()

        athlete_data = self.data[self.data['Athlete'] == athlete_name].copy()
        if athlete_data.empty:
            return pd.DataFrame()

        # Get best performance per event per year
        progression = athlete_data.groupby(['Year', 'Event Type']).agg({
            'Result': 'first',  # Best result
            'R.Sc': 'max',  # Best score
            'Type': lambda x: 'F' if 'F' in x.values else ('SF' if 'SF' in x.values else 'H'),
            'Competition': 'first',
            'Cat.': 'first'
        }).reset_index()

        return progression.sort_values(['Event Type', 'Year'])

    def get_round_analysis(self, athlete_name: str = None, event: str = None) -> pd.DataFrame:
        """Analyze round-by-round progression."""
        if self.data is None:
            self.load_data()

        df = self.data.copy()

        if athlete_name:
            df = df[df['Athlete'] == athlete_name]
        if event:
            df = df[df['Event Type'] == event]

        if df.empty:
            return pd.DataFrame()

        # Group by competition to see round progression
        round_analysis = []
        for comp in df['Competition'].unique():
            comp_data = df[df['Competition'] == comp]
            for athlete in comp_data['Athlete'].unique():
                athlete_comp = comp_data[comp_data['Athlete'] == athlete]

                heat_result = athlete_comp[athlete_comp['Type'] == 'H']['Result'].values
                semi_result = athlete_comp[athlete_comp['Type'] == 'SF']['Result'].values
                final_result = athlete_comp[athlete_comp['Type'] == 'F']['Result'].values
                final_place = athlete_comp[athlete_comp['Type'] == 'F']['Pl.'].values

                round_analysis.append({
                    'Athlete': athlete,
                    'Event': athlete_comp['Event Type'].iloc[0],
                    'Competition': comp,
                    'Category': athlete_comp['Cat.'].iloc[0],
                    'Heat': heat_result[0] if len(heat_result) > 0 else None,
                    'Semi': semi_result[0] if len(semi_result) > 0 else None,
                    'Final': final_result[0] if len(final_result) > 0 else None,
                    'Final_Place': final_place[0] if len(final_place) > 0 else None,
                    'Made_Final': len(final_result) > 0,
                    'Made_Semi': len(semi_result) > 0 or len(final_result) > 0,
                    'Year': athlete_comp['Year'].iloc[0] if 'Year' in athlete_comp.columns else None
                })

        return pd.DataFrame(round_analysis)

    def get_finals_history(self, event: str = None) -> pd.DataFrame:
        """Get history of finals appearances."""
        if self.data is None:
            self.load_data()

        finals = self.data[self.data['Type'] == 'F'].copy()

        if event:
            finals = finals[finals['Event Type'] == event]

        return finals.sort_values(['Year', 'Cat.', 'Pl.'])

    def calculate_medal_probability(self, athlete_pb: float, event: str,
                                     competition_level: str = 'A') -> Dict:
        """Calculate probability of making each round based on PB."""
        event_key = event.lower().replace(' ', '-').replace('metres', 'm')

        # Map common event names
        event_mapping = {
            '100m': '100m', '200m': '200m', '400m': '400m',
            '800m': '800m', '1500m': '1500m',
            'pole vault': 'pole-vault', 'pole-vault': 'pole-vault',
            'shot put': 'shot-put', 'shot-put': 'shot-put',
            'triple jump': 'triple-jump', 'triple-jump': 'triple-jump',
            '400m hurdles': '400mh', '400mh': '400mh',
            'javelin throw': 'javelin-throw', 'javelin-throw': 'javelin-throw',
            'hammer throw': 'hammer-throw', 'hammer-throw': 'hammer-throw',
        }

        mapped_event = None
        for key, value in event_mapping.items():
            if key in event_key:
                mapped_event = value
                break

        if not mapped_event or mapped_event not in CHAMPIONSHIP_STANDARDS:
            return {'error': f'No standards available for {event}'}

        standards = CHAMPIONSHIP_STANDARDS[mapped_event]['men']
        is_field = any(x in event.lower() for x in ['jump', 'vault', 'put', 'throw', 'hammer', 'javelin'])

        result = {
            'event': event,
            'athlete_pb': athlete_pb,
            'is_field_event': is_field
        }

        # Calculate probabilities based on how close PB is to standards
        for round_type in ['gold', 'finals', 'semis', 'qualification']:
            if round_type in standards:
                standard = standards[round_type].get('2024') or standards[round_type].get('2023')
                if standard:
                    if is_field:
                        gap = standard - athlete_pb
                        probability = max(0, min(100, 100 - (gap / standard * 200)))
                    else:
                        gap = athlete_pb - standard
                        probability = max(0, min(100, 100 - (gap / standard * 200)))

                    result[f'{round_type}_standard'] = standard
                    result[f'{round_type}_gap'] = gap
                    result[f'{round_type}_probability'] = round(probability, 1)

        return result

    def get_easy_points_opportunities(self) -> pd.DataFrame:
        """Identify events/competitions where KSA can score easy points."""
        if self.data is None:
            self.load_data()

        # Analyze historical medal/top 8 performances
        opportunities = []

        for event in self.get_events():
            event_data = self.data[self.data['Event Type'] == event]
            finals_data = event_data[event_data['Type'] == 'F']

            if finals_data.empty:
                continue

            # Get KSA best performers
            best_performers = finals_data.groupby('Athlete').agg({
                'R.Sc': 'max',
                'Pl.': lambda x: x.astype(str).str.replace('.', '').str.extract(r'(\d+)')[0].astype(float).min(),
                'Cat.': 'first'
            }).reset_index()

            for _, row in best_performers.iterrows():
                # Calculate opportunity score
                place = row['Pl.']
                if pd.notna(place) and place <= 8:
                    opportunity_score = (9 - place) * 10  # Higher score for better places
                    opportunities.append({
                        'Event': event,
                        'Athlete': row['Athlete'],
                        'Best_Place': int(place),
                        'Best_Score': row['R.Sc'],
                        'Competition_Level': row['Cat.'],
                        'Opportunity_Score': opportunity_score,
                        'Assessment': 'Medal Contender' if place <= 3 else 'Finalist' if place <= 8 else 'Development'
                    })

        return pd.DataFrame(opportunities).sort_values('Opportunity_Score', ascending=False)


# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

st.set_page_config(
    page_title="KSA Championship Progression",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styling
st.markdown(f"""
<style>
.stApp {{
    background-color: #0a0a0a !important;
}}
.block-container {{
    background-color: rgba(0, 0, 0, 0.8) !important;
    padding: 2rem;
    border-radius: 12px;
}}
h1, h2, h3 {{
    color: {TEAL_PRIMARY} !important;
}}
.medal-gold {{
    background: linear-gradient(135deg, #FFD700 0%, #B8860B 100%);
    color: #333;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}}
.medal-silver {{
    background: linear-gradient(135deg, #C0C0C0 0%, #A8A8A8 100%);
    color: #333;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}}
.medal-bronze {{
    background: linear-gradient(135deg, #CD7F32 0%, #A0522D 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}}
.finals-card {{
    background: rgba(0, 113, 103, 0.2);
    border: 2px solid {TEAL_PRIMARY};
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}}
.probability-high {{
    color: #00ff00;
    font-weight: bold;
}}
.probability-medium {{
    color: #ffff00;
    font-weight: bold;
}}
.probability-low {{
    color: #ff6600;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
            padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: white !important; margin: 0;">Championship Progression Dashboard</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
        Year-on-year athlete progression | Finals history | Round-by-round analysis | Medal probability
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    analyzer = ChampionshipAnalyzer()
    analyzer.load_data()
    return analyzer

analyzer = get_analyzer()

# Sidebar filters
st.sidebar.markdown(f"""
<div style="background: {TEAL_PRIMARY}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <h3 style="color: white; margin: 0;">Filters</h3>
</div>
""", unsafe_allow_html=True)

athletes = analyzer.get_athletes()
events = analyzer.get_events()

selected_athlete = st.sidebar.selectbox("Select Athlete", ["All Athletes"] + athletes)
selected_event = st.sidebar.selectbox("Select Event", ["All Events"] + events)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Year-on-Year Progression",
    "🏅 Finals History",
    "🔄 Round Analysis",
    "📊 Medal Probability",
    "🎯 Easy Points Opportunities"
])

# Tab 1: Year-on-Year Progression
with tab1:
    st.header("Year-on-Year Performance Progression")

    if selected_athlete != "All Athletes":
        progression = analyzer.get_athlete_progression(selected_athlete)

        if not progression.empty:
            st.subheader(f"{selected_athlete} - Performance Progression")

            # Filter by event if selected
            if selected_event != "All Events":
                progression = progression[progression['Event Type'] == selected_event]

            for event in progression['Event Type'].unique():
                event_prog = progression[progression['Event Type'] == event]

                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=event_prog['Year'],
                        y=event_prog['R.Sc'],
                        mode='lines+markers',
                        name='Performance Score',
                        line=dict(color=TEAL_PRIMARY, width=3),
                        marker=dict(size=12)
                    ))

                    fig.update_layout(
                        title=f"{event} - Score Progression",
                        xaxis_title="Year",
                        yaxis_title="World Athletics Score",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown(f"**{event} Results:**")
                    for _, row in event_prog.iterrows():
                        round_icon = "🥇" if row['Type'] == 'F' else "🔵" if row['Type'] == 'SF' else "⚪"
                        st.markdown(f"{round_icon} **{int(row['Year'])}**: {row['Result']} ({row['Cat.']}-level)")

        else:
            st.info("No progression data available for this athlete.")
    else:
        # Show all athletes progression summary
        st.subheader("All Athletes - Performance Trends")
        all_data = analyzer.data

        if all_data is not None and not all_data.empty:
            # Group by year and show improvement trends
            yearly_summary = all_data.groupby(['Year', 'Event Type']).agg({
                'R.Sc': 'max',
                'Athlete': 'nunique'
            }).reset_index()

            fig = px.bar(yearly_summary, x='Year', y='R.Sc', color='Event Type',
                        title="Best Scores by Event per Year",
                        barmode='group')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Finals History
with tab2:
    st.header("Finals Appearances & Results")

    finals = analyzer.get_finals_history(selected_event if selected_event != "All Events" else None)

    if not finals.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_finals = len(finals)
            st.metric("Total Finals", total_finals)

        with col2:
            medals = len(finals[finals['Pl.'].astype(str).str.contains('1|2|3', regex=True, na=False)])
            st.metric("Medal Finishes", medals)

        with col3:
            top_8 = len(finals[finals['Pl.'].astype(str).str.extract(r'(\d+)')[0].astype(float) <= 8])
            st.metric("Top 8 Finishes", top_8)

        with col4:
            unique_athletes = finals['Athlete'].nunique()
            st.metric("Athletes in Finals", unique_athletes)

        st.markdown("---")

        # Competition category breakdown
        st.subheader("Finals by Competition Level")

        cat_summary = finals.groupby('Cat.').agg({
            'Athlete': 'count',
            'Pl.': lambda x: (x.astype(str).str.extract(r'(\d+)')[0].astype(float) <= 3).sum()
        }).reset_index()
        cat_summary.columns = ['Category', 'Finals', 'Medals']

        # Add category descriptions
        cat_summary['Level'] = cat_summary['Category'].map(
            lambda x: COMPETITION_TIERS.get(x, {}).get('name', x)
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(cat_summary, x='Category', y='Finals',
                        title="Finals Appearances by Level",
                        color='Category',
                        color_discrete_map={k: v['color'] for k, v in COMPETITION_TIERS.items()})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Competition Levels:**")
            for cat, info in COMPETITION_TIERS.items():
                count = len(finals[finals['Cat.'] == cat])
                if count > 0:
                    st.markdown(f"<span style='color:{info['color']}'>●</span> **{cat}** - {info['name']}: {count} finals",
                               unsafe_allow_html=True)

        # Detailed finals table
        st.subheader("Detailed Finals Results")
        display_finals = finals[['Athlete', 'Event Type', 'Competition', 'Cat.', 'Pl.', 'Result', 'R.Sc', 'Year']].copy()
        display_finals = display_finals.sort_values(['Year', 'Cat.'], ascending=[False, True])
        st.dataframe(display_finals, use_container_width=True, hide_index=True)

    else:
        st.info("No finals data available.")

# Tab 3: Round Analysis
with tab3:
    st.header("Round-by-Round Progression")

    round_data = analyzer.get_round_analysis(
        athlete_name=selected_athlete if selected_athlete != "All Athletes" else None,
        event=selected_event if selected_event != "All Events" else None
    )

    if not round_data.empty:
        # Summary
        col1, col2, col3 = st.columns(3)

        with col1:
            made_final_pct = (round_data['Made_Final'].sum() / len(round_data)) * 100
            st.metric("Made Finals", f"{made_final_pct:.0f}%")

        with col2:
            made_semi_pct = (round_data['Made_Semi'].sum() / len(round_data)) * 100
            st.metric("Made Semis+", f"{made_semi_pct:.0f}%")

        with col3:
            total_comps = len(round_data)
            st.metric("Total Competitions", total_comps)

        st.markdown("---")

        # Round progression visualization
        st.subheader("Competition Round Progression")

        for _, row in round_data.iterrows():
            if row['Made_Final']:
                place_str = str(row['Final_Place']).replace('.', '') if pd.notna(row['Final_Place']) else '?'
                medal_class = 'medal-gold' if place_str in ['1', '1'] else ('medal-silver' if place_str == '2' else ('medal-bronze' if place_str == '3' else 'finals-card'))

                st.markdown(f"""
                <div class="{medal_class}">
                    <strong>{row['Athlete']}</strong> - {row['Event']} @ {row['Competition'][:50]}...<br>
                    Heat: {row['Heat'] or '-'} → Semi: {row['Semi'] or '-'} → <strong>Final: {row['Final']} (Place: {place_str})</strong>
                </div>
                """, unsafe_allow_html=True)

        # Conversion rates
        st.subheader("Conversion Rates")

        conversion_data = {
            'Stage': ['Heat→Semi', 'Semi→Final', 'Final→Medal', 'Final→Top 8'],
            'Rate': [
                round_data['Made_Semi'].mean() * 100,
                round_data[round_data['Made_Semi']]['Made_Final'].mean() * 100 if round_data['Made_Semi'].sum() > 0 else 0,
                len(round_data[round_data['Final_Place'].astype(str).str.extract(r'(\d+)')[0].astype(float) <= 3]) / max(1, round_data['Made_Final'].sum()) * 100,
                len(round_data[round_data['Final_Place'].astype(str).str.extract(r'(\d+)')[0].astype(float) <= 8]) / max(1, round_data['Made_Final'].sum()) * 100
            ]
        }

        fig = go.Figure(go.Bar(
            x=conversion_data['Stage'],
            y=conversion_data['Rate'],
            marker_color=[GRAY_BLUE, TEAL_LIGHT, GOLD_ACCENT, TEAL_PRIMARY]
        ))
        fig.update_layout(
            title="Round Progression Conversion Rates",
            yaxis_title="Conversion Rate (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No round analysis data available.")

# Tab 4: Medal Probability
with tab4:
    st.header("Medal & Qualification Probability")

    st.markdown("""
    Calculate the probability of making each round at major championships based on current personal bests
    compared to historical championship standards.
    """)

    # Get athlete PBs
    if os.path.exists(analyzer.profiles_db):
        conn = sqlite3.connect(analyzer.profiles_db)
        pbs = pd.read_sql("""
            SELECT a.full_name, p.event_name, p.pb_result
            FROM ksa_athletes a
            JOIN athlete_pbs p ON a.athlete_id = p.athlete_id
        """, conn)
        conn.close()

        if not pbs.empty:
            st.subheader("KSA Athletes - Championship Probability")

            for _, row in pbs.iterrows():
                athlete = row['full_name']
                event = row['event_name']
                pb = row['pb_result']

                # Parse PB
                try:
                    if ':' in str(pb):
                        parts = str(pb).split(':')
                        pb_val = float(parts[0]) * 60 + float(parts[1])
                    else:
                        pb_val = float(str(pb).replace('m', '').replace('pts', ''))
                except:
                    continue

                prob = analyzer.calculate_medal_probability(pb_val, event)

                if 'error' not in prob:
                    with st.expander(f"**{athlete}** - {event} (PB: {pb})"):
                        col1, col2, col3, col4 = st.columns(4)

                        for i, (round_type, col) in enumerate(zip(['gold', 'finals', 'semis'], [col1, col2, col3])):
                            if f'{round_type}_probability' in prob:
                                p = prob[f'{round_type}_probability']
                                standard = prob.get(f'{round_type}_standard', 'N/A')
                                gap = prob.get(f'{round_type}_gap', 0)

                                color_class = 'probability-high' if p > 60 else ('probability-medium' if p > 30 else 'probability-low')

                                with col:
                                    st.markdown(f"""
                                    <div style="background: rgba(0,0,0,0.5); padding: 1rem; border-radius: 8px; text-align: center;">
                                        <p style="color: #aaa; margin: 0;">{round_type.title()}</p>
                                        <p class="{color_class}" style="font-size: 1.5rem; margin: 0.5rem 0;">{p}%</p>
                                        <p style="color: #888; font-size: 0.8rem; margin: 0;">Standard: {standard}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

    else:
        st.warning("Athlete profiles database not found.")

# Tab 5: Easy Points Opportunities
with tab5:
    st.header("Strategic Points Opportunities")

    st.markdown("""
    Events and competitions where KSA athletes have historically performed well
    and have strong potential for medals/top 8 finishes.
    """)

    opportunities = analyzer.get_easy_points_opportunities()

    if not opportunities.empty:
        # Top opportunities
        st.subheader("Best Medal/Points Opportunities")

        medal_opps = opportunities[opportunities['Assessment'] == 'Medal Contender']
        finalist_opps = opportunities[opportunities['Assessment'] == 'Finalist']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="medal-gold">
                <h3 style="margin: 0;">🥇 Medal Contenders</h3>
            </div>
            """, unsafe_allow_html=True)

            if not medal_opps.empty:
                for _, row in medal_opps.iterrows():
                    st.markdown(f"""
                    <div class="finals-card">
                        <strong>{row['Athlete']}</strong><br>
                        {row['Event']} - Best: {row['Best_Place']}{'st' if row['Best_Place']==1 else 'nd' if row['Best_Place']==2 else 'rd' if row['Best_Place']==3 else 'th'} place<br>
                        <span style="color: {GOLD_ACCENT};">Level: {row['Competition_Level']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No current medal contenders identified")

        with col2:
            st.markdown(f"""
            <div style="background: {TEAL_PRIMARY}; padding: 1rem; border-radius: 8px; text-align: center;">
                <h3 style="color: white; margin: 0;">🎯 Finalists</h3>
            </div>
            """, unsafe_allow_html=True)

            if not finalist_opps.empty:
                for _, row in finalist_opps.head(10).iterrows():
                    st.markdown(f"""
                    <div class="finals-card">
                        <strong>{row['Athlete']}</strong><br>
                        {row['Event']} - Best: {row['Best_Place']}th place<br>
                        <span style="color: {TEAL_LIGHT};">Level: {row['Competition_Level']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No finalists identified")

        # Full opportunities table
        st.subheader("Full Opportunity Analysis")
        st.dataframe(opportunities, use_container_width=True, hide_index=True)

    else:
        st.info("No opportunities data available.")

# Footer
st.markdown(f"""
<hr style='margin-top: 30px; border: 1px solid #333;'>
<div style='text-align: center; color: #666; font-size: 0.85rem; padding: 1rem 0;'>
    Championship Progression Dashboard — <strong style="color: {TEAL_PRIMARY};">Team Saudi</strong>
</div>
""", unsafe_allow_html=True)
