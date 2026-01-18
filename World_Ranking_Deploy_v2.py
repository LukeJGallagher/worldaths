import streamlit as st
import pandas as pd
import sqlite3
import datetime
import os
import plotly.express as px
import plotly.graph_objects as go

###################################
# Team Saudi Brand Colors
###################################
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'
TEAL_LIGHT = '#009688'
GRAY_BLUE = '#78909C'

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
    h1, h2, h3, h4, h5, h6 {{
        color: {TEAL_PRIMARY} !important;
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
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_team_saudi_theme()

###################################
# 3) Database Paths
###################################
SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')

DB_MEN_RANK = os.path.join(SQL_DIR, 'rankings_men_all_events.db')
DB_WOMEN_RANK = os.path.join(SQL_DIR, 'rankings_women_all_events.db')
DB_KSA_MEN = os.path.join(SQL_DIR, 'ksa_modal_results_men.db')
DB_KSA_WOMEN = os.path.join(SQL_DIR, 'ksa_modal_results_women.db')
DB_PROFILES = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')

###################################
# 4) Data Loading Functions
###################################
@st.cache_data
def load_sqlite_table(db_path, table_name):
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_athlete_profiles():
    """Load all athlete profile data."""
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
        st.error(f"Error loading profiles: {e}")
        return None, None, None, None, None

###################################
# 5) Header
###################################
st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: white !important; margin: 0; font-size: 2rem;">Saudi Athletics Dashboard</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">World Rankings & Athlete Performance Analysis</p>
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
men_rankings = load_sqlite_table(DB_MEN_RANK, 'rankings_men_all_events')
women_rankings = load_sqlite_table(DB_WOMEN_RANK, 'rankings_women_all_events')

try:
    ksa_men_results = load_sqlite_table(DB_KSA_MEN, 'ksa_modal_results_men')
except:
    ksa_men_results = None

try:
    ksa_women_results = load_sqlite_table(DB_KSA_WOMEN, 'ksa_modal_results_women')
except:
    ksa_women_results = None

# Load athlete profiles
athletes_df, rankings_df, breakdown_df, pbs_df, progression_df = load_athlete_profiles()

###################################
# 7) Tabs
###################################
tab1, tab2, tab3, tab4 = st.tabs([
    'Athlete Profiles',
    'Combined Rankings',
    'Saudi Athletes Rankings',
    'Saudi Modal Results'
])

###################################
# Tab 1: Athlete Profiles (NEW)
###################################
with tab1:
    st.header('KSA Athlete Profiles')

    if athletes_df is None or athletes_df.empty:
        st.warning("No athlete profiles found. Run the scraper first: `python scrape_ksa_athlete_profiles.py`")
    else:
        # Sidebar filters
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            athlete_names = ["All Athletes"] + sorted(athletes_df['full_name'].dropna().unique().tolist())
            selected_athlete = st.selectbox("Select Athlete", athlete_names, key="profile_athlete")

        with col_filter2:
            gender_options = ["All", "Men", "Women"]
            selected_gender = st.selectbox("Gender", gender_options, key="profile_gender")

        # Filter athletes
        filtered_athletes = athletes_df.copy()
        if selected_gender != "All":
            filtered_athletes = filtered_athletes[filtered_athletes['gender'] == selected_gender.lower()]

        if selected_athlete != "All Athletes":
            filtered_athletes = filtered_athletes[filtered_athletes['full_name'] == selected_athlete]

            # Show single athlete detail view
            if len(filtered_athletes) == 1:
                athlete = filtered_athletes.iloc[0]
                athlete_id = athlete['athlete_id']

                # Profile Card
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

                # Rankings Section
                st.subheader("Current WPA Rankings")

                if rankings_df is not None and not rankings_df.empty:
                    athlete_rankings = rankings_df[rankings_df['athlete_id'] == athlete_id]

                    if not athlete_rankings.empty:
                        # Display as metric cards
                        cols = st.columns(min(len(athlete_rankings), 4))
                        for i, (_, rank) in enumerate(athlete_rankings.iterrows()):
                            with cols[i % 4]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <p style="color: {GOLD_ACCENT}; font-size: 0.9rem; margin: 0;">{rank['event_name']}</p>
                                    <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0.25rem 0;">
                                        #{int(rank['world_rank']) if pd.notna(rank['world_rank']) else 'N/A'}
                                    </p>
                                    <p style="color: #aaa; font-size: 0.85rem; margin: 0;">
                                        Score: {rank.get('ranking_score', 'N/A')}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No ranking data available for this athlete.")
                else:
                    st.info("No ranking data available.")

                # Ranking Breakdown
                st.subheader("Ranking Score Breakdown")

                if breakdown_df is not None and not breakdown_df.empty:
                    athlete_breakdown = breakdown_df[breakdown_df['athlete_id'] == athlete_id]

                    if not athlete_breakdown.empty:
                        # Format for display
                        display_breakdown = athlete_breakdown[[
                            'competition_date', 'competition_name', 'event_name',
                            'result_value', 'result_score', 'placing', 'place_score'
                        ]].copy()
                        display_breakdown.columns = ['Date', 'Competition', 'Event', 'Result', 'Result Score', 'Place', 'Place Score']

                        st.dataframe(display_breakdown.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)
                    else:
                        st.info("No ranking breakdown data available.")

                # Personal Bests
                st.subheader("Personal Bests")

                if pbs_df is not None and not pbs_df.empty:
                    athlete_pbs = pbs_df[pbs_df['athlete_id'] == athlete_id]

                    if not athlete_pbs.empty:
                        display_pbs = athlete_pbs[['event_name', 'pb_result', 'pb_date', 'pb_venue']].copy()
                        display_pbs.columns = ['Event', 'PB', 'Date', 'Venue']
                        st.dataframe(display_pbs, use_container_width=True, hide_index=True)
                    else:
                        st.info("No PB data available.")

                # Progression Chart
                st.subheader("Year-by-Year Progression")

                if progression_df is not None and not progression_df.empty:
                    athlete_prog = progression_df[progression_df['athlete_id'] == athlete_id]

                    if not athlete_prog.empty and 'year' in athlete_prog.columns:
                        # Get unique events for this athlete
                        prog_events = athlete_prog['event_name'].unique()

                        if len(prog_events) > 0:
                            selected_prog_event = st.selectbox("Select Event", prog_events, key="prog_event")
                            event_prog = athlete_prog[athlete_prog['event_name'] == selected_prog_event].sort_values('year')

                            if not event_prog.empty:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=event_prog['year'],
                                    y=event_prog['best_result'],
                                    mode='lines+markers',
                                    name='Outdoor',
                                    line=dict(color=TEAL_PRIMARY, width=3),
                                    marker=dict(size=10)
                                ))

                                if 'indoor_best' in event_prog.columns:
                                    indoor_data = event_prog[event_prog['indoor_best'].notna()]
                                    if not indoor_data.empty:
                                        fig.add_trace(go.Scatter(
                                            x=indoor_data['year'],
                                            y=indoor_data['indoor_best'],
                                            mode='lines+markers',
                                            name='Indoor',
                                            line=dict(color=GOLD_ACCENT, width=2, dash='dash'),
                                            marker=dict(size=8)
                                        ))

                                fig.update_layout(
                                    title=f"{selected_prog_event} Progression",
                                    xaxis_title="Year",
                                    yaxis_title="Result",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No progression data available.")

        else:
            # Show all athletes grid
            st.markdown(f"**Showing {len(filtered_athletes)} athletes**")

            # Display as cards grid
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
                                        padding: 1rem; margin-bottom: 1rem; min-height: 120px;">
                                <h4 style="margin: 0; color: {TEAL_PRIMARY} !important;">{athlete['full_name']}</h4>
                                <p style="color: #aaa; margin: 0.25rem 0; font-size: 0.9rem;">
                                    {athlete.get('primary_event', '')} | {athlete.get('gender', '').capitalize()}
                                </p>
                                <p style="color: #888; margin: 0; font-size: 0.8rem;">
                                    Status: {athlete.get('status', 'active').capitalize()}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

###################################
# Tab 2: Combined Rankings
###################################
with tab2:
    st.header('Combined Rankings')
    gender = st.selectbox('Select Gender', ['All', 'Men', 'Women'], index=1, key="combined_gender")

    if gender == 'Men':
        data = men_rankings.copy()
    elif gender == 'Women':
        data = women_rankings.copy()
    else:
        data = pd.concat([men_rankings, women_rankings])

    if not data.empty:
        events = sorted(data['Event Type'].dropna().unique())
        selected_event = st.selectbox("Select Event", options=events, key="combined_event_select")
        data = data[data['Event Type'] == selected_event]

        countries = sorted(data['Country'].dropna().unique())
        selected_country = st.multiselect("Select Country", options=countries, key="combined_country")
        if selected_country:
            data = data[data['Country'].isin(selected_country)]

        names = sorted(data['Name'].dropna().unique())
        selected_name = st.multiselect("Select Athlete", options=names, key="combined_name")
        if selected_name:
            data = data[data['Name'].isin(selected_name)]

        if not data.empty and 'Rank' in data.columns and 'Score' in data.columns:
            min_rank, max_rank = int(data['Rank'].min()), int(data['Rank'].max())
            min_score, max_score = int(data['Score'].min()), int(data['Score'].max())

            selected_rank = st.slider("Select Rank Range", min_rank, max_rank, (min_rank, max_rank))
            selected_score = st.slider("Select Score Range", min_score, max_score, (min_score, max_score))

            filtered = data[(data['Rank'].between(*selected_rank)) & (data['Score'].between(*selected_score))]
            filtered = filtered.drop_duplicates()
            st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

###################################
# Tab 3: Saudi Athletes Rankings
###################################
with tab3:
    st.header('Saudi Athletes Rankings')
    st.markdown(f"""
    <p style='color: #ccc; font-size: 0.9em;'>
    Note: Rankings are based on <strong style="color: {GOLD_ACCENT};">3 athletes per country per event</strong>.
    </p>
    """, unsafe_allow_html=True)

    saudi_men = men_rankings[men_rankings['Country'].str.upper().str.contains('KSA', na=False)]
    saudi_women = women_rankings[women_rankings['Country'].str.upper().str.contains('KSA', na=False)]
    saudi_combined = pd.concat([saudi_men, saudi_women])

    if not saudi_combined.empty:
        saudi_events = sorted(saudi_combined['Event Type'].dropna().unique())
        selected_event_saudi = st.selectbox("Select Event", options=["All"] + saudi_events, key="ksa_event_key")
        if selected_event_saudi != "All":
            saudi_combined = saudi_combined[saudi_combined['Event Type'] == selected_event_saudi]

        saudi_names = sorted(saudi_combined['Name'].dropna().unique())
        selected_name_saudi = st.selectbox("Select Athlete", options=["All"] + saudi_names, key="ksa_name_key")
        if selected_name_saudi != "All":
            saudi_combined = saudi_combined[saudi_combined['Name'] == selected_name_saudi]

        saudi_combined = saudi_combined.drop_duplicates()
        st.dataframe(saudi_combined.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("No Saudi athletes found in rankings.")

###################################
# Tab 4: Saudi Modal Results
###################################
with tab4:
    st.header('Saudi Modal Results')

    if ksa_men_results is not None and not ksa_men_results.empty:
        data = ksa_men_results.copy()
        gender_loaded = "Men"
    elif ksa_women_results is not None and not ksa_women_results.empty:
        data = ksa_women_results.copy()
        gender_loaded = "Women"
    else:
        data = pd.DataFrame()
        gender_loaded = None

    if not data.empty:
        data = data.drop_duplicates()
        st.markdown(f"<p style='color: #ccc;'>Showing data for: <strong style='color: {GOLD_ACCENT};'>{gender_loaded}</strong></p>", unsafe_allow_html=True)

        modal_events = sorted(data['Event Type'].dropna().unique())
        selected_event_modal = st.selectbox("Select Event", options=["All"] + modal_events, key="modal_event_key")
        data_event_filtered = data if selected_event_modal == "All" else data[data['Event Type'] == selected_event_modal]

        modal_names = sorted(data_event_filtered['Athlete'].dropna().unique())
        selected_name_modal = st.selectbox("Select Athlete", options=["All"] + modal_names, key="modal_name_key")
        if selected_name_modal != "All":
            data_event_filtered = data_event_filtered[data_event_filtered['Athlete'] == selected_name_modal]

        st.dataframe(data_event_filtered.reset_index(drop=True), use_container_width=True)
    else:
        st.warning('No modal results available.')

###################################
# Footer
###################################
st.markdown(f"""
    <hr style='margin-top: 30px; border: 1px solid #333;'>
    <div style='text-align: center; color: #666; font-size: 0.85rem; padding: 1rem 0;'>
        Saudi Athletics Dashboard — Created by <strong style="color: {TEAL_PRIMARY};">Luke Gallagher</strong> | Team Saudi
    </div>
""", unsafe_allow_html=True)
