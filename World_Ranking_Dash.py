import streamlit as st
import pandas as pd
import base64
import sqlite3
import datetime
import os

###################################
# 1) Streamlit Setup
###################################
st.set_page_config(
    page_title="Saudi Athletics Qualification Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

###################################
# 2) Dark Background and Title
###################################
def set_background_from_url(url):
    css = f"""
    <style>
    .stApp {{
        background-color: #111 !important;
        background-image: url("{url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white !important;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.65) !important;
        padding: 2rem;
        border-radius: 12px;
        color: white !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #00FF7F !important;
    }}
    label, .stTextInput label, .stSelectbox label, .stSlider label {{
        color: #DDD !important;
    }}
    .css-1n76uvr, .css-1siy2j7 {{
        background-color: #e63946 !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #aaa;
        background-color: #111;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: #00FF7F;
        border-bottom: 3px solid #00FF7F;
    }}
    .stDataFrame, .stTable {{
        background-color: rgba(255, 255, 255, 0.03) !important;
        color: white !important;
    }}
    .markdown-text-container {{
        color: white !important;
    }}
    .css-1d391kg {{
        background-color: #222 !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background_from_url("https://raw.githubusercontent.com/LukeJGallagher/Athletics/main/world_athletics/Background2.PNG")

# Absolute path to the SQL folder
SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')

# DB paths
DB_MEN_RANK = os.path.join(SQL_DIR, 'rankings_men_all_events.db')
DB_WOMEN_RANK = os.path.join(SQL_DIR, 'rankings_women_all_events.db')
DB_KSA_MEN = os.path.join(SQL_DIR, 'ksa_modal_results_men.db')
DB_KSA_WOMEN = os.path.join(SQL_DIR, 'ksa_modal_results_women.db')

# Show last modified time
if os.path.exists(DB_MEN_RANK):
    try:
        modified_time = os.path.getmtime(DB_MEN_RANK)
        modified_dt = datetime.datetime.fromtimestamp(modified_time)
        st.markdown(f"""
            <p style='text-align: center; color: #ccc; margin-top: -1rem;'>
                <em>Database last updated: {modified_dt.strftime('%d %b %Y, %H:%M')}</em>
            </p>
        """, unsafe_allow_html=True)
    except Exception:
        pass
else:
    st.warning(f"Database file not found at: `{DB_MEN_RANK}`")

@st.cache_data
def load_sqlite_table(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f'SELECT * FROM {table_name}', conn)

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

# Tabs
tab1, tab2, tab3 = st.tabs(['Combined Rankings', 'Saudi Athletes Rankings', 'Saudi Modal Results'])

with tab1:
    st.header('Combined Rankings')
    gender = st.selectbox('Select Gender', ['All', 'Men', 'Women'], index=1)

    if gender == 'Men':
        data = men_rankings.copy()
    elif gender == 'Women':
        data = women_rankings.copy()
    else:
        data = pd.concat([men_rankings, women_rankings])

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

    selected_rank = st.slider("Select Rank Range", int(data['Rank'].min()), int(data['Rank'].max()), (int(data['Rank'].min()), int(data['Rank'].max())))
    selected_score = st.slider("Select Score Range", int(data['Score'].min()), int(data['Score'].max()), (int(data['Score'].min()), int(data['Score'].max())))

    filtered = data[(data['Rank'].between(*selected_rank)) & (data['Score'].between(*selected_score))]
    filtered = filtered.drop_duplicates()
    st.dataframe(filtered.reset_index(drop=True))

with tab2:
    st.header('Saudi Athletes Rankings')
    st.markdown("""
    <p style='color: #ccc; font-size: 0.9em;'>
    Note: Rankings are based on <strong>3 athletes per country per event</strong>.
    </p>
    """, unsafe_allow_html=True)

    saudi_men = men_rankings[men_rankings['Country'].str.upper().str.contains('KSA')]
    saudi_women = women_rankings[women_rankings['Country'].str.upper().str.contains('KSA')]
    saudi_combined = pd.concat([saudi_men, saudi_women])

    saudi_events = sorted(saudi_combined['Event Type'].dropna().unique())
    selected_event_saudi = st.selectbox("Select Event", options=["All"] + saudi_events, key="ksa_event_key")
    if selected_event_saudi != "All":
        saudi_combined = saudi_combined[saudi_combined['Event Type'] == selected_event_saudi]

    saudi_names = sorted(saudi_combined['Name'].dropna().unique())
    selected_name_saudi = st.selectbox("Select Athlete", options=["All"] + saudi_names, key="ksa_name_key")
    if selected_name_saudi != "All":
        saudi_combined = saudi_combined[saudi_combined['Name'] == selected_name_saudi]

    saudi_combined = saudi_combined.drop_duplicates()
    st.dataframe(saudi_combined.reset_index(drop=True))

with tab3:
    st.header('Saudi Modal Results')
    if ksa_men_results is not None:
        data = ksa_men_results.copy()
        gender_loaded = "Men"
    elif ksa_women_results is not None:
        data = ksa_women_results.copy()
        gender_loaded = "Women"
    else:
        data = pd.DataFrame()
        gender_loaded = None

    if not data.empty:
        data = data.drop_duplicates()
        st.markdown(f"<p style='color: #ccc;'>Showing data for: <strong>{gender_loaded}</strong></p>", unsafe_allow_html=True)

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

st.markdown("""
    <hr style='margin-top: 30px; border: 1px solid #444;'>
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        Athletics Qualification Dashboard — Created by <strong>Luke Gallagher</strong>
    </div>
    """, unsafe_allow_html=True)

st.write('End of Application')
