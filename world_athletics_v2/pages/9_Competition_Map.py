"""
Competition Map - Global heatmap of competition quality by country.

Shows where the best competitions happen (by WA ranking points),
where KSA athletes compete abroad, and the highest-value venues
for maximising ranking points.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.theme import (
    get_theme_css,
    render_page_header,
    render_section_header,
    render_metric_card,
    render_sidebar,
    TEAL_PRIMARY,
    TEAL_DARK,
    TEAL_LIGHT,
    GOLD_ACCENT,
    PLOTLY_LAYOUT,
)
from components.filters import event_gender_picker, age_category_filter
from data.connector import get_connector
from data.event_utils import (
    display_to_db,
    normalize_event_for_match,
    format_event_name,
    EVENT_DB_TO_DISPLAY,
)

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Competition Map", page_icon="\U0001f5fa\ufe0f", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header("Competition Map", "Global heatmap of competition quality and ranking points")

dc = get_connector()

# ── IOC to ISO-3 Mapping ──────────────────────────────────────────────

IOC_TO_ISO3 = {
    "GER": "DEU", "SUI": "CHE", "NED": "NLD", "GRE": "GRC",
    "CRO": "HRV", "DEN": "DNK", "POR": "PRT", "RSA": "ZAF",
    "KSA": "SAU", "UAE": "ARE", "KUW": "KWT", "BRN": "BHR",
    "QAT": "QAT", "OMA": "OMN", "TPE": "TWN", "PHI": "PHL",
    "NGR": "NGA", "ALG": "DZA", "TAN": "TZA", "ZIM": "ZWE",
    "SKN": "KNA", "VIN": "VCT", "TTO": "TTO", "BAR": "BRB",
    "IRI": "IRN", "BUL": "BGR", "HAI": "HTI", "PAR": "PRY",
    "CHI": "CHL", "URU": "URY", "CRC": "CRI", "MAS": "MYS",
    "SIN": "SGP", "HKG": "HKG", "INA": "IDN", "SRI": "LKA",
    "ROU": "ROU", "MDA": "MDA", "LAT": "LVA", "SLO": "SVN",
    "MON": "MCO", "SUD": "SDN", "GUI": "GIN", "GAM": "GMB",
    "TOG": "TGO", "MTN": "MRT", "CHA": "TCD", "MAD": "MDG",
    "SEY": "SYC", "LES": "LSO", "BOT": "BWA", "SWZ": "SWZ",
    "ANG": "AGO", "MOZ": "MOZ", "ZAM": "ZMB", "MAW": "MWI",
    "BUR": "BFA", "NIG": "NER", "CGO": "COG", "SOL": "SLB",
    "VAN": "VUT", "SAM": "WSM", "FIJ": "FJI", "TGA": "TON",
    "PNG": "PNG", "GBS": "GNB", "MLI": "MLI", "NEP": "NPL",
    "BAN": "BGD", "MYA": "MMR", "CAM": "KHM", "LAO": "LAO",
    "BRU": "BRN", "MGL": "MNG", "TKM": "TKM", "KGZ": "KGZ",
    "TJK": "TJK", "UZB": "UZB", "KAZ": "KAZ", "AFG": "AFG",
    "PLE": "PSE", "LBN": "LBN", "SYR": "SYR", "JOR": "JOR",
    "IRQ": "IRQ", "YEM": "YEM", "BHU": "BTN", "MDV": "MDV",
    "TLS": "TLS", "PRK": "PRK",
}

# Country name lookup for display
COUNTRY_NAMES = {
    "USA": "United States", "GBR": "Great Britain", "FRA": "France",
    "GER": "Germany", "JPN": "Japan", "CHN": "China", "AUS": "Australia",
    "ITA": "Italy", "ESP": "Spain", "KSA": "Saudi Arabia", "QAT": "Qatar",
    "BRN": "Bahrain", "UAE": "United Arab Emirates", "KUW": "Kuwait",
    "OMA": "Oman", "IND": "India", "BRA": "Brazil", "CAN": "Canada",
    "RUS": "Russia", "KEN": "Kenya", "ETH": "Ethiopia", "RSA": "South Africa",
    "POL": "Poland", "BEL": "Belgium", "NED": "Netherlands", "SUI": "Switzerland",
    "SWE": "Sweden", "FIN": "Finland", "CZE": "Czech Republic", "TUR": "Turkey",
    "KOR": "South Korea", "THA": "Thailand", "MAS": "Malaysia", "SIN": "Singapore",
    "PHI": "Philippines", "INA": "Indonesia", "VIE": "Vietnam",
}


def ioc_to_iso3(code: str) -> str:
    """Convert IOC country code to ISO-3166 alpha-3 for Plotly choropleth."""
    return IOC_TO_ISO3.get(code, code)


def get_country_name(ioc_code: str) -> str:
    """Get readable country name from IOC code."""
    return COUNTRY_NAMES.get(ioc_code, ioc_code)


# ── Filters ───────────────────────────────────────────────────────────

st.markdown("#### Filters")
filter_row1 = st.columns([3, 1, 1, 2])

with filter_row1[0]:
    event, gender = event_gender_picker(key_prefix="cmap")

with filter_row1[1]:
    age_cat = age_category_filter(key="cmap_age")

with filter_row1[2]:
    # Environment filter
    env_option = st.selectbox(
        "Environment",
        ["All", "Outdoor", "Indoor"],
        key="cmap_env",
    )

with filter_row1[3]:
    year_range = st.slider(
        "Year Range",
        min_value=2001,
        max_value=2026,
        value=(2020, 2026),
        key="cmap_year",
    )

# Second filter row: Points slider + Competition category
filter_row2 = st.columns([3, 3])

with filter_row2[0]:
    points_range = st.slider(
        "WA Points Range",
        min_value=0,
        max_value=1400,
        value=(0, 1400),
        step=50,
        key="cmap_points",
        help="Filter results by WA ranking points scored",
    )

with filter_row2[1]:
    # WA Competition Categories (based on points potential)
    COMP_CATEGORIES = {
        "All Categories": None,
        "A - Diamond League / Majors (1000+)": 1000,
        "B - Continental Tour Gold (800-999)": 800,
        "C - Continental Tour Silver (600-799)": 600,
        "D - Continental Tour Bronze (400-599)": 400,
        "E - Area Permits (200-399)": 200,
        "F - Local (0-199)": 0,
    }
    comp_category = st.selectbox(
        "Competition Category",
        list(COMP_CATEGORIES.keys()),
        key="cmap_category",
        help="Filter by estimated competition tier (based on avg WA points)",
    )


# ── Build SQL Conditions ──────────────────────────────────────────────

def build_where_clause(
    event_filter: str,
    gender_filter: str,
    age_cat_filter: str | None,
    env_filter: str,
    year_min: int,
    year_max: int,
    nat_filter: str | None = None,
    points_min: int = 0,
    points_max: int = 1400,
) -> str:
    """Build WHERE clause for master queries with all filters applied."""
    conditions = [
        "venue IS NOT NULL",
        "resultscore IS NOT NULL",
        "resultscore > 0",
        f"regexp_extract(venue, '\\(([A-Z]{{2,4}})\\)', 1) != ''",
        f"year >= {year_min}",
        f"year <= {year_max}",
    ]

    # Points range filter
    if points_min > 0:
        conditions.append(f"resultscore >= {points_min}")
    if points_max < 1400:
        conditions.append(f"resultscore <= {points_max}")

    # Event filter - use normalized exact matching
    norm_event = normalize_event_for_match(display_to_db(event_filter))
    conditions.append(
        f"regexp_replace(LOWER(event), '[^0-9a-z]', '', 'g') = '{norm_event}'"
    )

    # Gender filter
    g = gender_filter.strip().upper()
    if g == "M":
        conditions.append("LOWER(gender) IN ('m', 'men', 'male')")
    else:
        conditions.append("LOWER(gender) IN ('f', 'w', 'women', 'female')")

    # Age category
    if age_cat_filter == "U20":
        conditions.append("age IS NOT NULL AND age < 20")
    elif age_cat_filter == "U23":
        conditions.append("age IS NOT NULL AND age < 23")
    elif age_cat_filter == "Senior":
        conditions.append("age IS NOT NULL AND age >= 23")

    # Environment
    if env_filter == "Outdoor":
        conditions.append("LOWER(environment) = 'outdoor'")
    elif env_filter == "Indoor":
        conditions.append("LOWER(environment) = 'indoor'")

    # Nationality
    if nat_filter:
        conditions.append(f"UPPER(nat) = '{nat_filter.upper()}'")

    return " AND ".join(conditions)


# ── Query: Country-Level Aggregation ──────────────────────────────────

where = build_where_clause(
    event, gender, age_cat, env_option, year_range[0], year_range[1],
    points_min=points_range[0], points_max=points_range[1],
)

country_sql = f"""
SELECT
    regexp_extract(venue, '\\(([A-Z]{{2,4}})\\)', 1) AS comp_country,
    COUNT(*) AS n_results,
    ROUND(AVG(resultscore), 0) AS avg_score,
    MAX(resultscore) AS max_score,
    MIN(resultscore) AS min_score,
    COUNT(DISTINCT competitor) AS n_athletes,
    COUNT(DISTINCT venue) AS n_venues,
    COUNT(DISTINCT year) AS n_years
FROM master
WHERE {where}
GROUP BY comp_country
HAVING COUNT(*) >= 3
ORDER BY avg_score DESC
"""

country_df = dc.query(country_sql)

if country_df.empty:
    st.warning("No data found for the selected filters. Try broadening the event, year range, or age category.")
    st.stop()

# Apply competition category filter on avg_score
cat_threshold = COMP_CATEGORIES.get(comp_category)
if cat_threshold is not None:
    # Category filter: show countries whose avg is in this tier range
    tier_ranges = {
        1000: (1000, 9999),
        800: (800, 999),
        600: (600, 799),
        400: (400, 599),
        200: (200, 399),
        0: (0, 199),
    }
    tier = tier_ranges.get(cat_threshold, (0, 9999))
    country_df = country_df[
        (country_df["avg_score"] >= tier[0]) & (country_df["avg_score"] <= tier[1])
    ]
    if country_df.empty:
        st.info(f"No countries match category '{comp_category}'. Try a broader filter.")
        st.stop()

# Convert IOC codes to ISO-3 for choropleth
country_df["iso3"] = country_df["comp_country"].apply(ioc_to_iso3)
country_df["country_name"] = country_df["comp_country"].apply(get_country_name)

# ── Summary Metrics ───────────────────────────────────────────────────

gender_label = "Men" if gender == "M" else "Women"
total_results = int(country_df["n_results"].sum())
total_countries = len(country_df)
overall_avg = int(country_df["n_results"].mul(country_df["avg_score"]).sum() / total_results) if total_results > 0 else 0
top_country = country_df.iloc[0]

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Total Results", f"{total_results:,}", "excellent")
with m2:
    render_metric_card("Countries", str(total_countries), "good")
with m3:
    render_metric_card("Avg WA Points", str(overall_avg), "neutral")
with m4:
    render_metric_card(
        "Top Country",
        f"{top_country['country_name']} ({int(top_country['avg_score'])} pts)",
        "gold",
    )

st.markdown("")

# ── World Heatmap ─────────────────────────────────────────────────────

render_section_header(
    "Competition Quality by Country",
    f"{event} {gender_label} | {year_range[0]}-{year_range[1]} | Average WA Ranking Points",
)

# Build custom colorscale from light teal to dark teal
teal_colorscale = [
    [0.0, "#e8f5e9"],
    [0.2, "#c8e6c9"],
    [0.4, "#81c784"],
    [0.6, "#4caf50"],
    [0.8, "#2A8F5C"],
    [1.0, TEAL_PRIMARY],
]

fig_map = px.choropleth(
    country_df,
    locations="iso3",
    color="avg_score",
    hover_name="country_name",
    hover_data={
        "iso3": False,
        "comp_country": True,
        "n_results": True,
        "avg_score": True,
        "max_score": True,
        "n_venues": True,
        "n_athletes": True,
    },
    labels={
        "avg_score": "Avg WA Points",
        "n_results": "Results",
        "max_score": "Max Points",
        "n_venues": "Venues",
        "n_athletes": "Athletes",
        "comp_country": "IOC Code",
    },
    color_continuous_scale=teal_colorscale,
    projection="natural earth",
)

fig_map.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", color="#333"),
    margin=dict(l=0, r=0, t=10, b=0),
    height=500,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="lightgray",
        showland=True,
        landcolor="#f8f9fa",
        showocean=True,
        oceancolor="#e8f4f8",
        showcountries=True,
        countrycolor="lightgray",
    ),
    coloraxis_colorbar=dict(
        title="Avg WA<br>Points",
        thickness=15,
        len=0.6,
    ),
)

st.plotly_chart(fig_map, use_container_width=True)

# ── Two-Column Layout: Top Countries | Best Venues ───────────────────

col_left, col_right = st.columns(2)

# ── Top Countries Table ───────────────────────────────────────────────

with col_left:
    render_section_header("Top 20 Competition Countries", "Ranked by average WA points")

    top_countries = country_df.head(20).copy()
    top_countries["Rank"] = range(1, len(top_countries) + 1)
    top_countries = top_countries.rename(columns={
        "country_name": "Country",
        "comp_country": "IOC",
        "n_results": "Results",
        "avg_score": "Avg Points",
        "max_score": "Max Points",
        "n_venues": "Venues",
        "n_athletes": "Athletes",
    })

    display_cols = ["Rank", "Country", "IOC", "Results", "Avg Points", "Max Points", "Venues"]
    try:
        styled = top_countries[display_cols].style.background_gradient(
            cmap="YlGn", subset=["Avg Points"]
        )
    except ImportError:
        # matplotlib not available on Streamlit Cloud
        styled = top_countries[display_cols]
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

# ── Best Venues Table ────────────────────────────────────────────────

with col_right:
    render_section_header("Best Venues for Ranking Points", "Highest average WA points per venue")

    venue_sql = f"""
    SELECT
        venue,
        regexp_extract(venue, '\\(([A-Z]{{2,4}})\\)', 1) AS country_code,
        COUNT(*) AS n_results,
        ROUND(AVG(resultscore), 0) AS avg_score,
        MAX(resultscore) AS max_score,
        COUNT(DISTINCT competitor) AS n_athletes,
        COUNT(DISTINCT year) AS n_years
    FROM master
    WHERE {where}
    GROUP BY venue
    HAVING COUNT(*) >= 5
    ORDER BY avg_score DESC
    LIMIT 25
    """

    venue_df = dc.query(venue_sql)

    if not venue_df.empty:
        venue_df["Rank"] = range(1, len(venue_df) + 1)

        # Shorten venue names for display (take the city part)
        def shorten_venue(v: str) -> str:
            """Extract a shorter venue description."""
            if len(v) > 50:
                parts = v.split(",")
                if len(parts) >= 2:
                    return parts[0][:35] + ", " + parts[-1].strip()
            return v

        venue_df["venue_short"] = venue_df["venue"].apply(shorten_venue)
        venue_df = venue_df.rename(columns={
            "venue_short": "Venue",
            "country_code": "Country",
            "n_results": "Results",
            "avg_score": "Avg Points",
            "max_score": "Max Points",
            "n_athletes": "Athletes",
        })

        display_venue_cols = ["Rank", "Venue", "Country", "Results", "Avg Points", "Max Points"]
        try:
            styled_v = venue_df[display_venue_cols].style.background_gradient(
                cmap="YlGn", subset=["Avg Points"]
            )
        except ImportError:
            styled_v = venue_df[display_venue_cols]
        st.dataframe(
            styled_v,
            use_container_width=True,
            hide_index=True,
            height=500,
        )
    else:
        st.info("No venues with 5+ results match the current filters.")


# ── Section: Where KSA Athletes Compete ──────────────────────────────

render_section_header(
    "Where KSA Athletes Compete",
    "Competition locations for Saudi athletes abroad",
)

ksa_where = build_where_clause(
    event, gender, age_cat, env_option, year_range[0], year_range[1],
    nat_filter="KSA", points_min=points_range[0], points_max=points_range[1],
)

ksa_country_sql = f"""
SELECT
    regexp_extract(venue, '\\(([A-Z]{{2,4}})\\)', 1) AS comp_country,
    COUNT(*) AS n_results,
    ROUND(AVG(resultscore), 0) AS avg_score,
    MAX(resultscore) AS max_score,
    COUNT(DISTINCT competitor) AS n_athletes,
    COUNT(DISTINCT venue) AS n_venues
FROM master
WHERE {ksa_where}
GROUP BY comp_country
ORDER BY n_results DESC
"""

ksa_df = dc.query(ksa_country_sql)

if ksa_df.empty:
    st.info("No KSA athlete data found for the selected event and filters. Try a different event or broader year range.")
else:
    ksa_df["iso3"] = ksa_df["comp_country"].apply(ioc_to_iso3)
    ksa_df["country_name"] = ksa_df["comp_country"].apply(get_country_name)

    # KSA summary metrics
    ksa_total = int(ksa_df["n_results"].sum())
    ksa_countries = len(ksa_df)
    ksa_abroad = ksa_df[ksa_df["comp_country"] != "KSA"]
    ksa_abroad_pct = int(ksa_abroad["n_results"].sum() / ksa_total * 100) if ksa_total > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("KSA Results", f"{ksa_total:,}", "excellent")
    with k2:
        render_metric_card("Countries Visited", str(ksa_countries), "good")
    with k3:
        render_metric_card("% Abroad", f"{ksa_abroad_pct}%", "neutral")
    with k4:
        if not ksa_abroad.empty:
            top_dest = ksa_abroad.iloc[0]
            render_metric_card(
                "Top Destination",
                f"{top_dest['country_name']} ({int(top_dest['n_results'])})",
                "gold",
            )
        else:
            render_metric_card("Top Destination", "N/A", "neutral")

    st.markdown("")

    # KSA competition map
    ksa_map_col, ksa_table_col = st.columns([3, 2])

    with ksa_map_col:
        fig_ksa = px.choropleth(
            ksa_df,
            locations="iso3",
            color="n_results",
            hover_name="country_name",
            hover_data={
                "iso3": False,
                "comp_country": True,
                "n_results": True,
                "avg_score": True,
                "n_athletes": True,
            },
            labels={
                "n_results": "Results",
                "avg_score": "Avg WA Points",
                "n_athletes": "KSA Athletes",
                "comp_country": "IOC Code",
            },
            color_continuous_scale=[
                [0.0, "#e0f2f1"],
                [0.3, TEAL_LIGHT],
                [0.7, TEAL_PRIMARY],
                [1.0, TEAL_DARK],
            ],
            projection="natural earth",
        )

        # Highlight KSA with a gold star marker
        ksa_row = ksa_df[ksa_df["comp_country"] == "KSA"]
        if not ksa_row.empty:
            fig_ksa.add_trace(go.Scattergeo(
                lat=[24.7136],
                lon=[46.6753],
                mode="markers+text",
                marker=dict(size=14, color=GOLD_ACCENT, symbol="star"),
                text=["KSA"],
                textposition="top center",
                textfont=dict(size=11, color=GOLD_ACCENT, family="Inter, sans-serif"),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig_ksa.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif", color="#333"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="lightgray",
                showland=True,
                landcolor="#f8f9fa",
                showocean=True,
                oceancolor="#e8f4f8",
                showcountries=True,
                countrycolor="lightgray",
            ),
            coloraxis_colorbar=dict(
                title="Results",
                thickness=12,
                len=0.5,
            ),
        )

        st.plotly_chart(fig_ksa, use_container_width=True)

    with ksa_table_col:
        ksa_table = ksa_df.copy()
        ksa_table["Rank"] = range(1, len(ksa_table) + 1)
        ksa_table = ksa_table.rename(columns={
            "country_name": "Country",
            "comp_country": "IOC",
            "n_results": "Results",
            "avg_score": "Avg Points",
            "max_score": "Max Points",
            "n_athletes": "Athletes",
            "n_venues": "Venues",
        })

        display_ksa_cols = ["Rank", "Country", "IOC", "Results", "Avg Points", "Athletes"]
        try:
            styled_k = ksa_table[display_ksa_cols].style.background_gradient(
                cmap="YlGn", subset=["Results"]
            )
        except ImportError:
            styled_k = ksa_table[display_ksa_cols]
        st.dataframe(
            styled_k,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

    # KSA athletes detail
    with st.expander("KSA Athletes - Competition Detail", expanded=False):
        ksa_detail_sql = f"""
        SELECT
            competitor AS Athlete,
            regexp_extract(venue, '\\(([A-Z]{{2,4}})\\)', 1) AS Country,
            venue AS Venue,
            resultscore AS WA_Points,
            result_numeric AS Mark,
            CAST(date AS DATE) AS Date
        FROM master
        WHERE {ksa_where}
        ORDER BY resultscore DESC
        LIMIT 100
        """
        ksa_detail = dc.query(ksa_detail_sql)
        if not ksa_detail.empty:
            st.dataframe(ksa_detail, use_container_width=True, hide_index=True)


# ── Section: Points Distribution by Country ──────────────────────────

render_section_header(
    "Points Distribution",
    "How competition quality varies across the top host countries",
)

# Get top 10 countries for box plot
top10_countries = country_df.head(10)["comp_country"].tolist()

if top10_countries:
    top10_list = ", ".join(f"'{c}'" for c in top10_countries)
    dist_sql = f"""
    SELECT
        regexp_extract(venue, '\\(([A-Z]{{2,4}})\\)', 1) AS comp_country,
        resultscore
    FROM master
    WHERE {where}
    AND regexp_extract(venue, '\\(([A-Z]{{2,4}})\\)', 1) IN ({top10_list})
    """

    dist_df = dc.query(dist_sql)

    if not dist_df.empty:
        dist_df["country_name"] = dist_df["comp_country"].apply(get_country_name)

        # Order by median score descending
        order = (
            dist_df.groupby("country_name")["resultscore"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig_box = px.box(
            dist_df,
            x="country_name",
            y="resultscore",
            color_discrete_sequence=[TEAL_PRIMARY],
            labels={"country_name": "Country", "resultscore": "WA Ranking Points"},
            category_orders={"country_name": order},
        )

        fig_box.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            showlegend=False,
            title=dict(
                text=f"WA Points Distribution - Top 10 Countries ({event} {gender_label})",
                font=dict(size=14, color="#333"),
            ),
        )
        fig_box.update_xaxes(showgrid=False)
        fig_box.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", title="WA Ranking Points")

        st.plotly_chart(fig_box, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: #999; font-size: 0.8rem;'>"
    f"Data from World Athletics master database | {year_range[0]}-{year_range[1]} | "
    f"Results with valid WA ranking points only"
    f"</p>",
    unsafe_allow_html=True,
)
