"""
800m Race Analysis — Research-backed pacing intelligence.

Deep dive into 800m race dynamics using split data from major championships.
Grounded in sports science research (Casado, Sandford, Hanley, Mytton, Renfree).

Data: World Athletics GraphQL API - getEventTimetableWithContent splits
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar,
    TEAL_PRIMARY, TEAL_DARK, TEAL_LIGHT, GOLD_ACCENT, GRAY_BLUE,
    CHART_COLORS, PLOTLY_LAYOUT,
)

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="800m Race Analysis", page_icon="🏃", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header(
    "800m Race Analysis",
    "Research-backed pacing intelligence from major championships"
)

# ── Constants ─────────────────────────────────────────────────────────

RUNNER_TYPE_COLORS = {
    "Negative Split": "#0077B6",
    "Even Pace": TEAL_PRIMARY,
    "Mild Positive": GOLD_ACCENT,
    "Positive Split": "#FFB800",
    "Aggressive Fade": "#dc3545",
}
RUNNER_TYPE_ORDER = ["Negative Split", "Even Pace", "Mild Positive", "Positive Split", "Aggressive Fade"]

CHAMP_CHRONOLOGICAL = [
    "Beijing 2015 WC", "Rio 2016 OG", "London 2017 WC", "Doha 2019 WC",
    "Tokyo 2021 OG", "Oregon 2022 WC", "Budapest 2023 WC", "Paris 2024 OG",
    "Tokyo 2025 WC",
]

# Race video links for championship 800m finals
# Olympics on olympics.com; World Championships on worldathletics.org/supersport
RACE_VIDEOS = {
    "Paris 2024 OG": {
        "M": "https://www.olympics.com/en/video/men-s-800m-final-athletics-olympic-games-paris-2024",
        "W": "https://www.olympics.com/en/video/women-s-800m-final-athletics-olympic-games-paris-2024",
        "label": "Paris 2024 Olympics",
    },
    "Tokyo 2025 WC": {
        "M": "https://worldathletics.org/competitions/world-athletics-championships/world-athletics-championships-tokyo-2025-7190593/results/men/800-metres/final/result",
        "W": "https://worldathletics.org/competitions/world-athletics-championships/world-athletics-championships-tokyo-2025-7190593/results/women/800-metres/final/result",
        "label": "Tokyo 2025 WC",
    },
    "Budapest 2023 WC": {
        "M": "https://worldathletics.org/competitions/world-athletics-championships/world-athletics-championships-budapest-2023-7138987/results/men/800-metres/final/result",
        "W": "https://worldathletics.org/competitions/world-athletics-championships/world-athletics-championships-budapest-2023-7138987/results/women/800-metres/final/result",
        "label": "Budapest 2023 WC",
    },
    "Oregon 2022 WC": {
        "M": "https://worldathletics.org/competitions/world-athletics-championships/world-athletics-championships-oregon22/results/men/800-metres/final/result",
        "W": "https://worldathletics.org/competitions/world-athletics-championships/world-athletics-championships-oregon22/results/women/800-metres/final/result",
        "label": "Oregon 2022 WC",
    },
    "Tokyo 2021 OG": {
        "M": "https://www.olympics.com/en/video/men-s-800m-final-athletics-olympic-games-tokyo-2020",
        "W": "https://www.olympics.com/en/video/women-s-800m-final-athletics-olympic-games-tokyo-2020",
        "label": "Tokyo 2021 Olympics",
    },
    "Doha 2019 WC": {
        "M": "https://worldathletics.org/competitions/world-athletics-championships/iaaf-world-athletics-championships-doha-2019/results/men/800-metres/final/result",
        "W": "https://worldathletics.org/competitions/world-athletics-championships/iaaf-world-athletics-championships-doha-2019/results/women/800-metres/final/result",
        "label": "Doha 2019 WC",
    },
    "London 2017 WC": {
        "M": "https://worldathletics.org/competitions/world-athletics-championships/iaaf-world-championships-london-2017/results/men/800-metres/final/result",
        "W": "https://worldathletics.org/competitions/world-athletics-championships/iaaf-world-championships-london-2017/results/women/800-metres/final/result",
        "label": "London 2017 WC",
    },
    "Rio 2016 OG": {
        "M": "https://www.olympics.com/en/video/men-s-800m-final-rio-2016-replays",
        "W": "https://www.olympics.com/en/video/women-s-800m-final-rio-2016-replays",
        "label": "Rio 2016 Olympics",
    },
    "Beijing 2015 WC": {
        "M": "https://worldathletics.org/competitions/world-athletics-championships/2015-world-championships/results/men/800-metres/final/result",
        "W": "https://worldathletics.org/competitions/world-athletics-championships/2015-world-championships/results/women/800-metres/final/result",
        "label": "Beijing 2015 WC",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────

def time_to_seconds(t: str) -> float:
    """Convert '1:45.09' or '52.16' to seconds."""
    if not t or not isinstance(t, str):
        return np.nan
    t = t.strip()
    try:
        if ":" in t:
            parts = t.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(t)
    except (ValueError, IndexError):
        return np.nan


def seconds_to_time(s: float) -> str:
    """Convert seconds to M:SS.ss format."""
    if pd.isna(s) or s <= 0:
        return ""
    mins = int(s // 60)
    secs = s - mins * 60
    if mins > 0:
        return f"{mins}:{secs:05.2f}"
    return f"{secs:.2f}"


def classify_runner_type(differential: float) -> str:
    """Classify runner based on lap differential (Sandford-inspired)."""
    if differential < -1.0:
        return "Negative Split"
    elif differential < 0.5:
        return "Even Pace"
    elif differential < 2.0:
        return "Mild Positive"
    elif differential < 4.0:
        return "Positive Split"
    else:
        return "Aggressive Fade"


def export_chart_button(fig: go.Figure, filename: str, key: str) -> None:
    """Add a download button for chart PNG export."""
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button(
            label="Export Chart (PNG)",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
            key=key,
        )
    except Exception:
        pass


# ── Data Loading ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data():
    """Load 800m championship results and splits from parquet files."""
    data_dir = Path(__file__).parent.parent / "data" / "scraped"

    results_path = data_dir / "800m_championship_results.parquet"
    splits_path = data_dir / "800m_championship_splits.parquet"

    if not results_path.exists() or not splits_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    df_results = pd.read_parquet(results_path)
    df_splits = pd.read_parquet(splits_path)

    df_results["time_sec"] = df_results["mark"].apply(time_to_seconds)
    df_splits["split_sec"] = df_splits["split_mark"].apply(time_to_seconds)
    df_splits["split_dist"] = (
        df_splits["split_name"].str.replace(" ", "").str.replace("m", "").astype(float)
    )

    return df_results, df_splits


@st.cache_data(ttl=3600)
def load_diamond_league():
    """Load Diamond League 800m results if available."""
    data_dir = Path(__file__).parent.parent / "data" / "scraped"
    dl_path = data_dir / "800m_diamond_league_results.parquet"

    if not dl_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(dl_path)
    df["time_sec"] = df["mark"].apply(time_to_seconds)
    return df


@st.cache_data(ttl=3600)
def build_lap_analysis(df_results: pd.DataFrame, df_splits: pd.DataFrame) -> pd.DataFrame:
    """Build per-athlete lap analysis with 1st/2nd lap times and differentials."""
    df_400 = df_splits[df_splits["split_dist"] == 400].copy()
    df_400 = df_400.rename(columns={"split_sec": "lap1_sec"})

    merged = df_400.merge(
        df_results[["championship", "phase", "sex", "name", "time_sec", "rank", "mark"]],
        on=["championship", "phase", "sex", "name"],
        how="inner",
    )

    merged["lap2_sec"] = merged["time_sec"] - merged["lap1_sec"]
    merged["differential"] = merged["lap2_sec"] - merged["lap1_sec"]
    merged["diff_pct"] = (merged["differential"] / merged["lap1_sec"]) * 100

    # Filter outliers
    merged = merged[
        (merged["lap1_sec"] > 20) & (merged["lap1_sec"] < 65)
        & (merged["lap2_sec"] > 20) & (merged["lap2_sec"] < 65)
        & (merged["time_sec"] > 80) & (merged["time_sec"] < 130)
    ].copy()

    merged["runner_type"] = merged["differential"].apply(classify_runner_type)

    # Speed per 200m segment (estimated from lap times)
    merged["speed_lap1"] = 400 / merged["lap1_sec"]  # m/s
    merged["speed_lap2"] = 400 / merged["lap2_sec"]

    # Coefficient of variation between laps
    merged["cv_pct"] = merged.apply(
        lambda r: np.std([r["lap1_sec"], r["lap2_sec"]]) / np.mean([r["lap1_sec"], r["lap2_sec"]]) * 100,
        axis=1,
    )

    return merged


# ── Load & Validate ───────────────────────────────────────────────────

df_results, df_splits = load_data()

if df_results.empty or df_splits.empty:
    st.error("No 800m split data found. Run the championship scraper first.")
    st.stop()

df_laps = build_lap_analysis(df_results, df_splits)

if df_laps.empty:
    st.error("Could not compute lap analysis — check split data quality.")
    st.stop()

# Diamond League data (optional — times only, no splits)
df_dl = load_diamond_league()


# ── Filters ───────────────────────────────────────────────────────────

col_f1, col_f2, col_f3 = st.columns([1, 2, 1])

with col_f1:
    gender = st.selectbox(
        "Gender", ["M", "W"],
        format_func=lambda x: "Men" if x == "M" else "Women",
        key="ra_gender",
    )

with col_f2:
    champ_options = ["All Championships"] + sorted(df_laps["championship"].unique().tolist())
    champ_filter = st.selectbox("Championship", champ_options, key="ra_champ")

with col_f3:
    phase_options = ["All Rounds", "Finals Only", "Semis Only", "Heats Only"]
    phase_filter = st.selectbox("Round", phase_options, key="ra_round")

gender_label = "Men's" if gender == "M" else "Women's"

# Apply filters
df = df_laps[df_laps["sex"] == gender].copy()
if champ_filter != "All Championships":
    df = df[df["championship"] == champ_filter]
if phase_filter == "Finals Only":
    df = df[df["phase"].str.contains("Final", case=False, na=False)]
elif phase_filter == "Semis Only":
    df = df[df["phase"].str.contains("Semi", case=False, na=False)]
elif phase_filter == "Heats Only":
    df = df[df["phase"].str.contains("Heat", case=False, na=False)]

st.markdown(f"### {gender_label} 800m — {champ_filter}" + (f" ({phase_filter})" if phase_filter != "All Rounds" else ""))

if df.empty:
    st.warning("No data for this filter combination.")
    st.stop()


# ── KPI Row ───────────────────────────────────────────────────────────

medal_df = df[df["rank"].between(1, 3)]
final_df = df[df["phase"].str.contains("Final", case=False, na=False)]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card("Races Analysed", str(len(df)), "neutral")
with col2:
    render_metric_card("Avg Finish", seconds_to_time(df["time_sec"].mean()), "excellent")
with col3:
    render_metric_card("Avg 1st Lap", seconds_to_time(df["lap1_sec"].mean()), "good")
with col4:
    render_metric_card("Avg 2nd Lap", seconds_to_time(df["lap2_sec"].mean()), "good")
with col5:
    render_metric_card("Avg Differential", f"{df['differential'].mean():+.2f}s", "gold")


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Pacing Profiles",
    "Split Differential",
    "Runner Types",
    "Tactical Positioning",
    "Championship Trends",
    "Sweet Spot Finder",
    "Athlete Deep Dive",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: PACING PROFILES (Seahorse Pattern - Casado et al.)
# ══════════════════════════════════════════════════════════════════════
with tab1:
    render_section_header(
        "Pacing Profiles",
        f"{gender_label} 800m — the 'Seahorse' pacing pattern (Casado et al.)"
    )

    # Estimated 200m segment analysis from lap data
    # We have 400m splits, so we can show L1 vs L2 speed comparison
    st.markdown("""
    Elite 800m runners display a characteristic **"seahorse" pacing pattern**: fast acceleration
    in the first 200m, deceleration through 200-400m, then gradual re-acceleration in the
    final 200m. Since we have 400m split data, we analyse first-lap vs second-lap speed profiles.
    """)

    col_a, col_b = st.columns([3, 1])

    with col_a:
        # Speed profile: medalists vs field
        fig_pace = go.Figure()

        segments = ["1st Lap (0-400m)", "2nd Lap (400-800m)"]

        # All athletes
        all_speeds = [df["speed_lap1"].mean(), df["speed_lap2"].mean()]
        fig_pace.add_trace(go.Scatter(
            x=segments, y=all_speeds,
            mode="lines+markers",
            line=dict(color=GRAY_BLUE, width=3),
            marker=dict(size=10),
            name=f"All Athletes (n={len(df)})",
        ))

        # Medalists
        if not medal_df.empty:
            medal_speeds = [medal_df["speed_lap1"].mean(), medal_df["speed_lap2"].mean()]
            fig_pace.add_trace(go.Scatter(
                x=segments, y=medal_speeds,
                mode="lines+markers",
                line=dict(color=GOLD_ACCENT, width=4),
                marker=dict(size=14, symbol="star"),
                name=f"Medalists (n={len(medal_df)})",
            ))

        # Finals non-medalists
        if not final_df.empty:
            final_non_medal = final_df[~final_df["rank"].between(1, 3)]
            if not final_non_medal.empty:
                final_speeds = [final_non_medal["speed_lap1"].mean(), final_non_medal["speed_lap2"].mean()]
                fig_pace.add_trace(go.Scatter(
                    x=segments, y=final_speeds,
                    mode="lines+markers",
                    line=dict(color=TEAL_PRIMARY, width=3, dash="dash"),
                    marker=dict(size=8),
                    name=f"Finalists 4th-8th (n={len(final_non_medal)})",
                ))

        fig_pace.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{gender_label} 800m Average Speed by Lap Segment",
            yaxis_title="Speed (m/s)",
            height=450,
        )
        fig_pace.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig_pace.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        st.plotly_chart(fig_pace, use_container_width=True)
        st.caption(
            "**Seahorse pattern** (Casado et al., 2021): Elite 800m runners start fast, "
            "decelerate mid-race, then attempt to re-accelerate in the final 200m. "
            "Medalists maintain higher speed in the second lap relative to their first."
        )
        export_chart_button(fig_pace, f"{gender_label.lower()}_800m_pacing_profile.png", "exp_t1_pace")

    with col_b:
        st.markdown(f"**{gender_label} Pacing Summary**")

        avg_diff = df["differential"].mean()
        avg_cv = df["cv_pct"].mean()

        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Avg 1st Lap | **{seconds_to_time(df['lap1_sec'].mean())}** |
        | Avg 2nd Lap | **{seconds_to_time(df['lap2_sec'].mean())}** |
        | Avg Differential | **{avg_diff:+.2f}s** |
        | Median CV% | **{df['cv_pct'].median():.1f}%** |
        | Positive Split % | **{(df['differential'] > 0).mean()*100:.0f}%** |
        """)

        if not medal_df.empty:
            st.markdown("**Medalist Pacing:**")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Avg 1st Lap | **{seconds_to_time(medal_df['lap1_sec'].mean())}** |
            | Avg 2nd Lap | **{seconds_to_time(medal_df['lap2_sec'].mean())}** |
            | Avg Differential | **{medal_df['differential'].mean():+.2f}s** |
            """)

    # Gender-specific research note
    with st.expander("Research Context"):
        if gender == "M":
            st.markdown("""
            **Hanley et al. (2021):** Men's 800m shows **progressive deceleration** — each
            200m segment is slower than the previous by ~0.5 seconds. Only 2 of 22 men's
            world records were run as negative splits. The optimal first-lap advantage for
            medalists is **2.2 ± 1.1 seconds** faster than the second lap (Mytton et al., 2018).
            """)
        else:
            st.markdown("""
            **Hanley et al. (2021):** Women's 800m shows a distinctly different pattern — after
            a fast first 200m, speed remains **almost constant** through the remaining segments.
            Women show much **less progressive deceleration** than men, possibly due to higher
            proportions of slow-twitch (Type I) muscle fibres that resist fatigue better.
            Women achieved 80% of world records at championships vs men's 46%.
            """)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: SPLIT DIFFERENTIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    render_section_header(
        "Split Differential Analysis",
        f"{gender_label} 800m — 1st lap vs 2nd lap comparison"
    )

    col_s1, col_s2 = st.columns([2, 1])

    with col_s1:
        fig_diff = go.Figure()

        # Even pace reference line
        min_lap = min(df["lap1_sec"].min(), df["lap2_sec"].min()) - 1
        max_lap = max(df["lap1_sec"].max(), df["lap2_sec"].max()) + 1
        fig_diff.add_trace(go.Scatter(
            x=[min_lap, max_lap], y=[min_lap, max_lap],
            mode="lines", line=dict(dash="dash", color="gray", width=1),
            name="Even Pace", showlegend=True,
        ))

        # Medalists as gold stars
        if not medal_df.empty:
            fig_diff.add_trace(go.Scatter(
                x=medal_df["lap1_sec"], y=medal_df["lap2_sec"],
                mode="markers",
                marker=dict(size=12, color=GOLD_ACCENT, symbol="star",
                            line=dict(width=1, color="black")),
                text=medal_df.apply(
                    lambda r: f"{r['name']}<br>{r['championship']}<br>"
                              f"{r['mark']} (#{int(r['rank'])})", axis=1
                ),
                hovertemplate="%{text}<br>L1: %{x:.2f}s | L2: %{y:.2f}s<extra></extra>",
                name="Medalists",
            ))

        # All others coloured by finish time
        non_medal = df[~df.index.isin(medal_df.index)]
        fig_diff.add_trace(go.Scatter(
            x=non_medal["lap1_sec"], y=non_medal["lap2_sec"],
            mode="markers",
            marker=dict(
                size=7, color=non_medal["time_sec"],
                colorscale=[[0, TEAL_PRIMARY], [0.5, GOLD_ACCENT], [1, "#dc3545"]],
                colorbar=dict(title="Finish (s)"),
                opacity=0.7,
            ),
            text=non_medal.apply(
                lambda r: f"{r['name']}<br>{r['championship']}<br>{r['mark']}", axis=1
            ),
            hovertemplate="%{text}<br>L1: %{x:.2f}s | L2: %{y:.2f}s<extra></extra>",
            name="All Athletes",
        ))

        fig_diff.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{gender_label} 800m — 1st Lap vs 2nd Lap (below line = negative split)",
            xaxis_title="1st Lap (s)", yaxis_title="2nd Lap (s)",
            height=500,
        )
        fig_diff.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig_diff.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        st.plotly_chart(fig_diff, use_container_width=True)
        export_chart_button(fig_diff, f"{gender_label.lower()}_800m_differential_scatter.png", "exp_t2_scatter")

    with col_s2:
        st.markdown(f"**{gender_label} Key Insights**")

        neg_count = (df["differential"] < 0).sum()
        pos_count = (df["differential"] > 0).sum()
        even_count = len(df) - neg_count - pos_count

        st.markdown(f"""
        - **{neg_count}** negative splits ({neg_count/len(df)*100:.0f}%)
        - **{pos_count}** positive splits ({pos_count/len(df)*100:.0f}%)
        - Avg differential: **{df['differential'].mean():+.2f}s**
        """)

        if not medal_df.empty:
            st.markdown(f"- Medal avg diff: **{medal_df['differential'].mean():+.2f}s**")

        if not final_df.empty:
            st.markdown("**Finals Only:**")
            final_medal = final_df[final_df["rank"].between(1, 3)]
            gold_df = final_df[final_df["rank"] == 1]
            st.markdown(f"""
            - Finals avg diff: **{final_df['differential'].mean():+.2f}s**
            - Medal winners diff: **{final_medal['differential'].mean():+.2f}s** ({len(final_medal)} races)
            - Gold winners diff: **{gold_df['differential'].mean():+.2f}s** ({len(gold_df)} races)
            """)

    # Histogram of differentials
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df["differential"], nbinsx=40,
        marker_color=TEAL_PRIMARY, opacity=0.8, name="All",
    ))
    if not medal_df.empty:
        fig_hist.add_trace(go.Histogram(
            x=medal_df["differential"], nbinsx=20,
            marker_color=GOLD_ACCENT, opacity=0.8, name="Medalists",
        ))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Even Pace")
    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{gender_label} Distribution of Lap Differentials (2nd Lap - 1st Lap)",
        xaxis_title="Differential (seconds)", yaxis_title="Count",
        barmode="overlay", height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "Positive values = 2nd lap slower (fade). Negative = 2nd lap faster (negative split). "
        "Research shows positive splits of 1.5-3.0s produce the fastest 800m times (Hanley et al.)."
    )

    # Binned analysis: optimal differential by performance level
    render_section_header("Optimal Differential by Performance Level", "")

    if gender == "M":
        bins = [80, 104, 106, 108, 110, 115, 130]
        labels = ["<1:44", "1:44-1:46", "1:46-1:48", "1:48-1:50", "1:50-1:55", "1:55+"]
    else:
        bins = [80, 116, 118, 120, 122, 127, 140]
        labels = ["<1:56", "1:56-1:58", "1:58-2:00", "2:00-2:02", "2:02-2:07", "2:07+"]

    df["perf_bin"] = pd.cut(df["time_sec"], bins=bins, labels=labels, right=False)
    bin_stats = df.groupby("perf_bin", observed=True).agg(
        avg_diff=("differential", "mean"),
        avg_lap1=("lap1_sec", "mean"),
        n=("name", "count"),
    ).reset_index()

    if not bin_stats.empty:
        fig_bins = go.Figure()
        fig_bins.add_trace(go.Bar(
            x=bin_stats["perf_bin"].astype(str),
            y=bin_stats["avg_diff"],
            marker_color=[TEAL_PRIMARY if d < 2 else GOLD_ACCENT if d < 3.5 else "#dc3545"
                          for d in bin_stats["avg_diff"]],
            text=bin_stats.apply(lambda r: f"{r['avg_diff']:+.2f}s (n={r['n']})", axis=1),
            textposition="outside",
        ))
        fig_bins.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_bins.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{gender_label} Average Differential by Finish Time Range",
            xaxis_title="Finish Time Range", yaxis_title="Avg Differential (s)",
            height=350,
        )
        st.plotly_chart(fig_bins, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3: RUNNER TYPE CLASSIFICATION (Sandford Speed Reserve)
# ══════════════════════════════════════════════════════════════════════
with tab3:
    render_section_header(
        "Runner Type Classification",
        f"{gender_label} 800m — inspired by Sandford et al. Speed Reserve model"
    )

    st.markdown(f"""
    Runners classified by their **lap differential** pattern, inspired by Sandford et al.'s
    Speed Reserve Ratio (SRR = MSS/MAS) model. While we classify from race data rather than
    lab testing, the differential reveals whether an athlete runs as a **speed type** (fast
    start, positive split) or **endurance type** (conservative start, even/negative split).
    """)

    st.markdown("""
    | Type | Differential | Profile | Research Parallel |
    |------|-------------|---------|-------------------|
    | **Negative Split** | < -1.0s | Faster 2nd lap | Endurance Type (SRR ≤ 1.47) |
    | **Even Pace** | -1.0s to +0.5s | Near-equal laps | 800m Specialist (SRR 1.48-1.57) |
    | **Mild Positive** | +0.5s to +2.0s | Slight fade — most common | Balanced |
    | **Positive Split** | +2.0s to +4.0s | Notable fade | Speed Type (SRR ≥ 1.58) |
    | **Aggressive Fade** | > +4.0s | Significant deceleration | Over-extended start |
    """)

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        # Donut chart
        type_counts = df["runner_type"].value_counts()
        ordered_types = [t for t in RUNNER_TYPE_ORDER if t in type_counts.index]

        fig_donut = go.Figure(data=[go.Pie(
            labels=ordered_types,
            values=[type_counts[t] for t in ordered_types],
            marker_colors=[RUNNER_TYPE_COLORS[t] for t in ordered_types],
            textinfo="label+percent",
            hole=0.45,
            textposition="outside",
        )])
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{gender_label} Runner Type Distribution",
            height=400,
            annotations=[dict(text=f"n={len(df)}", x=0.5, y=0.5, font_size=16, showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r2:
        # Stats table by type
        type_stats = df.groupby("runner_type").agg(
            count=("name", "count"),
            avg_time=("time_sec", "mean"),
            avg_diff=("differential", "mean"),
            best_time=("time_sec", "min"),
            medal_pct=("rank", lambda x: (x <= 3).sum() / len(x) * 100 if len(x) > 0 else 0),
            finals_pct=("phase", lambda x: x.str.contains("Final", case=False, na=False).sum() / len(x) * 100),
        ).reindex([t for t in RUNNER_TYPE_ORDER if t in df["runner_type"].values])

        display_stats = pd.DataFrame({
            "N": type_stats["count"].astype(int),
            "Avg Time": type_stats["avg_time"].apply(seconds_to_time),
            "Avg Diff": type_stats["avg_diff"].apply(lambda x: f"{x:+.2f}s"),
            "Best": type_stats["best_time"].apply(seconds_to_time),
            "Medal %": type_stats["medal_pct"].apply(lambda x: f"{x:.0f}%"),
            "Finals %": type_stats["finals_pct"].apply(lambda x: f"{x:.0f}%"),
        })
        st.dataframe(display_stats, use_container_width=True)

    # Box plot by runner type
    fig_box = px.box(
        df, x="runner_type", y="time_sec",
        category_orders={"runner_type": RUNNER_TYPE_ORDER},
        color="runner_type",
        color_discrete_map=RUNNER_TYPE_COLORS,
        labels={"time_sec": "Finish Time (s)", "runner_type": "Runner Type"},
    )
    fig_box.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{gender_label} Finish Time Distribution by Runner Type",
        height=400, showlegend=False,
    )
    fig_box.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig_box.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption(
        "Lower boxes = faster times. Even Pace and Mild Positive typically produce "
        "the fastest performances. Aggressive Fade suggests the athlete over-extended early."
    )
    export_chart_button(fig_box, f"{gender_label.lower()}_800m_runner_types_box.png", "exp_t3_box")

    # Medal distribution by type
    if not medal_df.empty:
        render_section_header("Which Runner Type Wins Medals?", "")
        medal_types = medal_df["runner_type"].value_counts()
        fig_medal_type = go.Figure(data=[go.Bar(
            x=[t for t in RUNNER_TYPE_ORDER if t in medal_types.index],
            y=[medal_types.get(t, 0) for t in RUNNER_TYPE_ORDER if t in medal_types.index],
            marker_color=[RUNNER_TYPE_COLORS.get(t, GRAY_BLUE) for t in RUNNER_TYPE_ORDER if t in medal_types.index],
            text=[str(medal_types.get(t, 0)) for t in RUNNER_TYPE_ORDER if t in medal_types.index],
            textposition="outside",
        )])
        fig_medal_type.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{gender_label} Medal Count by Runner Type",
            xaxis_title="Runner Type", yaxis_title="Medals",
            height=350,
        )
        st.plotly_chart(fig_medal_type, use_container_width=True)

    with st.expander("Research Context — Speed Reserve Classification"):
        st.markdown("""
        **Sandford et al. (2019):** Classified elite 800m runners into three subgroups using
        **Speed Reserve Ratio (SRR = MSS / MAS)**:

        - **Speed Type (SRR ≥ 1.58):** Faster 400m PB, forward position at 200/400/600m
        - **800m Specialist (SRR 1.48-1.57):** Balanced speed/endurance, slower first 200m
        - **Endurance Type (SRR ≤ 1.47):** Faster 1500m PB, back-of-field positioning

        Key finding: *Speed between 400-600m had the strongest positive relationship with
        800m performance in ALL groups* — the critical "make or break" segment.

        *Citation: Sandford, G.N. et al. (2019). Anaerobic Speed Reserve: A Key Component
        of Elite Male 800-m Running. Int J Sports Physiol Perform.*
        """)


# ══════════════════════════════════════════════════════════════════════
# TAB 4: TACTICAL POSITIONING (Mytton / Renfree)
# ══════════════════════════════════════════════════════════════════════
with tab4:
    render_section_header(
        "Tactical Positioning Analysis",
        f"{gender_label} 800m — position at 400m vs final result (Mytton & Renfree)"
    )

    # We can use rank and lap speed as proxy for tactical position
    # Faster 1st lap = more forward position early
    st.markdown("""
    Research shows that **position at 400m correlates with final placement**, especially for
    endurance-type runners (Renfree, r=0.54-0.66). Post-2011, medals shifted from "sit and kick"
    to **front-running** strategies (Mytton et al., 2018).
    """)

    # Correlation: 1st lap speed rank vs final position
    if not final_df.empty and len(final_df) >= 5:
        finals_work = final_df.copy()
        finals_work["lap1_rank"] = finals_work.groupby("championship")["lap1_sec"].rank(method="min")

        col_t1, col_t2 = st.columns([2, 1])

        with col_t1:
            fig_tact = go.Figure()

            # Scatter: 1st lap relative speed vs final position
            fig_tact.add_trace(go.Scatter(
                x=finals_work["lap1_rank"],
                y=finals_work["rank"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=[GOLD_ACCENT if r <= 3 else TEAL_PRIMARY for r in finals_work["rank"]],
                    line=dict(width=1, color="white"),
                ),
                text=finals_work.apply(
                    lambda r: f"{r['name']}<br>{r['championship']}<br>"
                              f"L1 rank: {int(r['lap1_rank'])} → Final: #{int(r['rank'])}", axis=1
                ),
                hovertemplate="%{text}<extra></extra>",
                name="Finals",
            ))

            # Correlation line
            if len(finals_work) >= 5:
                slope, intercept, r_val, p_val, _ = sp_stats.linregress(
                    finals_work["lap1_rank"], finals_work["rank"]
                )
                x_line = np.array([1, finals_work["lap1_rank"].max()])
                fig_tact.add_trace(go.Scatter(
                    x=x_line, y=slope * x_line + intercept,
                    mode="lines", line=dict(dash="dash", color="#dc3545", width=2),
                    name=f"Trend (r={r_val:.2f}, p={p_val:.3f})",
                ))

            fig_tact.update_layout(
                **PLOTLY_LAYOUT,
                title=f"{gender_label} First Lap Position Rank vs Final Placement",
                xaxis_title="1st Lap Speed Rank (1 = fastest first lap)",
                yaxis_title="Final Position",
                height=450,
            )
            fig_tact.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
            fig_tact.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", autorange="reversed")
            st.plotly_chart(fig_tact, use_container_width=True)
            st.caption(
                "Each dot = one finals performance. Gold stars = medalists. "
                "A positive correlation means faster first-lap runners tend to finish higher."
            )
            export_chart_button(fig_tact, f"{gender_label.lower()}_800m_tactical_position.png", "exp_t4_tact")

        with col_t2:
            st.markdown(f"**{gender_label} Front-Runner Advantage**")

            # % of medalists who had top-3 fastest first lap
            top3_lap1 = finals_work[finals_work["lap1_rank"] <= 3]
            top3_medals = top3_lap1[top3_lap1["rank"] <= 3]

            if len(medal_df) > 0:
                front_pct = len(top3_medals) / len(finals_work[finals_work["rank"] <= 3]) * 100 if len(finals_work[finals_work["rank"] <= 3]) > 0 else 0
                st.markdown(f"- Top-3 at 400m who medal: **{front_pct:.0f}%**")

            if len(finals_work) >= 5:
                st.markdown(f"- Correlation (r): **{r_val:.2f}**")
                st.markdown(f"- Statistical significance (p): **{p_val:.3f}**")
                sig = "Yes" if p_val < 0.05 else "No"
                st.markdown(f"- Significant (p < 0.05): **{sig}**")

            st.markdown("---")
            st.markdown("**Key Research Finding:**")
            st.markdown("""
            > *"Winners separate themselves NOT by speeding up more, but by
            **avoiding slowing** compared with competitors in the final 200m."*
            — Hanley et al. (2019)
            """)

        # Era comparison: pre-2019 vs post-2019
        render_section_header("Era Comparison", "How has 800m tactics evolved?")

        early_champs = ["Beijing 2015 WC", "Rio 2016 OG", "London 2017 WC", "Doha 2019 WC"]
        late_champs = ["Tokyo 2021 OG", "Oregon 2022 WC", "Budapest 2023 WC", "Paris 2024 OG", "Tokyo 2025 WC"]

        early_df = final_df[final_df["championship"].isin(early_champs)]
        late_df = final_df[final_df["championship"].isin(late_champs)]

        if not early_df.empty and not late_df.empty:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                render_metric_card(
                    "2015-2019 Avg Differential",
                    f"{early_df['differential'].mean():+.2f}s",
                    "neutral"
                )
            with col_e2:
                render_metric_card(
                    "2021-2025 Avg Differential",
                    f"{late_df['differential'].mean():+.2f}s",
                    "gold"
                )

            st.markdown(f"""
            | Era | Avg 1st Lap | Avg 2nd Lap | Avg Diff | Avg Finish |
            |-----|------------|------------|----------|-----------|
            | 2015-2019 | {seconds_to_time(early_df['lap1_sec'].mean())} | {seconds_to_time(early_df['lap2_sec'].mean())} | {early_df['differential'].mean():+.2f}s | {seconds_to_time(early_df['time_sec'].mean())} |
            | 2021-2025 | {seconds_to_time(late_df['lap1_sec'].mean())} | {seconds_to_time(late_df['lap2_sec'].mean())} | {late_df['differential'].mean():+.2f}s | {seconds_to_time(late_df['time_sec'].mean())} |
            """)
    else:
        st.info("Select 'Finals Only' round filter for best tactical analysis, or ensure enough finals data exists.")

    with st.expander("Research Context — Tactical Evolution"):
        st.markdown("""
        **Mytton et al. (2018):** A "changing of the guard" occurred around 2011 in men's 800m:
        - **Pre-2011:** 7 of 9 championship golds won with "sit and kick" (back-to-front)
        - **Post-2011:** Front runners now dominate medals
        - Medalists post-2011: 1st lap **2.2 ± 1.1s** faster than 2nd lap

        **Renfree:** Position at 400m and 600m predicts qualification for endurance-type
        runners (r = 0.54-0.66, p < 0.01). Forward position is MORE critical for endurance
        runners than speed-dominant types.

        *Citations: Mytton, G.J. et al. (2018). Int J Sports Physiol Perform.
        Renfree, A. et al. J Sports Sciences.*
        """)


# ══════════════════════════════════════════════════════════════════════
# TAB 5: CHAMPIONSHIP COMPARISON & TRENDS
# ══════════════════════════════════════════════════════════════════════
with tab5:
    render_section_header(
        "Championship Comparison",
        f"{gender_label} 800m — pacing trends across major championships"
    )

    champ_stats = df.groupby("championship").agg(
        avg_time=("time_sec", "mean"),
        avg_lap1=("lap1_sec", "mean"),
        avg_lap2=("lap2_sec", "mean"),
        avg_diff=("differential", "mean"),
        n=("name", "count"),
    ).sort_values("avg_time")

    # Grouped bar: lap times by championship
    fig_champ = go.Figure()
    fig_champ.add_trace(go.Bar(
        x=champ_stats.index, y=champ_stats["avg_lap1"],
        name="1st Lap", marker_color=TEAL_PRIMARY,
    ))
    fig_champ.add_trace(go.Bar(
        x=champ_stats.index, y=champ_stats["avg_lap2"],
        name="2nd Lap", marker_color=GOLD_ACCENT,
    ))
    fig_champ.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{gender_label} Average Lap Times by Championship",
        xaxis_title="Championship", yaxis_title="Time (s)",
        barmode="group", height=400,
    )
    st.plotly_chart(fig_champ, use_container_width=True)
    export_chart_button(fig_champ, f"{gender_label.lower()}_800m_championship_laps.png", "exp_t5_champ")

    # Differential trend (chronological)
    champ_order = [c for c in CHAMP_CHRONOLOGICAL if c in champ_stats.index]

    if champ_order:
        render_section_header("Differential Trend", "Is the 800m getting more positive-split?")

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=champ_order,
            y=[champ_stats.loc[c, "avg_diff"] for c in champ_order],
            mode="lines+markers",
            line=dict(color=TEAL_PRIMARY, width=3),
            marker=dict(size=10, color=GOLD_ACCENT),
            name="Avg Differential",
            text=[f"n={int(champ_stats.loc[c, 'n'])}" for c in champ_order],
            hovertemplate="%{x}<br>Avg Diff: %{y:+.2f}s<br>%{text}<extra></extra>",
        ))
        fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{gender_label} Average Differential Trend Over Championships",
            yaxis_title="Differential (s)", height=350,
        )
        fig_trend.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption(
            "Positive trend = races becoming more positive-split over time. "
            "Negative trend = races becoming more evenly paced."
        )

    # Finals data table
    render_section_header("Finals Results Table", "")
    finals = df[df["phase"].str.contains("Final", case=False, na=False)].copy()
    if not finals.empty:
        display_df = finals[["championship", "rank", "name", "country", "mark",
                             "lap1_sec", "lap2_sec", "differential", "runner_type"]].copy()
        display_df.columns = ["Championship", "Pos", "Athlete", "Nat", "Time",
                              "1st Lap (s)", "2nd Lap (s)", "Diff (s)", "Type"]
        display_df = display_df.sort_values(["Championship", "Pos"])

        # Highlight medalists
        def highlight_medals(row):
            pos = row.get("Pos")
            if pos == 1:
                return ["background-color: rgba(160, 142, 102, 0.25)"] * len(row)
            elif pos == 2:
                return ["background-color: rgba(192, 192, 192, 0.2)"] * len(row)
            elif pos == 3:
                return ["background-color: rgba(205, 127, 50, 0.2)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_medals, axis=1),
            use_container_width=True, hide_index=True, height=400,
        )
    else:
        st.info("No finals data available with current filters.")

    # Diamond League comparison
    if not df_dl.empty:
        dl_gender = df_dl[df_dl["sex"] == gender].copy()
        if not dl_gender.empty:
            render_section_header(
                "Diamond League Comparison",
                f"{gender_label} 800m — Championship vs Diamond League times (no splits for DL)"
            )

            dl_gender = dl_gender[dl_gender["time_sec"].between(80, 130)]

            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                render_metric_card("DL Races", str(len(dl_gender)), "neutral")
            with col_dl2:
                render_metric_card("DL Avg Time", seconds_to_time(dl_gender["time_sec"].mean()), "good")
            with col_dl3:
                champ_avg = df["time_sec"].mean()
                dl_avg = dl_gender["time_sec"].mean()
                diff = dl_avg - champ_avg
                render_metric_card("DL vs Champs", f"{diff:+.2f}s", "gold" if diff < 0 else "neutral")

            # Top DL results
            if len(dl_gender) > 0:
                top_dl = dl_gender.nsmallest(15, "time_sec")[
                    ["competition_name", "date", "name", "country", "mark", "rank"]
                ].copy()
                top_dl.columns = ["Meeting", "Date", "Athlete", "Nat", "Time", "Pos"]
                st.dataframe(top_dl, use_container_width=True, hide_index=True)

            st.caption(
                "Diamond League results are times only — no split data is published by World Athletics for DL meetings. "
                "Run `python -m scrapers.scrape_800m` to update DL data."
            )

    # Race video links
    render_section_header("Race Videos", f"Watch {gender_label} 800m championship finals")

    video_cols = st.columns(3)
    vid_idx = 0
    for champ_key in CHAMP_CHRONOLOGICAL:
        vinfo = RACE_VIDEOS.get(champ_key)
        if vinfo and gender in vinfo:
            with video_cols[vid_idx % 3]:
                url = vinfo[gender]
                source = "Olympics.com" if "olympics.com" in url else "World Athletics"
                st.markdown(f"""
                <a href="{url}" target="_blank" style="text-decoration: none;">
                    <div style="background: white; border-radius: 8px; padding: 0.75rem;
                         border-left: 4px solid {GOLD_ACCENT}; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                         margin-bottom: 0.5rem;">
                        <div style="font-weight: 600; color: {TEAL_PRIMARY};">{vinfo['label']}</div>
                        <div style="font-size: 0.85rem; color: #666;">{gender_label} 800m Final &middot; {source}</div>
                    </div>
                </a>
                """, unsafe_allow_html=True)
            vid_idx += 1

    st.caption("Videos hosted on Olympics.com (Olympic Games) and World Athletics (World Championships). Click to open in new tab.")


# ══════════════════════════════════════════════════════════════════════
# TAB 6: SWEET SPOT FINDER
# ══════════════════════════════════════════════════════════════════════
with tab6:
    render_section_header(
        "Sweet Spot Finder",
        f"{gender_label} 800m — find the optimal pacing strategy for any target time"
    )

    col_sw1, col_sw2 = st.columns([2, 1])

    with col_sw1:
        fig_sweet = go.Figure()

        fig_sweet.add_trace(go.Scatter(
            x=df["lap1_sec"], y=df["differential"],
            mode="markers",
            marker=dict(
                size=8, color=df["time_sec"],
                colorscale=[[0, TEAL_PRIMARY], [0.5, GOLD_ACCENT], [1, "#dc3545"]],
                colorbar=dict(title="Finish (s)"),
                opacity=0.7,
            ),
            text=df.apply(
                lambda r: f"{r['name']}<br>{r['championship']}<br>"
                          f"Time: {r['mark']}<br>L1: {r['lap1_sec']:.2f}s | "
                          f"L2: {r['lap2_sec']:.2f}s<br>Diff: {r['differential']:+.2f}s", axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
            name="Athletes",
        ))

        # Sweet zone
        fig_sweet.add_hrect(
            y0=-1.0, y1=1.5, fillcolor=TEAL_PRIMARY, opacity=0.08,
            annotation_text="Sweet Zone", annotation_position="top left",
        )
        fig_sweet.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

        fig_sweet.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{gender_label} 1st Lap Time vs Differential (green = optimal zone)",
            xaxis_title="1st Lap Time (s)",
            yaxis_title="Differential (2nd - 1st lap, seconds)",
            height=500,
        )
        fig_sweet.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig_sweet.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        st.plotly_chart(fig_sweet, use_container_width=True)
        export_chart_button(fig_sweet, f"{gender_label.lower()}_800m_sweet_spot.png", "exp_t6_sweet")

    with col_sw2:
        st.markdown(f"**{gender_label} Correlations**")

        corr = df[["lap1_sec", "differential", "time_sec"]].corr()
        r_lap_diff = corr.loc["lap1_sec", "differential"]
        r_diff_time = corr.loc["differential", "time_sec"]

        st.markdown(f"""
        | Relationship | r value | Meaning |
        |---|---|---|
        | Faster 1st lap → bigger fade | **{r_lap_diff:.3f}** | {"Strong" if abs(r_lap_diff) > 0.5 else "Moderate" if abs(r_lap_diff) > 0.3 else "Weak"} |
        | Bigger fade → slower finish | **{r_diff_time:.3f}** | {"Strong" if abs(r_diff_time) > 0.5 else "Moderate" if abs(r_diff_time) > 0.3 else "Weak"} |
        """)

        st.markdown("---")
        st.markdown("**Sweet Zone:** -1.0s to +1.5s differential")
        sweet_zone = df[(df["differential"] >= -1.0) & (df["differential"] <= 1.5)]
        st.markdown(f"- Athletes in zone: **{len(sweet_zone)}** ({len(sweet_zone)/len(df)*100:.0f}%)")
        if not sweet_zone.empty:
            st.markdown(f"- Their avg time: **{seconds_to_time(sweet_zone['time_sec'].mean())}**")
            outside = df[~df.index.isin(sweet_zone.index)]
            if not outside.empty:
                st.markdown(f"- Outside zone avg: **{seconds_to_time(outside['time_sec'].mean())}**")

    # Interactive target time finder
    render_section_header("Coach's Pacing Calculator", "Enter a target time to get optimal split recommendations")

    if gender == "M":
        min_val, max_val, default_val = 100, 120, 108
    else:
        min_val, max_val, default_val = 115, 135, 122

    target_time = st.slider(
        f"Target {gender_label} 800m finish time (seconds)",
        min_value=min_val, max_value=max_val,
        value=default_val, step=1,
        key="ra_target",
    )

    nearby = df[(df["time_sec"] >= target_time - 3) & (df["time_sec"] <= target_time + 3)]

    if not nearby.empty:
        rec_l1 = nearby["lap1_sec"].mean()
        rec_l2 = nearby["lap2_sec"].mean()
        rec_diff = nearby["differential"].mean()

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            render_metric_card("Target", seconds_to_time(target_time), "gold")
        with col_c2:
            render_metric_card("Recommended 1st Lap", seconds_to_time(rec_l1), "excellent")
        with col_c3:
            render_metric_card("Expected 2nd Lap", seconds_to_time(rec_l2), "good")
        with col_c4:
            render_metric_card("Typical Differential", f"{rec_diff:+.2f}s", "neutral")

        st.markdown(f"*Based on {len(nearby)} performances within ±3s of target ({seconds_to_time(target_time)})*")

        # Coach's recommendation
        st.markdown(f"""
        > **Coach's Recommendation:** To run **{seconds_to_time(target_time)}**, aim for a first lap of
        **{seconds_to_time(rec_l1)}** (±1s), allowing a controlled fade of **{abs(rec_diff):.1f}s** in the
        second lap. {"Avoid going out faster than " + seconds_to_time(rec_l1 - 2) + " — risk of aggressive fade." if rec_diff > 0 else "You can afford a slightly faster first lap at this level."}
        """)

        # Best performers in range
        best = nearby.nsmallest(10, "time_sec")[
            ["name", "country", "championship", "mark", "lap1_sec", "lap2_sec", "differential", "runner_type"]
        ].copy()
        best.columns = ["Athlete", "Nat", "Championship", "Time", "1st Lap", "2nd Lap", "Diff", "Type"]
        st.dataframe(best, use_container_width=True, hide_index=True)
    else:
        st.info("No performances found near this target time. Try adjusting the slider.")


# ══════════════════════════════════════════════════════════════════════
# TAB 7: ATHLETE DEEP DIVE
# ══════════════════════════════════════════════════════════════════════
with tab7:
    render_section_header(
        "Athlete Deep Dive",
        f"{gender_label} 800m — individual pacing profiles and consistency"
    )

    athlete_counts = df["name"].value_counts()
    athletes_with_data = athlete_counts[athlete_counts >= 2].index.tolist()

    if not athletes_with_data:
        st.info("Need at least 2 races per athlete for deep dive analysis. Try 'All Championships' filter.")
    else:
        selected_athlete = st.selectbox(
            "Select Athlete",
            athletes_with_data,
            format_func=lambda x: f"{x} ({athlete_counts[x]} races)",
            key="ra_athlete",
        )

        athlete_df = df[df["name"] == selected_athlete].sort_values("time_sec")

        if not athlete_df.empty:
            # Profile card
            col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)
            with col_a1:
                render_metric_card("Races", str(len(athlete_df)), "neutral")
            with col_a2:
                render_metric_card("Best Time", seconds_to_time(athlete_df["time_sec"].min()), "gold")
            with col_a3:
                render_metric_card("Avg Differential", f"{athlete_df['differential'].mean():+.2f}s", "excellent")
            with col_a4:
                dominant = athlete_df["runner_type"].mode().iloc[0]
                render_metric_card("Dominant Type", dominant, "good")
            with col_a5:
                best_pos = int(athlete_df["rank"].min()) if athlete_df["rank"].notna().any() else 0
                render_metric_card("Best Position", f"#{best_pos}" if best_pos > 0 else "N/A", "neutral")

            # Lap times across championships
            fig_ath = go.Figure()
            fig_ath.add_trace(go.Bar(
                x=athlete_df["championship"], y=athlete_df["lap1_sec"],
                name="1st Lap", marker_color=TEAL_PRIMARY,
            ))
            fig_ath.add_trace(go.Bar(
                x=athlete_df["championship"], y=athlete_df["lap2_sec"],
                name="2nd Lap", marker_color=GOLD_ACCENT,
            ))
            fig_ath.update_layout(
                **PLOTLY_LAYOUT,
                title=f"{selected_athlete} — Lap Times Across Championships",
                yaxis_title="Time (s)", barmode="group", height=400,
            )
            st.plotly_chart(fig_ath, use_container_width=True)

            # Pacing consistency
            render_section_header("Pacing Consistency", "")

            col_c1, col_c2 = st.columns([1, 1])

            with col_c1:
                cv = athlete_df["cv_pct"].mean()
                cohort_cv = df["cv_pct"].mean()
                st.markdown(f"""
                | Metric | Athlete | Cohort Avg |
                |--------|---------|-----------|
                | Pacing CV% | **{cv:.1f}%** | {cohort_cv:.1f}% |
                | Avg Differential | **{athlete_df['differential'].mean():+.2f}s** | {df['differential'].mean():+.2f}s |
                | Std Dev of Diff | **{athlete_df['differential'].std():.2f}s** | {df['differential'].std():.2f}s |
                | Fastest 1st Lap | **{seconds_to_time(athlete_df['lap1_sec'].min())}** | {seconds_to_time(df['lap1_sec'].min())} |
                """)

                consistency = "very consistent" if cv < 3 else "consistent" if cv < 5 else "variable" if cv < 8 else "highly variable"
                st.markdown(f"**Assessment:** {selected_athlete}'s pacing is **{consistency}** "
                            f"(CV = {cv:.1f}%, {'below' if cv < cohort_cv else 'above'} cohort average of {cohort_cv:.1f}%).")

            with col_c2:
                # Comparison to similar-time runners
                athlete_avg_time = athlete_df["time_sec"].mean()
                similar = df[(df["time_sec"] >= athlete_avg_time - 2) & (df["time_sec"] <= athlete_avg_time + 2)]
                if not similar.empty:
                    st.markdown(f"**vs Runners with similar times (±2s):**")
                    st.markdown(f"""
                    | Metric | {selected_athlete} | Similar Runners (n={len(similar)}) |
                    |--------|---------|-----------|
                    | Avg Diff | **{athlete_df['differential'].mean():+.2f}s** | {similar['differential'].mean():+.2f}s |
                    | Avg 1st Lap | **{seconds_to_time(athlete_df['lap1_sec'].mean())}** | {seconds_to_time(similar['lap1_sec'].mean())} |
                    """)

            # Full race detail table
            render_section_header("Race History", "")
            detail = athlete_df[
                ["championship", "phase", "rank", "mark", "lap1_sec", "lap2_sec",
                 "differential", "cv_pct", "runner_type"]
            ].copy()
            detail.columns = ["Championship", "Round", "Pos", "Time", "1st Lap", "2nd Lap",
                              "Diff", "CV%", "Type"]
            st.dataframe(detail, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"**{gender_label} 800m Race Analysis** | "
    f"{len(df_results)} results, {len(df_splits)} splits across "
    f"{df_results['championship'].nunique()} championships | "
    f"Data: World Athletics GraphQL API"
)
st.caption(
    "Research: Casado et al. (2021), Sandford et al. (2019), "
    "Hanley et al. (2019/2021), Mytton et al. (2018), Renfree et al."
)
