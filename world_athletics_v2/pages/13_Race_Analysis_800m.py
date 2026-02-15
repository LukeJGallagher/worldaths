"""
800m Race Analysis - Split Differentials, Runner Types & Sweet Spots

Analyses 800m splits from major championships (Olympics, World Championships)
to identify pacing strategies, runner types, and optimal differential patterns.

Data: World Athletics GraphQL API - getEventTimetableWithContent splits
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path

st.set_page_config(page_title="800m Race Analysis", page_icon="🏃", layout="wide")

# ── Brand colors ──────────────────────────────────────────────────────
SAUDI_GREEN = "#235032"
DARK_GREEN = "#1a3d25"
GOLD_ACCENT = "#a08e66"
LIGHT_GREEN = "#3a7050"
GRAY_BLUE = "#78909C"

PLOTLY_LAYOUT = {
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "font": {"family": "Inter, sans-serif", "color": "#333"},
    "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
}


# ── Helper: parse time string to seconds ──────────────────────────────
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
    if pd.isna(s):
        return ""
    mins = int(s // 60)
    secs = s - mins * 60
    if mins > 0:
        return f"{mins}:{secs:05.2f}"
    return f"{secs:.2f}"


# ── Load data ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    data_dir = Path(__file__).parent.parent / "data" / "scraped"

    results_path = data_dir / "800m_championship_results.parquet"
    splits_path = data_dir / "800m_championship_splits.parquet"

    if not results_path.exists() or not splits_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    df_results = pd.read_parquet(results_path)
    df_splits = pd.read_parquet(splits_path)

    # Parse times
    df_results["time_sec"] = df_results["mark"].apply(time_to_seconds)
    df_splits["split_sec"] = df_splits["split_mark"].apply(time_to_seconds)

    # Normalize split names (remove space inconsistencies)
    df_splits["split_dist"] = df_splits["split_name"].str.replace(" ", "").str.replace("m", "").astype(float)

    return df_results, df_splits


@st.cache_data(ttl=3600)
def build_lap_analysis(df_results: pd.DataFrame, df_splits: pd.DataFrame):
    """Build per-athlete lap analysis with 1st/2nd lap times and differentials."""

    # Get 400m splits for each athlete in each race
    df_400 = df_splits[df_splits["split_dist"] == 400].copy()
    df_400 = df_400.rename(columns={"split_sec": "lap1_sec"})

    # Merge with final results to get finish time
    merged = df_400.merge(
        df_results[["championship", "phase", "sex", "name", "time_sec", "rank", "mark"]],
        on=["championship", "phase", "sex", "name"],
        how="inner",
    )

    # Calculate 2nd lap
    merged["lap2_sec"] = merged["time_sec"] - merged["lap1_sec"]
    merged["differential"] = merged["lap2_sec"] - merged["lap1_sec"]
    merged["diff_pct"] = (merged["differential"] / merged["lap1_sec"]) * 100

    # Filter out bad data
    merged = merged[
        (merged["lap1_sec"] > 20) & (merged["lap1_sec"] < 65)
        & (merged["lap2_sec"] > 20) & (merged["lap2_sec"] < 65)
        & (merged["time_sec"] > 80) & (merged["time_sec"] < 130)
    ].copy()

    # Classify runner type
    def classify_runner(row):
        d = row["differential"]
        if d < -1.0:
            return "Negative Split"
        elif d < 0.5:
            return "Even Pace"
        elif d < 2.0:
            return "Mild Positive"
        elif d < 4.0:
            return "Positive Split"
        else:
            return "Aggressive Fade"

    merged["runner_type"] = merged.apply(classify_runner, axis=1)

    return merged


def render_header():
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {SAUDI_GREEN} 0%, {DARK_GREEN} 100%);
         padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid {GOLD_ACCENT};">
        <h2 style="color: white; margin: 0;">800m Race Analysis</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Split differentials, pacing strategies & runner types from major championships
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────
render_header()

df_results, df_splits = load_data()

if df_results.empty or df_splits.empty:
    st.error("No 800m split data found. Run the scraper first.")
    st.stop()

df_laps = build_lap_analysis(df_results, df_splits)

if df_laps.empty:
    st.error("Could not compute lap analysis - check split data quality.")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    gender = st.selectbox("Gender", ["M", "W"], format_func=lambda x: "Men" if x == "M" else "Women")
with col_f2:
    champ_options = ["All Championships"] + sorted(df_laps["championship"].unique().tolist())
    champ_filter = st.selectbox("Championship", champ_options)
with col_f3:
    phase_options = ["All Rounds"] + sorted(df_laps["phase"].unique().tolist())
    phase_filter = st.selectbox("Round", phase_options)

# Apply filters
df = df_laps[df_laps["sex"] == gender].copy()
if champ_filter != "All Championships":
    df = df[df["championship"] == champ_filter]
if phase_filter != "All Rounds":
    df = df[df["phase"] == phase_filter]

if df.empty:
    st.warning("No data for this filter combination.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────
st.markdown("---")

medal_df = df[df["rank"].between(1, 3)]
final_df = df[df["phase"].str.contains("Final", case=False, na=False)]

cols = st.columns(5)
with cols[0]:
    st.metric("Races Analysed", f"{len(df)}")
with cols[1]:
    st.metric("Avg Finish", seconds_to_time(df["time_sec"].mean()))
with cols[2]:
    st.metric("Avg 1st Lap", seconds_to_time(df["lap1_sec"].mean()))
with cols[3]:
    st.metric("Avg 2nd Lap", seconds_to_time(df["lap2_sec"].mean()))
with cols[4]:
    st.metric("Avg Differential", f"{df['differential'].mean():+.2f}s")

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Differential Analysis",
    "Runner Types",
    "Sweet Spot Finder",
    "Championship Comparison",
    "Athlete Deep Dive",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: DIFFERENTIAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("1st Lap vs 2nd Lap Differential")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Scatter: Lap 1 vs Lap 2 colored by finish time
        fig = go.Figure()

        # Even pace line
        min_lap = min(df["lap1_sec"].min(), df["lap2_sec"].min()) - 1
        max_lap = max(df["lap1_sec"].max(), df["lap2_sec"].max()) + 1
        fig.add_trace(go.Scatter(
            x=[min_lap, max_lap], y=[min_lap, max_lap],
            mode="lines", line=dict(dash="dash", color="gray", width=1),
            name="Even Pace", showlegend=True,
        ))

        # Medal winners
        if not medal_df.empty:
            fig.add_trace(go.Scatter(
                x=medal_df["lap1_sec"], y=medal_df["lap2_sec"],
                mode="markers",
                marker=dict(size=12, color=GOLD_ACCENT, symbol="star", line=dict(width=1, color="black")),
                text=medal_df.apply(lambda r: f"{r['name']}<br>{r['championship']}<br>{r['mark']} (#{int(r['rank'])})", axis=1),
                hovertemplate="%{text}<br>L1: %{x:.2f}s | L2: %{y:.2f}s<extra></extra>",
                name="Medalists",
            ))

        # All others
        non_medal = df[~df.index.isin(medal_df.index)]
        fig.add_trace(go.Scatter(
            x=non_medal["lap1_sec"], y=non_medal["lap2_sec"],
            mode="markers",
            marker=dict(
                size=7, color=non_medal["time_sec"],
                colorscale=[[0, SAUDI_GREEN], [0.5, GOLD_ACCENT], [1, "#dc3545"]],
                colorbar=dict(title="Finish<br>Time (s)"),
                opacity=0.7,
            ),
            text=non_medal.apply(lambda r: f"{r['name']}<br>{r['championship']}<br>{r['mark']}", axis=1),
            hovertemplate="%{text}<br>L1: %{x:.2f}s | L2: %{y:.2f}s<extra></extra>",
            name="All Athletes",
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="1st Lap vs 2nd Lap (below line = negative split)",
            xaxis_title="1st Lap (s)",
            yaxis_title="2nd Lap (s)",
            height=500,
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Key Insights**")

        neg_split = df[df["differential"] < 0]
        pos_split = df[df["differential"] > 0]

        st.markdown(f"""
        - **{len(neg_split)}** negative splits ({len(neg_split)/len(df)*100:.0f}%)
        - **{len(pos_split)}** positive splits ({len(pos_split)/len(df)*100:.0f}%)
        - Avg differential: **{df['differential'].mean():+.2f}s**
        - Medal avg diff: **{medal_df['differential'].mean():+.2f}s** (if medals exist)
        """)

        if not final_df.empty:
            st.markdown("**Finals Only:**")
            final_medal = final_df[final_df["rank"].between(1, 3)]
            st.markdown(f"""
            - Finals avg diff: **{final_df['differential'].mean():+.2f}s**
            - Medal winners diff: **{final_medal['differential'].mean():+.2f}s**
            - Gold winners diff: **{final_df[final_df['rank']==1]['differential'].mean():+.2f}s**
            """)

    # Histogram of differentials
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df["differential"],
        nbinsx=40,
        marker_color=SAUDI_GREEN,
        opacity=0.8,
        name="All",
    ))
    if not medal_df.empty:
        fig_hist.add_trace(go.Histogram(
            x=medal_df["differential"],
            nbinsx=20,
            marker_color=GOLD_ACCENT,
            opacity=0.8,
            name="Medalists",
        ))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Even Pace")
    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        title="Distribution of Lap Differentials (2nd Lap - 1st Lap)",
        xaxis_title="Differential (seconds)",
        yaxis_title="Count",
        barmode="overlay",
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: RUNNER TYPES
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Runner Type Classification")

    st.markdown("""
    | Type | Differential | Description |
    |------|-------------|-------------|
    | **Negative Split** | < -1.0s | Faster 2nd lap - rare at championship level |
    | **Even Pace** | -1.0s to +0.5s | Near-equal laps - optimal for most |
    | **Mild Positive** | +0.5s to +2.0s | Slight fade - most common |
    | **Positive Split** | +2.0s to +4.0s | Notable fade - aggressive 1st lap |
    | **Aggressive Fade** | > +4.0s | Significant deceleration |
    """)

    # Runner type distribution
    type_counts = df["runner_type"].value_counts()
    type_order = ["Negative Split", "Even Pace", "Mild Positive", "Positive Split", "Aggressive Fade"]
    type_colors = {
        "Negative Split": "#0077B6",
        "Even Pace": SAUDI_GREEN,
        "Mild Positive": GOLD_ACCENT,
        "Positive Split": "#FFB800",
        "Aggressive Fade": "#dc3545",
    }

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=[t for t in type_order if t in type_counts.index],
            values=[type_counts.get(t, 0) for t in type_order if t in type_counts.index],
            marker_colors=[type_colors.get(t, GRAY_BLUE) for t in type_order if t in type_counts.index],
            textinfo="label+percent",
            hole=0.4,
        )])
        fig_pie.update_layout(**PLOTLY_LAYOUT, title="Distribution of Runner Types", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Avg finish time by runner type
        type_stats = df.groupby("runner_type").agg(
            avg_time=("time_sec", "mean"),
            avg_diff=("differential", "mean"),
            count=("name", "count"),
            best_time=("time_sec", "min"),
            medal_pct=("rank", lambda x: (x <= 3).sum() / len(x) * 100 if len(x) > 0 else 0),
        ).reindex([t for t in type_order if t in df["runner_type"].values])

        type_stats["avg_time_fmt"] = type_stats["avg_time"].apply(seconds_to_time)
        type_stats["best_time_fmt"] = type_stats["best_time"].apply(seconds_to_time)

        st.dataframe(
            type_stats[["count", "avg_time_fmt", "avg_diff", "best_time_fmt", "medal_pct"]].rename(columns={
                "count": "N",
                "avg_time_fmt": "Avg Time",
                "avg_diff": "Avg Diff (s)",
                "best_time_fmt": "Best Time",
                "medal_pct": "Medal %",
            }),
            use_container_width=True,
        )

    # Box plot by runner type
    fig_box = px.box(
        df, x="runner_type", y="time_sec",
        category_orders={"runner_type": type_order},
        color="runner_type",
        color_discrete_map=type_colors,
        labels={"time_sec": "Finish Time (s)", "runner_type": "Runner Type"},
    )
    fig_box.update_layout(**PLOTLY_LAYOUT, title="Finish Time Distribution by Runner Type", height=400, showlegend=False)
    fig_box.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig_box.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    st.plotly_chart(fig_box, use_container_width=True)

    # Which type wins medals?
    if not medal_df.empty:
        st.subheader("Which Runner Type Wins Medals?")
        medal_types = medal_df["runner_type"].value_counts()
        fig_medal = go.Figure(data=[go.Bar(
            x=[t for t in type_order if t in medal_types.index],
            y=[medal_types.get(t, 0) for t in type_order if t in medal_types.index],
            marker_color=[type_colors.get(t, GRAY_BLUE) for t in type_order if t in medal_types.index],
        )])
        fig_medal.update_layout(**PLOTLY_LAYOUT, title="Medal Count by Runner Type", height=350,
                                xaxis_title="Runner Type", yaxis_title="Medals")
        st.plotly_chart(fig_medal, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: SWEET SPOT FINDER
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Is There a Sweet Spot?")
    st.markdown("Exploring the relationship between 1st lap pace, differential, and finish time.")

    # Heatmap: 1st lap time vs differential -> finish time
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_sweet = go.Figure()

        # Color by rank (lower = better)
        fig_sweet.add_trace(go.Scatter(
            x=df["lap1_sec"],
            y=df["differential"],
            mode="markers",
            marker=dict(
                size=8,
                color=df["time_sec"],
                colorscale=[[0, SAUDI_GREEN], [0.5, GOLD_ACCENT], [1, "#dc3545"]],
                colorbar=dict(title="Finish (s)"),
                opacity=0.7,
            ),
            text=df.apply(lambda r: f"{r['name']}<br>{r['championship']}<br>Time: {r['mark']}<br>L1: {r['lap1_sec']:.2f}s | L2: {r['lap2_sec']:.2f}s<br>Diff: {r['differential']:+.2f}s", axis=1),
            hovertemplate="%{text}<extra></extra>",
            name="Athletes",
        ))

        # Mark the "sweet spot" zone
        fig_sweet.add_hrect(y0=-1.0, y1=1.5, fillcolor=SAUDI_GREEN, opacity=0.08,
                            annotation_text="Sweet Spot Zone", annotation_position="top left")
        fig_sweet.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

        fig_sweet.update_layout(
            **PLOTLY_LAYOUT,
            title="1st Lap Speed vs Differential (green zone = optimal)",
            xaxis_title="1st Lap Time (s)",
            yaxis_title="Differential (2nd - 1st lap, seconds)",
            height=500,
        )
        fig_sweet.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig_sweet.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        st.plotly_chart(fig_sweet, use_container_width=True)

    with col2:
        st.markdown("**Sweet Spot Analysis**")

        # Bin by finish time and compute optimal differential
        time_bins = pd.cut(df["time_sec"], bins=6)
        bin_stats = df.groupby(time_bins, observed=True).agg(
            avg_diff=("differential", "mean"),
            avg_lap1=("lap1_sec", "mean"),
            n=("name", "count"),
        ).reset_index()

        st.markdown("**Optimal Differential by Performance Level:**")
        for _, row in bin_stats.iterrows():
            time_range = str(row["time_sec"])
            st.markdown(f"- {time_range}: **{row['avg_diff']:+.2f}s** diff (n={row['n']})")

        # Correlation
        corr = df[["lap1_sec", "differential", "time_sec"]].corr()
        st.markdown(f"""
        **Correlations:**
        - Faster 1st lap vs bigger fade: **r={corr.loc['lap1_sec', 'differential']:.3f}**
        - Differential vs finish time: **r={corr.loc['differential', 'time_sec']:.3f}**
        """)

    # Individual sweet spot - regression
    st.subheader("Optimal Pacing by Finish Time Target")

    target_time = st.slider(
        "Target finish time (seconds)",
        min_value=int(df["time_sec"].min()),
        max_value=int(df["time_sec"].max()),
        value=int(df["time_sec"].median()),
    )

    nearby = df[(df["time_sec"] >= target_time - 3) & (df["time_sec"] <= target_time + 3)]
    if not nearby.empty:
        cols = st.columns(4)
        with cols[0]:
            st.metric("Target", seconds_to_time(target_time))
        with cols[1]:
            st.metric("Optimal 1st Lap", seconds_to_time(nearby["lap1_sec"].mean()))
        with cols[2]:
            st.metric("Expected 2nd Lap", seconds_to_time(nearby["lap2_sec"].mean()))
        with cols[3]:
            st.metric("Typical Differential", f"{nearby['differential'].mean():+.2f}s")

        st.markdown(f"*Based on {len(nearby)} performances within ±3s of target*")

        # Show the best performers in this range
        best = nearby.nsmallest(10, "time_sec")[["name", "country", "championship", "mark", "lap1_sec", "lap2_sec", "differential", "runner_type"]]
        best.columns = ["Athlete", "Country", "Championship", "Time", "1st Lap", "2nd Lap", "Diff", "Type"]
        st.dataframe(best, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: CHAMPIONSHIP COMPARISON
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Pacing Trends Across Championships")

    # Compare differentials across championships
    champ_stats = df.groupby("championship").agg(
        avg_time=("time_sec", "mean"),
        avg_lap1=("lap1_sec", "mean"),
        avg_lap2=("lap2_sec", "mean"),
        avg_diff=("differential", "mean"),
        n=("name", "count"),
    ).sort_values("avg_time")

    fig_champ = go.Figure()
    fig_champ.add_trace(go.Bar(
        x=champ_stats.index,
        y=champ_stats["avg_lap1"],
        name="1st Lap",
        marker_color=SAUDI_GREEN,
    ))
    fig_champ.add_trace(go.Bar(
        x=champ_stats.index,
        y=champ_stats["avg_lap2"],
        name="2nd Lap",
        marker_color=GOLD_ACCENT,
    ))
    fig_champ.update_layout(
        **PLOTLY_LAYOUT,
        title="Average Lap Times by Championship",
        xaxis_title="Championship",
        yaxis_title="Time (s)",
        barmode="group",
        height=400,
    )
    st.plotly_chart(fig_champ, use_container_width=True)

    # Differential trend
    champ_order = ["Beijing 2015 WC", "Rio 2016 OG", "London 2017 WC", "Doha 2019 WC",
                   "Tokyo 2021 OG", "Oregon 2022 WC", "Budapest 2023 WC", "Paris 2024 OG", "Tokyo 2025 WC"]
    champ_order = [c for c in champ_order if c in champ_stats.index]

    if champ_order:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=champ_order,
            y=[champ_stats.loc[c, "avg_diff"] for c in champ_order],
            mode="lines+markers",
            line=dict(color=SAUDI_GREEN, width=3),
            marker=dict(size=10, color=GOLD_ACCENT),
            name="Avg Differential",
        ))
        fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            title="Average Differential Trend Over Championships",
            yaxis_title="Differential (s)",
            height=350,
        )
        fig_trend.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        st.plotly_chart(fig_trend, use_container_width=True)

    # Finals table
    st.subheader("Finals Data")
    finals = df[df["phase"].str.contains("Final", case=False, na=False)].copy()
    if not finals.empty:
        display_cols = finals[["championship", "rank", "name", "country", "mark",
                               "lap1_sec", "lap2_sec", "differential", "runner_type"]].copy()
        display_cols.columns = ["Championship", "Pos", "Athlete", "Nat", "Time",
                                "1st Lap (s)", "2nd Lap (s)", "Diff (s)", "Type"]
        display_cols = display_cols.sort_values(["Championship", "Pos"])
        st.dataframe(display_cols, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 5: ATHLETE DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Individual Athlete Analysis")

    # Athletes with most data points
    athlete_counts = df["name"].value_counts()
    athletes_with_data = athlete_counts[athlete_counts >= 2].index.tolist()

    if not athletes_with_data:
        st.info("Need at least 2 races per athlete for deep dive analysis.")
    else:
        selected_athlete = st.selectbox(
            "Select Athlete",
            athletes_with_data,
            format_func=lambda x: f"{x} ({athlete_counts[x]} races)",
        )

        athlete_df = df[df["name"] == selected_athlete].sort_values("time_sec")

        if not athlete_df.empty:
            # Stats
            cols = st.columns(5)
            with cols[0]:
                st.metric("Races", len(athlete_df))
            with cols[1]:
                st.metric("Best Time", seconds_to_time(athlete_df["time_sec"].min()))
            with cols[2]:
                st.metric("Avg Diff", f"{athlete_df['differential'].mean():+.2f}s")
            with cols[3]:
                dominant_type = athlete_df["runner_type"].mode().iloc[0] if not athlete_df.empty else "?"
                st.metric("Dominant Type", dominant_type)
            with cols[4]:
                st.metric("Best Pos", f"#{int(athlete_df['rank'].min())}" if athlete_df["rank"].notna().any() else "?")

            # Race history
            fig_athlete = go.Figure()
            fig_athlete.add_trace(go.Bar(
                x=athlete_df["championship"],
                y=athlete_df["lap1_sec"],
                name="1st Lap",
                marker_color=SAUDI_GREEN,
            ))
            fig_athlete.add_trace(go.Bar(
                x=athlete_df["championship"],
                y=athlete_df["lap2_sec"],
                name="2nd Lap",
                marker_color=GOLD_ACCENT,
            ))
            fig_athlete.update_layout(
                **PLOTLY_LAYOUT,
                title=f"{selected_athlete} - Lap Times Across Championships",
                yaxis_title="Time (s)",
                barmode="group",
                height=400,
            )
            st.plotly_chart(fig_athlete, use_container_width=True)

            # Detail table
            detail = athlete_df[["championship", "phase", "rank", "mark", "lap1_sec", "lap2_sec", "differential", "runner_type"]].copy()
            detail.columns = ["Championship", "Round", "Pos", "Time", "1st Lap", "2nd Lap", "Diff", "Type"]
            st.dataframe(detail, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Data: World Athletics GraphQL API | {len(df_results)} results, {len(df_splits)} splits across {df_results['championship'].nunique()} championships")
