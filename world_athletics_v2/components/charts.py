"""
Plotly chart builders with Team Saudi styling.

All charts return go.Figure objects ready for st.plotly_chart().
"""

from typing import List, Optional, Dict, Any

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .theme import (
    TEAL_PRIMARY, TEAL_DARK, TEAL_LIGHT, GOLD_ACCENT,
    GRAY_BLUE, CHART_COLORS, PLOTLY_LAYOUT, STATUS_DANGER,
)


def _apply_base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply Team Saudi base layout to any figure."""
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=16, color="#333")),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    return fig


def progression_chart(
    df: pd.DataFrame,
    x_col: str = "date",
    y_col: str = "mark",
    title: str = "Performance Progression",
    pb_value: Optional[float] = None,
    sb_value: Optional[float] = None,
    lower_is_better: bool = True,
    hover_cols: Optional[List[str]] = None,
    show_trend: bool = True,
) -> go.Figure:
    """Line chart showing marks over time with PB/SB lines, trend, and star on best mark."""
    fig = go.Figure()

    # Build hover text from extra columns
    hover_text = None
    if hover_cols:
        hover_parts = []
        for _, row in df.iterrows():
            parts = []
            for hc in hover_cols:
                if hc in df.columns:
                    val = row.get(hc)
                    if pd.notna(val) and str(val) not in ("None", "nan", ""):
                        parts.append(f"{hc}: {val}")
            hover_parts.append("<br>".join(parts))
        hover_text = hover_parts

    # Main trace with larger markers
    scatter_kwargs = dict(
        x=df[x_col], y=df[y_col],
        mode="lines+markers",
        line=dict(color=TEAL_PRIMARY, width=2.5),
        marker=dict(size=8, color=TEAL_PRIMARY, line=dict(width=1, color="white")),
        name="Performance",
    )
    if hover_text:
        scatter_kwargs["hovertext"] = hover_text
        scatter_kwargs["hovertemplate"] = "%{x}<br><b>%{y:.2f}</b><br>%{hovertext}<extra></extra>"
    else:
        scatter_kwargs["hovertemplate"] = "%{x}<br><b>%{y:.2f}</b><extra></extra>"
    fig.add_trace(go.Scatter(**scatter_kwargs))

    # Trend line (rolling average)
    y_vals = pd.to_numeric(df[y_col], errors="coerce")
    valid_mask = y_vals.notna()
    if show_trend and valid_mask.sum() >= 4:
        window = max(3, valid_mask.sum() // 4)
        rolling_avg = y_vals.rolling(window=window, min_periods=2, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df[x_col], y=rolling_avg,
            mode="lines",
            line=dict(color=GOLD_ACCENT, width=2, dash="dot"),
            name="Trend",
            opacity=0.7,
        ))

    # Highlight best performance with star
    if valid_mask.sum() > 0:
        best_idx = y_vals[valid_mask].idxmin() if lower_is_better else y_vals[valid_mask].idxmax()
        fig.add_trace(go.Scatter(
            x=[df.loc[best_idx, x_col]],
            y=[df.loc[best_idx, y_col]],
            mode="markers",
            marker=dict(symbol="star", size=14, color=GOLD_ACCENT, line=dict(width=1.5, color="#333")),
            name="Best",
        ))

    # PB reference line
    if pb_value is not None:
        fig.add_hline(
            y=pb_value, line_dash="dash", line_color=GOLD_ACCENT, line_width=1.5,
            annotation_text="PB", annotation_position="right",
            annotation_font=dict(color=GOLD_ACCENT, size=11),
        )

    # SB reference line
    if sb_value is not None:
        fig.add_hline(
            y=sb_value, line_dash="dot", line_color=TEAL_LIGHT, line_width=1.5,
            annotation_text="SB", annotation_position="right",
            annotation_font=dict(color=TEAL_LIGHT, size=11),
        )

    _apply_base_layout(fig, title)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    # Normal axis: fastest time at bottom, improvement = line goes down
    # This is the standard coaching convention in athletics
    return fig


def standards_waterfall(
    standards: Dict[str, float],
    athlete_mark: Optional[float] = None,
    title: str = "Championship Standards",
    lower_is_better: bool = True,
) -> go.Figure:
    """Horizontal bar chart: Heat -> Semi -> Final -> Medal standards."""
    labels = list(standards.keys())
    values = list(standards.values())

    colors = [GRAY_BLUE, TEAL_LIGHT, TEAL_PRIMARY, GOLD_ACCENT]
    while len(colors) < len(labels):
        colors.append(GRAY_BLUE)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker_color=colors[:len(labels)],
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))

    if athlete_mark is not None:
        fig.add_vline(
            x=athlete_mark, line_dash="dash", line_color=STATUS_DANGER, line_width=2,
            annotation_text=f"Athlete: {athlete_mark:.2f}",
        )

    _apply_base_layout(fig, title)
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=labels[::-1]))
    return fig


def gap_to_medal_chart(
    athletes: List[Dict],
    title: str = "Gap to Medal Standard",
    lower_is_better: bool = True,
) -> go.Figure:
    """Bar chart showing each KSA athlete's gap to medal standard."""
    names = [a["name"] for a in athletes]
    gaps = [a["gap"] for a in athletes]

    colors = []
    for g in gaps:
        if lower_is_better:
            colors.append(TEAL_PRIMARY if g <= 0 else (GOLD_ACCENT if g < 0.5 else STATUS_DANGER))
        else:
            colors.append(TEAL_PRIMARY if g >= 0 else (GOLD_ACCENT if g > -0.1 else STATUS_DANGER))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=gaps,
        marker_color=colors,
        text=[f"{g:+.2f}" for g in gaps],
        textposition="outside",
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="#333", line_width=1)
    _apply_base_layout(fig, title)
    return fig


def season_progression_chart(
    df: pd.DataFrame,
    title: str = "Season-by-Season Best",
    lower_is_better: bool = True,
) -> go.Figure:
    """Bar chart showing best mark per season."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["season"],
        y=df["best_mark"],
        marker_color=TEAL_PRIMARY,
        text=df["best_mark"].apply(lambda x: f"{x:.2f}"),
        textposition="outside",
    ))

    _apply_base_layout(fig, title)
    if lower_is_better:
        fig.update_yaxes(autorange="reversed")
    return fig


def h2h_comparison_chart(
    athlete1: Dict, athlete2: Dict,
    title: str = "Head-to-Head",
) -> go.Figure:
    """Side-by-side comparison between two athletes."""
    categories = ["Wins", "PB Score", "World Rank", "Form Score"]
    a1_values = [athlete1.get("wins", 0), athlete1.get("pb_score", 0),
                 athlete1.get("rank", 0), athlete1.get("form", 0)]
    a2_values = [athlete2.get("wins", 0), athlete2.get("pb_score", 0),
                 athlete2.get("rank", 0), athlete2.get("form", 0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=athlete1.get("name", "Athlete 1"),
        x=categories, y=a1_values,
        marker_color=TEAL_PRIMARY,
    ))
    fig.add_trace(go.Bar(
        name=athlete2.get("name", "Athlete 2"),
        x=categories, y=a2_values,
        marker_color=GOLD_ACCENT,
    ))

    _apply_base_layout(fig, title)
    fig.update_layout(barmode="group")
    return fig


def form_gauge(score: float, label: str = "Form Score") -> go.Figure:
    """Gauge chart showing current form score (0-100)."""
    color = TEAL_PRIMARY if score >= 70 else (GOLD_ACCENT if score >= 40 else STATUS_DANGER)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label, "font": {"size": 14, "color": "#333"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#f5f5f5"},
                {"range": [40, 70], "color": "#e8f5e9"},
                {"range": [70, 100], "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": GOLD_ACCENT, "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def ranking_trend_chart(
    dates: List[str],
    ranks: List[int],
    title: str = "Ranking Trend",
) -> go.Figure:
    """Line chart showing ranking position over time (inverted y-axis)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=ranks,
        mode="lines+markers",
        line=dict(color=TEAL_PRIMARY, width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(35, 80, 50, 0.1)",
    ))

    _apply_base_layout(fig, title)
    fig.update_yaxes(autorange="reversed", title="World Ranking")
    return fig


def championship_trends_chart(
    df: pd.DataFrame,
    title: str = "Championship Final Performances by Place",
    lower_is_better: bool = True,
) -> go.Figure:
    """Multi-line chart showing performance per finishing position across years.

    Args:
        df: Long-format DataFrame with columns: year, place, mark
        title: Chart title
        lower_is_better: Invert Y axis for time events
    """
    # Place colors: gold, silver, bronze, then chart palette
    place_colors = {
        1: GOLD_ACCENT,
        2: "#C0C0C0",
        3: "#CD7F32",
        4: TEAL_PRIMARY,
        5: TEAL_LIGHT,
        6: GRAY_BLUE,
        7: "#0077B6",
        8: STATUS_DANGER,
    }

    place_labels = {
        1: "1st (Gold)", 2: "2nd (Silver)", 3: "3rd (Bronze)",
        4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th",
    }

    fig = go.Figure()

    for place in sorted(df["place"].unique()):
        place_data = df[df["place"] == place].sort_values("year")
        color = place_colors.get(int(place), GRAY_BLUE)
        label = place_labels.get(int(place), f"{int(place)}th")
        line_width = 3 if place <= 3 else 2

        fig.add_trace(go.Scatter(
            x=place_data["year"],
            y=place_data["mark"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=line_width),
            marker=dict(size=7 if place <= 3 else 5, color=color),
        ))

    _apply_base_layout(fig, title)

    if lower_is_better:
        fig.update_yaxes(autorange="reversed", title="Performance")
    else:
        fig.update_yaxes(title="Performance")

    fig.update_xaxes(title="Championship Year", dtick=1)
    fig.update_layout(height=450)
    return fig


def place_distribution_chart(
    df: pd.DataFrame,
    country: str = "KSA",
    country_col: str = "nat",
    title: str = "Place Distribution",
) -> go.Figure:
    """Horizontal bar chart showing count of finishes by place.

    Args:
        df: Finals results with pos and country columns
        country: Country to highlight
        country_col: Column name for country code
        title: Chart title
    """
    # Parse positions
    positions = pd.to_numeric(df["pos"].astype(str).str.strip(), errors="coerce").dropna()
    positions = positions[positions.between(1, 8)].astype(int)

    all_counts = positions.value_counts().sort_index()

    # Country-specific counts
    country_mask = df[country_col].str.upper() == country.upper() if country_col in df.columns else pd.Series(False, index=df.index)
    country_pos = pd.to_numeric(df.loc[country_mask, "pos"].astype(str).str.strip(), errors="coerce").dropna()
    country_pos = country_pos[country_pos.between(1, 8)].astype(int)
    country_counts = country_pos.value_counts().sort_index()

    places = list(range(1, 9))
    place_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]

    fig = go.Figure()

    # All finalists
    fig.add_trace(go.Bar(
        y=place_labels,
        x=[all_counts.get(p, 0) for p in places],
        orientation="h",
        name="All Countries",
        marker_color=GRAY_BLUE,
        opacity=0.5,
    ))

    # Country-specific
    if len(country_counts) > 0:
        fig.add_trace(go.Bar(
            y=place_labels,
            x=[country_counts.get(p, 0) for p in places],
            orientation="h",
            name=country,
            marker_color=TEAL_PRIMARY,
        ))

    _apply_base_layout(fig, title)
    fig.update_layout(barmode="overlay", height=350)
    fig.update_xaxes(title="Number of Finishes")
    return fig


def wa_points_heatmap(
    df: pd.DataFrame,
    title: str = "WA Points Scoring Map",
) -> go.Figure:
    """Heatmap showing WA points by event x competition date.

    Shows WA points with the actual mark (time/distance) underneath.

    Args:
        df: Results DataFrame with columns: discipline/event, date, result_score, mark/result
    """
    # Determine column names
    event_col = "discipline" if "discipline" in df.columns else "event"
    score_col = "result_score" if "result_score" in df.columns else "resultscore"
    mark_col = "mark" if "mark" in df.columns else "result" if "result" in df.columns else None
    date_col = "date"

    if score_col not in df.columns or event_col not in df.columns:
        return go.Figure()

    work = df.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["_score"])

    if len(work) == 0:
        return go.Figure()

    # Parse dates and create month labels
    work["_date"] = pd.to_datetime(work[date_col], format="mixed", errors="coerce")
    work = work.dropna(subset=["_date"])
    work["_month"] = work["_date"].dt.strftime("%b %Y")
    work["_month_sort"] = work["_date"].dt.to_period("M")

    # Pivot: event rows x month columns, values = max WA points that month
    pivot = work.pivot_table(
        index=event_col,
        columns="_month_sort",
        values="_score",
        aggfunc="max",
    )

    # Build a matching mark pivot (the mark that corresponds to the best score)
    mark_text = None
    if mark_col and mark_col in work.columns:
        # For each event+month, get the mark from the row with the highest score
        best_idx = work.groupby([event_col, "_month_sort"])["_score"].idxmax()
        best_rows = work.loc[best_idx]
        mark_pivot = best_rows.pivot_table(
            index=event_col,
            columns="_month_sort",
            values=mark_col,
            aggfunc="first",
        )
        mark_pivot = mark_pivot.reindex(index=pivot.index, columns=pivot.columns)

        # Build combined text: "1045\n10.23" (points + mark)
        import numpy as np
        mark_text = []
        for i, ev in enumerate(pivot.index):
            row_texts = []
            for j, col in enumerate(pivot.columns):
                pts = pivot.iloc[i, j]
                mk = mark_pivot.iloc[i, j] if i < len(mark_pivot) and j < len(mark_pivot.columns) else None
                if pd.notna(pts):
                    pts_str = f"{pts:.0f}"
                    if pd.notna(mk):
                        row_texts.append(f"{pts_str}<br><sub>{mk}</sub>")
                    else:
                        row_texts.append(pts_str)
                else:
                    row_texts.append("")
            mark_text.append(row_texts)

    # Sort columns chronologically
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    # Rename columns to human-readable month labels
    pivot.columns = [str(c) for c in pivot.columns]

    # Custom Saudi green colorscale
    colorscale = [
        [0.0, "#f5f5f5"],
        [0.3, "#c8e6c9"],
        [0.5, "#81c784"],
        [0.7, "#2A8F5C"],
        [0.85, "#005430"],
        [1.0, GOLD_ACCENT],
    ]

    # Use combined text if available, otherwise just points
    if mark_text:
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=colorscale,
            text=mark_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate="Event: %{y}<br>Month: %{x}<br>WA Points: %{z:.0f}<extra></extra>",
            colorbar=dict(title="WA Pts", thickness=15),
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=colorscale,
            text=pivot.values.astype(str),
            texttemplate="%{text:.0f}",
            textfont={"size": 11},
            hoverongaps=False,
            hovertemplate="Event: %{y}<br>Month: %{x}<br>WA Points: %{z:.0f}<extra></extra>",
            colorbar=dict(title="WA Pts", thickness=15),
        ))

    _apply_base_layout(fig, title)
    fig.update_layout(
        height=max(300, 60 * len(pivot) + 100),
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def event_scoring_bars(
    df: pd.DataFrame,
    title: str = "WA Points by Event",
) -> go.Figure:
    """Horizontal bar chart showing PB WA points per event. Highlights best-scoring events."""
    event_col = "discipline" if "discipline" in df.columns else "event"
    score_col = "result_score" if "result_score" in df.columns else "resultscore"

    if score_col not in df.columns or event_col not in df.columns:
        return go.Figure()

    work = df.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["_score"])
    work = work.sort_values("_score", ascending=True)

    if len(work) == 0:
        return go.Figure()

    max_score = work["_score"].max()
    colors = []
    for s in work["_score"]:
        ratio = s / max_score if max_score > 0 else 0
        if ratio >= 0.95:
            colors.append(GOLD_ACCENT)    # Best scoring = gold
        elif ratio >= 0.85:
            colors.append(TEAL_PRIMARY)   # Strong scoring
        elif ratio >= 0.70:
            colors.append(TEAL_LIGHT)     # Good scoring
        else:
            colors.append(GRAY_BLUE)      # Lower scoring

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=work[event_col],
        x=work["_score"],
        orientation="h",
        marker_color=colors,
        text=work["_score"].apply(lambda x: f"{x:.0f}"),
        textposition="outside",
        hovertemplate="Event: %{y}<br>WA Points: %{x:.0f}<extra></extra>",
    ))

    _apply_base_layout(fig, title)
    fig.update_layout(
        height=max(250, 40 * len(work) + 80),
        xaxis_title="WA Ranking Points",
        yaxis_title="",
    )
    return fig


def competition_points_chart(
    df: pd.DataFrame,
    title: str = "Ranking Points by Competition Category",
) -> go.Figure:
    """Bar chart showing average ranking points by competition category."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["category"],
        y=df["avg_points"],
        marker_color=CHART_COLORS[:len(df)],
        text=df["avg_points"].apply(lambda x: f"{x:.0f}"),
        textposition="outside",
    ))

    _apply_base_layout(fig, title)
    fig.update_yaxes(title="Average Ranking Points")
    return fig
