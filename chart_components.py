"""
Chart Components for World Athletics Dashboard

Reusable Plotly chart components for:
- Season progression timelines
- Gap analysis visuals
- Probability gauges
- Competitor comparison bars
- Form trend sparklines

All charts styled with Team Saudi theme.

Adapted from Tilastopaja project (Altair) to use Plotly for consistency
with existing World Athletics dashboard.

NOTE: This module is designed to work with both World Athletics (this project)
and Tilastopaja data. Future database combination is planned.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime


# Team Saudi color palette - Official Saudi Green
COLORS = {
    'primary': '#005430',      # Saudi Green (official PMS 3425 C)
    'secondary': '#a08e66',    # Gold accent
    'gold': '#a08e66',         # Gold accent (alias)
    'dark': '#003d1f',         # Dark green
    'light': '#2A8F5C',        # Light green
    'gray': '#78909C',         # Gray blue
    'success': '#005430',      # Saudi Green (positive)
    'warning': '#FFB800',      # Gold/yellow
    'danger': '#dc3545',       # Red
    'medal_gold': '#FFD700',
    'medal_silver': '#C0C0C0',
    'medal_bronze': '#CD7F32',
    'background': 'white',
    'text': '#333333',
    'grid': 'lightgray'
}

# Chart typography - larger for presentations
CHART_FONTS = {
    'title': 20,
    'axis': 14,
    'legend': 13,
    'annotation': 12,
    'tick': 12
}


def get_base_layout() -> Dict:
    """Get base Plotly layout with Team Saudi styling."""
    return {
        'plot_bgcolor': COLORS['background'],
        'paper_bgcolor': COLORS['background'],
        'font': {
            'family': 'Inter, sans-serif',
            'color': COLORS['text'],
            'size': CHART_FONTS['tick']
        },
        'title': {'font': {'size': CHART_FONTS['title']}},
        'margin': {'l': 60, 'r': 30, 't': 60, 'b': 50},
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'y': -0.15,
            'font': {'size': CHART_FONTS['legend']}
        },
        'xaxis': {'tickfont': {'size': CHART_FONTS['tick']}},
        'yaxis': {'tickfont': {'size': CHART_FONTS['tick']}}
    }


def get_performance_hovertemplate(include_venue: bool = True) -> str:
    """Get rich hover template for performance charts."""
    if include_venue:
        return (
            '<b>%{customdata[0]}</b><br>'
            'Result: %{y:.2f}<br>'
            'Venue: %{customdata[1]}<br>'
            'Date: %{customdata[2]}<br>'
            '<extra></extra>'
        )
    return (
        '<b>%{x|%d %b %Y}</b><br>'
        'Result: %{y:.2f}<br>'
        '<extra></extra>'
    )


def add_medal_lines(fig: go.Figure, benchmarks: Dict[str, float], event_type: str = 'time'):
    """Add gold/silver/bronze standard lines to a chart."""
    medal_styles = {
        'gold': {'color': COLORS['medal_gold'], 'dash': 'solid', 'width': 2, 'label': 'Gold'},
        'silver': {'color': COLORS['medal_silver'], 'dash': 'dash', 'width': 2, 'label': 'Silver'},
        'bronze': {'color': COLORS['medal_bronze'], 'dash': 'dot', 'width': 2, 'label': 'Bronze'},
        'final': {'color': COLORS['gray'], 'dash': 'dashdot', 'width': 1.5, 'label': 'Final (8th)'}
    }

    for key, value in benchmarks.items():
        if value is not None and key in medal_styles:
            style = medal_styles[key]
            fig.add_hline(
                y=value,
                line_dash=style['dash'],
                line_color=style['color'],
                line_width=style['width'],
                annotation_text=f"{style['label']}: {value:.2f}",
                annotation_position='right',
                annotation_font_size=CHART_FONTS['annotation']
            )
    return fig


def season_progression_chart(
    performances: List[Dict],
    benchmarks: Dict[str, float] = None,
    event_type: str = 'time',
    title: str = 'Season Progression',
    height: int = 350
) -> go.Figure:
    """
    Create season progression line chart with benchmark overlays.

    Args:
        performances: List of dicts with 'date', 'result', 'competition' keys
        benchmarks: Dict with 'medal', 'final', 'semi', 'heat' lines
        event_type: 'time' (inverted y-axis) or 'distance'/'points'
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    if not performances:
        fig = go.Figure()
        fig.add_annotation(text="No performance data available", showarrow=False)
        fig.update_layout(height=height, **get_base_layout())
        return fig

    # Create DataFrame
    df = pd.DataFrame(performances)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    fig = go.Figure()

    # Add performance line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['result'],
        mode='lines+markers',
        name='Performance',
        line={'color': COLORS['primary'], 'width': 3},
        marker={'size': 10, 'color': COLORS['primary']},
        hovertemplate='%{x|%d %b %Y}<br>Result: %{y:.2f}<extra></extra>'
    ))

    # Add benchmark lines if provided
    if benchmarks:
        benchmark_styles = {
            'medal': {'color': COLORS['gold'], 'dash': 'dash', 'name': 'Medal Line'},
            'final': {'color': COLORS['primary'], 'dash': 'dot', 'name': 'Final Line'},
            'semi': {'color': COLORS['light'], 'dash': 'dashdot', 'name': 'Semi Line'},
            'heat': {'color': COLORS['gray'], 'dash': 'dot', 'name': 'Heat Line'}
        }

        for key, value in benchmarks.items():
            if value is not None and key in benchmark_styles:
                style = benchmark_styles[key]
                fig.add_hline(
                    y=value,
                    line_dash=style['dash'],
                    line_color=style['color'],
                    annotation_text=style['name'],
                    annotation_position='right'
                )

    # Configure layout
    layout = get_base_layout()
    layout.update({
        'title': title,
        'height': height,
        'xaxis': {
            'title': 'Date',
            'showgrid': True,
            'gridcolor': COLORS['grid']
        },
        'yaxis': {
            'title': 'Performance',
            'showgrid': True,
            'gridcolor': COLORS['grid'],
            'autorange': 'reversed' if event_type == 'time' else True
        }
    })

    fig.update_layout(**layout)
    return fig


def gap_analysis_chart(
    athlete_performance: float,
    benchmarks: Dict[str, float],
    event_type: str = 'time',
    title: str = 'Gap to Championship Benchmarks',
    height: int = 250
) -> go.Figure:
    """
    Create horizontal bar chart showing gap to each benchmark.

    Args:
        athlete_performance: Athlete's season best or projected performance
        benchmarks: Dict with benchmark values
        event_type: 'time', 'distance', or 'points'
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    benchmark_order = ['heat', 'semi', 'final', 'medal']
    benchmark_labels = {
        'medal': 'Medal Line',
        'final': 'Final Line',
        'semi': 'Semi Line',
        'heat': 'Heat Line'
    }

    data = []
    for key in benchmark_order:
        if key in benchmarks and benchmarks[key] is not None:
            target = benchmarks[key]
            if event_type == 'time':
                gap = athlete_performance - target  # Positive = behind
            else:
                gap = target - athlete_performance  # Positive = behind

            data.append({
                'benchmark': benchmark_labels.get(key, key),
                'gap': gap,
                'color': COLORS['success'] if gap < 0 else COLORS['danger']
            })

    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No benchmark data available", showarrow=False)
        fig.update_layout(height=height, **get_base_layout())
        return fig

    df = pd.DataFrame(data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['benchmark'],
        x=df['gap'],
        orientation='h',
        marker_color=df['color'],
        text=[f"{g:+.2f}" for g in df['gap']],
        textposition='outside',
        hovertemplate='%{y}: %{x:+.2f}<extra></extra>'
    ))

    # Add zero line
    fig.add_vline(x=0, line_color=COLORS['text'], line_width=2)

    layout = get_base_layout()
    layout.update({
        'title': title,
        'height': height,
        'xaxis': {
            'title': 'Gap (negative = ahead, positive = behind)',
            'showgrid': True,
            'gridcolor': COLORS['grid'],
            'zeroline': True
        },
        'yaxis': {
            'title': '',
            'categoryorder': 'array',
            'categoryarray': [benchmark_labels[k] for k in benchmark_order if k in benchmarks]
        },
        'showlegend': False
    })

    fig.update_layout(**layout)
    return fig


def probability_gauge(
    probabilities: Dict[str, float],
    title: str = 'Advancement Probability',
    height: int = 200
) -> go.Figure:
    """
    Create probability bars for each round.

    Args:
        probabilities: Dict with round names and probability percentages
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    round_order = ['heat', 'semi', 'final', 'medal']
    round_labels = {
        'heat': 'Make Heats',
        'semi': 'Make Semis',
        'final': 'Make Finals',
        'medal': 'Win Medal'
    }

    data = []
    for key in round_order:
        if key in probabilities:
            prob = probabilities[key]
            # Color based on probability
            if prob >= 70:
                color = COLORS['success']
            elif prob >= 40:
                color = COLORS['warning']
            else:
                color = COLORS['danger']

            data.append({
                'round': round_labels.get(key, key),
                'probability': prob,
                'color': color
            })

    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No probability data", showarrow=False)
        fig.update_layout(height=height, **get_base_layout())
        return fig

    df = pd.DataFrame(data)

    fig = go.Figure()

    # Background bars (100%)
    fig.add_trace(go.Bar(
        y=df['round'],
        x=[100] * len(df),
        orientation='h',
        marker_color=COLORS['grid'],
        showlegend=False,
        hoverinfo='skip'
    ))

    # Probability bars
    fig.add_trace(go.Bar(
        y=df['round'],
        x=df['probability'],
        orientation='h',
        marker_color=df['color'],
        text=[f"{p:.0f}%" for p in df['probability']],
        textposition='outside',
        showlegend=False,
        hovertemplate='%{y}: %{x:.0f}%<extra></extra>'
    ))

    layout = get_base_layout()
    layout.update({
        'title': title,
        'height': height,
        'barmode': 'overlay',
        'xaxis': {
            'title': 'Probability %',
            'range': [0, 110],
            'showgrid': True,
            'gridcolor': COLORS['grid']
        },
        'yaxis': {
            'title': '',
            'categoryorder': 'array',
            'categoryarray': list(reversed([round_labels[k] for k in round_order if k in probabilities]))
        },
        'showlegend': False
    })

    fig.update_layout(**layout)
    return fig


def competitor_comparison_chart(
    athlete_name: str,
    athlete_sb: float,
    competitors: List[Dict],
    event_type: str = 'time',
    title: str = 'Season Best Comparison',
    height: int = 400
) -> go.Figure:
    """
    Create horizontal bar chart comparing athlete to competitors.

    Args:
        athlete_name: Name of the main athlete
        athlete_sb: Athlete's season best
        competitors: List of competitor dicts with 'name', 'country', 'sb'
        event_type: 'time', 'distance', or 'points'
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    data = []

    # Add main athlete
    data.append({
        'name': f"{athlete_name} (KSA)",
        'sb': athlete_sb,
        'is_athlete': True
    })

    # Add competitors (limit to top 12)
    for comp in competitors[:12]:
        data.append({
            'name': f"{comp.get('name', 'Unknown')} ({comp.get('country', '')})",
            'sb': comp.get('sb', 0),
            'is_athlete': False
        })

    df = pd.DataFrame(data)

    # Sort by performance
    df = df.sort_values('sb', ascending=(event_type == 'time'))

    fig = go.Figure()

    # Color bars - highlight KSA athlete
    colors = [COLORS['primary'] if row['is_athlete'] else COLORS['gray']
              for _, row in df.iterrows()]

    fig.add_trace(go.Bar(
        y=df['name'],
        x=df['sb'],
        orientation='h',
        marker_color=colors,
        text=[f"{sb:.2f}" for sb in df['sb']],
        textposition='outside',
        hovertemplate='%{y}<br>SB: %{x:.2f}<extra></extra>'
    ))

    layout = get_base_layout()
    layout.update({
        'title': title,
        'height': height,
        'xaxis': {
            'title': 'Season Best',
            'showgrid': True,
            'gridcolor': COLORS['grid'],
            'autorange': 'reversed' if event_type == 'time' else True
        },
        'yaxis': {
            'title': '',
            'tickfont': {'size': 10}
        },
        'showlegend': False
    })

    fig.update_layout(**layout)
    return fig


def form_trend_chart(
    performances: List[Dict],
    event_type: str = 'time',
    title: str = 'Recent Form',
    height: int = 200
) -> go.Figure:
    """
    Create form trend sparkline with trend indicator.

    Args:
        performances: List of dicts with 'date', 'result' keys
        event_type: 'time', 'distance', or 'points'
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    if not performances:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        fig.update_layout(height=height, **get_base_layout())
        return fig

    df = pd.DataFrame(performances)
    df = df.sort_values('date') if 'date' in df.columns else df
    df['index'] = range(len(df))

    fig = go.Figure()

    # Performance line
    fig.add_trace(go.Scatter(
        x=df['index'],
        y=df['result'],
        mode='lines+markers',
        name='Performance',
        line={'color': COLORS['primary'], 'width': 3},
        marker={'size': 8, 'color': COLORS['primary']},
        hovertemplate='Result: %{y:.2f}<extra></extra>'
    ))

    # Add trend line using linear regression
    if len(df) >= 3:
        import numpy as np
        z = np.polyfit(df['index'], df['result'], 1)
        p = np.poly1d(z)
        trend_y = [p(x) for x in df['index']]

        fig.add_trace(go.Scatter(
            x=df['index'],
            y=trend_y,
            mode='lines',
            name='Trend',
            line={'color': COLORS['gold'], 'width': 2, 'dash': 'dash'}
        ))

    layout = get_base_layout()
    layout.update({
        'title': title,
        'height': height,
        'xaxis': {
            'title': 'Recent Performances',
            'showticklabels': False,
            'showgrid': False
        },
        'yaxis': {
            'title': '',
            'showgrid': True,
            'gridcolor': COLORS['grid'],
            'autorange': 'reversed' if event_type == 'time' else True
        },
        'showlegend': False
    })

    fig.update_layout(**layout)
    return fig


def create_form_score_gauge(
    score: float,
    title: str = 'Current Form',
    height: int = 180
) -> go.Figure:
    """
    Create a gauge chart for form score (0-100).

    Args:
        score: Form score from 0-100
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure object
    """
    # Determine color based on score
    if score >= 85:
        color = COLORS['primary']
        label = 'Excellent'
    elif score >= 70:
        color = COLORS['light']
        label = 'Good'
    elif score >= 50:
        color = COLORS['gold']
        label = 'Moderate'
    elif score >= 30:
        color = COLORS['warning']
        label = 'Low'
    else:
        color = COLORS['danger']
        label = 'Poor'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': f' ({label})', 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': '#ffebee'},
                {'range': [30, 50], 'color': '#fff3e0'},
                {'range': [50, 70], 'color': '#fffde7'},
                {'range': [70, 85], 'color': '#e0f2f1'},
                {'range': [85, 100], 'color': '#e0f7fa'}
            ],
            'threshold': {
                'line': {'color': COLORS['text'], 'width': 2},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    layout = get_base_layout()
    layout['height'] = height
    layout['margin'] = {'l': 20, 'r': 20, 't': 50, 'b': 20}

    fig.update_layout(**layout)
    return fig


def create_last_3_comps_display(
    recent_results: List[Dict],
    event_type: str = 'time'
) -> str:
    """
    Create HTML display for last 3 competitions.

    Args:
        recent_results: List of dicts with 'date', 'result', 'competition'
        event_type: 'time', 'distance', or 'points'

    Returns:
        HTML string for display
    """
    if not recent_results:
        return '<span style="color: #999;">No recent results</span>'

    results = recent_results[:3]
    parts = []

    for r in results:
        result = r.get('result', 0)
        if event_type == 'time':
            if result >= 60:
                mins = int(result // 60)
                secs = result % 60
                result_str = f"{mins}:{secs:05.2f}"
            else:
                result_str = f"{result:.2f}"
        else:
            result_str = f"{result:.2f}"

        parts.append(f'<span style="background: #005430; color: white; padding: 2px 6px; border-radius: 4px; margin-right: 4px;">{result_str}</span>')

    return ' '.join(parts)
