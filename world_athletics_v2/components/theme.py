"""
Team Saudi theme - single source of truth for all styling.

All colors, CSS, and branding components in one place.
Cached with @st.cache_resource to avoid re-computation.
"""

import base64
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Brand Colors (sampled from banner/logo assets) ───────────────────

TEAL_PRIMARY = "#235032"      # Exact banner/logo green
TEAL_DARK = "#1a3d25"
TEAL_LIGHT = "#3a7050"
GOLD_ACCENT = "#a08e66"
GRAY_BLUE = "#78909C"

# Status colors
STATUS_EXCELLENT = TEAL_PRIMARY
STATUS_GOOD = TEAL_LIGHT
STATUS_WARNING = "#FFB800"
STATUS_DANGER = "#dc3545"
STATUS_NEUTRAL = "#6c757d"

# Gradients
HEADER_GRADIENT = f"linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%)"
SIDEBAR_GRADIENT = f"linear-gradient(180deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%)"
GOLD_BAR = f"linear-gradient(90deg, {TEAL_PRIMARY} 0%, {GOLD_ACCENT} 50%, {TEAL_DARK} 100%)"

# Chart colors
CHART_COLORS = [TEAL_PRIMARY, GOLD_ACCENT, TEAL_LIGHT, "#0077B6", STATUS_WARNING, STATUS_DANGER]
PLOTLY_LAYOUT = {
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "font": {"family": "Inter, sans-serif", "color": "#333"},
    "margin": {"l": 10, "r": 10, "t": 40, "b": 30},
}


# ── Theme Assets ──────────────────────────────────────────────────────

# Look for assets in repo first (works on Streamlit Cloud), then local dev path
_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_LOCAL_THEME_DIR = Path(r"C:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\Theme")
THEME_DIR = _ASSETS_DIR if _ASSETS_DIR.exists() else _LOCAL_THEME_DIR


def _get_image_base64(image_path: Path) -> Optional[str]:
    """Load image as base64 string."""
    if image_path.exists():
        return base64.b64encode(image_path.read_bytes()).decode()
    return None


@st.cache_resource
def get_banner_b64() -> Optional[str]:
    return _get_image_base64(THEME_DIR / "team_saudi_banner.jpg")


@st.cache_resource
def get_logo_b64() -> Optional[str]:
    return _get_image_base64(THEME_DIR / "team_saudi_logo.jpg")


# ── CSS ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_theme_css() -> str:
    """Get cached theme CSS."""
    return f"""<style>
    /* Main font */
    html, body, [class*="css"] {{
        font-family: Inter, sans-serif;
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: {SIDEBAR_GRADIENT};
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label {{
        color: white !important;
    }}

    /* ── Sidebar nav items as styled buttons ── */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        padding: 0.5rem 0;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] li {{
        list-style: none;
        margin: 0.2rem 0.5rem;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        display: block;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.85) !important;
        text-decoration: none !important;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.05);
    }}
    /* Force ALL text inside nav links to white */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a p,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a div,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] li span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] li p {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
        background: rgba(255, 255, 255, 0.15);
        color: white !important;
        border-color: {GOLD_ACCENT};
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover p {{
        color: white !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: rgba(160, 142, 102, 0.25);
        color: white !important;
        border-color: {GOLD_ACCENT};
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] p {{
        color: white !important;
        font-weight: 600;
    }}
    /* Hide the default bullet/dot indicators */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span[data-testid="stIconMaterial"] {{
        display: none;
    }}
    /* Force white on any remaining sidebar text elements */
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}

    /* Streamlit metric styling */
    [data-testid="stMetric"] {{
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 3px solid {TEAL_PRIMARY};
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
    }}

    /* Dataframe styling */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>"""


# ── Reusable HTML Components ─────────────────────────────────────────

def render_page_header(title: str, subtitle: str = "") -> None:
    """Render a branded page header with banner image."""
    banner = get_banner_b64()
    if banner:
        st.markdown(f'''
        <div style="position: relative; border-radius: 12px; overflow: hidden; margin-bottom: 1.5rem;
             box-shadow: 0 8px 25px rgba(35, 80, 50, 0.25);">
            <img src="data:image/jpeg;base64,{banner}"
                 style="width: 100%; height: 160px; object-fit: cover; filter: brightness(0.7);">
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                 display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <h1 style="color: white; font-size: 2rem; margin: 0;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">{title}</h1>
                    {"<p style='color: " + GOLD_ACCENT + "; font-size: 1rem; margin: 0.5rem 0 0 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>" + subtitle + "</p>" if subtitle else ""}
                </div>
            </div>
            <div style="position: absolute; bottom: 0; left: 0; right: 0; height: 4px;
                 background: {GOLD_BAR};"></div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div style="background: {HEADER_GRADIENT}; padding: 1.5rem; border-radius: 8px;
             margin-bottom: 1.5rem; border-left: 4px solid {GOLD_ACCENT};">
            <h1 style="color: white; margin: 0; font-size: 2rem;">{title}</h1>
            {"<p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>" + subtitle + "</p>" if subtitle else ""}
        </div>
        ''', unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "") -> None:
    """Render a section header with teal gradient."""
    st.markdown(f'''
    <div style="background: {HEADER_GRADIENT}; padding: 1rem; border-radius: 8px;
         margin: 1.5rem 0 1rem 0; border-left: 4px solid {GOLD_ACCENT};">
        <h3 style="color: white; margin: 0;">{title}</h3>
        {"<p style='color: rgba(255,255,255,0.8); margin: 0.25rem 0 0 0; font-size: 0.9rem;'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    ''', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, status: str = "neutral") -> None:
    """Render a colored metric card."""
    colors = {
        "excellent": TEAL_PRIMARY,
        "good": TEAL_LIGHT,
        "warning": STATUS_WARNING,
        "danger": STATUS_DANGER,
        "neutral": STATUS_NEUTRAL,
        "gold": GOLD_ACCENT,
    }
    color = colors.get(status, STATUS_NEUTRAL)
    st.markdown(f"""
    <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center;">
        <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">{label}</p>
        <p style="color: white; margin: 0.25rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{value}</p>
    </div>
    """, unsafe_allow_html=True)


def _get_last_scraped() -> str:
    """Get the last scraped date from parquet file modification times."""
    import os
    from datetime import datetime
    data_dir = Path(__file__).parent.parent / "data" / "scraped"
    if not data_dir.exists():
        return ""
    latest = 0
    for f in data_dir.glob("*.parquet"):
        mtime = os.path.getmtime(f)
        if mtime > latest:
            latest = mtime
    if latest > 0:
        return datetime.fromtimestamp(latest).strftime("%d %b %Y")
    return ""


def render_sidebar() -> None:
    """Render the branded sidebar with logo."""
    logo = get_logo_b64()
    if logo:
        st.sidebar.markdown(f'''
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 0.5rem;">
            <img src="data:image/jpeg;base64,{logo}"
                 style="width: 160px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        </div>
        ''', unsafe_allow_html=True)

    last_scraped = _get_last_scraped()
    scraped_html = ""
    if last_scraped:
        scraped_html = f'<p style="color: rgba(255,255,255,0.5); font-size: 0.65rem; margin: 0.15rem 0 0 0;">Last updated: {last_scraped}</p>'

    st.sidebar.markdown(f'''
    <div style="text-align: center; padding: 0.25rem 0 0.75rem 0;">
        <p style="color: {GOLD_ACCENT}; font-size: 0.8rem; margin: 0; font-weight: 500;">World Athletics Intelligence</p>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.7rem; margin: 0.15rem 0 0 0;">v2.0</p>
        {scraped_html}
    </div>
    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 0;">
    ''', unsafe_allow_html=True)

    # Footer: attribution
    st.sidebar.markdown(f'''
    <div style="position: fixed; bottom: 0.5rem; left: 0; width: var(--sidebar-width, 21rem);
         text-align: center; padding: 0.5rem;">
        <p style="color: rgba(255,255,255,0.4); font-size: 0.6rem; margin: 0;">
            App by Luke Gallagher
        </p>
    </div>
    ''', unsafe_allow_html=True)


def apply_theme() -> None:
    """Apply the full Team Saudi theme. Call once at app startup."""
    st.set_page_config(
        page_title="World Athletics - Team Saudi",
        page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏃</text></svg>",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    render_sidebar()
