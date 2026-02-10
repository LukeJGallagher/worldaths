"""
Loading components for Saudi Athletics Dashboard.
Provides skeleton UI with shimmer animation and branded loading header.
"""
import streamlit as st

# Team Saudi Brand Colors
TEAL_PRIMARY = '#007167'
TEAL_DARK = '#005a51'
GOLD_ACCENT = '#a08e66'


def get_shimmer_css() -> str:
    """Return CSS for shimmer animation effect."""
    return """
    <style>
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: 200px 0; }
    }

    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 400px 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 4px;
    }

    .skeleton-card {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 400px 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .skeleton-text {
        height: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }

    .skeleton-title {
        height: 1.5rem;
        width: 60%;
        margin-bottom: 1rem;
    }

    .loading-progress-bar {
        height: 4px;
        background: rgba(255,255,255,0.2);
        border-radius: 2px;
        overflow: hidden;
        margin-top: 0.5rem;
    }

    .loading-progress-fill {
        height: 100%;
        background: #a08e66;
        border-radius: 2px;
        transition: width 0.3s ease;
    }
    </style>
    """


def render_loading_header(current_step: int, total_steps: int, message: str):
    """Render branded loading header with progress bar."""
    progress_pct = int((current_step / total_steps) * 100)

    st.markdown(get_shimmer_css(), unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
         padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="color: white; margin: 0; font-family: Inter, sans-serif;">
                    Saudi Athletics Intelligence
                </h2>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    {message} ({current_step}/{total_steps})
                </p>
            </div>
            <div style="text-align: right; min-width: 80px;">
                <span style="color: {GOLD_ACCENT}; font-size: 1.5rem; font-weight: bold;">
                    {progress_pct}%
                </span>
            </div>
        </div>
        <div class="loading-progress-bar">
            <div class="loading-progress-fill" style="width: {progress_pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_skeleton_card(width: str = "100%", height: str = "100px"):
    """Render a skeleton placeholder card with shimmer effect."""
    st.markdown(f"""
    <div class="skeleton-card" style="width: {width}; height: {height};">
        <div class="skeleton skeleton-title"></div>
        <div class="skeleton skeleton-text" style="width: 80%;"></div>
        <div class="skeleton skeleton-text" style="width: 60%;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_skeleton_table(rows: int = 5, cols: int = 4):
    """Render a skeleton placeholder table with shimmer effect."""
    header_html = "".join([f'<div class="skeleton" style="height: 1rem; flex: 1; margin: 0 4px;"></div>' for _ in range(cols)])

    rows_html = ""
    for _ in range(rows):
        row_cells = "".join([f'<div class="skeleton" style="height: 0.8rem; flex: 1; margin: 0 4px;"></div>' for _ in range(cols)])
        rows_html += f'<div style="display: flex; padding: 0.5rem 0;">{row_cells}</div>'

    st.markdown(f"""
    <div style="background: white; border-radius: 8px; padding: 1rem; border: 1px solid #e0e0e0;">
        <div style="display: flex; padding: 0.5rem 0; border-bottom: 2px solid #e0e0e0; margin-bottom: 0.5rem;">
            {header_html}
        </div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)


def render_skeleton_metrics(count: int = 4):
    """Render skeleton metric cards in a row."""
    cards_html = ""
    for _ in range(count):
        cards_html += f"""
        <div class="skeleton-card" style="flex: 1; margin: 0 0.5rem; min-height: 80px;">
            <div class="skeleton skeleton-text" style="width: 50%; height: 0.7rem;"></div>
            <div class="skeleton" style="width: 70%; height: 1.5rem; margin-top: 0.5rem;"></div>
        </div>
        """

    st.markdown(f"""
    <div style="display: flex; margin: 1rem 0;">
        {cards_html}
    </div>
    """, unsafe_allow_html=True)


def render_loading_placeholder(tab_name: str):
    """Render a full loading placeholder for a tab."""
    st.markdown(get_shimmer_css(), unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p style="font-size: 1.1rem;">Loading {tab_name}...</p>
    </div>
    """, unsafe_allow_html=True)

    render_skeleton_metrics(4)
    render_skeleton_table(5, 4)


def init_loading_state():
    """Initialize loading state in session state."""
    if 'loading' not in st.session_state:
        st.session_state.loading = {
            'profiles': False,
            'benchmarks': False,
            'road_to_tokyo': False,
            'master': False,
            'wittw': False,
            'current_step': 0,
            'total_steps': 5,
            'message': 'Initializing...',
            'complete': False
        }


def update_loading_state(key: str, message: str):
    """Update loading state after a data source loads."""
    if 'loading' not in st.session_state:
        init_loading_state()

    st.session_state.loading[key] = True
    st.session_state.loading['current_step'] += 1
    st.session_state.loading['message'] = message

    # Check if all complete
    required = ['profiles', 'benchmarks', 'road_to_tokyo', 'master', 'wittw']
    if all(st.session_state.loading.get(k) for k in required):
        st.session_state.loading['complete'] = True


def is_data_loaded(key: str) -> bool:
    """Check if a specific data source is loaded."""
    if 'loading' not in st.session_state:
        return False
    return st.session_state.loading.get(key, False)


def is_loading_complete() -> bool:
    """Check if all data loading is complete."""
    if 'loading' not in st.session_state:
        return False
    return st.session_state.loading.get('complete', False)
