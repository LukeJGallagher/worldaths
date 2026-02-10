# Loading Experience Design

*Saudi Athletics Dashboard - Progressive Loading with Skeleton UI*
*Created: 1 February 2026*

## Overview

Improve the user load experience for the Saudi Athletics Dashboard by implementing:
- Skeleton UI with shimmer effects (feels faster psychologically)
- Progressive data reveal (users can explore loaded sections while others finish)
- Team Saudi branding throughout the loading experience
- Clear progress feedback showing current step

## Architecture

### Loading States

**1. Immediate (0ms)**
- Team Saudi branded header with logo appears instantly
- Tab bar shows with all 11 tabs visible but disabled
- Skeleton placeholders appear in the active tab area

**2. Progressive (as each file loads)**
- Loading bar under header: "Loading athlete profiles... (2/5)"
- As each data source completes, tabs that use it become enabled
- Currently loading tab shows animated shimmer effect

**3. Complete**
- Loading bar disappears
- All tabs enabled
- Dashboard fully interactive

### Data Loading Order

Files load in this order (fastest to slowest):

| Order | File | Size | Tabs Enabled |
|-------|------|------|--------------|
| 1 | ksa_profiles.parquet | ~100KB | Athlete Profiles |
| 2 | benchmarks.parquet | ~14KB | Event Standards |
| 3 | road_to_tokyo.parquet | ~350KB | World Champs Qualification |
| 4 | master.parquet | ~33MB | Combined Rankings, Saudi Rankings, Major Games, WITTW, AI Analyst, Coach View, Project East |
| 5 | WITTW Analyzer init | N/A | Final initialization |

## Visual Components

### Header Bar (always visible)

```
+-------------------------------------------------------------+
|  [Logo]  Saudi Athletics Intelligence    Loading... (2/5)   |
|  ================----------  40%                            |
+-------------------------------------------------------------+
```

- Team Saudi gradient background (#007167 -> #005a51)
- Gold accent progress bar (#a08e66)
- Shows current step text

### Skeleton Placeholders

CSS shimmer animation with:
- Gray rectangles matching real content shapes
- Left-to-right gradient animation
- Subtle pulse effect

### Tab States

- **Loading**: Grayed out text, no hover effect
- **Ready**: Full color, clickable
- **Active**: Underline accent in gold

## Implementation

### New File: loading_components.py

```python
# ~100 lines
def render_loading_header(current_step: int, total_steps: int, message: str)
def render_skeleton_card(width: str, height: str)
def render_skeleton_table(rows: int, cols: int)
def get_shimmer_css() -> str
```

### Session State Tracking

```python
st.session_state.loading = {
    'profiles': False,      # True when loaded
    'benchmarks': False,
    'road_to_tokyo': False,
    'master': False,
    'current_step': 0,
    'total_steps': 5
}
```

### Modified Files

**World_Ranking_Deploy_v3.py**
- Replace current st.status block with new loading components
- Track loading state per data source
- Wrap tab content with "show skeleton or real data" logic

**data_connector.py**
- Keep existing @st.cache_data for actual data caching
- No callback needed - use session state updates

## CSS Shimmer Animation

```css
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
```

## Team Saudi Branding

Colors used:
- Primary Teal: #007167 (header background)
- Dark Teal: #005a51 (gradient end)
- Gold Accent: #a08e66 (progress bar, active tab underline)
- Font: Inter, sans-serif

## Testing Checklist

- [ ] Header appears immediately on page load
- [ ] Progress bar updates as each file loads
- [ ] Tabs enable progressively as data becomes available
- [ ] Skeleton placeholders show shimmer animation
- [ ] Works on Streamlit Cloud (Azure mode)
- [ ] Works locally (local mode)
- [ ] Cached loads skip loading screen entirely
