# Race Intelligence & Enhanced Analytics Design

> **Created:** 2026-01-25
> **Status:** Approved
> **Estimated:** ~1,000 lines of new code

## Overview

Comprehensive enhancement to the Saudi Athletics Dashboard adding race simulation, competitor intelligence, improved charts, and detailed athlete profiles.

---

## Component 1: Standards Progression (Tab 5)

**Location:** Tab 5 "World Champs Qualification" - new expandable section

### Features
- Year-over-year qualification standard trends
- Line chart showing standards over time (2019-2024+)
- Multi-championship comparison (Olympic, World Champs, Asian Games)
- Trend insight (e.g., "Standards tightening by ~0.02s per cycle")

### UI Elements
- Event dropdown
- Gender dropdown
- Championship filter (All / Olympic / World / Asian)
- Line chart with legend
- Table with year-by-year breakdown

### Data Source
- Extend `benchmarks.parquet` with historical years
- Or scrape from World Athletics qualification pages

---

## Component 2: Competitor Form Cards (Tab 2)

**Location:** Tab 2 "Athlete Profiles" - new section below H2H comparison

### Features
- Top 5-10 competitors for the selected KSA athlete's primary event
- Form score (0-100) with status icons
- Best 2 races with dates and venues
- Last competition with recency
- Visual comparison bar vs KSA athlete

### Form Score Calculation (0-100)
```
40% weight: Recent average vs PB (closer = higher)
30% weight: Trend direction (improving/stable/declining)
20% weight: Recency of last competition (within 30 days = bonus)
10% weight: Competition quality (Diamond League > local meet)
```

### Form Status Icons
- 🔥 Hot (Form 85+, competed within 14 days)
- 📈 Rising (Form 70+, improving trend)
- ── Stable (Form 60-70, consistent)
- 📉 Cooling (Form <60 or no recent results)

### Card Layout
```
┌─────────────────────────────────────────────────────┐
│ 1. Kirani JAMES (GRN)                    Form: 92  │
│    PB: 43.74  |  Avg (Last 5): 44.12  |  🔥 Hot    │
│                                                     │
│    Best 2 Races:                                    │
│    ⭐ 43.95  Paris DL         14 Jun 2024          │
│    ⭐ 44.05  Doha DL          10 May 2024          │
│                                                     │
│    Last: 44.18 @ Monaco DL (8 days ago)            │
│    ████████████████████░░░░ vs Al-Yaqoub          │
└─────────────────────────────────────────────────────┘
```

---

## Component 3: Race Intelligence Tab (New Tab 11)

**Location:** New tab after Project East 2026

### Section A: Form Rankings Table

Interactive table showing all athletes in an event ranked by current form (not just PB).

**Columns:**
- Rank
- Athlete name
- Nationality (with flag)
- Form score + status icon
- Average (last 5)
- Last competition (days ago)
- Checkbox for race preview selection

**Filters:**
- Top N (10/20/50/All)
- Region (All/Asia/Europe/Americas/Africa)
- Search by athlete name

**KSA Highlighting:** Green background for Saudi athletes

### Section B: Race Preview Builder

Build hypothetical races and see advancement probabilities.

**Features:**
- Championship selector (Asian Games 2026, World Championships 2027, LA 2028)
- Select up to 8 athletes (manual or auto-fill top 8 + KSA)
- Round-by-round progression table

**Progression Table:**
```
│ Athlete         │   HEATS      │   SEMI       │  FINAL   │
│                 │ Prob │ Need  │ Prob │ Need  │Prob│Need │
├─────────────────┼──────┼───────┼──────┼───────┼────┼─────┤
│ M. Al-Yaqoub    │ 95%  │<45.80 │ 72%  │<45.20 │45% │<44.60│
```

**Medal Zone:**
```
│ Athlete         │ 🥇 Gold  │ 🥈 Silver │ 🥉 Bronze │ Gap  │
│                 │  43.50   │   43.85   │   44.20   │      │
├─────────────────┼──────────┼───────────┼───────────┼──────┤
│ M. Al-Yaqoub    │    5%    │    10%    │    15%    │-0.69 │
```

**Target Time Logic:**
- Heats: Historical auto-qualifier time (top 3 per heat)
- Semis: Typical semi-final qualifying time (8th fastest overall)
- Final: Historical 8th place in finals
- Medal standards: Average from last 3 championships

**Insight Box:** Auto-generated summary of key targets for KSA athletes

---

## Component 4: Chart Redesign

**Location:** `chart_components.py` - global configuration update

### Typography Updates
```python
CHART_CONFIG = {
    'title_font_size': 20,        # Was 16
    'axis_font_size': 14,         # Was 12
    'legend_font_size': 13,       # Was 11
    'annotation_font_size': 12,   # Was 10
}
```

### Simplified Color Palette
```python
'colors': {
    'primary': '#005430',      # Saudi Green (main data)
    'secondary': '#a08e66',    # Gold (comparisons)
    'benchmark': '#2A8F5C',    # Light green (standards)
    'neutral': '#78909C',      # Gray (grid/secondary)
}
```

### Medal Standard Lines
```python
'gold_line': {'color': '#FFD700', 'dash': 'solid', 'width': 2},
'silver_line': {'color': '#C0C0C0', 'dash': 'dash', 'width': 2},
'bronze_line': {'color': '#CD7F32', 'dash': 'dot', 'width': 2},
```

### Interactive Tooltips
```python
'hovertemplate': '''
    <b>%{customdata[0]}</b><br>
    Result: %{y:.2f}<br>
    Venue: %{customdata[1]}<br>
    Date: %{customdata[2]}<br>
    <extra></extra>
'''
```

### Comparison Overlays
- Toggle to show/hide multiple athletes on same chart
- Toggle to show/hide medal standard lines

---

## Component 5: Career Milestones Timeline (Tab 2)

**Location:** Tab 2 "Athlete Profiles" - new expandable section

### Features
- Visual timeline grouped by year
- Auto-detected milestone types
- Filterable by milestone type

### Milestone Types
- ⭐ Personal Bests (each time PB is broken)
- 🏆 Championship titles (1st place at national/continental/world level)
- 🥇🥈🥉 Medals at major championships
- 📍 Career firsts (first sub-X, first DL, first WC appearance)
- 🎯 Standard achieved (first time meeting Olympic/WC standard)

### Layout
```
2024 ──●────────●─────────────●──────────────────────────
       │        │             │
       │        │             └─ Jun: 44.89 Doha DL
       │        └─ Apr: First Diamond League appearance
       └─ Feb: 45.12 Asian Indoor (Bronze 🥉)

2023 ────●──────────────●───────────────●────────────────
         │              │               │
         │              │               └─ Aug: 44.95 WC
         │              └─ Jul: PB! 44.78 ⭐
         └─ Mar: First sub-45 (44.98)
```

### Controls
- Expand All / Collapse
- Filter: PBs Only
- Filter: Medals Only
- Filter: Championships Only

### Data Source
- Extracted from `master.parquet` results history
- Cross-reference with `benchmarks.parquet` for standard achievements

---

## Implementation Order

| Priority | Component | Effort | Dependencies |
|----------|-----------|--------|--------------|
| 1 | Chart Redesign | ~100 lines | None (foundation) |
| 2 | Form Score Function | ~80 lines | None |
| 3 | Competitor Form Cards | ~200 lines | Form Score |
| 4 | Race Intelligence Tab | ~400 lines | Form Score, Charts |
| 5 | Standards Progression | ~150 lines | Charts |
| 6 | Career Timeline | ~150 lines | None |

**Total: ~1,080 lines**

---

## Data Requirements

### New Functions Needed in `data_connector.py`
```python
def get_athlete_form_score(athlete_id: str) -> dict
def get_top_competitors_by_form(event: str, gender: str, limit: int) -> DataFrame
def get_historical_standards(event: str, gender: str) -> DataFrame
def get_round_qualifying_times(championship: str, event: str) -> dict
def get_athlete_milestones(athlete_id: str) -> DataFrame
```

### Parquet Schema Extensions
- `benchmarks.parquet`: Add historical years (2019-2024)
- Or create new `historical_standards.parquet`

---

## Success Criteria

1. Coaches can see competitor form at a glance
2. Race previews show clear probability and target times
3. Charts are readable on projectors (larger fonts)
4. Career progression is visually clear
5. All components use Team Saudi branding consistently
