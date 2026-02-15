# AI Analytics Ideas (from Tilastopaja project)

Reference patterns from the other athletics dashboard's AI module for future enhancement.

## Key Ideas to Consider

### 1. Natural Language to SQL
- AI generates DuckDB SQL from natural language questions
- JSON response format: `{explanation, sql, chart_type, chart_code, follow_ups}`
- Auto-retry on SQL errors (Binder Error, missing columns)
- Name suggestion when query returns empty results
- System prompt includes full schema + domain knowledge

### 2. Auto-Generated Charts
- AI writes Plotly chart code in its JSON response
- Fallback: auto-chart based on `chart_type` + data columns
- Team Saudi color sequence applied to all charts

### 3. Pre-Built Query Tabs (Instant, No AI Roundtrip)
- **Standards Gap**: KSA PBs & WA Points, Best by Event, Recent Form, Improving Athletes
- **Rival Watch**: KSA vs Rivals, Top 20 Asia, Asian Games 2023, Form Trend
- **Championship History**: AG 2023, World Champs, Olympics, Best Performances, Medals, Asian Champs
- Each has 4-6 buttons that run direct SQL queries instantly

### 4. Data Summary Caching
- `_get_data_summary()` builds compact reference of KSA athlete names + events
- Injected as system message so AI knows exact names/events in database
- Cached in `st.session_state['ai_data_summary']`

### 5. Smart Name Detection
- `_detect_name_words()` finds capitalized words likely to be athlete names
- Enhanced prompt with LIKE wildcards: `WHERE Athlete_Name LIKE '%LastName%'`
- Cross-references with actual database names for exact match hints

### 6. Free Model Optimizations
- MAX_HISTORY = 4 (small context windows)
- Truncated explanations in history (200 chars)
- SQL reminders injected at end of every user message
- Context document truncated to first 800 lines

### 7. Competition ID Mapping
```python
CHAMPIONSHIP_IDS = {
    "Asian Games Hangzhou 2023": "13048549",
    "Asian Games Jakarta 2018": "12911586",
    "WC Tokyo 2025": "13112510",
    "WC Budapest 2023": "13046619",
    "WC Oregon 2022": "13002354",
    "WC Doha 2019": "12935526",
}
```

### 8. Event Type Auto-Detection
```python
TIME_EVENTS = {'100m', '200m', '400m', '800m', '1500m', '5000m', '10000m',
               '110m Hurdles', '100m Hurdles', '400m Hurdles', '3000m Steeplechase',
               'Marathon', '20km Race Walk', '35km Race Walk'}
# For time events: sort ASC (lower=better), use MIN
# For field events: sort DESC (higher=better), use MAX
```

## Implementation Priority

1. **Already done in v2**: NotebookLM backend, quick analysis tabs, backend selector
2. **Next phase**: Natural language to SQL (requires DuckDB schema mapping)
3. **Future**: Auto-chart generation, name detection, competition ID mapping
