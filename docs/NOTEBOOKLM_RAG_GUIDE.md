# NotebookLM + MCP: AI-Powered Analytics Guide

**A comprehensive guide for integrating NotebookLM as a RAG backend for Team Saudi sports analytics projects.**

---

## Overview

This guide explains how to replace traditional RAG (Retrieval-Augmented Generation) systems with Google's NotebookLM, using MCP (Model Context Protocol) for seamless integration with AI tools like Claude Code.

### Why NotebookLM over Custom RAG?

| Factor | Custom RAG | NotebookLM |
|--------|-----------|------------|
| **Setup Time** | Hours (embeddings, vector DB, chunking) | 5 minutes |
| **Infrastructure** | Sentence-transformers, FAISS/Chroma | None - Google handles it |
| **Startup Time** | 30-60 seconds (model loading) | Instant |
| **Hallucinations** | Possible retrieval gaps | Refuses if info not in docs |
| **Cost** | Compute + API costs | Free |
| **Maintenance** | Embeddings drift, reindexing | Zero |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI TOOL (Claude Code/Cursor)                │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
        ┌───────────┴───────────┐   ┌──────────┴──────────┐
        ▼                       ▼   ▼                      ▼
┌───────────────────┐   ┌───────────────────┐   ┌─────────────────┐
│  LIVE DATA        │   │  NOTEBOOKLM MCP   │   │ AUTO-GENERATED  │
│  (Structured)     │   │  (Documents)      │   │ BRIEFINGS       │
│                   │   │                   │   │                 │
│ • Parquet files   │   │ • Rulebooks       │   │ • Weekly summaries
│ • DuckDB queries  │   │ • Research PDFs   │   │ • Gap analyses  │
│ • Real-time data  │   │ • Historical docs │   │ • Rankings      │
│                   │   │                   │   │                 │
│ Fast SQL queries  │   │ Semantic search   │   │ Uploaded to     │
│                   │   │ + synthesis       │   │ NotebookLM      │
└───────────────────┘   └───────────────────┘   └─────────────────┘
```

### Key Insight

NotebookLM works with **documents**, not databases. The solution:

1. **Keep DuckDB** for live structured data queries (fast, works well)
2. **Generate briefings** from structured data → upload to NotebookLM
3. **Use NotebookLM** for document understanding (rulebooks, research, context)

---

## Setup Guide

### Step 1: Create Google Account (if needed)

1. Go to [accounts.google.com](https://accounts.google.com)
2. Create a free account (or use existing)

### Step 2: Access NotebookLM

1. Go to [notebooklm.google.com](https://notebooklm.google.com)
2. Sign in with Google account
3. Create your first notebook (e.g., "KSA Athletics Intelligence")

### Step 3: Install notebooklm-py

```bash
pip install notebooklm-py
```

### Step 4: Authenticate

```bash
notebooklm login
```

This opens a browser window for Google authentication. One-time setup.

### Step 5: Install NotebookLM MCP (for Claude Code)

```bash
claude mcp add notebooklm npx notebooklm-mcp@latest
```

---

## Workflow for Sports Projects

### A. Initial Setup (One-Time)

1. **Create project notebook** in NotebookLM
2. **Upload static documents**:
   - Rulebooks (PDF)
   - Historical analysis reports
   - Competition rules
   - Reference materials

### B. Automated Pipeline (Weekly/Daily)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Scrape Data │────▶│  Generate   │────▶│  Upload to  │
│ (existing)  │     │  Briefings  │     │  NotebookLM │
└─────────────┘     └─────────────┘     └─────────────┘
```

### C. Query Flow

```
User: "What does Athlete X need to medal at Asian Games?"

Claude Code:
├── Query Parquet: Get athlete's current stats
├── Query NotebookLM: Get medal standards + historical context
└── Synthesize: Combine data + context for answer
```

---

## Briefing Generator Template

Create `generate_briefings.py` in your project:

```python
"""
Generate Markdown Briefings from Data for NotebookLM Upload.
Customize for your sport/project.
"""

import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration - CUSTOMIZE THESE
PROJECT_NAME = "Your Sport Name"
PARQUET_DIR = Path("Data/parquet")
BRIEFINGS_DIR = Path("briefings")
BRIEFINGS_DIR.mkdir(exist_ok=True)


def load_parquet(filename: str) -> pd.DataFrame:
    """Load Parquet file using DuckDB."""
    filepath = PARQUET_DIR / filename
    if not filepath.exists():
        return pd.DataFrame()
    conn = duckdb.connect()
    return conn.execute(f"SELECT * FROM '{filepath}'").df()


def generate_athlete_overview() -> str:
    """Generate athlete overview briefing."""
    df = load_parquet("athletes.parquet")
    if df.empty:
        return f"# {PROJECT_NAME} - Athlete Overview\n\nNo data available."

    today = datetime.now().strftime('%d %B %Y')

    briefing = f"""# {PROJECT_NAME} - Athlete Overview
*Generated: {today}*

## Summary
- **Total Athletes**: {len(df)}
- **Active**: {len(df[df['status'] == 'active'])}

## Top Athletes by Ranking

| Rank | Athlete | Event | Score |
|------|---------|-------|-------|
"""
    # Add your athletes table here
    for _, row in df.head(20).iterrows():
        briefing += f"| {row.get('rank', 'N/A')} | {row.get('name', 'N/A')} | {row.get('event', 'N/A')} | {row.get('score', 'N/A')} |\n"

    return briefing


def generate_gap_analysis() -> str:
    """Generate gap to standards analysis."""
    # CUSTOMIZE: Load your benchmark/standards data
    athletes = load_parquet("athletes.parquet")
    standards = load_parquet("standards.parquet")

    today = datetime.now().strftime('%d %B %Y')

    briefing = f"""# {PROJECT_NAME} - Gap Analysis
*Generated: {today}*

## Medal Standards

| Event | Gold | Silver | Bronze |
|-------|------|--------|--------|
"""
    # Add your standards table
    for _, row in standards.head(20).iterrows():
        briefing += f"| {row.get('event', 'N/A')} | {row.get('gold', 'N/A')} | {row.get('silver', 'N/A')} | {row.get('bronze', 'N/A')} |\n"

    return briefing


def generate_competition_focus() -> str:
    """Generate competition focus briefing."""
    today = datetime.now().strftime('%d %B %Y')

    return f"""# {PROJECT_NAME} - Competition Focus
*Generated: {today}*

## Upcoming Events
- [List your upcoming competitions]

## Priority Athletes
- [List priority athletes for upcoming events]

## Key Competitors
- [List main competitors to watch]
"""


def generate_all_briefings():
    """Generate all briefings and save to files."""
    print("Generating briefings...")

    briefings = {
        "01_athlete_overview.md": generate_athlete_overview(),
        "02_gap_analysis.md": generate_gap_analysis(),
        "03_competition_focus.md": generate_competition_focus(),
    }

    for filename, content in briefings.items():
        filepath = BRIEFINGS_DIR / filename
        filepath.write_text(content, encoding='utf-8')
        print(f"  Created: {filepath}")

    print(f"\nAll briefings saved to: {BRIEFINGS_DIR}")


if __name__ == "__main__":
    generate_all_briefings()
```

---

## Upload Script Template

Create `upload_to_notebooklm.py`:

```python
"""
Upload briefings to NotebookLM.
"""

import subprocess
from pathlib import Path

BRIEFINGS_DIR = Path("briefings")
DOCUMENTS_DIR = Path("documents")
NOTEBOOK_NAME = "Your Project Name"  # CUSTOMIZE


def upload_file(filepath: Path, notebook: str) -> bool:
    """Upload file using notebooklm CLI."""
    result = subprocess.run(
        ["notebooklm", "source", "add", str(filepath), "--notebook", notebook],
        capture_output=True, text=True
    )
    return result.returncode == 0


def main():
    # Upload briefings
    for file in BRIEFINGS_DIR.glob("*.md"):
        print(f"Uploading: {file.name}")
        upload_file(file, NOTEBOOK_NAME)

    # Upload PDFs
    for file in DOCUMENTS_DIR.glob("*.pdf"):
        print(f"Uploading: {file.name}")
        upload_file(file, NOTEBOOK_NAME)


if __name__ == "__main__":
    main()
```

---

## GitHub Actions Automation

Add to `.github/workflows/weekly_sync.yml`:

```yaml
- name: Generate briefings
  run: python generate_briefings.py

- name: Upload to NotebookLM
  env:
    GOOGLE_AUTH_TOKEN: ${{ secrets.NOTEBOOKLM_TOKEN }}
  run: python upload_to_notebooklm.py
```

**Note**: For GitHub Actions, you'll need to handle authentication via service account or stored token.

---

## MCP Integration with Claude Code

Once NotebookLM MCP is installed, Claude Code can:

1. **List notebooks**: See all your NotebookLM notebooks
2. **Query notebooks**: Ask questions against uploaded sources
3. **Get citations**: Responses include source references

### Example Prompts

```
"Query my KSA Athletics notebook: What are the medal standards for 100m?"

"Search the rulebooks for qualification requirements"

"What does the historical data say about medal progression?"
```

---

## Best Practices

### 1. Briefing Structure

- **Use Markdown tables** - NotebookLM parses them well
- **Include dates** - Shows freshness of data
- **Summarize key stats** - Don't dump raw data
- **Add context** - Explain what numbers mean

### 2. Document Organization

```
NotebookLM Notebook: "Project Name"
├── Briefings (auto-updated weekly)
│   ├── Athlete Overview
│   ├── Gap Analysis
│   └── Competition Focus
├── Static Documents (manual upload)
│   ├── Rulebook.pdf
│   ├── Historical Analysis.pdf
│   └── Standards Reference.pdf
└── Research (as needed)
    ├── Competitor Analysis.pdf
    └── Training Reports.pdf
```

### 3. Update Frequency

| Content Type | Update Frequency |
|--------------|------------------|
| Briefings (from Parquet) | Weekly |
| Rulebooks | When rules change |
| Competition standards | Seasonally |
| Research reports | As produced |

---

## Troubleshooting

### "notebooklm: command not found"

```bash
pip install notebooklm-py
```

### Authentication Failed

```bash
notebooklm logout
notebooklm login
```

### Upload Errors

1. Check file size (NotebookLM has limits)
2. Ensure file is readable (not locked)
3. Try uploading via web UI first

### MCP Not Working

```bash
claude mcp remove notebooklm
claude mcp add notebooklm npx notebooklm-mcp@latest
```

---

## Project Checklist

- [ ] Google account created
- [ ] NotebookLM accessed at [notebooklm.google.com](https://notebooklm.google.com)
- [ ] `notebooklm-py` installed
- [ ] Authenticated via `notebooklm login`
- [ ] Project notebook created
- [ ] Static documents uploaded (PDFs, rulebooks)
- [ ] `generate_briefings.py` customized
- [ ] `upload_to_notebooklm.py` configured
- [ ] MCP installed: `claude mcp add notebooklm npx notebooklm-mcp@latest`
- [ ] Test query working

---

## Team Saudi Projects Using This Approach

| Project | Notebook Name | Key Sources |
|---------|---------------|-------------|
| Athletics | KSA Athletics Intelligence | Rankings, Rulebooks, Standards |
| Swimming | KSA Swimming Analytics | FINA Rules, Meet Results |
| Fencing | Fencing Performance | FIE Rules, Rankings |
| Taekwondo | Taekwondo Dashboard | WT Rules, Points System |

---

## Resources

- [NotebookLM](https://notebooklm.google.com) - Google's AI research assistant
- [notebooklm-py](https://github.com/teng-lin/notebooklm-py) - Python API
- [NotebookLM MCP](https://github.com/PleasePrompto/notebooklm-mcp) - MCP server
- [MCP Protocol](https://modelcontextprotocol.io) - Model Context Protocol docs

---

*Document created: February 2026*
*Author: Team Saudi Performance Analysis*
