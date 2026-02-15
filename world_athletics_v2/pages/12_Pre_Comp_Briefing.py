"""
Pre-Competition Briefing Report.

Replicates QAF/Aspire Academy competition analysis slide design
with Team Saudi branding. Shows KSA athlete profile, recent results,
and competitor intelligence for a selected championship.
"""

import base64
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from components.theme import (
    TEAL_PRIMARY, TEAL_DARK, TEAL_LIGHT, GOLD_ACCENT,
    render_page_header, render_sidebar, get_theme_css, get_logo_b64,
)
from data.connector import DataConnector
from data.event_utils import format_event_name, get_event_type


# ── IOC 3-letter → ISO 2-letter for emoji flags ─────────────────────

IOC_TO_ISO2 = {
    "KSA": "SA", "IND": "IN", "CHN": "CN", "JPN": "JP", "KOR": "KR",
    "IRI": "IR", "IRQ": "IQ", "KGZ": "KG", "KAZ": "KZ", "TJK": "TJ",
    "TKM": "TM", "UZB": "UZ", "QAT": "QA", "BRN": "BH", "UAE": "AE",
    "OMA": "OM", "KUW": "KW", "MAC": "MO", "HKG": "HK", "TPE": "TW",
    "MAS": "MY", "SGP": "SG", "PHI": "PH", "INA": "ID", "THA": "TH",
    "VIE": "VN", "MYA": "MM", "LAO": "LA", "CAM": "KH", "BRU": "BN",
    "PRK": "KP", "MGL": "MN", "AFG": "AF", "PAK": "PK", "BAN": "BD",
    "SRI": "LK", "NEP": "NP", "MDV": "MV", "LBN": "LB", "SYR": "SY",
    "JOR": "JO", "PLE": "PS", "YEM": "YE",
    "USA": "US", "GBR": "GB", "JAM": "JM", "CAN": "CA", "AUS": "AU",
    "NZL": "NZ", "RSA": "ZA", "NGR": "NG", "KEN": "KE", "ETH": "ET",
    "GER": "DE", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NED": "NL",
    "SUI": "CH", "SWE": "SE", "NOR": "NO", "DEN": "DK", "FIN": "FI",
    "POL": "PL", "CZE": "CZ", "GRE": "GR", "CRO": "HR", "SRB": "RS",
    "BUL": "BG", "ROU": "RO", "HUN": "HU", "TUR": "TR", "POR": "PT",
    "BEL": "BE", "AUT": "AT", "IRL": "IE", "ISR": "IL", "UKR": "UA",
    "LTU": "LT", "LAT": "LV", "EST": "EE", "BLR": "BY", "GEO": "GE",
    "ARM": "AM", "AZE": "AZ", "CYP": "CY",
    "CUB": "CU", "BRA": "BR", "MEX": "MX", "COL": "CO", "ARG": "AR",
    "CHI": "CL", "PER": "PE", "ECU": "EC", "DOM": "DO", "PUR": "PR",
    "TTO": "TT", "BAH": "BS", "BAR": "BB", "PAN": "PA", "CRC": "CR",
    "MAR": "MA", "TUN": "TN", "ALG": "DZ", "EGY": "EG", "CMR": "CM",
    "SEN": "SN", "GHA": "GH", "CIV": "CI", "BOT": "BW", "NAM": "NA",
    "UGA": "UG", "NIG": "NE", "MOZ": "MZ", "ZIM": "ZW", "TAN": "TZ",
    "NOR": "NO", "LBA": "LY", "SLE": "SL",
}


CHAMPIONSHIPS = {
    "Asian Games 2026 (Nagoya)": {"region": "asia", "label": "Asian Games 2026"},
    "World Championships 2025 (Tokyo)": {"region": None, "label": "World Championships 2025"},
    "Olympics 2028 (Los Angeles)": {"region": None, "label": "Olympics 2028"},
}


# ── Data Overrides (bad records to exclude) ─────────────────────────
# Format: {athlete_name_substring: [discipline_substring, ...]}
EXCLUDE_RESULTS = {
    "ATAFI": ["Discus"],
}


# ── Helper Functions ─────────────────────────────────────────────────


def _flag(code: str) -> str:
    """Convert IOC country code to emoji flag."""
    iso2 = IOC_TO_ISO2.get(code.upper(), code[:2].upper()) if code else ""
    if len(iso2) == 2:
        return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso2)
    return ""


def _parse_mark(mark) -> Optional[float]:
    """Parse a mark string to float seconds/metres."""
    if mark is None or str(mark).strip() in ("", "-", "None", "nan"):
        return None
    m = str(mark).strip()
    try:
        return float(m)
    except ValueError:
        try:
            parts = m.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        except (ValueError, TypeError):
            pass
    return None


def _format_mark(seconds: float, event: str) -> str:
    """Format seconds to display string."""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = seconds - mins * 60
        return f"{mins}:{secs:05.2f}"
    return f"{seconds:.2f}"


def _safe(val) -> bool:
    """Check if a value is non-null and non-empty."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    return str(val).strip() not in ("", "None", "nan", "-")


# ── Data Gathering ───────────────────────────────────────────────────


def gather_athlete_data(
    dc: DataConnector, athlete_name: str, event_display: str,
) -> Dict[str, Any]:
    """Gather all data for the KSA athlete."""
    athletes = dc.get_ksa_athletes()
    athlete_row = athletes[athletes["full_name"] == athlete_name]
    if athlete_row.empty:
        return {}
    ath = athlete_row.iloc[0]

    data: Dict[str, Any] = {
        "name": athlete_name,
        "event_display": event_display,
        "athlete_id": ath.get("athlete_id"),
        "photo_url": ath.get("photo_url", ""),
        "world_rank": ath.get("best_world_rank"),
        "country": ath.get("country_code", "KSA"),
        "sb": "-",
        "pb": "-",
        "latest_mark": "-",
        "latest_date": "-",
        "latest_place": "",
        "perf_count": 0,
        "last3_avg_str": "-",
        "results": pd.DataFrame(),
    }

    # PBs for the event
    pbs = dc.get_ksa_athlete_pbs(athlete_name)
    if pbs is not None and not pbs.empty:
        disc_col = "discipline" if "discipline" in pbs.columns else "event"
        mark_col = "mark" if "mark" in pbs.columns else "result"
        for _, row in pbs.iterrows():
            if format_event_name(str(row.get(disc_col, ""))) == event_display:
                data["pb"] = str(row.get(mark_col, "-"))
                break

    # Season bests - try current year first, then fall back to previous year
    current_year = datetime.now().year
    for try_season in [current_year, current_year - 1]:
        sbs = dc.get_ksa_athlete_season_bests(athlete_name, season=try_season)
        if sbs is not None and not sbs.empty:
            disc_col = "discipline" if "discipline" in sbs.columns else "event"
            mark_col = "mark" if "mark" in sbs.columns else "result"
            for _, row in sbs.iterrows():
                if format_event_name(str(row.get(disc_col, ""))) == event_display:
                    data["sb"] = str(row.get(mark_col, "-"))
                    if try_season < current_year:
                        data["sb_note"] = f"({try_season} SB)"
                    break
        if data["sb"] != "-":
            break

    # Results (load all for athlete, filter by event in Python)
    results = dc.get_ksa_results(athlete_name, limit=50)
    if results is not None and not results.empty:
        disc_col = "discipline" if "discipline" in results.columns else "event"
        mask = results[disc_col].apply(
            lambda d: format_event_name(str(d)) == event_display
        )
        results = results[mask].copy()
        data["results"] = results
        data["perf_count"] = len(results)

        if not results.empty:
            data["latest_mark"] = str(results.iloc[0].get("mark", "-"))
            data["latest_date"] = str(results.iloc[0].get("date", "-"))
            data["latest_place"] = str(results.iloc[0].get("place", ""))

            # Compute marks for averages and SB fallback
            mark_col = "mark" if "mark" in results.columns else "result"
            marks = []
            for _, row in results.iterrows():
                val = _parse_mark(row.get(mark_col))
                if val is not None:
                    marks.append(val)

            # Last 3 average
            if len(marks) >= 3:
                avg3 = sum(marks[:3]) / 3
                data["last3_avg_str"] = _format_mark(avg3, event_display)
            elif marks:
                avg = sum(marks) / len(marks)
                data["last3_avg_str"] = _format_mark(avg, event_display)

            # SB fallback: best mark from results when season_bests empty
            if data["sb"] == "-" and marks:
                is_time = get_event_type(event_display) == "time"
                best = min(marks) if is_time else max(marks)
                data["sb"] = _format_mark(best, event_display)
                data["sb_note"] = "(from results)"

    return data


def gather_competitors(
    dc: DataConnector,
    event_display: str,
    gender: str = "M",
    region: Optional[str] = None,
    limit: int = 25,
    max_per_country: Optional[int] = None,
) -> pd.DataFrame:
    """Gather competitor data for the event, filtered by gender.

    Uses get_top_performers() from master data for correct gender matching
    since rivals.parquet currently contains Women's data for all events.
    Falls back to rivals.parquet if master data is unavailable.

    Args:
        max_per_country: If set, keep only top N athletes per country.
            Realistic for Asian Games (typically 3 per country per event).
    """
    from data.event_utils import ASIAN_COUNTRY_CODES, get_event_type

    # Use master.parquet for accurate gender-filtered competitors
    country_codes = list(ASIAN_COUNTRY_CODES) if region == "asia" else None
    # Fetch more than needed so per-country filter still yields enough rows
    fetch_limit = limit * 3 if max_per_country else limit
    performers = dc.get_top_performers(
        event=event_display, gender=gender,
        country_codes=country_codes, limit=fetch_limit,
    )
    if performers is not None and not performers.empty:
        result = performers
    else:
        # Fall back to rivals.parquet (may have wrong gender)
        rivals = dc.get_rivals(event=event_display, gender=gender, region=region, limit=fetch_limit)
        if rivals is None or rivals.empty:
            rivals = dc.get_rivals(event=event_display, limit=fetch_limit)
        result = rivals if rivals is not None else pd.DataFrame()

    # Apply per-country limit
    if max_per_country and not result.empty:
        nat_col = "country_code" if "country_code" in result.columns else "nat"
        if nat_col in result.columns:
            # Data is already sorted by best mark; keep top N per country
            result = (
                result.groupby(nat_col, sort=False)
                .head(max_per_country)
                .reset_index(drop=True)
            )
            # Re-sort globally and limit
            mark_col = "best_mark_numeric" if "best_mark_numeric" in result.columns else "ranking_score"
            if mark_col in result.columns:
                et = get_event_type(event_display)
                result = result.sort_values(
                    mark_col, ascending=(et == "time"), na_position="last"
                ).head(limit)

    return result


# ── HTML Report Builder ──────────────────────────────────────────────


def build_report_html(
    athlete_data: Dict[str, Any],
    competitors: pd.DataFrame,
    championship_label: str,
    logo_b64: Optional[str] = None,
    standalone: bool = False,
) -> str:
    """Build the full HTML report matching the QAF/Aspire slide design."""

    name = athlete_data.get("name", "Unknown")
    event_display = athlete_data.get("event_display", "")
    sb = athlete_data.get("sb", "-")
    sb_note = athlete_data.get("sb_note", "")
    pb = athlete_data.get("pb", "-")
    world_rank = athlete_data.get("world_rank")
    world_rank_str = str(int(world_rank)) if _safe(world_rank) else "-"
    latest_mark = athlete_data.get("latest_mark", "-")
    latest_date = athlete_data.get("latest_date", "-")
    latest_place = athlete_data.get("latest_place", "")
    perf_count = athlete_data.get("perf_count", 0)
    last3_avg_str = athlete_data.get("last3_avg_str", "-")

    # Last perf display
    last_perf_str = latest_mark
    if _safe(latest_place):
        last_perf_str = f"{latest_mark} ({latest_place})"

    # Photo
    photos_dir = Path(__file__).parent.parent / "data" / "scraped" / "photos"
    aid = athlete_data.get("athlete_id")
    local_photo = photos_dir / f"{aid}.jpg" if aid else None

    if local_photo and local_photo.exists():
        photo_b64 = base64.b64encode(local_photo.read_bytes()).decode()
        photo_html = (
            f'<img src="data:image/jpeg;base64,{photo_b64}" '
            f'style="width: 120px; height: 140px; object-fit: cover; '
            f'border-radius: 8px; border: 2px solid rgba(255,255,255,0.3);">'
        )
    elif _safe(athlete_data.get("photo_url")):
        photo_html = (
            f'<img src="{athlete_data["photo_url"]}" '
            f'style="width: 120px; height: 140px; object-fit: cover; '
            f'border-radius: 8px; border: 2px solid rgba(255,255,255,0.3);" '
            f'onerror="this.style.display=\'none\'">'
        )
    else:
        photo_html = (
            '<div style="width: 120px; height: 140px; background: rgba(255,255,255,0.1); '
            'border-radius: 8px; display: flex; align-items: center; justify-content: center; '
            'color: rgba(255,255,255,0.4); font-size: 2.5rem;">&#9786;</div>'
        )

    # Logo
    logo_html = ""
    if logo_b64:
        logo_html = (
            f'<img src="data:image/jpeg;base64,{logo_b64}" '
            f'style="height: 55px; border-radius: 6px;">'
        )

    # ── Athlete results table rows ──
    results_html = ""
    results_df = athlete_data.get("results")
    if results_df is not None and not results_df.empty:
        for idx, (_, row) in enumerate(results_df.head(8).iterrows()):
            mark = str(row.get("mark", "-"))
            venue = str(row.get("venue", row.get("competition", "-")))
            if len(venue) > 30:
                venue = venue[:27] + "..."
            date_str = str(row.get("date", "-"))
            row_bg = "background: rgba(255,255,255,0.05);" if idx % 2 == 0 else ""
            results_html += (
                f'<tr style="{row_bg}">'
                f'<td style="padding: 5px 10px; color: white; font-size: 0.82rem;">{mark}</td>'
                f'<td style="padding: 5px 10px; color: rgba(255,255,255,0.8); font-size: 0.82rem;">{venue}</td>'
                f'<td style="padding: 5px 10px; color: rgba(255,255,255,0.8); font-size: 0.82rem;">{date_str}</td>'
                f'</tr>'
            )
    else:
        results_html = (
            '<tr><td colspan="3" style="padding: 10px; color: rgba(255,255,255,0.5); '
            'text-align: center; font-size: 0.85rem;">No results available</td></tr>'
        )

    # ── Competitors table rows ──
    competitors_html = ""
    athlete_sb_numeric = _parse_mark(sb) if sb != "-" else _parse_mark(pb)
    event_type = get_event_type(event_display)
    lower_is_better = event_type == "time"
    yr = str(datetime.now().year)[-2:]

    if not competitors.empty:
        # Compute ranking position within the filtered set
        comp_sorted = competitors.copy()
        # Handle both data sources: rivals.parquet and get_top_performers()
        is_master_data = "best_mark_numeric" in comp_sorted.columns
        if is_master_data:
            # From get_top_performers() - already sorted by best_mark
            pass
        elif "ranking_score" in comp_sorted.columns:
            comp_sorted = comp_sorted.sort_values(
                "ranking_score", ascending=False, na_position="last"
            )
        elif "world_rank" in comp_sorted.columns:
            comp_sorted = comp_sorted.sort_values(
                "world_rank", ascending=True, na_position="last"
            )
        comp_sorted = comp_sorted.reset_index(drop=True)
        comp_sorted["_asia_rank"] = range(1, len(comp_sorted) + 1)

        for idx, (_, rival) in enumerate(comp_sorted.iterrows()):
            comp_name = str(rival.get("full_name", ""))
            comp_nat = str(rival.get("country_code", ""))
            flag = _flag(comp_nat)

            # SB / PB - handle both data sources
            if is_master_data:
                best_num = rival.get("best_mark_numeric")
                display_sb = _format_mark(float(best_num), event_display) if _safe(best_num) else "-"
            else:
                comp_sb = str(rival.get("sb_mark", "")) if _safe(rival.get("sb_mark")) else ""
                comp_pb = str(rival.get("pb_mark", "")) if _safe(rival.get("pb_mark")) else ""
                display_sb = comp_sb or comp_pb or "-"

            asia_rank = rival.get("_asia_rank", "")
            asia_rank_str = str(int(asia_rank)) if _safe(asia_rank) else "-"

            perf_cnt = rival.get("performances_count")
            perf_str = str(int(perf_cnt)) if _safe(perf_cnt) else "-"

            best5 = str(rival.get("best5_avg", "")) if _safe(rival.get("best5_avg")) else "-"

            latest = str(rival.get("latest_mark", "")) if _safe(rival.get("latest_mark")) else "-"
            latest_dt = str(rival.get("latest_date", "")) if _safe(rival.get("latest_date")) else ""
            # For master data, use latest_venue
            if is_master_data and latest == "-":
                latest_venue = str(rival.get("latest_venue", "")) if _safe(rival.get("latest_venue")) else ""
                if latest_venue:
                    latest = latest_venue[:20]

            # Compute gap vs KSA
            gap_str = "-"
            if athlete_sb_numeric:
                comp_numeric = None
                if is_master_data and _safe(rival.get("best_mark_numeric")):
                    comp_numeric = float(rival["best_mark_numeric"])
                elif display_sb != "-":
                    comp_numeric = _parse_mark(display_sb)
                if comp_numeric:
                    gap = comp_numeric - athlete_sb_numeric
                    if lower_is_better:
                        gap_str = f"{gap:+.2f}" if abs(gap) < 60 else "-"
                    else:
                        gap_str = f"{-gap:+.2f}" if abs(gap) < 100 else "-"

            # Last perf with date
            last_perf_display = latest
            if latest_dt and str(latest_dt) not in ("-", "None", "nan", "NaT", ""):
                dt_str = str(latest_dt)[:10]
                last_perf_display = f"{display_sb} ({dt_str})" if is_master_data else f"{latest} ({dt_str[:6]})"

            # Row styling
            is_ksa = comp_nat.upper() in ("KSA", "SAU")
            if is_ksa:
                row_bg = f"background: rgba(160, 142, 102, 0.35);"
            elif idx % 2 == 0:
                row_bg = "background: rgba(255,255,255,0.06);"
            else:
                row_bg = "background: rgba(255,255,255,0.02);"

            competitors_html += (
                f'<tr style="{row_bg}">'
                f'<td style="padding: 6px 10px; color: white; font-size: 0.82rem; white-space: nowrap;">{comp_name}</td>'
                f'<td style="padding: 6px 6px; color: white; font-size: 0.82rem; text-align: center;">{flag} {comp_nat}</td>'
                f'<td style="padding: 6px 6px; color: white; font-size: 0.82rem; text-align: center; font-weight: 600;">{display_sb}</td>'
                f'<td style="padding: 6px 6px; color: white; font-size: 0.82rem; text-align: center;">{gap_str}</td>'
                f'<td style="padding: 6px 6px; color: white; font-size: 0.82rem; text-align: center;">{asia_rank_str}</td>'
                f'<td style="padding: 6px 6px; color: white; font-size: 0.82rem; text-align: center;">{perf_str}</td>'
                f'<td style="padding: 6px 6px; color: white; font-size: 0.82rem; text-align: center;">{best5}</td>'
                f'<td style="padding: 6px 6px; color: white; font-size: 0.82rem; text-align: right; white-space: nowrap;">{last_perf_display}</td>'
                f'</tr>'
            )
    else:
        competitors_html = (
            '<tr><td colspan="8" style="padding: 15px; color: rgba(255,255,255,0.5); '
            'text-align: center; font-size: 0.85rem;">No competitor data. '
            'Run: python -m scrapers.scrape_rival_profiles</td></tr>'
        )

    # ── Build full HTML ──
    standalone_head = (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<style>* { margin: 0; padding: 0; box-sizing: border-box; }\n'
        'body { font-family: Inter, -apple-system, sans-serif; background: #111; padding: 1.5rem; }\n'
        '@media print { body { background: white; padding: 0; } }</style>\n'
        '</head><body>\n'
    ) if standalone else ""

    standalone_foot = "\n</body></html>" if standalone else ""

    html = f"""{standalone_head}
<div style="background: linear-gradient(135deg, {TEAL_DARK} 0%, {TEAL_PRIMARY} 40%, {TEAL_DARK} 100%);
     border-radius: 12px; padding: 1.5rem 2rem; font-family: Inter, -apple-system, sans-serif;
     max-width: 1400px; margin: 0 auto;">

    <!-- ═══ HEADER ═══ -->
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.2rem;
         border-bottom: 3px solid {GOLD_ACCENT}; padding-bottom: 1rem;">
        <div>
            <h1 style="color: white; margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.5px;">
                {championship_label} Competitor Analysis
            </h1>
            <h2 style="color: {GOLD_ACCENT}; margin: 0.3rem 0 0 0; font-size: 1.15rem; font-weight: 500;">
                {name} ({event_display})
            </h2>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            {logo_html}
        </div>
    </div>

    <!-- ═══ TWO-COLUMN LAYOUT ═══ -->
    <div style="display: flex; gap: 1.5rem; align-items: flex-start;">

        <!-- LEFT COLUMN: Athlete Info + Results -->
        <div style="width: 33%; min-width: 280px; flex-shrink: 0;">

            <!-- Photo + Info Grid -->
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                <div style="flex-shrink: 0;">{photo_html}</div>
                <div style="flex: 1;">
                    <div style="background: rgba(255,255,255,0.08); border-radius: 8px; padding: 0.7rem;
                         border: 1px solid rgba(255,255,255,0.12);">
                        <h4 style="color: {GOLD_ACCENT}; margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600;">
                            Athlete Information
                        </h4>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 3px 0; width: 50%;">
                                    <div style="color: white; font-size: 1.05rem; font-weight: 700;">{sb if sb != "-" else pb}</div>
                                    <div style="color: rgba(255,255,255,0.55); font-size: 0.68rem;">SB {sb_note}</div>
                                </td>
                                <td style="padding: 3px 0;">
                                    <div style="color: white; font-size: 1.05rem; font-weight: 700;">{last_perf_str}</div>
                                    <div style="color: rgba(255,255,255,0.55); font-size: 0.68rem;">Last Perf. (Date)</div>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">
                                    <div style="color: white; font-size: 1.05rem; font-weight: 700;">-</div>
                                    <div style="color: rgba(255,255,255,0.55); font-size: 0.68rem;">Asia Rank</div>
                                </td>
                                <td style="padding: 3px 0;">
                                    <div style="color: white; font-size: 1.05rem; font-weight: 700;">{world_rank_str}</div>
                                    <div style="color: rgba(255,255,255,0.55); font-size: 0.68rem;">World Rank</div>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">
                                    <div style="color: white; font-size: 1.05rem; font-weight: 700;">{last3_avg_str}</div>
                                    <div style="color: rgba(255,255,255,0.55); font-size: 0.68rem;">Last 3 Avg.</div>
                                </td>
                                <td style="padding: 3px 0;">
                                    <div style="color: white; font-size: 1.05rem; font-weight: 700;">{perf_count}</div>
                                    <div style="color: rgba(255,255,255,0.55); font-size: 0.68rem;"># Perf. in '{yr}</div>
                                </td>
                            </tr>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Athlete Results Table -->
            <div style="background: rgba(255,255,255,0.08); border-radius: 8px; padding: 0.7rem;
                 border: 1px solid rgba(255,255,255,0.12);">
                <h4 style="color: {GOLD_ACCENT}; margin: 0 0 0.4rem 0; font-size: 0.9rem; font-weight: 600;">
                    {name.split()[-1] if name else ''} Results
                </h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                            <th style="padding: 5px 10px; color: rgba(255,255,255,0.7); font-size: 0.75rem; text-align: left; font-weight: 600;">Perf.</th>
                            <th style="padding: 5px 10px; color: rgba(255,255,255,0.7); font-size: 0.75rem; text-align: left; font-weight: 600;">City</th>
                            <th style="padding: 5px 10px; color: rgba(255,255,255,0.7); font-size: 0.75rem; text-align: left; font-weight: 600;">Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- RIGHT COLUMN: Competitors Table -->
        <div style="flex: 1; overflow-x: auto;">
            <div style="background: rgba(255,255,255,0.08); border-radius: 8px; padding: 0.7rem;
                 border: 1px solid rgba(255,255,255,0.12);">
                <h4 style="color: {GOLD_ACCENT}; margin: 0 0 0.4rem 0; font-size: 0.9rem; font-weight: 600;">
                    Competitors
                </h4>
                <table style="width: 100%; border-collapse: collapse; min-width: 700px;">
                    <thead>
                        <tr style="background: rgba(255,255,255,0.1); border-bottom: 2px solid rgba(255,255,255,0.2);">
                            <th style="padding: 7px 10px; color: rgba(255,255,255,0.9); font-size: 0.75rem; text-align: left; font-weight: 600;">Athlete</th>
                            <th style="padding: 7px 6px; color: rgba(255,255,255,0.9); font-size: 0.75rem; text-align: center; font-weight: 600;">Nat.</th>
                            <th style="padding: 7px 6px; color: rgba(255,255,255,0.9); font-size: 0.75rem; text-align: center; font-weight: 600;">SB</th>
                            <th style="padding: 7px 6px; color: rgba(255,255,255,0.9); font-size: 0.75rem; text-align: center; font-weight: 600;">vs KSA Ath.</th>
                            <th style="padding: 7px 6px; color: rgba(255,255,255,0.9); font-size: 0.75rem; text-align: center; font-weight: 600;">Asia Rank</th>
                            <th style="padding: 7px 6px; color: rgba(255,255,255,0.9); font-size: 0.75rem; text-align: center; font-weight: 600;"># Perf in '{yr}</th>
                            <th style="padding: 7px 6px; color: rgba(255,255,255,0.9); font-size: 0.75rem; text-align: center; font-weight: 600;">Last 3 Avg. Perf.</th>
                            <th style="padding: 7px 6px; color: rgba(255,255,255,0.9); font-size: 0.75rem; text-align: right; font-weight: 600;">Last Perf. (Date)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {competitors_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- ═══ FOOTER ═══ -->
    <div style="margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.1);
         display: flex; justify-content: space-between; align-items: center;">
        <div style="color: rgba(255,255,255,0.4); font-size: 0.72rem;">
            Team Saudi Performance Analysis &mdash; Generated {datetime.now().strftime('%d %b %Y')}
        </div>
        <div style="color: rgba(255,255,255,0.4); font-size: 0.72rem;">
            Data: World Athletics | Season {datetime.now().year}
        </div>
    </div>
</div>{standalone_foot}"""

    return html


# ══════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Pre-Comp Briefing", page_icon="📋", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()
render_page_header(
    "Pre-Competition Briefing",
    "Competition analysis & competitor intelligence",
)

dc = DataConnector()
ksa = dc.get_ksa_athletes()

if ksa.empty:
    st.warning("No KSA athlete data loaded.")
    st.stop()

# ── Selectors ────────────────────────────────────────────────────────

col0, col1, col2, col3, col4 = st.columns([1, 2, 2, 2, 1])

with col0:
    pcr_gender = st.selectbox("Gender", ["Men", "Women"], key="pcr_gender")
    _pcr_g = "M" if pcr_gender == "Men" else "F"

with col1:
    # Filter athletes by selected gender
    if "gender" in ksa.columns:
        _male_codes = {"M", "male", "men", "Male", "Men"}
        _female_codes = {"F", "female", "women", "Female", "Women", "W"}
        if _pcr_g == "M":
            _gender_mask = ksa["gender"].astype(str).isin(_male_codes)
        else:
            _gender_mask = ksa["gender"].astype(str).isin(_female_codes)
        ksa_filtered = ksa[_gender_mask]
    else:
        ksa_filtered = ksa
    athlete_names = sorted(ksa_filtered["full_name"].dropna().unique().tolist())
    if not athlete_names:
        athlete_names = sorted(ksa["full_name"].dropna().unique().tolist())
    selected_athlete = st.selectbox("KSA Athlete", athlete_names, key="pcr_athlete")

with col2:
    # Auto-populate event from athlete's primary event
    athlete_row = ksa[ksa["full_name"] == selected_athlete]
    primary = athlete_row.iloc[0].get("primary_event", "") if not athlete_row.empty else ""
    primary_display = format_event_name(str(primary))

    # Get all events this athlete has PBs for
    athlete_pbs = dc.get_ksa_athlete_pbs(selected_athlete)
    event_options = []
    if primary_display and primary_display != "Overall Ranking":
        event_options.append(primary_display)
    # Build excluded events for this athlete from EXCLUDE_RESULTS
    excluded_events = set()
    for name_key, disc_list in EXCLUDE_RESULTS.items():
        if name_key.upper() in selected_athlete.upper():
            for d in disc_list:
                excluded_events.add(format_event_name(d).lower())

    if athlete_pbs is not None and not athlete_pbs.empty:
        disc_col = "discipline" if "discipline" in athlete_pbs.columns else "event"
        for _, row in athlete_pbs.iterrows():
            e = format_event_name(str(row.get(disc_col, "")))
            if e and e not in event_options and e != "Overall Ranking":
                if e.lower() not in excluded_events:
                    event_options.append(e)
    if not event_options:
        event_options = ["100m"]

    selected_event = st.selectbox("Event", event_options, key="pcr_event")

with col3:
    selected_champ = st.selectbox(
        "Championship", list(CHAMPIONSHIPS.keys()), key="pcr_champ"
    )

with col4:
    max_per_country_options = ["All", "1", "2", "3", "4", "5"]
    max_per_country_sel = st.selectbox(
        "Per Country",
        max_per_country_options,
        index=3,  # Default to 3 (realistic for Asian Games)
        key="pcr_per_country",
    )
    _max_per_country = int(max_per_country_sel) if max_per_country_sel != "All" else None

champ_config = CHAMPIONSHIPS[selected_champ]

# ── Generate Report ──────────────────────────────────────────────────

st.markdown("---")

# Use gender from the dropdown selector
_ath_gender = _pcr_g

with st.spinner("Generating briefing..."):
    athlete_data = gather_athlete_data(dc, selected_athlete, selected_event)
    competitors = gather_competitors(
        dc,
        event_display=selected_event,
        gender=_ath_gender,
        region=champ_config["region"],
        limit=25,
        max_per_country=_max_per_country,
    )

if not athlete_data:
    st.warning(f"Could not load data for {selected_athlete}.")
    st.stop()

# Render inline using st.components.v1.html for reliable rendering
import streamlit.components.v1 as components

logo_b64 = get_logo_b64()
report_html = build_report_html(
    athlete_data, competitors, champ_config["label"], logo_b64, standalone=True,
)
components.html(report_html, height=700, scrolling=True)

# ── Export Buttons ───────────────────────────────────────────────────

st.markdown("")
exp1, exp2, exp3 = st.columns([1, 1, 3])

with exp1:
    standalone_html = build_report_html(
        athlete_data, competitors, champ_config["label"], logo_b64, standalone=True,
    )
    safe_name = selected_athlete.replace(" ", "_").lower()
    st.download_button(
        "Download HTML",
        standalone_html,
        file_name=f"briefing_{safe_name}_{selected_event.replace(' ', '_')}.html",
        mime="text/html",
        key="pcr_dl_html",
    )

with exp2:
    st.markdown(
        '<p style="color: #666; font-size: 0.85rem; padding-top: 0.5rem;">'
        'Print to PDF: Open HTML &rarr; Ctrl+P</p>',
        unsafe_allow_html=True,
    )
