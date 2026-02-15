"""
Pre-Competition Report Generator.
Generates branded PDF and HTML reports for coaches.
"""

import io
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from data.event_utils import get_event_type, format_event_name
from analytics.form_engine import (
    calculate_form_score,
    project_performance,
    detect_trend,
    calculate_advancement_probability,
    calculate_advancement_probability_heuristic,
    RECENCY_WEIGHTS,
)
from analytics.standards import calculate_standards_from_toplist, gap_analysis
from components.report_components import (
    format_mark_display,
    format_gap_display,
    get_status_color,
    get_status_label,
    get_trend_arrow,
    get_trend_color,
    build_standards_rows,
    build_championship_target_rows,
    build_rivals_rows,
    get_championship_targets,
    TEAL_PRIMARY,
    TEAL_DARK,
    GOLD_ACCENT,
    STATUS_DANGER,
    STATUS_NEUTRAL,
)

logger = logging.getLogger(__name__)


class PreCompReportGenerator:
    """Generates pre-competition reports in PDF or HTML format."""

    def __init__(self, dc):
        """Initialize with DataConnector instance.

        Args:
            dc: A DataConnector instance (from data.connector).
        """
        self.dc = dc

    # ── Public API ──────────────────────────────────────────────────────

    def generate(
        self,
        athlete_name: str,
        event: str,
        competition: Optional[str] = None,
        format: str = "pdf",
    ):
        """Generate a pre-competition report.

        Args:
            athlete_name: Full name of the KSA athlete.
            event: Display event name (e.g. '100m').
            competition: Optional competition name for the header.
            format: 'pdf' or 'html'.

        Returns:
            bytes (PDF) or str (HTML).
        """
        data = self._gather_data(athlete_name, event)
        data["competition"] = competition or "Upcoming Competition"
        data["generated_date"] = datetime.now().strftime("%d %B %Y")

        if format == "pdf":
            return self._build_pdf(data)
        return self._build_html(data)

    # ── Data Gathering ──────────────────────────────────────────────────

    def _gather_data(self, athlete_name: str, event: str) -> dict:
        """Collect all data needed for the report."""
        event_type = get_event_type(event)
        lower_is_better = event_type == "time"

        athlete_info = self._get_athlete_info(athlete_name, event)
        pb_mark = self._get_pb(athlete_name, event)
        sb_mark = self._get_sb(athlete_name, event)

        # Recent results
        results = self.dc.get_ksa_results(
            athlete_name=athlete_name, discipline=event, limit=10,
        )
        recent_marks = self._extract_numeric_marks(results, limit=8)

        # Form analysis
        form_score = (
            calculate_form_score(recent_marks, pb=pb_mark, lower_is_better=lower_is_better)
            if recent_marks
            else 50.0
        )

        projection = (
            project_performance(recent_marks, lower_is_better=lower_is_better)
            if recent_marks
            else {"projected_mark": None, "confidence_low": None, "confidence_high": None, "trend": "unknown"}
        )

        trend = detect_trend(recent_marks, lower_is_better) if len(recent_marks) >= 3 else "unknown"

        # Standards from season toplist
        gender = self._infer_gender(athlete_info, event)
        toplist = self.dc.get_season_toplist(event, gender, limit=100)
        standards = self._compute_standards(toplist, lower_is_better)

        # Gap analysis
        athlete_mark = sb_mark or pb_mark
        gaps = {}
        if athlete_mark and standards:
            gaps = gap_analysis(athlete_mark, standards, lower_is_better=lower_is_better)

        # Advancement probabilities
        probabilities = self._compute_probabilities(
            projection, standards, recent_marks, lower_is_better,
            athlete_pb=pb_mark,
        )

        # Rivals
        rivals = self.dc.get_rivals(event=event, gender=gender, limit=10)

        # Championship-specific targets (Asian Games 2026, WC Tokyo 2025, LA 2028)
        event_display = format_event_name(event) if event != format_event_name(event) else event
        championship_targets = get_championship_targets(event_display)
        championship_target_rows = build_championship_target_rows(
            championship_targets, athlete_mark, event_type, lower_is_better,
        )

        return {
            "athlete": athlete_info,
            "event": event,
            "event_display": event_display,
            "event_type": event_type,
            "lower_is_better": lower_is_better,
            "pb": pb_mark,
            "sb": sb_mark,
            "recent_marks": recent_marks,
            "recent_results": results,
            "form_score": form_score,
            "projection": projection,
            "trend": trend,
            "standards": standards,
            "gaps": gaps,
            "probabilities": probabilities,
            "rivals": rivals,
            "championship_targets": championship_targets,
            "championship_target_rows": championship_target_rows,
        }

    def _get_athlete_info(self, athlete_name: str, event: str) -> dict:
        """Look up athlete profile from KSA athletes data."""
        athletes = self.dc.get_ksa_athletes()
        if athletes is not None and not athletes.empty:
            name_col = "full_name" if "full_name" in athletes.columns else "competitor"
            match = athletes[athletes[name_col].str.contains(athlete_name, case=False, na=False)]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "name": row.get("full_name", row.get("competitor", athlete_name)),
                    "country": row.get("country_code", row.get("nat", "KSA")),
                    "world_rank": row.get("best_world_rank", "-"),
                    "ranking_score": row.get("best_ranking_score", row.get("best_score", "-")),
                    "primary_event": row.get("primary_event", event),
                }
        return {
            "name": athlete_name,
            "country": "KSA",
            "world_rank": "-",
            "ranking_score": "-",
            "primary_event": event,
        }

    def _get_pb(self, athlete_name: str, event: str) -> Optional[float]:
        """Get personal best for the given event."""
        pbs = self.dc.get_ksa_athlete_pbs(athlete_name)
        return self._find_mark_for_event(pbs, event)

    def _get_sb(self, athlete_name: str, event: str) -> Optional[float]:
        """Get season best for the given event."""
        sbs = self.dc.get_ksa_athlete_season_bests(athlete_name)
        return self._find_mark_for_event(sbs, event)

    def _find_mark_for_event(self, df: pd.DataFrame, event: str) -> Optional[float]:
        """Find a numeric mark for a given event in a PB/SB dataframe."""
        if df is None or df.empty:
            return None
        disc_col = "discipline" if "discipline" in df.columns else "event"
        mark_col = "mark" if "mark" in df.columns else "result"
        if disc_col not in df.columns or mark_col not in df.columns:
            return None
        event_norm = event.lower().replace(" ", "")
        for _, row in df.iterrows():
            disc = str(row.get(disc_col, "")).lower().replace(" ", "")
            if event_norm in disc:
                val = pd.to_numeric(row.get(mark_col), errors="coerce")
                if not pd.isna(val):
                    return float(val)
        return None

    @staticmethod
    def _extract_numeric_marks(results: pd.DataFrame, limit: int = 8) -> list:
        """Extract numeric marks from a results DataFrame."""
        if results is None or results.empty:
            return []
        mark_col = "mark" if "mark" in results.columns else "result"
        if mark_col not in results.columns:
            return []
        numeric = pd.to_numeric(results[mark_col], errors="coerce").dropna()
        return numeric.tolist()[:limit]

    @staticmethod
    def _infer_gender(athlete_info: dict, event: str) -> str:
        """Infer gender from athlete info or event name."""
        primary = athlete_info.get("primary_event", "")
        if "Women" in primary or "women" in primary:
            return "F"
        return "M"

    @staticmethod
    def _compute_standards(toplist: pd.DataFrame, lower_is_better: bool) -> dict:
        """Calculate standards from toplist, handling empty data."""
        if toplist is None or toplist.empty:
            return {}
        mark_col = "mark" if "mark" in toplist.columns else "result"
        if mark_col not in toplist.columns:
            mark_col = "result_numeric" if "result_numeric" in toplist.columns else None
        if mark_col is None:
            return {}
        return calculate_standards_from_toplist(
            toplist, mark_col=mark_col, lower_is_better=lower_is_better,
        )

    @staticmethod
    def _compute_probabilities(
        projection: dict,
        standards: dict,
        recent_marks: list,
        lower_is_better: bool,
        athlete_pb: float = None,
    ) -> dict:
        """Calculate advancement probabilities for each round.

        Uses scipy normal CDF when projected marks are close to standards.
        Falls back to heuristic PB-based estimate when all scipy probs are ~0%.
        """
        probabilities = {}
        proj_mark = projection.get("projected_mark")
        if not standards:
            return probabilities

        # Try scipy-based probability using projection
        if proj_mark:
            std_dev = float(np.std(recent_marks)) if recent_marks else 0.1
            if std_dev <= 0:
                std_dev = 0.1

            for level, mark in standards.items():
                if mark is None:
                    continue
                try:
                    prob = calculate_advancement_probability(
                        proj_mark, mark, std_dev, lower_is_better,
                    )
                    probabilities[level] = round(prob * 100, 1)
                except Exception as e:
                    logger.debug("Probability calc failed for %s: %s", level, e)
                    probabilities[level] = None

        # If all probabilities are 0% or None, use PB-based heuristic
        real_probs = [v for v in probabilities.values() if v is not None and v > 0]
        if not real_probs and athlete_pb:
            for level, mark in standards.items():
                if mark is None:
                    continue
                try:
                    prob = calculate_advancement_probability_heuristic(
                        athlete_pb, mark, lower_is_better,
                    )
                    probabilities[level] = round(prob * 100, 1)
                except Exception as e:
                    logger.debug("Heuristic prob failed for %s: %s", level, e)
                    probabilities[level] = 0.0

        # Ensure no None values (replace with 0.0)
        for level in list(probabilities.keys()):
            if probabilities[level] is None:
                probabilities[level] = 0.0

        return probabilities

    # ── PDF Builder ─────────────────────────────────────────────────────

    def _build_pdf(self, data: dict) -> bytes:
        """Build a branded PDF report using ReportLab Platypus."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, white, black
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm,
        )

        # Colors
        teal = HexColor(TEAL_PRIMARY)
        teal_dark = HexColor(TEAL_DARK)
        gold = HexColor(GOLD_ACCENT)
        danger = HexColor(STATUS_DANGER)
        neutral = HexColor(STATUS_NEUTRAL)

        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            "HeaderWhite", parent=styles["Title"],
            textColor=white, fontSize=18, alignment=TA_CENTER, spaceAfter=2,
        ))
        styles.add(ParagraphStyle(
            "SubHeaderWhite", parent=styles["Normal"],
            textColor=white, fontSize=11, alignment=TA_CENTER, spaceAfter=0,
        ))
        styles.add(ParagraphStyle(
            "SectionTitle", parent=styles["Heading2"],
            textColor=teal, fontSize=13, spaceBefore=10, spaceAfter=4,
            borderWidth=0, borderPadding=0,
        ))
        styles.add(ParagraphStyle(
            "CellText", parent=styles["Normal"], fontSize=9, leading=12,
        ))
        styles.add(ParagraphStyle(
            "CellTextCenter", parent=styles["Normal"],
            fontSize=9, leading=12, alignment=TA_CENTER,
        ))
        styles.add(ParagraphStyle(
            "FooterText", parent=styles["Normal"],
            textColor=neutral, fontSize=8, alignment=TA_CENTER,
        ))

        elements = []

        # ── 1. Header ──
        athlete = data["athlete"]
        header_data = [[
            Paragraph("Team Saudi Performance Analysis", styles["HeaderWhite"]),
        ], [
            Paragraph(
                f"{athlete['name']}  |  {data['event_display']}  |  {data['competition']}",
                styles["SubHeaderWhite"],
            ),
        ], [
            Paragraph(f"Generated {data['generated_date']}", styles["SubHeaderWhite"]),
        ]]
        header_table = Table(header_data, colWidths=[doc.width])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), teal),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 8 * mm))

        # ── 2. Athlete Snapshot ──
        elements.append(Paragraph("Athlete Snapshot", styles["SectionTitle"]))

        et = data["event_type"]
        pb_str = format_mark_display(data["pb"], et) if data["pb"] else "-"
        sb_str = format_mark_display(data["sb"], et) if data["sb"] else "-"
        wr_str = str(athlete.get("world_rank", "-"))
        rs_str = str(athlete.get("ranking_score", "-"))
        form_str = f"{data['form_score']:.0f} / 100"

        snapshot_data = [
            ["Personal Best", "Season Best", "World Rank", "Ranking Score", "Form Score"],
            [pb_str, sb_str, wr_str, rs_str, form_str],
        ]
        snapshot_table = Table(snapshot_data, colWidths=[doc.width / 5] * 5)
        snapshot_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), teal),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, 1), 11),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(snapshot_table)
        elements.append(Spacer(1, 6 * mm))

        # ── 3. Championship Standards ──
        if data["gaps"]:
            elements.append(Paragraph("Championship Standards", styles["SectionTitle"]))

            std_header = ["Level", "Standard", "Athlete Mark", "Gap", "Status"]
            std_rows = build_standards_rows(
                data["gaps"], data["sb"] or data["pb"] or 0, et, data["lower_is_better"],
            )

            table_data = [std_header]
            row_colors = []
            for r in std_rows:
                table_data.append([r["Level"], r["Standard"], r["Athlete"], r["Gap"], r["Status"]])
                row_colors.append(HexColor(r["Color"]))

            std_table = Table(table_data, colWidths=[doc.width * f for f in [0.18, 0.22, 0.22, 0.20, 0.18]])
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            for i, color in enumerate(row_colors):
                style_cmds.append(("TEXTCOLOR", (4, i + 1), (4, i + 1), color))
                style_cmds.append(("FONTNAME", (4, i + 1), (4, i + 1), "Helvetica-Bold"))

            std_table.setStyle(TableStyle(style_cmds))
            elements.append(std_table)
            elements.append(Spacer(1, 6 * mm))

        # ── 3b. Championship Targets (Asian Games, WC, Olympics) ──
        champ_rows = data.get("championship_target_rows", [])
        if champ_rows:
            elements.append(Paragraph("Championship Targets", styles["SectionTitle"]))

            ct_header = ["Championship", "Level", "Target", "Athlete", "Gap", "Status"]
            ct_data = [ct_header]
            ct_colors = []
            for r in champ_rows:
                ct_data.append([
                    r["Championship"].split("(")[0].strip(),  # Shorten name
                    r["Level"],
                    r["Target"],
                    r["Athlete"],
                    r["Gap"],
                    r["Status"],
                ])
                ct_colors.append(HexColor(r["Color"]))

            ct_table = Table(
                ct_data,
                colWidths=[doc.width * f for f in [0.22, 0.12, 0.16, 0.16, 0.18, 0.16]],
            )
            ct_style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f8f9fa")]),
            ]
            for i, color in enumerate(ct_colors):
                ct_style_cmds.append(("TEXTCOLOR", (5, i + 1), (5, i + 1), color))
                ct_style_cmds.append(("FONTNAME", (5, i + 1), (5, i + 1), "Helvetica-Bold"))

            ct_table.setStyle(TableStyle(ct_style_cmds))
            elements.append(ct_table)
            elements.append(Spacer(1, 6 * mm))

        # ── 4. Form & Projection ──
        proj = data["projection"]
        if proj.get("projected_mark") is not None:
            elements.append(Paragraph("Form & Projection", styles["SectionTitle"]))

            trend = data["trend"]
            arrow = get_trend_arrow(trend)
            proj_mark_str = format_mark_display(proj["projected_mark"], et)
            conf_low_str = format_mark_display(proj["confidence_low"], et) if proj["confidence_low"] else "-"
            conf_high_str = format_mark_display(proj["confidence_high"], et) if proj["confidence_high"] else "-"
            conf_range = f"{conf_low_str} - {conf_high_str}"
            trend_str = f"{arrow} {trend.title()}"

            proj_data = [
                ["Projected Mark", "Confidence Range", "Trend"],
                [proj_mark_str, conf_range, trend_str],
            ]
            proj_table = Table(proj_data, colWidths=[doc.width / 3] * 3)
            proj_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, 1), 11),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(proj_table)
            elements.append(Spacer(1, 6 * mm))

        # ── 5. Recent Results ──
        results_df = data["recent_results"]
        if results_df is not None and not results_df.empty:
            elements.append(Paragraph("Recent Results", styles["SectionTitle"]))

            date_col = "date" if "date" in results_df.columns else None
            venue_col = "venue" if "venue" in results_df.columns else "competition" if "competition" in results_df.columns else None
            mark_col = "mark" if "mark" in results_df.columns else "result"
            place_col = "place" if "place" in results_df.columns else "pos" if "pos" in results_df.columns else None

            res_header = ["Date", "Venue", "Mark", "Place"]
            res_data = [res_header]

            for _, row in results_df.head(8).iterrows():
                d = str(row.get(date_col, "-")) if date_col else "-"
                v = str(row.get(venue_col, "-")) if venue_col else "-"
                # Truncate long venue names
                if len(v) > 30:
                    v = v[:27] + "..."
                m = str(row.get(mark_col, "-"))
                p = str(row.get(place_col, "-")) if place_col else "-"
                res_data.append([d, v, m, p])

            res_table = Table(
                res_data,
                colWidths=[doc.width * f for f in [0.20, 0.40, 0.20, 0.20]],
            )
            res_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f8f9fa")]),
            ]))
            elements.append(res_table)
            elements.append(Spacer(1, 6 * mm))

        # ── 6. Advancement Probability ──
        if data["probabilities"]:
            elements.append(Paragraph("Advancement Probability", styles["SectionTitle"]))

            # Build a simple text-based probability display
            prob_rows = [["Round", "Probability", "Assessment"]]
            for level in ["Heat", "Semi-Final", "Final", "Medal"]:
                prob = data["probabilities"].get(level)
                if prob is None:
                    continue
                if prob >= 70:
                    assessment = "Strong"
                elif prob >= 40:
                    assessment = "Possible"
                else:
                    assessment = "Unlikely"
                prob_rows.append([level, f"{prob:.0f}%", assessment])

            if len(prob_rows) > 1:
                prob_table = Table(
                    prob_rows,
                    colWidths=[doc.width * f for f in [0.30, 0.35, 0.35]],
                )

                prob_style_cmds = [
                    ("BACKGROUND", (0, 0), (-1, 0), teal),
                    ("TEXTCOLOR", (0, 0), (-1, 0), white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]

                # Color-code probability values
                for i in range(1, len(prob_rows)):
                    raw_prob = data["probabilities"].get(prob_rows[i][0])
                    if raw_prob is not None:
                        if raw_prob >= 70:
                            c = teal
                        elif raw_prob >= 40:
                            c = gold
                        else:
                            c = danger
                        prob_style_cmds.append(("TEXTCOLOR", (1, i), (2, i), c))
                        prob_style_cmds.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))

                prob_table.setStyle(TableStyle(prob_style_cmds))
                elements.append(prob_table)
                elements.append(Spacer(1, 6 * mm))

        # ── 7. Rival Threat Summary ──
        rival_rows = build_rivals_rows(data["rivals"])
        if rival_rows:
            elements.append(Paragraph("Rival Threat Summary", styles["SectionTitle"]))

            rival_header = ["Name", "Country", "World Rank", "Score"]
            rival_data = [rival_header]
            for r in rival_rows:
                rival_data.append([
                    str(r["Name"]), str(r["Country"]),
                    str(r["World Rank"]), str(r["Score"]),
                ])

            rival_table = Table(
                rival_data,
                colWidths=[doc.width * f for f in [0.35, 0.15, 0.25, 0.25]],
            )
            rival_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f8f9fa")]),
            ]))
            elements.append(rival_table)
            elements.append(Spacer(1, 6 * mm))

        # ── 8. Footer ──
        elements.append(HRFlowable(
            width="100%", thickness=1, color=HexColor("#dddddd"), spaceAfter=4,
        ))
        elements.append(Paragraph(
            f"Team Saudi Performance Analysis  |  Generated {data['generated_date']}",
            styles["FooterText"],
        ))

        doc.build(elements)
        return buf.getvalue()

    # ── HTML Builder ────────────────────────────────────────────────────

    def _build_html(self, data: dict) -> str:
        """Build an inline-styled HTML report for st.markdown() or file export."""
        athlete = data["athlete"]
        et = data["event_type"]
        proj = data["projection"]

        pb_str = format_mark_display(data["pb"], et) if data["pb"] else "-"
        sb_str = format_mark_display(data["sb"], et) if data["sb"] else "-"
        form_str = f"{data['form_score']:.0f}"

        html_parts = []

        # ── Base styles ──
        html_parts.append("""
<style>
  .report-container { font-family: Inter, sans-serif; color: #333; max-width: 900px; margin: 0 auto; }
  .report-container table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }
  .report-container th { background: #007167; color: white; padding: 8px 12px; font-size: 0.85rem; text-align: center; }
  .report-container td { padding: 8px 12px; text-align: center; border-bottom: 1px solid #eee; font-size: 0.85rem; }
  .report-container tr:nth-child(even) { background: #f8f9fa; }
</style>
""")

        # ── Header ──
        html_parts.append(f"""
<div class="report-container">
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
     padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 1.6rem;">Team Saudi Performance Analysis</h1>
  <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1rem;">
    {athlete['name']}  &middot;  {data['event_display']}  &middot;  {data['competition']}
  </p>
  <p style="color: {GOLD_ACCENT}; margin: 0.25rem 0 0 0; font-size: 0.85rem;">
    Generated {data['generated_date']}
  </p>
  <div style="height: 3px; background: linear-gradient(90deg, {TEAL_PRIMARY} 0%, {GOLD_ACCENT} 50%, {TEAL_DARK} 100%);
       margin-top: 1rem; border-radius: 2px;"></div>
</div>
""")

        # ── Athlete Snapshot ──
        html_parts.append(f"""
<h3 style="color: {TEAL_PRIMARY}; border-bottom: 2px solid {GOLD_ACCENT}; padding-bottom: 4px;">
  Athlete Snapshot
</h3>
<div style="display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap;">
  {self._html_metric_card("Personal Best", pb_str)}
  {self._html_metric_card("Season Best", sb_str)}
  {self._html_metric_card("World Rank", str(athlete.get("world_rank", "-")))}
  {self._html_metric_card("Ranking Score", str(athlete.get("ranking_score", "-")))}
  {self._html_metric_card("Form Score", f"{form_str}/100", highlight=True)}
</div>
""")

        # ── Championship Standards ──
        if data["gaps"]:
            std_rows = build_standards_rows(
                data["gaps"], data["sb"] or data["pb"] or 0, et, data["lower_is_better"],
            )
            html_parts.append(f"""
<h3 style="color: {TEAL_PRIMARY}; border-bottom: 2px solid {GOLD_ACCENT}; padding-bottom: 4px;">
  Championship Standards
</h3>
<table>
  <thead><tr><th>Level</th><th>Standard</th><th>Athlete Mark</th><th>Gap</th><th>Status</th></tr></thead>
  <tbody>
""")
            for r in std_rows:
                badge_color = r["Color"]
                html_parts.append(f"""
    <tr>
      <td style="font-weight: 600;">{r['Level']}</td>
      <td>{r['Standard']}</td>
      <td>{r['Athlete']}</td>
      <td>{r['Gap']}</td>
      <td><span style="background: {badge_color}; color: white; padding: 2px 8px;
           border-radius: 4px; font-size: 0.8rem; font-weight: 600;">{r['Status']}</span></td>
    </tr>
""")
            html_parts.append("  </tbody>\n</table>\n")

        # ── Championship Targets ──
        champ_rows = data.get("championship_target_rows", [])
        if champ_rows:
            html_parts.append(f"""
<h3 style="color: {TEAL_PRIMARY}; border-bottom: 2px solid {GOLD_ACCENT}; padding-bottom: 4px;">
  Championship Targets
</h3>
<table>
  <thead><tr><th style="text-align:left;">Championship</th><th>Level</th><th>Target</th><th>Athlete</th><th>Gap</th><th>Status</th></tr></thead>
  <tbody>
""")
            for r in champ_rows:
                badge_color = r["Color"]
                html_parts.append(f"""
    <tr>
      <td style="text-align:left; font-weight: 500;">{r['Championship']}</td>
      <td>{r['Level']}</td>
      <td>{r['Target']}</td>
      <td>{r['Athlete']}</td>
      <td>{r['Gap']}</td>
      <td><span style="background: {badge_color}; color: white; padding: 2px 8px;
           border-radius: 4px; font-size: 0.8rem; font-weight: 600;">{r['Status']}</span></td>
    </tr>
""")
            html_parts.append("  </tbody>\n</table>\n")

        # ── Form & Projection ──
        if proj.get("projected_mark") is not None:
            trend = data["trend"]
            arrow = get_trend_arrow(trend)
            trend_color = get_trend_color(trend)
            proj_mark_str = format_mark_display(proj["projected_mark"], et)
            conf_low = format_mark_display(proj["confidence_low"], et) if proj["confidence_low"] else "-"
            conf_high = format_mark_display(proj["confidence_high"], et) if proj["confidence_high"] else "-"

            html_parts.append(f"""
<h3 style="color: {TEAL_PRIMARY}; border-bottom: 2px solid {GOLD_ACCENT}; padding-bottom: 4px;">
  Form & Projection
</h3>
<div style="display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap;">
  {self._html_metric_card("Projected Mark", proj_mark_str)}
  {self._html_metric_card("Confidence Range", f"{conf_low} - {conf_high}")}
  <div style="flex: 1; min-width: 140px; background: {trend_color}; padding: 0.75rem;
       border-radius: 8px; text-align: center;">
    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem;">Trend</p>
    <p style="color: white; margin: 4px 0 0 0; font-size: 1.3rem; font-weight: bold;">
      {arrow} {trend.title()}
    </p>
  </div>
</div>
""")

        # ── Recent Results ──
        results_df = data["recent_results"]
        if results_df is not None and not results_df.empty:
            date_col = "date" if "date" in results_df.columns else None
            venue_col = "venue" if "venue" in results_df.columns else "competition" if "competition" in results_df.columns else None
            mark_col = "mark" if "mark" in results_df.columns else "result"
            place_col = "place" if "place" in results_df.columns else "pos" if "pos" in results_df.columns else None

            html_parts.append(f"""
<h3 style="color: {TEAL_PRIMARY}; border-bottom: 2px solid {GOLD_ACCENT}; padding-bottom: 4px;">
  Recent Results
</h3>
<table>
  <thead><tr><th>Date</th><th>Venue</th><th>Mark</th><th>Place</th></tr></thead>
  <tbody>
""")
            for _, row in results_df.head(8).iterrows():
                d = str(row.get(date_col, "-")) if date_col else "-"
                v = str(row.get(venue_col, "-")) if venue_col else "-"
                m = str(row.get(mark_col, "-"))
                p = str(row.get(place_col, "-")) if place_col else "-"
                html_parts.append(f"    <tr><td>{d}</td><td style='text-align:left;'>{v}</td><td>{m}</td><td>{p}</td></tr>\n")
            html_parts.append("  </tbody>\n</table>\n")

        # ── Advancement Probability ──
        if data["probabilities"]:
            html_parts.append(f"""
<h3 style="color: {TEAL_PRIMARY}; border-bottom: 2px solid {GOLD_ACCENT}; padding-bottom: 4px;">
  Advancement Probability
</h3>
<div style="margin-bottom: 1.5rem;">
""")
            for level in ["Heat", "Semi-Final", "Final", "Medal"]:
                prob = data["probabilities"].get(level)
                if prob is None:
                    continue
                bar_color = TEAL_PRIMARY if prob >= 70 else (GOLD_ACCENT if prob >= 40 else STATUS_DANGER)
                clamped = max(0, min(prob, 100))
                html_parts.append(f"""
  <div style="margin-bottom: 8px;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
      <span style="font-weight: 600; font-size: 0.85rem;">{level}</span>
      <span style="font-weight: 700; font-size: 0.85rem; color: {bar_color};">{prob:.0f}%</span>
    </div>
    <div style="background: #e9ecef; border-radius: 4px; height: 18px; overflow: hidden;">
      <div style="width: {clamped:.0f}%; height: 100%; background: {bar_color};
           border-radius: 4px; transition: width 0.5s;"></div>
    </div>
  </div>
""")
            html_parts.append("</div>\n")

        # ── Rival Threat Summary ──
        rival_rows = build_rivals_rows(data["rivals"])
        if rival_rows:
            html_parts.append(f"""
<h3 style="color: {TEAL_PRIMARY}; border-bottom: 2px solid {GOLD_ACCENT}; padding-bottom: 4px;">
  Rival Threat Summary
</h3>
<table>
  <thead><tr><th style="text-align:left;">Name</th><th>Country</th><th>World Rank</th><th>Score</th></tr></thead>
  <tbody>
""")
            for r in rival_rows:
                html_parts.append(
                    f"    <tr><td style='text-align:left;'>{r['Name']}</td>"
                    f"<td>{r['Country']}</td><td>{r['World Rank']}</td><td>{r['Score']}</td></tr>\n"
                )
            html_parts.append("  </tbody>\n</table>\n")

        # ── Footer ──
        html_parts.append(f"""
<div style="border-top: 1px solid #ddd; padding-top: 0.75rem; margin-top: 2rem; text-align: center;">
  <p style="color: {STATUS_NEUTRAL}; font-size: 0.8rem; margin: 0;">
    Team Saudi Performance Analysis  &middot;  Generated {data['generated_date']}
  </p>
</div>
</div>
""")

        return "".join(html_parts)

    # ── HTML Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _html_metric_card(label: str, value: str, highlight: bool = False) -> str:
        """Render an inline-styled metric card."""
        bg = GOLD_ACCENT if highlight else TEAL_PRIMARY
        return (
            f'<div style="flex: 1; min-width: 140px; background: {bg}; padding: 0.75rem;'
            f' border-radius: 8px; text-align: center;">'
            f'<p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem;">{label}</p>'
            f'<p style="color: white; margin: 4px 0 0 0; font-size: 1.3rem; font-weight: bold;">{value}</p>'
            f'</div>'
        )
