"""
Data connector for World Athletics v2.

Provides DuckDB-powered queries over Parquet files with Azure/local dual-mode.
Refactored from data_connector.py with typed returns and v2 schema support.

Usage:
    from data.connector import DataConnector
    dc = DataConnector()
    athletes = dc.get_ksa_athletes()
    rankings = dc.get_world_rankings("100m", "M")
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

import duckdb
import pandas as pd
from dotenv import load_dotenv

# Load .env from project root (world_athletics/)
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logger = logging.getLogger(__name__)

# Data directories
LOCAL_DATA_DIR = Path(__file__).parent / "scraped"
LEGACY_DATA_DIR = Path(__file__).parent.parent.parent / "Data" / "parquet"

CACHE_TTL = 3600  # 1 hour


def _normalize_gender_for_legacy(gender: str) -> str:
    """Map gender codes to legacy master format ('men'/'women')."""
    g = gender.strip().lower()
    if g in ("m", "male", "men"):
        return "men"
    if g in ("f", "w", "female", "women"):
        return "women"
    return g


def _event_to_db_format(event: str) -> str:
    """Convert display event name to DB format before normalization.

    Pages pass display names like '100m', but legacy data uses '100-metres'.
    This converts display -> db first so normalization matches correctly.
    """
    from data.event_utils import display_to_db
    db_name = display_to_db(event)
    # If display_to_db returns something different, use it; else keep original
    return db_name if db_name != event else event


class DataConnector:
    """DuckDB-powered data access with Azure/local dual-mode."""

    def __init__(self, force_local: bool = False):
        self._conn = duckdb.connect(":memory:")
        self._force_local = force_local or os.environ.get("FORCE_LOCAL_MODE", "").lower() == "true"
        self._data_mode = "local"  # Will be set to "azure" if Azure available
        self._views_registered = set()

        # Try Azure, fall back to local
        if not self._force_local:
            self._try_azure_setup()

        self._register_views()

    def _try_azure_setup(self):
        """Try to set up Azure Blob Storage access."""
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            try:
                import streamlit as st
                conn_str = st.secrets.get("AZURE_STORAGE_CONNECTION_STRING")
            except Exception:
                pass

        if conn_str:
            self._data_mode = "azure"
            self._azure_conn_str = conn_str
            logger.info("Data mode: Azure Blob Storage")
        else:
            logger.info("Data mode: Local files")

    def _register_views(self):
        """Register DuckDB views for available parquet files."""
        # v2 scraped data
        v2_files = {
            "ksa_athletes": "ksa_athletes.parquet",
            "ksa_pbs": "ksa_personal_bests.parquet",
            "ksa_results": "ksa_results.parquet",
            "world_rankings": "world_rankings.parquet",
            "mens_rankings": "mens_rankings.parquet",
            "top_rankings": "top_rankings.parquet",
            "season_toplists": "season_toplists.parquet",
            "calendar": "calendar.parquet",
            "upcoming": "upcoming.parquet",
            "recent_results": "recent_results.parquet",
            "rivals": "rivals.parquet",
            "qualifications": "qualifications.parquet",
        }

        for view_name, filename in v2_files.items():
            path = LOCAL_DATA_DIR / filename
            if path.exists():
                try:
                    self._conn.execute(
                        f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{path}')"
                    )
                    self._views_registered.add(view_name)
                except Exception as e:
                    logger.warning(f"Failed to register view {view_name}: {e}")

        # Legacy data (master.parquet from v1)
        legacy_master = LEGACY_DATA_DIR / "master.parquet"
        if legacy_master.exists():
            try:
                self._conn.execute(
                    f"CREATE OR REPLACE VIEW master AS SELECT * FROM read_parquet('{legacy_master}')"
                )
                self._views_registered.add("master")
            except Exception as e:
                logger.warning(f"Failed to register legacy master: {e}")

        legacy_benchmarks = LEGACY_DATA_DIR / "benchmarks.parquet"
        if legacy_benchmarks.exists():
            try:
                self._conn.execute(
                    f"CREATE OR REPLACE VIEW benchmarks AS SELECT * FROM read_parquet('{legacy_benchmarks}')"
                )
                self._views_registered.add("benchmarks")
            except Exception as e:
                logger.warning(f"Failed to register legacy benchmarks: {e}")

        legacy_ksa_profiles = LEGACY_DATA_DIR / "ksa_profiles.parquet"
        if legacy_ksa_profiles.exists():
            try:
                self._conn.execute(
                    f"CREATE OR REPLACE VIEW ksa_profiles AS SELECT * FROM read_parquet('{legacy_ksa_profiles}')"
                )
                self._views_registered.add("ksa_profiles")
            except Exception as e:
                logger.warning(f"Failed to register legacy ksa_profiles: {e}")

        legacy_road_to_tokyo = LEGACY_DATA_DIR / "road_to_tokyo.parquet"
        if legacy_road_to_tokyo.exists():
            try:
                self._conn.execute(
                    f"CREATE OR REPLACE VIEW road_to_tokyo AS SELECT * FROM read_parquet('{legacy_road_to_tokyo}')"
                )
                self._views_registered.add("road_to_tokyo")
            except Exception as e:
                logger.warning(f"Failed to register legacy road_to_tokyo: {e}")

        logger.info(f"Registered views: {', '.join(sorted(self._views_registered))}")

    def query(self, sql: str) -> pd.DataFrame:
        """Execute raw SQL query and return DataFrame."""
        try:
            return self._conn.execute(sql).fetchdf()
        except Exception as e:
            logger.error(f"Query failed: {e}\nSQL: {sql[:200]}")
            return pd.DataFrame()

    @property
    def available_views(self) -> set:
        return self._views_registered.copy()

    # ── KSA Athletes ──────────────────────────────────────────────────

    def get_ksa_athletes(self, gender: Optional[str] = None) -> pd.DataFrame:
        """Get all KSA athlete profiles. Falls back to legacy ksa_profiles."""
        # Try v2 data first
        if "ksa_athletes" in self._views_registered:
            sql = "SELECT * FROM ksa_athletes"
            if gender:
                sql += f" WHERE gender = '{gender.lower()}'"
            sql += " ORDER BY best_ranking_score DESC NULLS LAST"
            result = self.query(sql)
            if not result.empty:
                return result

        # Fall back to legacy ksa_profiles
        if "ksa_profiles" in self._views_registered:
            sql = "SELECT * FROM ksa_profiles"
            if gender:
                g = _normalize_gender_for_legacy(gender)
                sql += f" WHERE LOWER(gender) = '{g}'"
            sql += " ORDER BY best_world_rank ASC NULLS LAST"
            return self.query(sql)

        return pd.DataFrame()

    def get_ksa_athlete_pbs(self, athlete_name: Optional[str] = None) -> pd.DataFrame:
        """Get personal bests for KSA athletes. Falls back to master legacy data."""
        # Try v2 data first
        if "ksa_pbs" in self._views_registered:
            sql = "SELECT * FROM ksa_pbs"
            if athlete_name:
                safe_name = athlete_name.replace("'", "''")
                sql += f" WHERE full_name ILIKE '%{safe_name}%'"
            sql += " ORDER BY result_score DESC NULLS LAST"
            result = self.query(sql)
            if not result.empty:
                return result

        # Fall back to legacy master: best result per event per KSA competitor
        if "master" in self._views_registered:
            conditions = ["nat = 'KSA'", "result_numeric IS NOT NULL"]
            if athlete_name:
                safe_name = athlete_name.replace("'", "''")
                conditions.append(f"competitor ILIKE '%{safe_name}%'")

            # Use a subquery to get best (min for time events is tricky, so get
            # both min and max result_numeric per event and let the caller decide).
            # For simplicity, get the row with the best resultscore per event.
            sql = f"""
                WITH ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY competitor, event
                            ORDER BY resultscore DESC NULLS LAST
                        ) AS rn
                    FROM master
                    WHERE {' AND '.join(conditions)}
                )
                SELECT competitor, event, result, result_numeric, resultscore,
                       wind, venue, date, gender, rank
                FROM ranked
                WHERE rn = 1
                ORDER BY resultscore DESC NULLS LAST
            """
            return self.query(sql)

        return pd.DataFrame()

    def get_ksa_athlete_season_bests(
        self,
        athlete_name: Optional[str] = None,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get season bests per event for a KSA athlete.

        Returns best mark per event for the given season (default: latest year).
        """
        if "master" not in self._views_registered:
            return pd.DataFrame()

        from data.event_utils import get_event_type
        import datetime

        if season is None:
            season = datetime.datetime.now().year

        conditions = ["nat = 'KSA'", "result_numeric IS NOT NULL", f"year = {season}"]
        if athlete_name:
            safe_name = athlete_name.replace("'", "''")
            conditions.append(f"competitor ILIKE '%{safe_name}%'")

        sql = f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY competitor, event
                        ORDER BY resultscore DESC NULLS LAST
                    ) AS rn
                FROM master
                WHERE {' AND '.join(conditions)}
            )
            SELECT competitor, event, result, result_numeric, resultscore,
                   wind, venue, date, gender, rank
            FROM ranked
            WHERE rn = 1
            ORDER BY resultscore DESC NULLS LAST
        """
        return self.query(sql)

    def get_ksa_athlete_top5_avg(
        self,
        athlete_name: str,
        discipline: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get average of top 5 performances per event for a KSA athlete.

        Returns one row per event with: event, top5_avg, top5_best, top5_worst, n_performances.
        """
        if "master" not in self._views_registered:
            return pd.DataFrame()

        conditions = ["nat = 'KSA'", "result_numeric IS NOT NULL"]
        safe_name = athlete_name.replace("'", "''")
        conditions.append(f"competitor ILIKE '%{safe_name}%'")

        if discipline:
            norm_disc = re.sub(r'[^0-9a-z]', '', _event_to_db_format(discipline).lower())
            conditions.append(
                f"regexp_replace(LOWER(event), '[^0-9a-z]', '', 'g') = '{norm_disc}'"
            )

        # Get top 5 by resultscore per event
        sql = f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY event
                        ORDER BY resultscore DESC NULLS LAST
                    ) AS rn
                FROM master
                WHERE {' AND '.join(conditions)}
            ),
            top5 AS (
                SELECT * FROM ranked WHERE rn <= 5
            )
            SELECT
                event,
                ROUND(AVG(result_numeric), 3) AS top5_avg,
                MIN(result_numeric) AS top5_best,
                MAX(result_numeric) AS top5_worst,
                COUNT(*) AS n_performances,
                ROUND(AVG(resultscore), 0) AS avg_wa_points
            FROM top5
            GROUP BY event
            ORDER BY avg_wa_points DESC NULLS LAST
        """
        return self.query(sql)

    def get_ksa_athlete_year_progression(
        self,
        athlete_name: str,
        discipline: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get year-by-year best marks for a KSA athlete.

        Returns: year, event, best_mark, best_numeric, best_score, n_comps.
        """
        if "master" not in self._views_registered:
            return pd.DataFrame()

        conditions = ["nat = 'KSA'", "result_numeric IS NOT NULL"]
        safe_name = athlete_name.replace("'", "''")
        conditions.append(f"competitor ILIKE '%{safe_name}%'")

        if discipline:
            norm_disc = re.sub(r'[^0-9a-z]', '', _event_to_db_format(discipline).lower())
            conditions.append(
                f"regexp_replace(LOWER(event), '[^0-9a-z]', '', 'g') = '{norm_disc}'"
            )

        sql = f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY event, year
                        ORDER BY resultscore DESC NULLS LAST
                    ) AS rn
                FROM master
                WHERE {' AND '.join(conditions)}
            )
            SELECT year, event, result AS best_mark, result_numeric AS best_numeric,
                   resultscore AS best_score, venue, date
            FROM ranked
            WHERE rn = 1
            ORDER BY event, year DESC
        """
        # Also get competition counts per year
        count_sql = f"""
            SELECT year, event, COUNT(*) AS n_comps
            FROM master
            WHERE {' AND '.join(conditions)}
            GROUP BY year, event
        """

        main = self.query(sql)
        counts = self.query(count_sql)

        if not main.empty and not counts.empty:
            return main.merge(counts, on=["year", "event"], how="left")
        return main

    def get_ksa_results(
        self,
        athlete_name: Optional[str] = None,
        discipline: Optional[str] = None,
        age_category: Optional[str] = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Get competition results for KSA athletes. Falls back to master legacy data."""
        # Try v2 data first
        if "ksa_results" in self._views_registered:
            conditions = []
            if athlete_name:
                safe_name = athlete_name.replace("'", "''")
                conditions.append(f"full_name ILIKE '%{safe_name}%'")
            if discipline:
                safe_disc = discipline.replace("'", "''")
                conditions.append(f"discipline ILIKE '%{safe_disc}%'")

            sql = "SELECT * FROM ksa_results"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += f" ORDER BY TRY_STRPTIME(date, '%d %b %Y') DESC NULLS LAST LIMIT {limit}"
            result = self.query(sql)
            if not result.empty:
                return result

        # Fall back to legacy master filtered by nat='KSA'
        if "master" in self._views_registered:
            conditions = ["nat = 'KSA'"]
            if athlete_name:
                safe_name = athlete_name.replace("'", "''")
                conditions.append(f"competitor ILIKE '%{safe_name}%'")
            if discipline:
                # Convert display name to db format, then normalize for matching
                norm_disc = re.sub(r'[^0-9a-z]', '', _event_to_db_format(discipline).lower())
                conditions.append(
                    f"regexp_replace(LOWER(event), '[^0-9a-z]', '', 'g') = '{norm_disc}'"
                )

            # Age category filter (only legacy master has 'age' column)
            if age_category:
                if age_category == "U20":
                    conditions.append("age < 20")
                elif age_category == "U23":
                    conditions.append("age < 23")
                elif age_category == "Senior":
                    conditions.append("age >= 23")

            sql = f"SELECT * FROM master WHERE {' AND '.join(conditions)}"
            sql += f" ORDER BY date DESC LIMIT {limit}"
            return self.query(sql)

        return pd.DataFrame()

    # ── World Rankings ────────────────────────────────────────────────

    def get_world_rankings(
        self,
        event: Optional[str] = None,
        gender: Optional[str] = None,
        country: Optional[str] = None,
        age_category: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get world rankings with filters. Falls back to master legacy data."""
        # For Men: check mens_rankings first (composite from profiles + top rankings)
        if gender and gender.upper() == "M" and "mens_rankings" in self._views_registered:
            conditions = []
            if event:
                safe_event = event.replace("'", "''")
                # mens_rankings uses 'event' col with format "Men's 100m"
                conditions.append(f"event ILIKE '%{safe_event}%'")
            if country:
                conditions.append(f"country = '{country.upper()}'")

            sql = "SELECT * FROM mens_rankings"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += f" ORDER BY rank ASC LIMIT {limit}"
            result = self.query(sql)
            if not result.empty:
                return result

        # Try v2 data (world_rankings has Women's from API)
        if "world_rankings" in self._views_registered:
            conditions = []
            if event:
                safe_event = event.replace("'", "''")
                conditions.append(f"event = '{safe_event}'")
            if gender:
                conditions.append(f"gender = '{gender.upper()}'")
            if country:
                conditions.append(f"country = '{country.upper()}'")

            sql = "SELECT * FROM world_rankings"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += f" ORDER BY rank ASC LIMIT {limit}"
            result = self.query(sql)
            if not result.empty:
                return result

        # Fall back to legacy master (2.3M rows of world rankings data)
        if "master" in self._views_registered:
            conditions = []
            if event:
                # Convert display name to db format, then normalize for matching
                norm_event = re.sub(r'[^0-9a-z]', '', _event_to_db_format(event).lower())
                conditions.append(
                    f"regexp_replace(LOWER(event), '[^0-9a-z]', '', 'g') = '{norm_event}'"
                )
            if gender:
                g = _normalize_gender_for_legacy(gender)
                conditions.append(f"LOWER(gender) = '{g}'")
            if country:
                conditions.append(f"UPPER(nat) = '{country.upper()}'")

            # Age category filter (only legacy master has 'age' column)
            if age_category:
                if age_category == "U20":
                    conditions.append("age < 20")
                elif age_category == "U23":
                    conditions.append("age < 23")
                elif age_category == "Senior":
                    conditions.append("age >= 23")

            sql = "SELECT * FROM master"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += f" ORDER BY rank ASC LIMIT {limit}"
            return self.query(sql)

        return pd.DataFrame()

    def get_ksa_rankings(
        self,
        event: Optional[str] = None,
        gender: Optional[str] = None,
        age_category: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get rankings for KSA athletes only. Falls back to master where nat='KSA'."""
        return self.get_world_rankings(
            event=event, gender=gender, country="KSA",
            age_category=age_category, limit=500,
        )

    # ── Season Toplists ───────────────────────────────────────────────

    def get_season_toplist(
        self,
        event: str,
        gender: str,
        season: Optional[int] = None,
        region: Optional[str] = None,
        age_category: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get season toplist for an event. Falls back to master legacy data."""
        # Try v2 data first
        if "season_toplists" in self._views_registered:
            safe_event = event.replace("'", "''")
            conditions = [f"event = '{safe_event}'", f"gender = '{gender.upper()}'"]

            if season:
                conditions.append(f"season = {season}")

            sql = f"SELECT * FROM season_toplists WHERE {' AND '.join(conditions)}"
            sql += f" ORDER BY place ASC LIMIT {limit}"
            result = self.query(sql)
            if not result.empty:
                return result

        # Fall back to legacy master: filter by year, event, gender, sorted by result_numeric
        if "master" in self._views_registered:
            from data.event_utils import get_event_type
            # Convert display name to db format, then normalize for matching
            norm_event = re.sub(r'[^0-9a-z]', '', _event_to_db_format(event).lower())
            g = _normalize_gender_for_legacy(gender)
            conditions = [
                f"regexp_replace(LOWER(event), '[^0-9a-z]', '', 'g') = '{norm_event}'",
                f"LOWER(gender) = '{g}'",
                "result_numeric IS NOT NULL",
            ]

            if season:
                conditions.append(f"year = {season}")

            # Age category filter (only legacy master has 'age' column)
            if age_category:
                if age_category == "U20":
                    conditions.append("age < 20")
                elif age_category == "U23":
                    conditions.append("age < 23")
                elif age_category == "Senior":
                    conditions.append("age >= 23")

            # Determine sort order: time events -> ASC, field/points events -> DESC
            event_type = get_event_type(event)
            sort_order = "ASC" if event_type == "time" else "DESC"

            # Filter outliers using median-based approach
            # Get one result per competitor-event (best mark), then sort
            sql = f"""
                WITH filtered AS (
                    SELECT *
                    FROM master
                    WHERE {' AND '.join(conditions)}
                ),
                stats AS (
                    SELECT MEDIAN(result_numeric) AS med
                    FROM filtered
                ),
                clean AS (
                    SELECT f.*
                    FROM filtered f, stats s
                    WHERE f.result_numeric BETWEEN s.med * 0.2 AND s.med * 5.0
                )
                SELECT *
                FROM clean
                ORDER BY result_numeric {sort_order}
                LIMIT {limit}
            """
            return self.query(sql)

        return pd.DataFrame()

    # ── Championship Results (from legacy master) ────────────────────

    def get_championship_results(
        self,
        event: Optional[str] = None,
        gender: Optional[str] = None,
        country: Optional[str] = None,
        competition_keywords: Optional[List[str]] = None,
        finals_only: bool = False,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Get historical championship results from master.parquet.

        Args:
            event: Display event name (e.g. '100m')
            gender: 'M' or 'F'
            country: Country code filter (e.g. 'KSA')
            competition_keywords: List of keywords to match competition names
                (e.g. ['Olympic', 'Asian Games'])
            finals_only: If True, only return final results (pos is plain number 1-8)
            limit: Max rows to return

        Returns:
            DataFrame with: competitor, nat, event, result, result_numeric, pos,
            venue, date, year, competition (venue used as proxy)
        """
        if "master" not in self._views_registered:
            return pd.DataFrame()

        conditions = ["result_numeric IS NOT NULL"]

        if event:
            norm_event = re.sub(r'[^0-9a-z]', '', _event_to_db_format(event).lower())
            conditions.append(
                f"regexp_replace(LOWER(event), '[^0-9a-z]', '', 'g') = '{norm_event}'"
            )
        if gender:
            g = _normalize_gender_for_legacy(gender)
            conditions.append(f"LOWER(gender) = '{g}'")
        if country:
            conditions.append(f"UPPER(nat) = '{country.upper()}'")

        if competition_keywords:
            keyword_conds = []
            for kw in competition_keywords:
                safe_kw = kw.replace("'", "''")
                keyword_conds.append(f"venue ILIKE '%{safe_kw}%'")
            conditions.append(f"({' OR '.join(keyword_conds)})")

        if finals_only:
            # Finals positions are plain numbers (1-8), not 1sf3, 2h1, etc.
            conditions.append("regexp_matches(CAST(pos AS VARCHAR), '^[0-9]+$')")

        from data.event_utils import get_event_type
        event_type = get_event_type(event) if event else "time"
        sort_order = "ASC" if event_type == "time" else "DESC"

        sql = f"""
            WITH filtered AS (
                SELECT
                    competitor, nat, event, result, result_numeric,
                    CAST(pos AS VARCHAR) AS pos, venue, date,
                    EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) AS year
                FROM master
                WHERE {' AND '.join(conditions)}
            ),
            stats AS (
                SELECT MEDIAN(result_numeric) AS med FROM filtered
            ),
            clean AS (
                SELECT f.*
                FROM filtered f, stats s
                WHERE f.result_numeric BETWEEN s.med * 0.2 AND s.med * 5.0
            )
            SELECT * FROM clean
            ORDER BY result_numeric {sort_order}
            LIMIT {limit}
        """
        return self.query(sql)

    # ── Competitions ──────────────────────────────────────────────────

    def get_calendar(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ranking_category: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get competition calendar."""
        if "calendar" not in self._views_registered:
            return pd.DataFrame()

        conditions = []
        if start_date:
            conditions.append(f"start_date >= '{start_date}'")
        if end_date:
            conditions.append(f"start_date <= '{end_date}'")
        if ranking_category:
            safe_cat = ranking_category.replace("'", "''")
            conditions.append(f"ranking_category = '{safe_cat}'")

        sql = "SELECT * FROM calendar"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY start_date ASC"
        return self.query(sql)

    def get_upcoming_competitions(self) -> pd.DataFrame:
        """Get upcoming competitions."""
        if "upcoming" not in self._views_registered:
            return pd.DataFrame()
        return self.query("SELECT * FROM upcoming ORDER BY start_date ASC")

    def get_recent_results(self, limit: int = 20) -> pd.DataFrame:
        """Get recent results feed."""
        if "recent_results" not in self._views_registered:
            return pd.DataFrame()
        return self.query(f"SELECT * FROM recent_results ORDER BY date DESC LIMIT {limit}")

    # ── Rivals ────────────────────────────────────────────────────────

    def get_rivals(
        self,
        event: Optional[str] = None,
        gender: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Get rival athletes for an event."""
        if "rivals" not in self._views_registered:
            return pd.DataFrame()

        conditions = []
        if event:
            safe_event = event.replace("'", "''")
            conditions.append(f"event = '{safe_event}'")
        if gender:
            conditions.append(f"gender = '{gender.upper()}'")
        if region:
            safe_region = region.replace("'", "''")
            conditions.append(f"region = '{safe_region}'")

        sql = "SELECT * FROM rivals"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += f" ORDER BY world_rank ASC LIMIT {limit}"
        return self.query(sql)

    # ── Qualifications ────────────────────────────────────────────────

    def get_qualifications(
        self,
        competition_id: Optional[int] = None,
        country: Optional[str] = None,
        event: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get championship qualification status. Falls back to road_to_tokyo legacy data."""
        # Try v2 data first
        if "qualifications" in self._views_registered:
            conditions = []
            if competition_id:
                conditions.append(f"competition_id = {competition_id}")
            if country:
                conditions.append(f"country = '{country.upper()}'")
            if event:
                safe_event = event.replace("'", "''")
                conditions.append(f"event ILIKE '%{safe_event}%'")

            sql = "SELECT * FROM qualifications"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            result = self.query(sql)
            if not result.empty:
                return result

        # Fall back to legacy road_to_tokyo
        # Note: In road_to_tokyo, 'Athlete' column = country code, 'Status' = athlete name
        if "road_to_tokyo" in self._views_registered:
            conditions = []
            if event:
                safe_event = event.replace("'", "''")
                conditions.append(f"Actual_Event_Name ILIKE '%{safe_event}%'")
            if country:
                safe_country = country.upper()
                conditions.append(f"UPPER(Athlete) = '{safe_country}'")

            sql = "SELECT * FROM road_to_tokyo"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            return self.query(sql)

        return pd.DataFrame()

    # ── Legacy Data (from v1 master.parquet) ──────────────────────────

    def get_legacy_rankings(
        self,
        event: Optional[str] = None,
        gender: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Query the legacy 2.3M row master.parquet from v1."""
        if "master" not in self._views_registered:
            return pd.DataFrame()

        conditions = []
        if event:
            from data.event_utils import normalize_event_for_match
            # Use exact match after normalization
            safe_event = event.replace("'", "''")
            conditions.append(f"event = '{safe_event}'")
        if gender:
            g = _normalize_gender_for_legacy(gender)
            conditions.append(f"LOWER(gender) = '{g}'")

        sql = "SELECT * FROM master"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += f" ORDER BY rank ASC LIMIT {limit}"
        return self.query(sql)

    def get_benchmarks(self) -> pd.DataFrame:
        """Get championship benchmark standards from v1."""
        if "benchmarks" not in self._views_registered:
            return pd.DataFrame()
        return self.query("SELECT * FROM benchmarks")

    # ── Legacy Direct Access ─────────────────────────────────────────

    def get_ksa_profiles_legacy(self) -> pd.DataFrame:
        """Query legacy ksa_profiles.parquet directly."""
        if "ksa_profiles" not in self._views_registered:
            return pd.DataFrame()
        return self.query(
            "SELECT * FROM ksa_profiles ORDER BY best_world_rank ASC NULLS LAST"
        )

    def get_qualifications_legacy(
        self,
        event: Optional[str] = None,
        country: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query legacy road_to_tokyo.parquet directly.

        Note: In road_to_tokyo, 'Athlete' column = country code, 'Status' = athlete name.
        """
        if "road_to_tokyo" not in self._views_registered:
            return pd.DataFrame()

        conditions = []
        if event:
            safe_event = event.replace("'", "''")
            conditions.append(f"Actual_Event_Name ILIKE '%{safe_event}%'")
        if country:
            conditions.append(f"UPPER(Athlete) = '{country.upper()}'")

        sql = "SELECT * FROM road_to_tokyo"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return self.query(sql)

    # ── Discovery / Utility ──────────────────────────────────────────

    def get_event_list(self) -> List[str]:
        """Return distinct events from available data (display format)."""
        from data.event_utils import format_event_name

        events = set()

        # Try v2 views first
        for view in ("world_rankings", "season_toplists", "ksa_results"):
            if view in self._views_registered:
                try:
                    df = self.query(f"SELECT DISTINCT event FROM {view}")
                    if not df.empty:
                        events.update(df["event"].dropna().tolist())
                except Exception:
                    pass

        # Fall back to / supplement with legacy master
        if "master" in self._views_registered:
            try:
                df = self.query("SELECT DISTINCT event FROM master")
                if not df.empty:
                    events.update(df["event"].dropna().tolist())
            except Exception:
                pass

        # Convert to display names and sort
        display_events = sorted({format_event_name(e) for e in events})
        return display_events

    def get_athlete_list(self, country: Optional[str] = None) -> List[str]:
        """Return distinct athlete names from available data."""
        athletes = set()

        # Try v2 ksa_athletes
        if "ksa_athletes" in self._views_registered:
            try:
                sql = "SELECT DISTINCT full_name FROM ksa_athletes"
                if country:
                    sql += f" WHERE UPPER(country) = '{country.upper()}'"
                df = self.query(sql)
                if not df.empty:
                    athletes.update(df["full_name"].dropna().tolist())
            except Exception:
                pass

        # Try legacy ksa_profiles
        if "ksa_profiles" in self._views_registered:
            try:
                sql = "SELECT DISTINCT full_name FROM ksa_profiles"
                if country:
                    sql += f" WHERE UPPER(country_code) = '{country.upper()}'"
                df = self.query(sql)
                if not df.empty:
                    athletes.update(df["full_name"].dropna().tolist())
            except Exception:
                pass

        # Fall back to / supplement with legacy master
        if "master" in self._views_registered:
            try:
                sql = "SELECT DISTINCT competitor FROM master"
                if country:
                    sql += f" WHERE UPPER(nat) = '{country.upper()}'"
                df = self.query(sql)
                if not df.empty:
                    athletes.update(df["competitor"].dropna().tolist())
            except Exception:
                pass

        return sorted(athletes)

    # ── Diagnostics ───────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get connector status and data summary."""
        status = {
            "mode": self._data_mode,
            "views": sorted(self._views_registered),
            "counts": {},
        }
        for view in self._views_registered:
            try:
                count = self._conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
                status["counts"][view] = count
            except Exception:
                status["counts"][view] = -1

        return status

    def close(self):
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()


# ── Module-level singleton for Streamlit ───────────────────────────────

_connector: Optional[DataConnector] = None


def get_connector() -> DataConnector:
    """Get or create the module-level DataConnector singleton."""
    global _connector
    if _connector is None:
        _connector = DataConnector()
    return _connector
