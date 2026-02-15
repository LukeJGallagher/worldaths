"""
World Athletics GraphQL API Client.

Async client for the World Athletics GraphQL API with rate limiting,
retry logic, and typed response helpers.

Usage:
    async with WAClient() as client:
        rankings = await client.get_world_rankings("100m")
        athlete = await client.get_athlete_profile(14560359)
"""

import asyncio
import platform
import logging
from typing import Any, Dict, List, Optional, Union

import httpx

from .queries import (
    SEARCH_COMPETITORS,
    GET_SINGLE_COMPETITOR,
    GET_COMPETITOR_RESULTS_BY_DISCIPLINE,
    GET_COMPETITOR_RESULTS_BY_DATE,
    GET_COMPETITOR_MAJOR_CHAMPIONSHIPS,
    GET_COMPETITOR_TOP10,
    GET_COMPETITOR_WINNING_STREAK,
    GET_COMPETITOR_SEASON_BESTS,
    GET_COMPETITOR_HONOUR_SUMMARY,
    HEAD_TO_HEAD,
    GET_WORLD_RANKINGS,
    GET_TOP_LIST,
    GET_ALL_TIME_LIST,
    GET_RANKING_SCORE_CALCULATION,
    GET_TOP_RANKINGS,
    GET_WORLD_RANKINGS_CHANGES,
    GET_LEADING_ATHLETES,
    GET_CALENDAR_EVENTS,
    GET_CALENDAR_COMPETITION_RESULTS,
    GET_UPCOMING_COMPETITIONS,
    GET_RECENT_RESULTS,
    GET_CHAMPIONSHIP_QUALIFICATIONS,
    GET_LATEST_QUALIFIED_COMPETITORS,
    GET_RECORDS_BY_CATEGORY,
    GET_RECORDS_CATEGORIES,
    GET_LATEST_RECORDS,
)
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# API configuration (updated 2026-02-13 - WA migrated from prod.aws to edge.aws)
# Key rotates periodically - set WA_GRAPHQL_URL and WA_API_KEY env vars to override
import os as _os

GRAPHQL_URL = _os.environ.get(
    "WA_GRAPHQL_URL",
    "https://graphql-prod-4843.edge.aws.worldathletics.org/graphql",
)
API_KEY = _os.environ.get("WA_API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://worldathletics.org",
    "Referer": "https://worldathletics.org/",
    "x-api-key": API_KEY,
    "x-amz-user-agent": "aws-amplify/3.0.2",
}


class WAClient:
    """Async World Athletics GraphQL API client."""

    def __init__(self, max_per_second: float = 3.0, max_retries: int = 3):
        self._client: Optional[httpx.AsyncClient] = None
        self._limiter = RateLimiter(max_per_second=max_per_second, max_retries=max_retries)
        self._max_retries = max_retries

    async def __aenter__(self):
        self._client = httpx.AsyncClient(verify=False, timeout=30)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _execute(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query with rate limiting and retry."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with WAClient() as client:'")

        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire()
            try:
                response = await self._client.post(
                    GRAPHQL_URL,
                    headers=HEADERS,
                    json={"query": query, "variables": variables or {}},
                )
                response.raise_for_status()
                result = response.json()

                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown GraphQL error")
                    # Don't retry validation/auth errors - they won't succeed
                    is_permanent = any(
                        kw in error_msg
                        for kw in (
                            "VariableTypeMismatch", "Not Authorized",
                            "ValidationError", "FieldUndefined",
                            "MissingFieldArgument", "SubSelectionRequired",
                        )
                    )
                    if is_permanent:
                        raise GraphQLError(error_msg, result["errors"])
                    logger.warning(f"GraphQL error (attempt {attempt + 1}): {error_msg}")
                    if attempt < self._max_retries:
                        await asyncio.sleep(self._limiter.get_backoff_delay(attempt))
                        continue
                    raise GraphQLError(error_msg, result["errors"])

                return result.get("data", {})

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP {e.response.status_code} (attempt {attempt + 1})")
                if attempt < self._max_retries:
                    await asyncio.sleep(self._limiter.get_backoff_delay(attempt))
                    continue
                raise
            except httpx.RequestError as e:
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self._max_retries:
                    await asyncio.sleep(self._limiter.get_backoff_delay(attempt))
                    continue
                raise

        return {}

    # ── Athlete Search & Profiles ──────────────────────────────────────

    async def search_athletes(
        self,
        query: Optional[str] = None,
        country_code: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> List[Dict]:
        """Search for athletes by name, country, or gender."""
        variables: Dict[str, Any] = {}
        if query:
            variables["query"] = query
        if country_code:
            variables["countryCode"] = country_code
        if gender:
            variables["gender"] = gender

        data = await self._execute(SEARCH_COMPETITORS, variables)
        # API now returns a list directly (no 'results' wrapper)
        result = data.get("searchCompetitors")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("results") or []
        return []

    async def search_all_athletes_for_country(self, country_code: str) -> List[Dict]:
        """Fetch all athletes for a country."""
        results = await self.search_athletes(country_code=country_code)
        seen_ids = set()
        all_athletes = []
        for athlete in results:
            aid = athlete.get("aaAthleteId") or athlete.get("iaafId")
            if aid and aid not in seen_ids:
                seen_ids.add(aid)
                all_athletes.append(athlete)

        logger.info(f"Found {len(all_athletes)} athletes for {country_code}")
        return all_athletes

    async def get_athlete_profile(self, athlete_id: int) -> Optional[Dict]:
        """Get comprehensive athlete profile: PBs, SBs, rankings, honours."""
        data = await self._execute(GET_SINGLE_COMPETITOR, {"id": athlete_id})
        return data.get("getSingleCompetitor")

    async def get_athlete_results_by_discipline(
        self, athlete_id: int, year: Optional[int] = None
    ) -> Optional[Dict]:
        """Get athlete results grouped by event."""
        variables: Dict[str, Any] = {"id": athlete_id}
        if year:
            variables["resultsByYear"] = year
        data = await self._execute(GET_COMPETITOR_RESULTS_BY_DISCIPLINE, variables)
        return data.get("getSingleCompetitorResultsDiscipline")

    async def get_athlete_results_by_date(
        self, athlete_id: int, year: Optional[int] = None
    ) -> Optional[Dict]:
        """Get athlete results ordered by date (recent form)."""
        variables: Dict[str, Any] = {"id": athlete_id}
        if year:
            variables["resultsByYear"] = year
        data = await self._execute(GET_COMPETITOR_RESULTS_BY_DATE, variables)
        return data.get("getSingleCompetitorResultsDate")

    async def get_athlete_major_championships(
        self, athlete_id: int, url_slug: str = ""
    ) -> Optional[Dict]:
        """Get athlete's major championship results only.

        Requires urlSlug (String!) per new API schema.
        """
        data = await self._execute(
            GET_COMPETITOR_MAJOR_CHAMPIONSHIPS,
            {"id": athlete_id, "urlSlug": url_slug},
        )
        return data.get("getSingleCompetitorMajorChampionships")

    async def get_athlete_top10(
        self, athlete_id: int, discipline: Optional[str] = None
    ) -> Optional[Dict]:
        """Get athlete's top 10 all-time performances."""
        variables: Dict[str, Any] = {"id": athlete_id}
        if discipline:
            variables["allTimePersonalTop10Discipline"] = discipline
        data = await self._execute(GET_COMPETITOR_TOP10, variables)
        return data.get("getSingleCompetitorAllTimePersonalTop10")

    async def get_athlete_winning_streak(
        self,
        athlete_id: int,
        discipline: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        final_only: bool = False,
    ) -> Optional[Dict]:
        """Get athlete's winning streak data."""
        variables: Dict[str, Any] = {"id": athlete_id}
        # New API prefixes all args with 'winningStreaks'
        if discipline:
            variables["winningStreaksDisciplineOption"] = discipline
        if start_date:
            variables["winningStreaksStartDate"] = start_date
        if end_date:
            variables["winningStreaksEndDate"] = end_date
        variables["winningStreaksFinalOnly"] = final_only
        data = await self._execute(GET_COMPETITOR_WINNING_STREAK, variables)
        return data.get("getSingleCompetitorWinningStreak")

    async def get_athlete_season_bests(
        self, athlete_id: int, season: Optional[int] = None
    ) -> Optional[Dict]:
        """Get athlete's season bests."""
        variables: Dict[str, Any] = {"id": athlete_id}
        if season:
            variables["seasonsBestsSeason"] = season
        data = await self._execute(GET_COMPETITOR_SEASON_BESTS, variables)
        return data.get("getSingleCompetitorSeasonBests")

    async def get_athlete_honours(self, athlete_id: int, url_slug: str = "") -> Optional[List]:
        """Get athlete's honours/medals summary.

        Requires urlSlug (String!) per new API schema.
        """
        data = await self._execute(
            GET_COMPETITOR_HONOUR_SUMMARY,
            {"id": athlete_id, "urlSlug": url_slug},
        )
        return data.get("getSingleCompetitorHonourSummary")

    async def head_to_head(
        self,
        athlete1_id: int,
        athlete2_id: int,
        discipline: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        final_only: bool = False,
    ) -> Optional[Dict]:
        """Get head-to-head record between two athletes."""
        variables: Dict[str, Any] = {
            "id": athlete1_id,
            "headToHeadOpponent": athlete2_id,
        }
        # New API prefixes all args with 'headToHead'
        if discipline:
            variables["headToHeadDiscipline"] = discipline
        if start_date:
            variables["headToHeadStartDate"] = start_date
        if end_date:
            variables["headToHeadEndDate"] = end_date
        if final_only:
            variables["headToHeadFinalOnly"] = final_only
        data = await self._execute(HEAD_TO_HEAD, variables)
        return data.get("headToHead")

    # ── Rankings & Toplists ────────────────────────────────────────────

    async def get_world_rankings(
        self,
        event_group: str,
        region_type: Optional[str] = None,
        region: Optional[str] = None,
        rank_date: Optional[str] = None,
        limit: int = 100,
        limit_by_country: Optional[int] = None,
    ) -> Optional[Dict]:
        """Get world rankings for an event group.

        NOTE: 'sex' and 'page' args were REMOVED from the API.
        Gender is now encoded in the eventGroup parameter.

        Args:
            event_group: e.g. "10K", "100m", "HJ"
            region_type: "world", "area", "country"
            region: e.g. "asia", "KSA"
            rank_date: ISO datetime e.g. "2026-02-13T00:00:00.000Z"
            limit_by_country: max athletes per country
        """
        variables: Dict[str, Any] = {
            "eventGroup": event_group,
            "limit": limit,
        }
        if region_type:
            variables["regionType"] = region_type
        if region:
            variables["region"] = region
        if rank_date:
            variables["rankDate"] = rank_date
        if limit_by_country is not None:
            variables["limitByCountry"] = limit_by_country

        data = await self._execute(GET_WORLD_RANKINGS, variables)
        return data.get("getWorldRankings")

    async def get_top_list(
        self,
        discipline_code: str,
        gender: str,
        year: Optional[int] = None,
        environment: str = "outdoor",
        region_type: Optional[str] = None,
        region: Optional[str] = None,
        age_category: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
    ) -> Optional[Dict]:
        """Get season toplist for an event.

        NOTE: API now uses AAapiQuery input type. Params are wrapped
        into a 'query' object with a 'field' sort parameter.

        Args:
            discipline_code: e.g. "100", "200", "HJ", "LJ"
            gender: "M" or "F"
            year: season year (defaults to current)
            age_category: e.g. "U20", "U23", "senior"
        """
        query_obj: Dict[str, Any] = {
            "disciplineCode": discipline_code,
            "gender": gender,
            "environment": environment,
            "page": page,
            "limit": limit,
        }
        if year:
            query_obj["season"] = year  # AAapiQuery uses 'season' not 'year'
        if region_type:
            query_obj["regionType"] = region_type
        if region:
            query_obj["region"] = region
        if age_category:
            query_obj["ageCategory"] = age_category

        variables = {"query": query_obj, "field": "resultScore"}
        data = await self._execute(GET_TOP_LIST, variables)
        return data.get("getTopList")

    async def get_all_time_list(
        self,
        discipline_code: str,
        gender: str,
        environment: str = "outdoor",
        region_type: Optional[str] = None,
        region: Optional[str] = None,
        age_category: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
    ) -> Optional[Dict]:
        """Get all-time toplist for an event (uses AAapiQuery input type)."""
        query_obj: Dict[str, Any] = {
            "disciplineCode": discipline_code,
            "gender": gender,
            "environment": environment,
            "page": page,
            "limit": limit,
        }
        if region_type:
            query_obj["regionType"] = region_type
        if region:
            query_obj["region"] = region
        if age_category:
            query_obj["ageCategory"] = age_category

        variables = {"query": query_obj, "field": "resultScore"}
        data = await self._execute(GET_ALL_TIME_LIST, variables)
        return data.get("getAllTimeList")

    async def get_ranking_score_breakdown(self, athlete_id: int) -> Optional[List]:
        """Get how an athlete's ranking score is calculated."""
        data = await self._execute(GET_RANKING_SCORE_CALCULATION, {"athleteId": athlete_id})
        return data.get("getRankingScoreCalculation")

    async def get_top_rankings(self, limit: int = 50) -> Optional[Dict]:
        """Get #1 ranked athletes per event for BOTH genders.

        This is the only authorized query that returns men's ranking data.
        Returns dict with 'moduleTitle' and 'rankings' list.
        Each ranking has: competitorName, competitorId, eventName,
        eventUrlSlug, sexCode (M/W), score, countryCode.
        """
        data = await self._execute(GET_TOP_RANKINGS, {"limit": limit, "all": True})
        return data.get("getTopRankings")

    async def get_ranking_movers(self, limit: int = 20) -> Optional[Dict]:
        """Get athletes who moved most in rankings.

        Returns dict with 'moduleTitle', 'moduleSubtitle', 'changes' list.
        Each change has: competitorName, eventName, eventUrlSlug, sexCode,
        place, previousPlace, improvement, score, countryCode.
        """
        data = await self._execute(GET_WORLD_RANKINGS_CHANGES, {"all": True, "limit": limit})
        return data.get("getWorldRankingsChanges")

    async def get_leading_athletes(self, limit: int = 20) -> Optional[Dict]:
        """Get current world-leading athletes.

        Returns dict with 'eventResults' list. Each has: eventName,
        disciplineUrlSlug, sexCode, environment, ageCategory, season,
        and 'results' list with mark, countryCode, competitor {name, id}.
        """
        data = await self._execute(GET_LEADING_ATHLETES, {"limit": limit})
        return data.get("getLeadingAthletes")

    # ── Competitions & Results ─────────────────────────────────────────

    async def get_calendar_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        region_type: Optional[str] = None,
        region_id: Optional[int] = None,
        discipline_id: Optional[int] = None,
        ranking_category_id: Optional[int] = None,
        permit_level_id: Optional[int] = None,
        query: Optional[str] = None,
        competition_group_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Optional[Dict]:
        """Get competition calendar with filters.

        NOTE: API changed 'region: String' to 'regionId: Int',
        and other IDs from String to Int.
        """
        variables: Dict[str, Any] = {"limit": limit, "offset": offset}
        if start_date:
            variables["startDate"] = start_date
        if end_date:
            variables["endDate"] = end_date
        if region_type:
            variables["regionType"] = region_type
        if region_id is not None:
            variables["regionId"] = region_id
        if discipline_id is not None:
            variables["disciplineId"] = discipline_id
        if ranking_category_id is not None:
            variables["rankingCategoryId"] = ranking_category_id
        if permit_level_id is not None:
            variables["permitLevelId"] = permit_level_id
        if query:
            variables["query"] = query
        if competition_group_id is not None:
            variables["competitionGroupId"] = competition_group_id

        data = await self._execute(GET_CALENDAR_EVENTS, variables)
        return data.get("getCalendarEvents")

    async def get_competition_results(
        self,
        competition_id: int,
        day: Optional[int] = None,
        event_id: Optional[int] = None,
    ) -> Optional[Dict]:
        """Get full results for a competition (competition viewer)."""
        variables: Dict[str, Any] = {"competitionId": competition_id}
        if day is not None:
            variables["day"] = day
        if event_id is not None:
            variables["eventId"] = event_id

        data = await self._execute(GET_CALENDAR_COMPETITION_RESULTS, variables)
        return data.get("getCalendarCompetitionResults")

    async def get_upcoming_competitions(self, today: Optional[str] = None) -> Optional[List]:
        """Get upcoming competitions.

        Returns a list of groups, each with 'label' and 'competitions' list.
        Competitions have: competitionId, name, venue, startDate, endDate,
        dateRange, isNextEvent, urlSlug.
        """
        variables = {}
        if today:
            variables["today"] = today
        data = await self._execute(GET_UPCOMING_COMPETITIONS, variables)
        return data.get("getUpcomingCompetitions")

    async def get_recent_results(self, limit: int = 20) -> Optional[Dict]:
        """Get most recent global results.

        Returns dict with 'results' list. Each result has: id, iaafId,
        name, venue, startDate, endDate, event (WAWEvent object).
        """
        data = await self._execute(GET_RECENT_RESULTS, {"limit": limit})
        return data.get("getRecentResults")

    # ── Championship Qualification ─────────────────────────────────────

    async def get_championship_qualifications(
        self,
        competition_id: int,
        event_id: Optional[int] = None,
        country: Optional[str] = None,
        qualification_type: Optional[str] = None,
    ) -> Optional[Dict]:
        """Get championship qualification status."""
        variables: Dict[str, Any] = {"competitionId": competition_id}
        if event_id:
            variables["eventId"] = event_id
        if country:
            variables["country"] = country
        if qualification_type:
            variables["qualificationType"] = qualification_type

        data = await self._execute(GET_CHAMPIONSHIP_QUALIFICATIONS, variables)
        return data.get("getChampionshipQualifications")

    async def get_latest_qualified(
        self, competition_id: int, limit: int = 50, offset: int = 0
    ) -> Optional[Dict]:
        """Get recently qualified athletes for a championship."""
        data = await self._execute(
            GET_LATEST_QUALIFIED_COMPETITORS,
            {"competitionId": competition_id, "limit": limit, "offset": offset},
        )
        return data.get("getLatestQualifiedCompetitors")

    # ── Records ────────────────────────────────────────────────────────

    async def get_records_categories(self) -> Optional[List]:
        """Get all record category IDs and names.

        NOTE: API now returns list directly (no 'categories' wrapper).
        """
        data = await self._execute(GET_RECORDS_CATEGORIES)
        return data.get("getRecordsCategories")

    async def get_records_by_category(self, category_id: int) -> Optional[List]:
        """Get records for a specific category (world, area, national).

        NOTE: API now returns list directly (no 'records' wrapper).
        """
        data = await self._execute(GET_RECORDS_BY_CATEGORY, {"categoryId": category_id})
        return data.get("getRecordsDetailByCategory")

    async def get_latest_records(self, limit: int = 20, days: int = 30) -> Optional[List]:
        """Get recently broken records."""
        data = await self._execute(GET_LATEST_RECORDS, {"limit": limit, "days": days})
        return (data.get("getLatestRecords") or {}).get("records")

    # ── Batch Helpers ──────────────────────────────────────────────────

    async def get_athlete_profiles_batch(
        self, athlete_ids: List[int], progress_callback=None
    ) -> List[Dict]:
        """Fetch multiple athlete profiles with progress tracking."""
        results = []
        for i, aid in enumerate(athlete_ids):
            profile = await self.get_athlete_profile(aid)
            if profile:
                results.append({"id": aid, **profile})
            if progress_callback:
                progress_callback(i + 1, len(athlete_ids))
        return results

    async def get_deep_athlete_data(
        self, athlete_id: int, url_slug: str = ""
    ) -> Dict[str, Any]:
        """Fetch ALL available data for a single athlete (deep scrape).

        Each sub-query is wrapped in try/except so one failure doesn't
        block the others. Partial data is still returned.

        Args:
            athlete_id: WA athlete ID (aaAthleteId).
            url_slug: Athlete URL slug (required by some queries).
        """
        result: Dict[str, Any] = {"id": athlete_id}

        # Profile is critical - if this fails, return minimal result
        try:
            result["profile"] = await self.get_athlete_profile(athlete_id)
        except Exception:
            result["profile"] = None

        # All other queries are optional - best-effort
        for key, coro in [
            ("results_by_discipline", self.get_athlete_results_by_discipline(athlete_id)),
            ("major_championships", self.get_athlete_major_championships(athlete_id, url_slug)),
            ("top10_performances", self.get_athlete_top10(athlete_id)),
            ("season_bests", self.get_athlete_season_bests(athlete_id)),
            ("honours", self.get_athlete_honours(athlete_id, url_slug)),
            ("ranking_breakdown", self.get_ranking_score_breakdown(athlete_id)),
        ]:
            try:
                result[key] = await coro
            except Exception:
                result[key] = None

        return result


class GraphQLError(Exception):
    """Raised when the GraphQL API returns errors."""

    def __init__(self, message: str, errors: List[Dict]):
        super().__init__(message)
        self.errors = errors


# ── Convenience runner for sync contexts ───────────────────────────────

def run_async(coro):
    """Run an async function from sync code (handles Windows event loop)."""
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)
