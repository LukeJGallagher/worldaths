"""
World Athletics GraphQL query strings.

Updated 2026-02-13 for new API schema at graphql-prod-4843.edge.aws.
Key changes from old schema:
- searchCompetitors: returns list directly (no 'results' wrapper)
- getWorldRankings: 'sex' arg removed (gender encoded in eventGroup)
- getTopList/getAllTimeList: now use AAapiQuery input type
- honours: 'categoryName' -> 'eventName', 'competitionId' added
- winningStreak: args prefixed with 'winningStreaks'
- headToHead: args prefixed with 'headToHead'
- resultsOrderBy: 'resultsByYearOrderByDate' -> 'resultsByYearOrderBy'
"""

# ── Athlete Search & Profiles ──────────────────────────────────────────

SEARCH_COMPETITORS = """
query SearchCompetitors($query: String, $gender: GenderType, $disciplineCode: String,
    $environment: String, $countryCode: String) {
  searchCompetitors(query: $query, gender: $gender, disciplineCode: $disciplineCode,
      environment: $environment, countryCode: $countryCode) {
    aaAthleteId
    familyName
    givenName
    birthDate
    disciplines
    iaafId
    gender
    country
    urlSlug
  }
}
"""

GET_SINGLE_COMPETITOR = """
query GetSingleCompetitor($id: Int, $urlSlug: String) {
  getSingleCompetitor(id: $id, urlSlug: $urlSlug) {
    basicData {
      iaafId
      aaId
      firstName
      lastName
      fullName
      familyName
      givenName
      friendlyName
      sexCode
      sexName
      countryCode
      countryName
      countryFullName
      birthDate
      birthPlace
      birthPlaceCountryName
      biography
      urlSlug
      male
    }
    primaryMedia {
      id
      title
      urlSlug
      credit
      fileNameUrl
      sourceWidth
      sourceHeight
      type
      format
      hosting
    }
    personalBests {
      results {
        discipline
        mark
        wind
        venue
        date
        resultScore
        records
        indoor
        notLegal
      }
    }
    seasonsBests {
      results {
        discipline
        mark
        wind
        venue
        date
        resultScore
        records
        indoor
        notLegal
      }
    }
    honours {
      categoryName
      withWind
      withDrop
      results {
        competition
        discipline
        mark
        place
        venue
        date
        indoor
        competitionId
        eventName
      }
    }
    worldRankings {
      current {
        eventGroup
        male
        place
        rankingScore
      }
      best {
        eventGroup
        place
        weeks
      }
    }
  }
}
"""

GET_COMPETITOR_RESULTS_BY_DISCIPLINE = """
query GetSingleCompetitorResultsDiscipline($id: Int, $resultsByYear: Int,
    $resultsByYearOrderBy: String) {
  getSingleCompetitorResultsDiscipline(id: $id, resultsByYear: $resultsByYear,
      resultsByYearOrderBy: $resultsByYearOrderBy) {
    parameters {
      resultsByYear
    }
    activeYears
    resultsByEvent {
      indoor
      discipline
      withWind
      results {
        date
        competition
        venue
        country
        category
        race
        place
        mark
        wind
        notLegal
        resultScore
        remark
      }
    }
  }
}
"""

GET_COMPETITOR_RESULTS_BY_DATE = """
query GetSingleCompetitorResultsDate($id: Int, $resultsByYear: Int,
    $resultsByYearOrderBy: String) {
  getSingleCompetitorResultsDate(id: $id, resultsByYear: $resultsByYear,
      resultsByYearOrderBy: $resultsByYearOrderBy) {
    parameters {
      resultsByYear
    }
    activeYears
    resultsByDate {
      date
      competition
      venue
      indoor
      discipline
      country
      category
      race
      place
      mark
      wind
      notLegal
      resultScore
      remark
    }
  }
}
"""

GET_COMPETITOR_MAJOR_CHAMPIONSHIPS = """
query GetSingleCompetitorMajorChampionships($id: Int, $urlSlug: String!) {
  getSingleCompetitorMajorChampionships(id: $id, urlSlug: $urlSlug) {
    parameters {
      majorChampionshipsByCategory
    }
    results {
      category
      results {
        id
        discipline
        competition
        venue
        race
        place
        result
        wind
        date
        eventId
        competitionId
      }
    }
  }
}
"""

GET_COMPETITOR_TOP10 = """
query GetSingleCompetitorAllTimePersonalTop10($id: Int,
    $allTimePersonalTop10Discipline: Int) {
  getSingleCompetitorAllTimePersonalTop10(id: $id,
      allTimePersonalTop10Discipline: $allTimePersonalTop10Discipline) {
    disciplines {
      id
      name
    }
    results {
      discipline
      result
      wind
      date
      competition
      country
      category
      race
      place
      score
      records
      remark
    }
  }
}
"""

GET_COMPETITOR_WINNING_STREAK = """
query GetSingleCompetitorWinningStreak($id: Int,
    $winningStreaksDisciplineOption: String,
    $winningStreaksStartDate: String, $winningStreaksEndDate: String,
    $winningStreaksFinalOnly: Boolean) {
  getSingleCompetitorWinningStreak(id: $id,
      winningStreaksDisciplineOption: $winningStreaksDisciplineOption,
      winningStreaksStartDate: $winningStreaksStartDate,
      winningStreaksEndDate: $winningStreaksEndDate,
      winningStreaksFinalOnly: $winningStreaksFinalOnly) {
    discipline
    startDate
    endDate
    totalWins
    totalLosses
    currentStreak
    longestStreak
    results {
      date
      competition
      venue
      place
      mark
      discipline
    }
  }
}
"""

GET_COMPETITOR_SEASON_BESTS = """
query GetSingleCompetitorSeasonBests($id: Int, $seasonsBestsSeason: Int) {
  getSingleCompetitorSeasonBests(id: $id, seasonsBestsSeason: $seasonsBestsSeason) {
    parameters {
      seasonsBestsSeason
    }
    activeSeasons
    results {
      discipline
      mark
      wind
      venue
      date
      resultScore
      records
      indoor
      notLegal
    }
  }
}
"""

GET_COMPETITOR_HONOUR_SUMMARY = """
query GetSingleCompetitorHonourSummary($id: Int, $urlSlug: String!) {
  getSingleCompetitorHonourSummary(id: $id, urlSlug: $urlSlug) {
    count
    honour
  }
}
"""

HEAD_TO_HEAD = """
query HeadToHead($id: Int, $headToHeadOpponent: Int,
    $headToHeadDiscipline: String,
    $headToHeadStartDate: String, $headToHeadEndDate: String,
    $headToHeadFinalOnly: Boolean) {
  headToHead(id: $id, headToHeadOpponent: $headToHeadOpponent,
      headToHeadDiscipline: $headToHeadDiscipline,
      headToHeadStartDate: $headToHeadStartDate,
      headToHeadEndDate: $headToHeadEndDate,
      headToHeadFinalOnly: $headToHeadFinalOnly) {
    athlete1 {
      id
      name
      countryCode
      wins
      results {
        date
        competition
        venue
        place
        mark
      }
    }
    athlete2 {
      id
      name
      countryCode
      wins
      results {
        date
        competition
        venue
        place
        mark
      }
    }
  }
}
"""

# ── Rankings & Toplists ────────────────────────────────────────────────

# Note: 'sex' argument was REMOVED. Gender is now encoded in eventGroup
# e.g. eventGroup="100m" returns men, eventGroup needs gender prefix or
# gender is embedded in the event group naming
GET_WORLD_RANKINGS = """
query GetWorldRankings($eventGroup: String, $regionType: String,
    $region: String, $rankDate: AWSDateTime, $limit: Int, $limitByCountry: Int) {
  getWorldRankings(eventGroup: $eventGroup, regionType: $regionType,
      region: $region, rankDate: $rankDate, limit: $limit,
      limitByCountry: $limitByCountry) {
    eventGroup
    rankDate
    rankings {
      id
      place
      competitorName
      competitorUrlSlug
      competitorBirthDate
      countryCode
      rankingScore
      disciplineCodes
      countryPlace
      previousPlace
      previousRankingScore
    }
  }
}
"""

# Note: getTopList now uses AAapiQuery input type, but we pass individual fields
# that get wrapped into the query object. If this fails, we need to use the
# AAapiQuery format instead.
GET_TOP_LIST = """
query GetTopList($query: AAapiQuery!, $field: String!) {
  getTopList(query: $query, field: $field) {
    page
    pages
    payload {
      position
      place
      achieverPosition
      result
      achiever
      nationality
      venue
      date
      resultScore
    }
  }
}
"""

GET_ALL_TIME_LIST = """
query GetAllTimeList($query: AAapiQuery!, $field: String!) {
  getAllTimeList(query: $query, field: $field) {
    page
    pages
    payload {
      position
      place
      achieverPosition
      result
      achiever
      nationality
      venue
      date
      resultScore
    }
  }
}
"""

GET_RANKING_SCORE_CALCULATION = """
query GetRankingScoreCalculation($athleteId: Int) {
  getRankingScoreCalculation(athleteId: $athleteId) {
    eventGroup
    rankingScore
    averagePerformanceScore
    place
    male
    athlete
    country
    results {
      date
      competition
      country
      category
      discipline
      mark
      place
      wind
      resultScore
      placingScore
      performanceScore
    }
  }
}
"""

GET_TOP_RANKINGS = """
query GetTopRankings($limit: Int, $all: Boolean) {
  getTopRankings(limit: $limit, all: $all) {
    moduleTitle
    rankings {
      rankingCalculationId
      competitorId
      competitorName
      eventName
      eventUrlSlug
      sexCode
      score
      countryCode
    }
  }
}
"""

GET_WORLD_RANKINGS_CHANGES = """
query GetWorldRankingsChanges($all: Boolean, $limit: Int) {
  getWorldRankingsChanges(all: $all, limit: $limit) {
    moduleTitle
    moduleSubtitle
    changes {
      rankingCalculationId
      place
      previousPlace
      improvement
      eventName
      eventUrlSlug
      score
      competitorId
      competitorName
      sexCode
      countryCode
    }
  }
}
"""

GET_LEADING_ATHLETES = """
query GetLeadingAthletes($limit: Int, $countryOfResidence: String) {
  getLeadingAthletes(limit: $limit, countryOfResidence: $countryOfResidence) {
    eventResults {
      eventName
      environment
      ageCategory
      season
      disciplineUrlSlug
      disciplineTypeUrlSlug
      sexCode
      results {
        mark
        countryCode
        competitor {
          name
          id
          urlSlug
          birthDate
        }
      }
    }
  }
}
"""

# ── Competitions & Results ─────────────────────────────────────────────

# Note: 'region' was replaced with 'regionId: Int'
GET_CALENDAR_EVENTS = """
query GetCalendarEvents($startDate: String, $endDate: String, $regionType: String,
    $regionId: Int, $disciplineId: Int,
    $rankingCategoryId: Int, $permitLevelId: Int, $query: String,
    $competitionGroupId: Int, $limit: Int, $offset: Int,
    $showOptionsWithNoHits: Boolean, $hideCompetitionsWithNoResults: Boolean) {
  getCalendarEvents(startDate: $startDate, endDate: $endDate,
      regionType: $regionType, regionId: $regionId,
      disciplineId: $disciplineId, rankingCategoryId: $rankingCategoryId,
      permitLevelId: $permitLevelId, query: $query,
      competitionGroupId: $competitionGroupId, limit: $limit, offset: $offset,
      showOptionsWithNoHits: $showOptionsWithNoHits,
      hideCompetitionsWithNoResults: $hideCompetitionsWithNoResults) {
    hits
    results {
      id
      iaafId
      name
      venue
      area
      country
      rankingCategory
      disciplines
      competitionGroup
      competitionSubgroup
      startDate
      endDate
      dateRange
      season
      hasResults
      hasApiResults
      hasStartlist
      hasCompetitionInformation
    }
  }
}
"""

GET_CALENDAR_COMPETITION_RESULTS = """
query GetCalendarCompetitionResults($competitionId: Int, $day: Int,
    $eventId: Int) {
  getCalendarCompetitionResults(competitionId: $competitionId, day: $day,
      eventId: $eventId) {
    competition {
      id
      name
      venue
      area
      startDate
      endDate
    }
    eventTitles {
      eventTitle {
        id
        name
        gender
        isRelay
      }
      events {
        event
        eventId
        gender
        isRelay
        perpiId
        phases {
          phase
          phaseId
          combined
          results {
            competitor {
              name
              teamMembers
              aaAthleteId
              iaafId
              urlSlug
              birthDate
              country
              hasProfile
            }
            mark
            wind
            place
            points
            remark
            qualified
            records
            details {
              event
              mark
              wind
              place
              points
              records
            }
          }
        }
      }
    }
  }
}
"""

GET_UPCOMING_COMPETITIONS = """
query GetUpcomingCompetitions($today: String) {
  getUpcomingCompetitions(today: $today) {
    label
    competitions {
      competitionId
      name
      venue
      startDate
      endDate
      dateRange
      isNextEvent
      urlSlug
    }
  }
}
"""

GET_RECENT_RESULTS = """
query GetRecentResults($limit: Int) {
  getRecentResults(limit: $limit) {
    results {
      id
      iaafId
      name
      venue
      startDate
      endDate
      event {
        id
        name
        venue
        countryCode
        countryName
        areaCode
        areaName
        categoryCode
        categoryName
        startDate
        endDate
      }
    }
  }
}
"""

# ── Championship Qualification ─────────────────────────────────────────

GET_CHAMPIONSHIP_QUALIFICATIONS = """
query GetChampionshipQualifications($competitionId: Int, $eventId: Int,
    $country: String, $qualificationType: String) {
  getChampionshipQualifications(competitionId: $competitionId,
      eventId: $eventId, country: $country,
      qualificationType: $qualificationType) {
    qualifications {
      event
      eventId
      gender
      entryStandard
      qualificationType
      qualifiedAthletes {
        athlete
        aaAthleteId
        country
        mark
        venue
        date
        qualificationType
      }
    }
  }
}
"""

GET_LATEST_QUALIFIED_COMPETITORS = """
query GetLatestQualifiedCompetitors($competitionId: Int, $limit: Int,
    $offset: Int) {
  getLatestQualifiedCompetitors(competitionId: $competitionId,
      limit: $limit, offset: $offset) {
    competitors {
      athlete
      aaAthleteId
      country
      event
      mark
      venue
      date
      qualificationType
    }
  }
}
"""

# ── Records ────────────────────────────────────────────────────────────

GET_RECORDS_BY_CATEGORY = """
query GetRecordsDetailByCategory($categoryId: Int) {
  getRecordsDetailByCategory(categoryId: $categoryId) {
    discipline
    gender
    indoor
    mark
    wind
    competitor
    country
    venue
    date
    records
  }
}
"""

GET_RECORDS_CATEGORIES = """
query GetRecordsCategories {
  getRecordsCategories {
    id
    name
  }
}
"""

GET_LATEST_RECORDS = """
query GetLatestRecords($limit: Int, $days: Int) {
  getLatestRecords(limit: $limit, days: $days) {
    records {
      discipline
      gender
      indoor
      mark
      wind
      competitor
      country
      venue
      date
      records
    }
  }
}
"""
