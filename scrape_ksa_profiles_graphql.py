"""
Enhanced KSA Athlete Profile Scraper using World Athletics GraphQL API
Fetches comprehensive athlete data including:
- Profile photos
- All personal bests
- Season bests
- Competition history
- Honors and medals
- World rankings
"""

import asyncio
import sqlite3
import csv
import platform
import os
import re
from typing import List, Dict, Optional, Any
from datetime import datetime

import httpx
from tqdm.asyncio import tqdm_asyncio

# Configuration
COUNTRY_CODE = "KSA"
SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
PROFILES_DB = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')
OUTPUT_CSV = os.path.join(SQL_DIR, f'{COUNTRY_CODE}_complete_profiles.csv')

# GraphQL endpoint and headers
GRAPHQL_URL = "https://graphql-prod-4752.prod.aws.worldathletics.org/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://worldathletics.org",
    "Referer": "https://worldathletics.org/",
    "x-api-key": "da2-qmxd4dl6zfbihixs5ik7uhwor4",
}


async def get_athlete_ids_for_country(client: httpx.AsyncClient, country_code: str) -> List[Dict]:
    """Get list of all athlete IDs and basic info for a country."""
    print(f"Fetching athlete list for country: {country_code}...")
    athletes = []
    page = 0

    search_query = """
    query SearchAthletes($countryCode: String, $page: Int) {
      searchAthletes(countryCode: $countryCode, page: $page) {
        results {
          id
          givenName
          familyName
          iaafId
          urlSlug
          disciplines
          countryCode
        }
      }
    }
    """

    while True:
        variables = {"countryCode": country_code, "page": page}
        try:
            response = await client.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={"query": search_query, "variables": variables},
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("data", {}).get("searchAthletes", {}).get("results", [])

            if not results:
                break

            for athlete in results:
                if athlete and 'id' in athlete:
                    athletes.append({
                        'id': athlete['id'],
                        'given_name': athlete.get('givenName'),
                        'family_name': athlete.get('familyName'),
                        'iaaf_id': athlete.get('iaafId'),
                        'url_slug': athlete.get('urlSlug'),
                        'disciplines': athlete.get('disciplines'),
                        'country_code': athlete.get('countryCode')
                    })

            page += 1
            await asyncio.sleep(0.3)

        except httpx.RequestError as e:
            print(f"Error fetching page {page}: {e}")
            break

    # Remove duplicates by ID
    seen = set()
    unique_athletes = []
    for a in athletes:
        if a['id'] not in seen:
            seen.add(a['id'])
            unique_athletes.append(a)

    print(f"Found {len(unique_athletes)} unique athletes.")
    return unique_athletes


async def fetch_complete_profile(client: httpx.AsyncClient, athlete_id: int) -> Optional[Dict]:
    """Fetch comprehensive athlete profile with all available data."""

    # Comprehensive GraphQL query for full profile
    graphql_query = """
    query GetSingleCompetitor($id: Int!) {
      getSingleCompetitor(id: $id) {
        basicData {
          iaafId
          firstName
          lastName
          fullName
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
          results {
            categoryName
            competition
            discipline
            mark
            place
            venue
            date
          }
        }
        worldRankings {
          current {
            eventGroup
            place
            rankingScore
          }
          best {
            eventGroup
            place
            rankingScore
          }
        }
      }
    }
    """

    variables = {"id": athlete_id}

    try:
        response = await client.post(
            GRAPHQL_URL,
            headers=HEADERS,
            json={"query": graphql_query, "variables": variables},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        profile = data.get("data", {}).get("getSingleCompetitor")
        if not profile:
            return None

        basic = profile.get("basicData", {}) or {}
        media = profile.get("primaryMedia", {}) or {}
        pbs = profile.get("personalBests", {}) or {}
        sbs = profile.get("seasonsBests", {}) or {}
        honours = profile.get("honours", {}) or {}
        rankings = profile.get("worldRankings", {}) or {}

        # Find primary discipline (highest scoring PB)
        primary_discipline = None
        primary_pb = None
        highest_score = -1

        pb_results = pbs.get("results", []) or []
        all_pbs = []
        for pb in pb_results:
            score = pb.get("resultScore") or 0
            all_pbs.append({
                'discipline': pb.get("discipline"),
                'mark': pb.get("mark"),
                'score': score,
                'date': pb.get("date"),
                'venue': pb.get("venue"),
                'wind': pb.get("wind"),
                'indoor': pb.get("indoor", False)
            })
            if score > highest_score:
                highest_score = score
                primary_discipline = pb.get("discipline")
                primary_pb = pb.get("mark")

        # Get current world ranking
        current_rankings = rankings.get("current", []) or []
        best_ranking = None
        best_ranking_score = 0
        for r in current_rankings:
            if r and r.get("rankingScore", 0) > best_ranking_score:
                best_ranking = r.get("place")
                best_ranking_score = r.get("rankingScore", 0)

        # Count honors/medals
        honour_results = honours.get("results", []) or []
        gold_count = sum(1 for h in honour_results if h and str(h.get("place", "")).strip() == "1")
        silver_count = sum(1 for h in honour_results if h and str(h.get("place", "")).strip() == "2")
        bronze_count = sum(1 for h in honour_results if h and str(h.get("place", "")).strip() == "3")

        # Get photo URL
        photo_url = None
        if media and media.get("fileNameUrl"):
            photo_url = media.get("fileNameUrl")

        return {
            "athlete_id": athlete_id,
            "iaaf_id": basic.get("iaafId"),
            "first_name": basic.get("firstName"),
            "last_name": basic.get("lastName"),
            "full_name": basic.get("fullName"),
            "gender": basic.get("sexName", "").lower() if basic.get("sexName") else None,
            "country_code": basic.get("countryCode"),
            "country_name": basic.get("countryName"),
            "birth_date": basic.get("birthDate"),
            "birth_place": basic.get("birthPlace"),
            "biography": basic.get("biography"),
            "url_slug": basic.get("urlSlug"),
            "photo_url": photo_url,
            "primary_discipline": primary_discipline,
            "primary_pb": primary_pb,
            "highest_score": highest_score if highest_score > 0 else None,
            "world_rank": best_ranking,
            "ranking_score": best_ranking_score if best_ranking_score > 0 else None,
            "gold_medals": gold_count,
            "silver_medals": silver_count,
            "bronze_medals": bronze_count,
            "total_medals": gold_count + silver_count + bronze_count,
            "personal_bests": all_pbs,
            "honours": honour_results,
            "season_bests": sbs.get("results", []) or [],
            "scraped_at": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error fetching athlete {athlete_id}: {e}")
        return None


def ensure_db_schema(conn: sqlite3.Connection):
    """Ensure database has all required columns."""
    cursor = conn.cursor()

    # Add new columns if they don't exist
    new_columns = [
        ("iaaf_id", "INTEGER"),
        ("photo_url", "TEXT"),
        ("birth_place", "TEXT"),
        ("biography", "TEXT"),
        ("gold_medals", "INTEGER DEFAULT 0"),
        ("silver_medals", "INTEGER DEFAULT 0"),
        ("bronze_medals", "INTEGER DEFAULT 0"),
        ("total_medals", "INTEGER DEFAULT 0"),
        ("url_slug", "TEXT"),
        ("ranking_score", "REAL"),
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE ksa_athletes ADD COLUMN {col_name} {col_type}")
            print(f"Added column: {col_name}")
        except:
            pass  # Column exists

    conn.commit()


def update_database(conn: sqlite3.Connection, profiles: List[Dict]):
    """Update the athlete profiles database with scraped data."""
    cursor = conn.cursor()

    updated = 0
    added = 0

    for profile in profiles:
        if not profile:
            continue

        athlete_id = str(profile.get("athlete_id") or profile.get("iaaf_id"))
        full_name = profile.get("full_name")

        if not full_name:
            continue

        # Check if athlete exists by name or ID
        cursor.execute(
            "SELECT athlete_id FROM ksa_athletes WHERE full_name = ? OR world_athletics_id = ?",
            (full_name, athlete_id)
        )
        existing = cursor.fetchone()

        # Generate profile image URL (use actual photo or fallback to avatar)
        photo = profile.get("photo_url")
        if not photo:
            photo = f"https://ui-avatars.com/api/?name={full_name.replace(' ', '+')}&background=007167&color=fff&size=128"

        if existing:
            # Update existing
            cursor.execute("""
                UPDATE ksa_athletes SET
                    iaaf_id = COALESCE(?, iaaf_id),
                    date_of_birth = COALESCE(?, date_of_birth),
                    birth_place = COALESCE(?, birth_place),
                    biography = COALESCE(?, biography),
                    primary_event = COALESCE(?, primary_event),
                    profile_image_url = COALESCE(?, profile_image_url),
                    photo_url = ?,
                    best_score = CASE WHEN ? > COALESCE(best_score, 0) THEN ? ELSE best_score END,
                    best_world_rank = CASE WHEN ? IS NOT NULL AND (? < COALESCE(best_world_rank, 9999)) THEN ? ELSE best_world_rank END,
                    ranking_score = CASE WHEN ? > COALESCE(ranking_score, 0) THEN ? ELSE ranking_score END,
                    gold_medals = COALESCE(?, gold_medals),
                    silver_medals = COALESCE(?, silver_medals),
                    bronze_medals = COALESCE(?, bronze_medals),
                    total_medals = COALESCE(?, total_medals),
                    url_slug = COALESCE(?, url_slug),
                    status = 'active',
                    updated_at = ?
                WHERE full_name = ? OR world_athletics_id = ?
            """, (
                profile.get("iaaf_id"),
                profile.get("birth_date"),
                profile.get("birth_place"),
                profile.get("biography"),
                profile.get("primary_discipline"),
                photo,
                profile.get("photo_url"),
                profile.get("highest_score"), profile.get("highest_score"),
                profile.get("world_rank"), profile.get("world_rank"), profile.get("world_rank"),
                profile.get("ranking_score"), profile.get("ranking_score"),
                profile.get("gold_medals"),
                profile.get("silver_medals"),
                profile.get("bronze_medals"),
                profile.get("total_medals"),
                profile.get("url_slug"),
                datetime.now().isoformat(),
                full_name, athlete_id
            ))
            updated += 1
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO ksa_athletes (
                    athlete_id, full_name, gender, date_of_birth, birth_place, biography,
                    primary_event, profile_image_url, photo_url, country_code, status,
                    world_athletics_id, iaaf_id, best_score, best_world_rank, ranking_score,
                    gold_medals, silver_medals, bronze_medals, total_medals, url_slug,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'KSA', 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                athlete_id,
                full_name,
                profile.get("gender"),
                profile.get("birth_date"),
                profile.get("birth_place"),
                profile.get("biography"),
                profile.get("primary_discipline"),
                photo,
                profile.get("photo_url"),
                athlete_id,
                profile.get("iaaf_id"),
                profile.get("highest_score"),
                profile.get("world_rank"),
                profile.get("ranking_score"),
                profile.get("gold_medals"),
                profile.get("silver_medals"),
                profile.get("bronze_medals"),
                profile.get("total_medals"),
                profile.get("url_slug"),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            added += 1

        # Update PBs
        for pb in profile.get("personal_bests", []):
            if not pb.get("discipline"):
                continue
            cursor.execute("""
                INSERT OR REPLACE INTO athlete_pbs (
                    athlete_id, event_name, pb_result, pb_date, pb_venue, is_indoor
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                athlete_id,
                pb.get("discipline"),
                pb.get("mark"),
                pb.get("date"),
                pb.get("venue"),
                1 if pb.get("indoor") else 0
            ))

        # Update rankings
        if profile.get("world_rank"):
            cursor.execute("""
                INSERT OR REPLACE INTO athlete_rankings (
                    athlete_id, event_name, world_rank, ranking_score, rank_date
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                athlete_id,
                profile.get("primary_discipline"),
                profile.get("world_rank"),
                profile.get("ranking_score"),
                datetime.now().strftime('%Y-%m-%d')
            ))

    conn.commit()
    return added, updated


def save_to_csv(profiles: List[Dict], output_path: str):
    """Save profiles to CSV for reference."""
    if not profiles:
        return

    # Flatten for CSV
    csv_data = []
    for p in profiles:
        if not p:
            continue
        csv_data.append({
            'athlete_id': p.get('athlete_id'),
            'iaaf_id': p.get('iaaf_id'),
            'full_name': p.get('full_name'),
            'gender': p.get('gender'),
            'country': p.get('country_code'),
            'birth_date': p.get('birth_date'),
            'birth_place': p.get('birth_place'),
            'primary_event': p.get('primary_discipline'),
            'primary_pb': p.get('primary_pb'),
            'highest_score': p.get('highest_score'),
            'world_rank': p.get('world_rank'),
            'ranking_score': p.get('ranking_score'),
            'gold_medals': p.get('gold_medals'),
            'silver_medals': p.get('silver_medals'),
            'bronze_medals': p.get('bronze_medals'),
            'total_medals': p.get('total_medals'),
            'photo_url': p.get('photo_url'),
            'url_slug': p.get('url_slug'),
            'biography': (p.get('biography') or '')[:200],
            'pb_count': len(p.get('personal_bests', [])),
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Saved {len(csv_data)} profiles to {output_path}")


async def main():
    """Main function to scrape and update athlete profiles."""
    print("=" * 60)
    print("KSA Athlete Profile Scraper - World Athletics GraphQL API")
    print("=" * 60)

    async with httpx.AsyncClient(verify=False) as client:
        # Step 1: Get all KSA athlete IDs
        athletes = await get_athlete_ids_for_country(client, COUNTRY_CODE)

        if not athletes:
            print("No athletes found. Exiting.")
            return

        # Step 2: Fetch complete profiles
        print(f"\nFetching complete profiles for {len(athletes)} athletes...")

        tasks = [fetch_complete_profile(client, a['id']) for a in athletes]
        profiles = await tqdm_asyncio.gather(*tasks, desc="Scraping Profiles")

        # Filter out None results
        valid_profiles = [p for p in profiles if p is not None]
        print(f"\nSuccessfully scraped {len(valid_profiles)} profiles.")

    # Step 3: Update database
    print("\nUpdating database...")
    conn = sqlite3.connect(PROFILES_DB)
    ensure_db_schema(conn)
    added, updated = update_database(conn, valid_profiles)

    # Print summary
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ksa_athletes")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ksa_athletes WHERE photo_url IS NOT NULL AND photo_url != ''")
    with_photos = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ksa_athletes WHERE total_medals > 0")
    with_medals = cursor.fetchone()[0]

    conn.close()

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Athletes added: {added}")
    print(f"Athletes updated: {updated}")
    print(f"Total in database: {total}")
    print(f"With profile photos: {with_photos}")
    print(f"With medals: {with_medals}")

    # Save CSV backup
    save_to_csv(valid_profiles, OUTPUT_CSV)

    print("=" * 60)


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
