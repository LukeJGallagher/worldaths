"""
KSA Athlete Profile Scraper
Scrapes athlete profiles from World Athletics using Selenium
Captures GraphQL responses for detailed profile data
"""

from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os
import sqlite3
import re
from datetime import datetime

# Configuration
SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
DB_PATH = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'athlete_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wait times (seconds)
PAGE_LOAD_WAIT = 5
TAB_CLICK_WAIT = 2
FINAL_WAIT = 3


def get_ksa_athletes_from_rankings():
    """Get KSA athletes with their IDs from rankings database."""
    rankings_db = os.path.join(SQL_DIR, 'rankings_men_all_events.db')
    women_rankings_db = os.path.join(SQL_DIR, 'rankings_women_all_events.db')

    athletes = {}

    # Men's rankings
    if os.path.exists(rankings_db):
        conn = sqlite3.connect(rankings_db)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT Name, "Profile URL"
            FROM rankings_men_all_events
            WHERE Country LIKE '%KSA%'
        ''')
        for row in cursor.fetchall():
            name, profile_url = row
            if profile_url:
                # Extract athlete ID from URL like: /athletes/saudi-arabia/abdulaziz-abdui-atafi-15017843
                match = re.search(r'-(\d+)$', profile_url)
                if match:
                    athlete_id = match.group(1)
                    # Extract slug from URL
                    slug_match = re.search(r'/athletes/[^/]+/(.+)-\d+$', profile_url)
                    slug = slug_match.group(1) if slug_match else name.lower().replace(' ', '-')
                    athletes[athlete_id] = {
                        'name': name,
                        'slug': slug,
                        'gender': 'men',
                        'profile_url': profile_url
                    }
        conn.close()

    # Women's rankings
    if os.path.exists(women_rankings_db):
        conn = sqlite3.connect(women_rankings_db)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT DISTINCT Name, "Profile URL"
                FROM rankings_women_all_events
                WHERE Country LIKE '%KSA%'
            ''')
            for row in cursor.fetchall():
                name, profile_url = row
                if profile_url:
                    match = re.search(r'-(\d+)$', profile_url)
                    if match:
                        athlete_id = match.group(1)
                        slug_match = re.search(r'/athletes/[^/]+/(.+)-\d+$', profile_url)
                        slug = slug_match.group(1) if slug_match else name.lower().replace(' ', '-')
                        athletes[athlete_id] = {
                            'name': name,
                            'slug': slug,
                            'gender': 'women',
                            'profile_url': profile_url
                        }
        except sqlite3.OperationalError:
            pass
        conn.close()

    return athletes


def scrape_athlete_profile(driver, athlete_id, athlete_slug, athlete_name):
    """Scrape a single athlete's profile data."""

    url = f"https://worldathletics.org/athletes/saudi-arabia/{athlete_slug}-{athlete_id}"
    print(f"\n{'='*60}")
    print(f"Scraping: {athlete_name}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    # Clear previous requests
    del driver.requests

    driver.get(url)
    time.sleep(PAGE_LOAD_WAIT)

    # Get profile image URL from page
    profile_image_url = None
    try:
        img_elem = driver.find_element(By.CSS_SELECTOR, "img.profileBasicInfo_athleteImage__2PIzn")
        profile_image_url = img_elem.get_attribute('src')
    except:
        try:
            img_elem = driver.find_element(By.CSS_SELECTOR, ".athlete-profile img")
            profile_image_url = img_elem.get_attribute('src')
        except:
            pass

    # Click Statistics tab first
    try:
        stats_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Statistics')]"))
        )
        stats_tab.click()
        print("  Clicked 'Statistics' tab")
        time.sleep(TAB_CLICK_WAIT)
    except Exception as e:
        print(f"  Could not click Statistics tab: {e}")

    # Click through all tabs to trigger GraphQL requests
    tab_texts = ["Profile", "Progression", "Honours", "PBs", "SBs", "Rankings"]
    for tab_name in tab_texts:
        try:
            tab = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, f"//button[contains(., '{tab_name}')]"))
            )
            tab.click()
            print(f"  Clicked tab: {tab_name}")
            time.sleep(TAB_CLICK_WAIT)

            # Expand any dropdowns
            dropdowns = driver.find_elements(By.CSS_SELECTOR, "button[aria-haspopup='listbox']")
            for dd in dropdowns[:2]:  # Limit to avoid too many clicks
                try:
                    dd.click()
                    time.sleep(0.5)
                except:
                    continue

            # Click "Show more" buttons
            show_more = driver.find_elements(By.XPATH, "//button[contains(., 'Show more')]")
            for btn in show_more[:2]:
                try:
                    btn.click()
                    time.sleep(0.5)
                except:
                    continue

        except Exception as e:
            print(f"  Could not click tab: {tab_name}")

    # Wait for final responses
    time.sleep(FINAL_WAIT)

    # Collect GraphQL responses
    collected_data = {
        'athlete_id': athlete_id,
        'profile_image_url': profile_image_url,
        'scraped_at': datetime.now().isoformat()
    }

    for request in driver.requests:
        if (
            request.method == "POST"
            and "graphql" in request.url
            and request.response
            and request.response.status_code == 200
        ):
            try:
                body = request.response.body.decode("utf-8", errors="ignore")
                payload = json.loads(body)
                op_name = request.body.decode("utf-8", errors="ignore")

                # Map GraphQL operations to data keys
                op_mapping = {
                    'getSingleCompetitor': 'profile',
                    'getSingleCompetitorPB': 'pbs',
                    'getSingleCompetitorProgression': 'progression',
                    'getSingleCompetitorHonours': 'honours',
                    'getSingleCompetitorWorldRanking': 'rankings',
                    'getSingleCompetitorSeasonBests': 'season_bests',
                    'getSingleCompetitorResultsByLimit': 'results_by_limit'
                }

                for op_key, data_key in op_mapping.items():
                    if op_key in op_name and data_key not in collected_data:
                        collected_data[data_key] = payload
                        print(f"  Captured: {op_key}")

            except Exception as e:
                pass

    return collected_data


def save_athlete_data_to_db(athlete_data, gender):
    """Save scraped athlete data to SQLite database."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    athlete_id = athlete_data.get('athlete_id')
    if not athlete_id:
        conn.close()
        return

    # Extract profile info
    profile_info = athlete_data.get('profile', {}).get('data', {}).get('getSingleCompetitor', {})
    basic_info = profile_info.get('basicData', {}) if profile_info else {}

    full_name = basic_info.get('givenName', '') + ' ' + basic_info.get('familyName', '')
    full_name = full_name.strip() or athlete_data.get('name', '')
    dob = basic_info.get('birthDate', '')
    country = basic_info.get('countryCode', 'KSA')

    # Get primary discipline
    primary_disciplines = profile_info.get('primaryDisciplines', []) if profile_info else []
    primary_event = primary_disciplines[0] if primary_disciplines else ''

    # Update or insert athlete
    cursor.execute('''
        INSERT OR REPLACE INTO ksa_athletes
        (athlete_id, full_name, gender, date_of_birth, primary_event, profile_image_url, country_code, status, last_scraped)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
    ''', (
        athlete_id,
        full_name,
        gender,
        dob,
        primary_event,
        athlete_data.get('profile_image_url'),
        country,
        datetime.now().isoformat()
    ))

    # Save rankings data
    rankings_data = athlete_data.get('rankings', {}).get('data', {}).get('getSingleCompetitorWorldRanking', {})
    if rankings_data:
        # Current rankings
        current = rankings_data.get('current', [])
        for rank_item in current:
            cursor.execute('''
                INSERT INTO athlete_rankings
                (athlete_id, event_name, world_rank, ranking_score, average_score, rank_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                athlete_id,
                rank_item.get('eventGroup', ''),
                rank_item.get('place'),
                rank_item.get('score'),
                rank_item.get('averageScore'),
                datetime.now().strftime('%Y-%m-%d')
            ))

        # Breakdown of ranking scores
        breakdown = rankings_data.get('breakdown', [])
        for item in breakdown:
            cursor.execute('''
                INSERT INTO ranking_breakdown
                (athlete_id, event_name, competition_date, competition_name, result_value,
                 result_score, placing, place_score, performance_score, competition_category, rank_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                athlete_id,
                item.get('eventGroup', ''),
                item.get('date', ''),
                item.get('competition', ''),
                item.get('result', ''),
                item.get('resultScore'),
                item.get('place'),
                item.get('placeScore'),
                item.get('performanceScore'),
                item.get('category', ''),
                datetime.now().strftime('%Y-%m-%d')
            ))

    # Save PBs
    pbs_data = athlete_data.get('pbs', {}).get('data', {}).get('getSingleCompetitorPB', {})
    if pbs_data:
        pbs_list = pbs_data.get('personalBests', [])
        for pb in pbs_list:
            cursor.execute('''
                INSERT INTO athlete_pbs
                (athlete_id, event_name, pb_result, pb_date, pb_venue, is_indoor)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                athlete_id,
                pb.get('discipline', ''),
                pb.get('result', ''),
                pb.get('date', ''),
                pb.get('venue', ''),
                1 if pb.get('indoor') else 0
            ))

    # Save progression
    progression_data = athlete_data.get('progression', {}).get('data', {}).get('getSingleCompetitorProgression', {})
    if progression_data:
        progression_list = progression_data.get('progression', [])
        for prog in progression_list:
            cursor.execute('''
                INSERT INTO athlete_progression
                (athlete_id, event_name, year, best_result, indoor_best)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                athlete_id,
                prog.get('discipline', ''),
                prog.get('year'),
                prog.get('outdoor', ''),
                prog.get('indoor', '')
            ))

    conn.commit()
    conn.close()

    print(f"  Saved to database: {full_name}")


def save_raw_json(athlete_id, data):
    """Save raw JSON data for debugging/backup."""
    filename = os.path.join(OUTPUT_DIR, f"{athlete_id}_data.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved raw JSON: {filename}")


def main():
    print("="*60)
    print("KSA Athlete Profile Scraper")
    print("="*60)

    # Get athletes to scrape
    athletes = get_ksa_athletes_from_rankings()
    print(f"\nFound {len(athletes)} KSA athletes to scrape")

    if not athletes:
        print("No athletes found in rankings database.")
        print("Please run the rankings scraper first (World Rankings_All.py)")
        return

    # Setup Chrome driver
    print("\nStarting Chrome browser...")
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # Uncomment for headless mode:
    # options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    try:
        scraped_count = 0
        failed_count = 0

        for athlete_id, athlete_info in athletes.items():
            try:
                data = scrape_athlete_profile(
                    driver,
                    athlete_id,
                    athlete_info['slug'],
                    athlete_info['name']
                )

                if data:
                    # Add name to data for database
                    data['name'] = athlete_info['name']
                    save_athlete_data_to_db(data, athlete_info['gender'])
                    save_raw_json(athlete_id, data)
                    scraped_count += 1

            except Exception as e:
                print(f"  ERROR scraping {athlete_info['name']}: {e}")
                failed_count += 1
                continue

        print("\n" + "="*60)
        print("SCRAPING COMPLETE")
        print(f"  Successful: {scraped_count}")
        print(f"  Failed: {failed_count}")
        print("="*60)

    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == '__main__':
    main()
