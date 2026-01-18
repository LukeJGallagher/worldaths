"""
KSA Athlete Profile Scraper v2
Scrapes athlete profiles from World Athletics using standard Selenium
Extracts data directly from page elements
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
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
                match = re.search(r'-(\d+)$', profile_url)
                if match:
                    athlete_id = match.group(1)
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
    """Scrape a single athlete's profile data from page elements."""

    url = f"https://worldathletics.org/athletes/saudi-arabia/{athlete_slug}-{athlete_id}"
    print(f"\n{'='*60}")
    print(f"Scraping: {athlete_name}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    driver.get(url)
    time.sleep(PAGE_LOAD_WAIT)

    collected_data = {
        'athlete_id': athlete_id,
        'name': athlete_name,
        'scraped_at': datetime.now().isoformat()
    }

    # Get profile image URL
    try:
        img_elem = driver.find_element(By.CSS_SELECTOR, "img[alt*='athlete'], .athlete-image img, img.profileBasicInfo_athleteImage__2PIzn")
        collected_data['profile_image_url'] = img_elem.get_attribute('src')
        print(f"  Got profile image")
    except:
        collected_data['profile_image_url'] = None

    # Get basic info from profile header
    try:
        # DOB
        dob_elem = driver.find_elements(By.XPATH, "//*[contains(text(), 'DOB')]/following-sibling::*")
        if dob_elem:
            collected_data['date_of_birth'] = dob_elem[0].text.strip()
    except:
        pass

    # Click Statistics tab
    try:
        stats_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Statistics')]"))
        )
        stats_tab.click()
        print("  Clicked Statistics tab")
        time.sleep(TAB_CLICK_WAIT)
    except:
        print("  Could not find Statistics tab")

    # Get Rankings data
    try:
        rankings_tab = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Rankings')]"))
        )
        rankings_tab.click()
        time.sleep(TAB_CLICK_WAIT)
        print("  Clicked Rankings tab")

        # Extract rankings table
        rankings_data = []
        ranking_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr, .ranking-row")
        for row in ranking_rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 3:
                    rankings_data.append({
                        'event': cells[0].text.strip() if len(cells) > 0 else '',
                        'rank': cells[1].text.strip() if len(cells) > 1 else '',
                        'score': cells[2].text.strip() if len(cells) > 2 else ''
                    })
            except:
                continue

        if rankings_data:
            collected_data['rankings'] = rankings_data
            print(f"  Found {len(rankings_data)} ranking entries")

    except Exception as e:
        print(f"  Rankings tab error: {e}")

    # Get PBs data
    try:
        pbs_tab = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'PBs')]"))
        )
        pbs_tab.click()
        time.sleep(TAB_CLICK_WAIT)
        print("  Clicked PBs tab")

        # Extract PBs table
        pbs_data = []
        pb_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        for row in pb_rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 2:
                    pbs_data.append({
                        'event': cells[0].text.strip() if len(cells) > 0 else '',
                        'result': cells[1].text.strip() if len(cells) > 1 else '',
                        'date': cells[2].text.strip() if len(cells) > 2 else '',
                        'venue': cells[3].text.strip() if len(cells) > 3 else ''
                    })
            except:
                continue

        if pbs_data:
            collected_data['pbs'] = pbs_data
            print(f"  Found {len(pbs_data)} PB entries")

    except Exception as e:
        print(f"  PBs tab error: {e}")

    # Get Progression data
    try:
        prog_tab = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Progression')]"))
        )
        prog_tab.click()
        time.sleep(TAB_CLICK_WAIT)
        print("  Clicked Progression tab")

        # Extract progression table
        prog_data = []
        prog_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        for row in prog_rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 2:
                    prog_data.append({
                        'year': cells[0].text.strip() if len(cells) > 0 else '',
                        'result': cells[1].text.strip() if len(cells) > 1 else '',
                        'indoor': cells[2].text.strip() if len(cells) > 2 else ''
                    })
            except:
                continue

        if prog_data:
            collected_data['progression'] = prog_data
            print(f"  Found {len(prog_data)} progression entries")

    except Exception as e:
        print(f"  Progression tab error: {e}")

    return collected_data


def save_athlete_data_to_db(athlete_data, gender):
    """Save scraped athlete data to SQLite database."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    athlete_id = athlete_data.get('athlete_id')
    if not athlete_id:
        conn.close()
        return

    full_name = athlete_data.get('name', '')
    dob = athlete_data.get('date_of_birth', '')

    # Update athlete record
    cursor.execute('''
        UPDATE ksa_athletes
        SET full_name = ?,
            gender = ?,
            date_of_birth = ?,
            profile_image_url = ?,
            status = 'active',
            last_scraped = ?
        WHERE athlete_id = ?
    ''', (
        full_name,
        gender,
        dob,
        athlete_data.get('profile_image_url'),
        datetime.now().isoformat(),
        athlete_id
    ))

    # If no rows updated, insert new
    if cursor.rowcount == 0:
        cursor.execute('''
            INSERT INTO ksa_athletes
            (athlete_id, full_name, gender, date_of_birth, profile_image_url, status, last_scraped)
            VALUES (?, ?, ?, ?, ?, 'active', ?)
        ''', (
            athlete_id,
            full_name,
            gender,
            dob,
            athlete_data.get('profile_image_url'),
            datetime.now().isoformat()
        ))

    # Save rankings
    rankings = athlete_data.get('rankings', [])
    for rank in rankings:
        try:
            world_rank = int(rank.get('rank', '0').replace(',', '')) if rank.get('rank') else None
            score = float(rank.get('score', '0').replace(',', '')) if rank.get('score') else None

            cursor.execute('''
                INSERT INTO athlete_rankings
                (athlete_id, event_name, world_rank, ranking_score, rank_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                athlete_id,
                rank.get('event', ''),
                world_rank,
                score,
                datetime.now().strftime('%Y-%m-%d')
            ))
        except Exception as e:
            pass

    # Save PBs
    pbs = athlete_data.get('pbs', [])
    for pb in pbs:
        cursor.execute('''
            INSERT INTO athlete_pbs
            (athlete_id, event_name, pb_result, pb_date, pb_venue)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            athlete_id,
            pb.get('event', ''),
            pb.get('result', ''),
            pb.get('date', ''),
            pb.get('venue', '')
        ))

    # Save progression
    progression = athlete_data.get('progression', [])
    for prog in progression:
        try:
            year = int(prog.get('year', '0')) if prog.get('year') else None
            cursor.execute('''
                INSERT INTO athlete_progression
                (athlete_id, event_name, year, best_result, indoor_best)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                athlete_id,
                '',  # Event will be determined by context
                year,
                prog.get('result', ''),
                prog.get('indoor', '')
            ))
        except:
            pass

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
    print("KSA Athlete Profile Scraper v2")
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
    options = Options()
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
