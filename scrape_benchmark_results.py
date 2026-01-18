"""
Benchmark Results Scraper
Scrapes top results from major championships for comparison with KSA athletes
Target competitions: World Championships, Asian Games, Asian Championships, GCC competitions
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import sqlite3
import os
from datetime import datetime

# Configuration
SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
DB_PATH = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')

# Major competitions to scrape (competition ID, name, type)
MAJOR_COMPETITIONS = [
    # World Championships
    ('7174037', 'World Athletics Championships Budapest 2023', 'World Championships'),
    ('7137507', 'World Athletics Championships Oregon 2022', 'World Championships'),

    # Asian Games
    ('7186657', 'Asian Games Hangzhou 2023', 'Asian Games'),

    # Asian Championships
    ('7186656', 'Asian Athletics Championships Bangkok 2023', 'Asian Championships'),

    # GCC - These IDs need to be found
    # Add GCC competition IDs here when available
]

# Events to track (must match World Athletics event IDs)
TRACK_EVENTS = [
    '100m', '200m', '400m', '800m', '1500m', '5000m', '10000m',
    '110mh', '100mh', '400mh', '3000msc',
    'high-jump', 'pole-vault', 'long-jump', 'triple-jump',
    'shot-put', 'discus-throw', 'hammer-throw', 'javelin-throw',
    'marathon', '20km-race-walking'
]


def scrape_competition_results(driver, competition_id, competition_name, competition_type):
    """Scrape results from a specific competition."""

    print(f"\n{'='*60}")
    print(f"Scraping: {competition_name}")
    print(f"Type: {competition_type}")
    print(f"{'='*60}")

    results = []
    base_url = f"https://worldathletics.org/competition/calendar-results/results/{competition_id}"

    driver.get(base_url)
    time.sleep(3)

    # Find all event links
    try:
        event_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/results/']")
        event_urls = list(set([link.get_attribute('href') for link in event_links if '/results/' in link.get_attribute('href')]))
        print(f"Found {len(event_urls)} event pages")
    except Exception as e:
        print(f"Error finding events: {e}")
        return results

    for event_url in event_urls[:20]:  # Limit to prevent very long scrapes
        try:
            driver.get(event_url)
            time.sleep(2)

            # Get event name from page
            try:
                event_name = driver.find_element(By.CSS_SELECTOR, "h1, .event-title").text
            except:
                event_name = "Unknown Event"

            # Get round info
            try:
                round_elem = driver.find_element(By.CSS_SELECTOR, ".round-name, .phase-name")
                round_name = round_elem.text
            except:
                round_name = "Final"

            # Get results table
            try:
                rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr, .results-table tr")

                for row in rows[:10]:  # Top 10 per event
                    try:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 4:
                            place = cells[0].text.strip()
                            athlete_name = cells[1].text.strip()
                            country = cells[2].text.strip() if len(cells) > 2 else ''
                            result = cells[3].text.strip() if len(cells) > 3 else ''

                            if place and athlete_name:
                                results.append({
                                    'competition_name': competition_name,
                                    'competition_date': '',  # Will be set from page
                                    'event_name': event_name,
                                    'round': round_name,
                                    'place': int(place) if place.isdigit() else None,
                                    'athlete_name': athlete_name,
                                    'country': country,
                                    'result_value': result
                                })
                    except Exception as row_error:
                        continue

                print(f"  {event_name}: {len(results)} results")

            except Exception as table_error:
                print(f"  Could not parse results table: {table_error}")

        except Exception as e:
            print(f"  Error loading event page: {e}")
            continue

    return results


def scrape_world_rankings_top_performers(driver, event, gender='men', top_n=20):
    """Scrape top performers from world rankings for a specific event."""

    print(f"\n  Scraping top {top_n} in {event} ({gender})...")

    results = []
    url = f"https://worldathletics.org/world-rankings/{event}/{gender}"

    driver.get(url)
    time.sleep(3)

    try:
        # Find results table
        rows = driver.find_elements(By.CSS_SELECTOR, "table.records-table tbody tr")

        for i, row in enumerate(rows[:top_n]):
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 5:
                    rank = cells[0].text.strip()
                    name = cells[1].text.strip()
                    dob = cells[2].text.strip() if len(cells) > 2 else ''
                    country = cells[3].text.strip() if len(cells) > 3 else ''
                    score = cells[4].text.strip() if len(cells) > 4 else ''

                    results.append({
                        'competition_name': f'World Rankings - {event.upper()}',
                        'competition_date': datetime.now().strftime('%Y-%m-%d'),
                        'event_name': event,
                        'round': 'Current Ranking',
                        'place': int(rank) if rank.isdigit() else i + 1,
                        'athlete_name': name,
                        'country': country,
                        'result_value': score
                    })

            except Exception as e:
                continue

        print(f"    Found {len(results)} athletes")

    except Exception as e:
        print(f"    Error: {e}")

    return results


def save_benchmark_results(results):
    """Save benchmark results to database."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Clear existing benchmark data (refresh)
    cursor.execute('DELETE FROM benchmark_results')

    for result in results:
        cursor.execute('''
            INSERT INTO benchmark_results
            (competition_name, competition_date, event_name, round, place, athlete_name, country, result_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('competition_name'),
            result.get('competition_date'),
            result.get('event_name'),
            result.get('round'),
            result.get('place'),
            result.get('athlete_name'),
            result.get('country'),
            result.get('result_value')
        ))

    conn.commit()
    conn.close()

    print(f"\nSaved {len(results)} benchmark results to database")


def main():
    print("="*60)
    print("Benchmark Results Scraper")
    print("="*60)

    # Setup Chrome driver
    print("\nStarting Chrome browser...")
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    all_results = []

    try:
        # Scrape world rankings top performers for key events
        print("\n" + "="*60)
        print("SCRAPING WORLD RANKINGS TOP PERFORMERS")
        print("="*60)

        key_events = ['100m', '200m', '400m', '800m', '1500m', '110mh', '400mh', 'long-jump', 'high-jump']

        for event in key_events:
            for gender in ['men', 'women']:
                results = scrape_world_rankings_top_performers(driver, event, gender, top_n=20)
                all_results.extend(results)

        # Save all results
        if all_results:
            save_benchmark_results(all_results)

        print("\n" + "="*60)
        print("SCRAPING COMPLETE")
        print(f"  Total benchmark results: {len(all_results)}")
        print("="*60)

    except Exception as e:
        print(f"Error during scraping: {e}")

    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == '__main__':
    main()
