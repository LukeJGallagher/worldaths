"""
Scrape profile images for KSA athletes from World Athletics
"""

import sqlite3
import requests
from bs4 import BeautifulSoup
import os
import time
import re
from datetime import datetime

SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
PROFILES_DB = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')

# Headers to mimic browser request
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}


def extract_athlete_id_from_url(profile_url):
    """Extract athlete ID from World Athletics profile URL."""
    if not profile_url:
        return None
    match = re.search(r'-(\d+)$', profile_url)
    if match:
        return match.group(1)
    return None


def get_profile_image_url(profile_url):
    """Fetch athlete profile page and extract profile image URL."""
    if not profile_url:
        return None

    try:
        response = requests.get(profile_url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for profile image in various possible locations
            # Method 1: Look for img with specific class
            img = soup.find('img', class_='profileBasicInfo_profileImg__3B3hY')
            if img and img.get('src'):
                return img['src']

            # Method 2: Look for avatar image
            img = soup.find('img', {'alt': lambda x: x and 'avatar' in x.lower()})
            if img and img.get('src'):
                return img['src']

            # Method 3: Look for profile image container
            profile_container = soup.find('div', class_=lambda x: x and 'profileImg' in str(x))
            if profile_container:
                img = profile_container.find('img')
                if img and img.get('src'):
                    return img['src']

            # Method 4: Look for any large image near the top
            hero = soup.find('div', class_=lambda x: x and 'hero' in str(x).lower())
            if hero:
                img = hero.find('img')
                if img and img.get('src'):
                    return img['src']

            # Method 5: Search for common athlete image patterns in script tags
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    # Look for imageUrl in JSON data
                    match = re.search(r'"imageUrl"\s*:\s*"([^"]+)"', script.string)
                    if match:
                        return match.group(1)
                    # Look for avatarUrl
                    match = re.search(r'"avatarUrl"\s*:\s*"([^"]+)"', script.string)
                    if match:
                        return match.group(1)

            print(f"  Could not find image on {profile_url}")

    except requests.RequestException as e:
        print(f"  Error fetching {profile_url}: {e}")

    return None


def scrape_profile_images():
    """Scrape profile images for all KSA athletes."""
    print("=" * 60)
    print("Scraping Profile Images from World Athletics")
    print("=" * 60)

    conn = sqlite3.connect(PROFILES_DB)
    cursor = conn.cursor()

    # Get athletes with profile URLs
    cursor.execute("""
        SELECT athlete_id, full_name, profile_url, profile_image_url
        FROM ksa_athletes
        WHERE profile_url IS NOT NULL
    """)
    athletes = cursor.fetchall()

    print(f"Found {len(athletes)} athletes with profile URLs")

    updated = 0
    for athlete_id, name, profile_url, existing_image in athletes:
        # Skip if already has image
        if existing_image:
            print(f"  Skipping {name} (already has image)")
            continue

        print(f"Processing: {name}")
        image_url = get_profile_image_url(profile_url)

        if image_url:
            cursor.execute("""
                UPDATE ksa_athletes
                SET profile_image_url = ?, updated_at = ?
                WHERE athlete_id = ?
            """, (image_url, datetime.now().isoformat(), athlete_id))
            conn.commit()
            updated += 1
            print(f"  [OK] Found image: {image_url[:60]}...")
        else:
            print(f"  [X] No image found")

        # Be respectful with rate limiting
        time.sleep(1)

    print(f"\nUpdated {updated} athlete images")
    conn.close()


def generate_default_avatar_urls():
    """Generate default World Athletics avatar URLs based on athlete IDs."""
    print("\n" + "=" * 60)
    print("Generating Default Avatar URLs")
    print("=" * 60)

    conn = sqlite3.connect(PROFILES_DB)
    cursor = conn.cursor()

    # Get athletes without images but with profile URLs
    cursor.execute("""
        SELECT athlete_id, full_name, profile_url
        FROM ksa_athletes
        WHERE profile_url IS NOT NULL
        AND (profile_image_url IS NULL OR profile_image_url = '')
    """)
    athletes = cursor.fetchall()

    print(f"Found {len(athletes)} athletes needing images")

    # World Athletics uses a standard avatar URL pattern
    # https://worldathletics.org/athletes/headshot/{athleteId}
    updated = 0
    for athlete_id, name, profile_url in athletes:
        wa_id = extract_athlete_id_from_url(profile_url)
        if wa_id:
            # Try the standard avatar URL
            avatar_url = f"https://assets.worldathletics.org/athletes/headshot/{wa_id}.jpg"

            # Verify the URL works
            try:
                response = requests.head(avatar_url, headers=HEADERS, timeout=5)
                if response.status_code == 200:
                    cursor.execute("""
                        UPDATE ksa_athletes
                        SET profile_image_url = ?, updated_at = ?
                        WHERE athlete_id = ?
                    """, (avatar_url, datetime.now().isoformat(), athlete_id))
                    updated += 1
                    print(f"  [OK] {name}: {avatar_url}")
                else:
                    print(f"  [X] {name}: Avatar not found (status {response.status_code})")
            except requests.RequestException as e:
                print(f"  [X] {name}: Error checking URL: {e}")

            time.sleep(0.5)

    conn.commit()
    print(f"\nUpdated {updated} athlete images")
    conn.close()


def print_summary():
    """Print summary of profile images."""
    conn = sqlite3.connect(PROFILES_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM ksa_athletes")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ksa_athletes WHERE profile_image_url IS NOT NULL AND profile_image_url != ''")
    with_images = cursor.fetchone()[0]

    print("\n" + "=" * 60)
    print("PROFILE IMAGE SUMMARY")
    print("=" * 60)
    print(f"Total athletes: {total}")
    print(f"With profile images: {with_images}")
    print(f"Missing images: {total - with_images}")

    if with_images > 0:
        print("\nAthletes with images:")
        cursor.execute("SELECT full_name, profile_image_url FROM ksa_athletes WHERE profile_image_url IS NOT NULL LIMIT 5")
        for name, url in cursor.fetchall():
            print(f"  {name}: {url[:50]}...")

    conn.close()


if __name__ == '__main__':
    # First try scraping from profile pages
    scrape_profile_images()

    # Then try generating default avatar URLs
    generate_default_avatar_urls()

    # Print summary
    print_summary()
