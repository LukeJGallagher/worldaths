"""
Scrape athlete profile photos from worldathletics.org using Playwright.

The GraphQL API returns null for primaryMedia on most athletes,
so we scrape the actual website to get profile photos.

Usage:
    python -m scrapers.scrape_photos              # Scrape all KSA athletes
    python -m scrapers.scrape_photos --athlete-id 14523471  # Single athlete
"""

import argparse
import asyncio
import os
import platform
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped"
PHOTOS_DIR = OUTPUT_DIR / "photos"
WA_BASE = "https://worldathletics.org/athletes"


async def get_ksa_url_slugs() -> list[dict]:
    """Get URL slugs for all KSA athletes via the search API."""
    from api.wa_client import WAClient

    async with WAClient(max_per_second=2.0) as client:
        results = await client.search_all_athletes_for_country("KSA")
        athletes = []
        for r in results:
            slug = r.get("urlSlug")
            aid = r.get("aaAthleteId")
            name = f"{r.get('givenName', '')} {r.get('familyName', '')}".strip()
            if slug and aid:
                athletes.append({
                    "athlete_id": str(aid),
                    "url_slug": slug,
                    "full_name": name,
                })
        return athletes


async def scrape_photo_url(page, url_slug: str) -> str | None:
    """Visit an athlete's WA page and extract the profile photo URL."""
    url = f"{WA_BASE}/{url_slug}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        # Wait for content to load
        await page.wait_for_timeout(2000)

        # Strategy 1: og:image meta tag (most reliable)
        og_image = await page.query_selector('meta[property="og:image"]')
        if og_image:
            content = await og_image.get_attribute("content")
            if content and "worldathletics" in content and "default" not in content.lower():
                return content

        # Strategy 2: Main profile image in the athlete header
        selectors = [
            'img[class*="competitor-profile"]',
            'img[class*="athlete-profile"]',
            'img[class*="headshot"]',
            '.profileImage img',
            '.competitor-header img',
            '[data-testid="athlete-image"] img',
            '.profilePhoto img',
        ]
        for selector in selectors:
            img = await page.query_selector(selector)
            if img:
                src = await img.get_attribute("src")
                if src and "default" not in src.lower() and "placeholder" not in src.lower():
                    return src

        # Strategy 3: First large image in the page header area
        imgs = await page.query_selector_all("img")
        for img in imgs[:10]:  # Only check first 10 images
            src = await img.get_attribute("src")
            alt = await img.get_attribute("alt") or ""
            width = await img.get_attribute("width")
            if src and ("athlete" in alt.lower() or "profile" in alt.lower()):
                return src
            if src and width and int(width) >= 150 and "logo" not in src.lower():
                return src

        return None
    except Exception as e:
        print(f"  Error scraping {url_slug}: {e}")
        return None


async def download_photo(photo_url: str, athlete_id: str) -> str | None:
    """Download a photo and save locally. Returns the local path."""
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine file extension
    ext = ".jpg"
    if ".png" in photo_url.lower():
        ext = ".png"
    elif ".webp" in photo_url.lower():
        ext = ".webp"

    local_path = PHOTOS_DIR / f"{athlete_id}{ext}"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(photo_url)
            if resp.status_code == 200 and len(resp.content) > 1000:  # Skip tiny/placeholder images
                local_path.write_bytes(resp.content)
                return str(local_path)
    except Exception as e:
        print(f"  Download error for {athlete_id}: {e}")

    return None


async def scrape_all_photos(athlete_ids: list[str] | None = None):
    """Scrape photos for all KSA athletes."""
    from playwright.async_api import async_playwright

    print("=" * 60)
    print("Scraping Athlete Photos from worldathletics.org")
    print("=" * 60)

    # Get URL slugs
    print("\n1. Getting athlete URL slugs from search API...")
    athletes = await get_ksa_url_slugs()
    print(f"   Found {len(athletes)} KSA athletes with URL slugs")

    if athlete_ids:
        athletes = [a for a in athletes if a["athlete_id"] in athlete_ids]
        print(f"   Filtered to {len(athletes)} requested athletes")

    if not athletes:
        print("   No athletes to scrape. Exiting.")
        return

    # Launch browser
    print("\n2. Launching browser...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        # Scrape each athlete
        print(f"\n3. Scraping {len(athletes)} athlete pages...")
        results = {}
        for i, ath in enumerate(athletes):
            name = ath["full_name"]
            slug = ath["url_slug"]
            aid = ath["athlete_id"]
            print(f"   [{i+1}/{len(athletes)}] {name} ({slug})...", end="")

            photo_url = await scrape_photo_url(page, slug)

            if photo_url:
                local = await download_photo(photo_url, aid)
                if local:
                    results[aid] = {"photo_url": photo_url, "local_path": local}
                    print(f" OK ({Path(local).name})")
                else:
                    results[aid] = {"photo_url": photo_url, "local_path": None}
                    print(f" URL found but download failed")
            else:
                print(" no photo found")

        await browser.close()

    # Update parquet with photo URLs
    print(f"\n4. Updating ksa_athletes.parquet with photo URLs...")
    parquet_path = OUTPUT_DIR / "ksa_athletes.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        updated = 0
        for _, row in df.iterrows():
            aid = str(row.get("athlete_id", ""))
            if aid in results:
                info = results[aid]
                if info["photo_url"]:
                    df.loc[df["athlete_id"].astype(str) == aid, "photo_url"] = info["photo_url"]
                    updated += 1

        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"   Updated {updated}/{len(df)} athletes with photo URLs")
    else:
        print("   ksa_athletes.parquet not found, skipping update")

    # Summary
    found = sum(1 for v in results.values() if v["photo_url"])
    downloaded = sum(1 for v in results.values() if v["local_path"])
    print(f"\n{'=' * 60}")
    print(f"PHOTO SCRAPE COMPLETE")
    print(f"  Photos found: {found}/{len(athletes)}")
    print(f"  Downloaded:   {downloaded}/{len(athletes)}")
    print(f"  Photo dir:    {PHOTOS_DIR}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Scrape athlete photos")
    parser.add_argument("--athlete-id", nargs="*", help="Specific athlete ID(s) to scrape")
    args = parser.parse_args()

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(scrape_all_photos(athlete_ids=args.athlete_id))


if __name__ == "__main__":
    main()
