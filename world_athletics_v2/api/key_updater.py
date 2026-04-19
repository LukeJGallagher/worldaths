"""
Auto-detect World Athletics GraphQL API endpoint and key.

Uses Playwright to intercept network requests from worldathletics.org.
The API endpoint and key rotate every few months. This module detects
the current credentials by loading an athlete page and capturing the
GraphQL requests made by the frontend.

Usage:
    # As a module
    from api.key_updater import detect_credentials, update_env_file
    creds = detect_credentials()
    update_env_file(creds)

    # From command line
    python -m api.key_updater              # Detect and print
    python -m api.key_updater --update     # Detect and update .env
    python -m api.key_updater --test-only  # Test current credentials
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict

import requests

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent / ".env"

# Test URL for validation
TEST_QUERY = '{ searchCompetitors(query: "bakheet", environment: "outdoor") { givenName familyName } }'


def detect_credentials(timeout_ms: int = 15000) -> Optional[Dict[str, str]]:
    """Intercept GraphQL credentials from worldathletics.org using Playwright.

    Returns dict with 'url' and 'key', or None if detection fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None

    api_info = {}

    def handle_request(request):
        url = request.url
        if "graphql" in url.lower() and "worldathletics" in url.lower():
            headers = request.headers
            api_key = headers.get("x-api-key", "")
            if api_key and "url" not in api_info:
                api_info["url"] = url
                api_info["key"] = api_key

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("request", handle_request)

            # Load athlete page that triggers GraphQL calls
            page.goto(
                "https://worldathletics.org/athletes/saudi-arabia/sami-bakheet-15036553",
                timeout=timeout_ms,
            )
            page.wait_for_timeout(5000)
            browser.close()
    except Exception as e:
        logger.error(f"Playwright detection failed: {e}")
        return None

    if api_info.get("url") and api_info.get("key"):
        logger.info(f"Detected endpoint: {api_info['url']}")
        logger.info(f"Detected API key: {api_info['key'][:10]}...")
        return api_info

    logger.warning("No GraphQL requests intercepted")
    return None


def test_credentials(url: str, key: str) -> bool:
    """Test if the given endpoint/key combination works."""
    try:
        r = requests.post(
            url,
            json={"query": TEST_QUERY},
            headers={
                "x-api-key": key,
                "x-amz-user-agent": "aws-amplify/3.0.2",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            if "data" in data and data["data"].get("searchCompetitors"):
                return True
            # Query succeeded but may have field errors - still means key works
            if "errors" in data and "Not Authorized" not in str(data["errors"]):
                return True
        return False
    except Exception as e:
        logger.warning(f"Credential test failed: {e}")
        return False


def get_current_credentials() -> Dict[str, str]:
    """Read current credentials from environment or .env file."""
    url = os.environ.get("WA_GRAPHQL_URL", "")
    key = os.environ.get("WA_API_KEY", "")

    if not url or not key:
        # Try reading .env directly
        if ENV_FILE.exists():
            content = ENV_FILE.read_text()
            url_match = re.search(r"WA_GRAPHQL_URL=(.+)", content)
            key_match = re.search(r"WA_API_KEY=(.+)", content)
            if url_match:
                url = url_match.group(1).strip()
            if key_match:
                key = key_match.group(1).strip()

    return {"url": url, "key": key}


def update_env_file(creds: Dict[str, str]) -> bool:
    """Update .env file with new credentials."""
    if not ENV_FILE.exists():
        logger.warning(f".env file not found at {ENV_FILE}")
        return False

    content = ENV_FILE.read_text()
    new_content = re.sub(
        r"WA_API_KEY=.+",
        f"WA_API_KEY={creds['key']}",
        content,
    )
    new_content = re.sub(
        r"WA_GRAPHQL_URL=.+",
        f"WA_GRAPHQL_URL={creds['url']}",
        new_content,
    )

    ENV_FILE.write_text(new_content)
    logger.info(f"Updated {ENV_FILE}")
    return True


def ensure_valid_credentials() -> Dict[str, str]:
    """Check current credentials and auto-detect new ones if they've expired.

    This is the main entry point for the pipeline. Call this before scraping.
    Returns valid credentials dict with 'url' and 'key'.
    Raises RuntimeError if credentials cannot be obtained.
    """
    current = get_current_credentials()

    # Test current credentials
    if current["url"] and current["key"]:
        if test_credentials(current["url"], current["key"]):
            print(f"  API credentials valid: {current['url'].split('/')[-2]}")
            return current
        print("  Current API credentials expired. Detecting new ones...")

    # Auto-detect new credentials
    new_creds = detect_credentials()
    if not new_creds:
        raise RuntimeError(
            "Failed to detect World Athletics API credentials. "
            "Ensure Playwright + Chromium are installed: "
            "pip install playwright && playwright install chromium"
        )

    # Validate new credentials
    if not test_credentials(new_creds["url"], new_creds["key"]):
        raise RuntimeError(
            f"Detected credentials failed validation: {new_creds['url']}"
        )

    # Update .env file
    update_env_file(new_creds)

    # Update in-memory environment
    os.environ["WA_GRAPHQL_URL"] = new_creds["url"]
    os.environ["WA_API_KEY"] = new_creds["key"]

    print(f"  New API credentials detected and saved: {new_creds['url'].split('/')[-2]}")
    return new_creds


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="World Athletics API key management")
    parser.add_argument("--update", action="store_true", help="Update .env with detected credentials")
    parser.add_argument("--test-only", action="store_true", help="Only test current credentials")
    args = parser.parse_args()

    if args.test_only:
        current = get_current_credentials()
        if current["url"] and current["key"]:
            ok = test_credentials(current["url"], current["key"])
            print(f"Endpoint: {current['url']}")
            print(f"API Key:  {current['key'][:10]}...")
            print(f"Status:   {'VALID' if ok else 'EXPIRED'}")
            sys.exit(0 if ok else 1)
        else:
            print("No credentials found in environment or .env")
            sys.exit(1)

    print("Detecting World Athletics API credentials...")
    creds = detect_credentials()
    if not creds:
        print("FAILED: Could not detect credentials")
        sys.exit(1)

    print(f"Endpoint: {creds['url']}")
    print(f"API Key:  {creds['key']}")

    ok = test_credentials(creds["url"], creds["key"])
    print(f"Valid:    {ok}")

    if args.update and ok:
        update_env_file(creds)
        print(f"Updated:  {ENV_FILE}")
