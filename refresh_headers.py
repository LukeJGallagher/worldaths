from seleniumwire import webdriver
import time
import json
import os
import sys

OUTPUT_FILE = "refreshed_headers.json"
ATHLETE_URL = "https://worldathletics.org/athletes/saudi-arabia/abdulaziz-abdui-atafi-15017843"

def build_cookie_header(cookies):
    return "; ".join([f"{c['name']}={c['value']}" for c in cookies])

def extract_headers_and_cookies():
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = webdriver.Chrome(options=options)

        print("🌐 Visiting:", ATHLETE_URL)
        driver.get(ATHLETE_URL)
        time.sleep(12)  # Wait longer if needed

        headers = {}
        for request in driver.requests:
            if request.method == "POST" and "graphql" in request.url:
                headers = dict(request.headers)
                break

        cookies = driver.get_cookies()
        driver.quit()

        if not headers:
            print("❌ No GraphQL request found. Try increasing wait time.")
            sys.exit(1)

        headers["Cookie"] = build_cookie_header(cookies)
        keep_keys = [
            "Content-Type", "User-Agent", "Referer", "Origin",
            "x-api-key", "Accept", "Accept-Encoding", "Accept-Language",
            "Cookie"
        ]
        cleaned = {k: v for k, v in headers.items() if k in keep_keys}

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)

        print(f"✅ Saved refreshed headers to `{OUTPUT_FILE}`")
        sys.exit(0)

    except Exception as e:
        print("❌ Error while refreshing headers:", e)
        sys.exit(1)

if __name__ == "__main__":
    extract_headers_and_cookies()
