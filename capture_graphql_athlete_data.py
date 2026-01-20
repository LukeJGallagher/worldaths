from seleniumwire import webdriver
import time
import json
import os

# === Configuration ===
athlete_id = "15017843"
athlete_slug = "abdulaziz-abdui-atafi"
output_file = f"athlete_{athlete_id}_graphql_live.json"
wait_time = 15  # seconds to allow page and requests to load

# === Setup Chrome ===
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
# options.add_argument("--headless")  # Uncomment if running in background
driver = webdriver.Chrome(options=options)

try:
    # === Load athlete profile page ===
    url = f"https://worldathletics.org/athletes/saudi-arabia/{athlete_slug}-{athlete_id}"
    print(f"🌐 Opening page: {url}")
    driver.get(url)
    time.sleep(wait_time)

    # === Capture only meaningful GraphQL POSTs ===
    found = False
    for request in driver.requests:
        if (
            request.method == "POST"
            and "graphql" in request.url
            and request.response
            and request.response.status_code == 200
            and b"getSingleCompetitor" in request.body  # Key filter
        ):
            try:
                print("📦 Intercepted GraphQL POST with athlete data...")
                body = request.response.body
                data = json.loads(body.decode("utf-8", errors="ignore"))
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"✅ Saved data to: {output_file}")
                found = True
                break
            except Exception as e:
                print("⚠️ Failed to parse GraphQL response:", e)

    if not found:
        print("❌ No matching athlete GraphQL query found. Try increasing wait time or check athlete ID.")

finally:
    driver.quit()
