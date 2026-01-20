from seleniumwire import webdriver
import time
import json
import os

# Setup
athlete_id = "15017843"
athlete_slug = "abdulaziz-abdui-atafi"
output_dir = f"athlete_{athlete_id}_graphql"
os.makedirs(output_dir, exist_ok=True)

# Launch Chrome
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

try:
    url = f"https://worldathletics.org/athletes/saudi-arabia/{athlete_slug}-{athlete_id}"
    print(f"🌐 Opened: {url}")
    driver.get(url)

    print("💡 Manually click through tabs: Statistics > Profile, Progression, Honours, PBs, SBs, Rankings...")
    print("⌛ Waiting for you to finish clicking tabs and dropdowns...")
    time.sleep(60)  # Give yourself time to manually explore tabs

    # Save all GraphQL responses
    seen = set()
    for req in driver.requests:
        if req.method == "POST" and "graphql" in req.url and req.response and req.response.status_code == 200:
            try:
                text = req.body.decode("utf-8", errors="ignore")
                for key in ["getSingleCompetitor", "getSingleCompetitorPB", "getSingleCompetitorProgression",
                            "getSingleCompetitorHonours", "getSingleCompetitorWorldRanking",
                            "getSingleCompetitorSeasonBests", "getSingleCompetitorResultsByLimit"]:
                    if key in text and key not in seen:
                        body = req.response.body.decode("utf-8", errors="ignore")
                        data = json.loads(body)
                        path = os.path.join(output_dir, f"{athlete_id}_{key}.json")
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        seen.add(key)
                        print(f"✅ Saved: {path}")
            except Exception as e:
                print(f"⚠️ Error saving request: {e}")

finally:
    driver.quit()
    print("✅ Done.")
