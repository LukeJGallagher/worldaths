from seleniumwire import webdriver
import time
import json
import os

athlete_id = "15017843"
athlete_slug = "abdulaziz-abdui-atafi"
output_path = f"athlete_{athlete_id}_graphql_live.json"

# Setup browser
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

try:
    # Step 1: Load athlete page and let frontend make GraphQL requests
    url = f"https://worldathletics.org/athletes/saudi-arabia/{athlete_slug}-{athlete_id}"
    print(f"🌐 Loading page: {url}")
    driver.get(url)
    time.sleep(12)

    # Step 2: Intercept GraphQL POST response
    found = False
    for request in driver.requests:
        if (
            request.method == "POST"
            and "graphql" in request.url
            and request.response
            and request.response.status_code == 200
        ):
            try:
                body = request.response.body
                data = json.loads(body.decode("utf-8", errors="ignore"))
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"✅ Captured and saved GraphQL data to: {output_path}")
                found = True
                break
            except Exception as e:
                print("⚠️ Error decoding intercepted response:", e)

    if not found:
        print("❌ No valid GraphQL response found. Try increasing wait time or reloading page.")

finally:
    driver.quit()
