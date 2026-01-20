
from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os

athlete_id = "15017843"
athlete_slug = "abdulaziz-abdui-atafi"
wait_time = 3
output_dir = f"athlete_{athlete_id}_graphql"

os.makedirs(output_dir, exist_ok=True)

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

try:
    url = f"https://worldathletics.org/athletes/saudi-arabia/{athlete_slug}-{athlete_id}"
    driver.get(url)
    print(f"🌐 Loaded: {url}")
    time.sleep(5)

    # Click all tabs one by one
    tab_texts = ["Profile", "Progression", "Honours", "PBs", "SBs", "Rankings"]
    for tab_name in tab_texts:
        try:
            tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//button[contains(., '{tab_name}')]"))
            )
            tab.click()
            print(f"🟢 Clicked tab: {tab_name}")
            time.sleep(wait_time)

            # Expand dropdowns or tables if present
            dropdowns = driver.find_elements(By.CSS_SELECTOR, "button[aria-haspopup='listbox']")
            for dd in dropdowns:
                try:
                    dd.click()
                    time.sleep(1)
                except:
                    continue

            # Expand any "Show More" or "+” rows
            show_more = driver.find_elements(By.XPATH, "//button[contains(., 'Show more')]")
            for btn in show_more:
                try:
                    btn.click()
                    time.sleep(1)
                except:
                    continue

        except Exception as e:
            print(f"⚠️ Could not click tab: {tab_name} - {e}")

    # Wait for any late requests to finish
    time.sleep(5)

    # Capture and save GraphQL POST responses
    seen_ops = set()
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
                for key in ["getSingleCompetitor", "getSingleCompetitorPB", "getSingleCompetitorProgression",
                            "getSingleCompetitorHonours", "getSingleCompetitorWorldRanking",
                            "getSingleCompetitorSeasonBests", "getSingleCompetitorResultsByLimit"]:
                    if key in op_name and key not in seen_ops:
                        filename = f"{output_dir}/{athlete_id}_{key}.json"
                        with open(filename, "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2)
                        seen_ops.add(key)
                        print(f"✅ Saved: {filename}")
            except Exception as e:
                print(f"⚠️ Failed to save GraphQL response: {e}")

finally:
    driver.quit()
    print("✅ Done.")

