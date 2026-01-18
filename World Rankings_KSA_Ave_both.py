import os
import csv
import time
import sqlite3
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIG ---
rank_date = "2025-04-01"
num_pages = 3
region_type = "countries"
region_code = "ksa"
base_url = "https://worldathletics.org/world-rankings"
save_dir = os.path.join("world_athletics", "Data")
os.makedirs(save_dir, exist_ok=True)
db_file = os.path.join(save_dir, "ksa_modal_results.db")

# --- EVENTS BY GENDER ---
event_lists = {
    "men": [
        "100m", "200m", "400m", "800m", "1500m", "5000m", "10000m",
        "110mh", "400mh", "3000msc", "high-jump", "pole-vault", "long-jump", "triple-jump",
        "shot-put", "discus-throw", "hammer-throw", "javelin-throw", "road-running",
        "marathon", "20km-race-walking", "35km-race-walking", "decathlon", "cross-country",
        "50km-race-walking"
    ],
    "women": [
        "100m", "200m", "400m", "800m", "1500m", "5000m", "10000m",
        "100mh", "400mh", "3000msc", "high-jump", "pole-vault", "long-jump", "triple-jump",
        "shot-put", "discus-throw", "hammer-throw", "javelin-throw", "road-running",
        "marathon", "race-walking", "35km-race-walking", "heptathlon", "cross-country",
        "50km-race-walking"
    ]
}

# --- SCRAPER ---
def scrape_ksa_modals(gender, event_list):
    print(f"\n🎯 Scraping KSA modal data for: {gender.upper()}")
    all_modal_data = []
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 15)

    for event in event_list:
        print(f"\n🔁 Event: {event}")
        for page in range(1, num_pages + 1):
            url = f"{base_url}/{event}/{gender}?regionType={region_type}&region={region_code}&rankDate={rank_date}&page={page}"
            print(f"🌍 Loading: {url}")
            try:
                driver.get(url)
                time.sleep(2)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.records-table")))
            except Exception as e:
                print(f"⚠️ Skipping page {page} — table not found: {e}")
                continue

            rows = driver.find_elements(By.CSS_SELECTOR, "table.records-table tbody tr")
            print(f"✅ Found {len(rows)} athlete rows")

            for i, row in enumerate(rows):
                try:
                    print(f"🔍 Clicking modal row {i+1}")
                    driver.execute_script("arguments[0].click();", row)
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.modal-dialog.modal--ranking-points")))
                    time.sleep(1)

                    modal = driver.find_element(By.CSS_SELECTOR, "div.modal-dialog.modal--ranking-points")
                    athlete_name = modal.find_element(By.CSS_SELECTOR, "h2#mediaLabel").text.strip()

                    try:
                        avg_score = modal.find_element(By.CSS_SELECTOR, ".rank-points__average span").text.strip()
                        rank_score = modal.find_element(By.CSS_SELECTOR, ".rank-points__score span").text.strip()
                    except:
                        avg_score = ""
                        rank_score = ""

                    perf_rows = modal.find_elements(By.CSS_SELECTOR, "table.records-table.dark tbody tr")
                    if not perf_rows:
                        print(f"⚠️ No performance rows for {athlete_name}")

                    for pr in perf_rows:
                        cells = pr.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 12:
                            row_data = {
                                "Gender": gender,
                                "Athlete": athlete_name,
                                "Event Type": event,
                                "Date": cells[0].text.strip(),
                                "Competition": cells[1].text.strip(),
                                "Cnt.": cells[2].text.strip(),
                                "Cat.": cells[3].text.strip(),
                                "Event": cells[4].text.strip(),
                                "Type": cells[5].text.strip(),
                                "Pl.": cells[6].text.strip(),
                                "Result": cells[7].text.strip(),
                                "R.Sc": cells[8].text.strip(),
                                "WR": cells[9].text.strip(),
                                "Pl.Sc": cells[10].text.strip(),
                                "Pf.Sc": cells[11].text.strip(),
                                "Avg Score": avg_score,
                                "Rank Score": rank_score
                            }
                            all_modal_data.append(row_data)
                            print(f"📋 Row {len(all_modal_data)}: {athlete_name} — {cells[0].text.strip()} — {cells[7].text.strip()}")
                        else:
                            print(f"⚠️ Skipping row with insufficient cells")

                    close_btn = modal.find_element(By.CSS_SELECTOR, "button.close")
                    driver.execute_script("arguments[0].click();", close_btn)
                    time.sleep(1)

                except Exception as e:
                    print(f"❌ Modal failed for {gender}, event {event}, row {i+1}: {e}")
                    continue

    driver.quit()

    # --- SAVE ---
    if all_modal_data:
        df = pd.DataFrame(all_modal_data)
        df.dropna(subset=["Athlete", "Date", "Event"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        csv_file = os.path.join(save_dir, f"ksa_modal_results_{gender}.csv")
        df.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"\n📅 Saved {len(df)} rows to {csv_file}")

        conn = sqlite3.connect(db_file)
        table_name = f"ksa_modal_results_{gender}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        print(f"💾 Also saved to SQLite database: {db_file} (table: {table_name})")
    else:
        print(f"⚠️ No data collected for gender: {gender} — nothing written to file or database.")

# --- RUN BOTH GENDERS ---
for gender, events in event_lists.items():
    scrape_ksa_modals(gender, events)
