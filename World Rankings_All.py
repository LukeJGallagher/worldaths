import os
import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIG ---
rank_date = "2025-04-01"
num_pages = 3  # Pages per event
base_url = "https://worldathletics.org/world-rankings"
save_dir = os.path.join("world_athletics", "Data")
os.makedirs(save_dir, exist_ok=True)

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

# --- SCRAPE FUNCTION ---
def scrape_gender(gender, events):
    print(f"\n🎯 Starting {gender.upper()} events...")
    all_data = []
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 15)

    for event in events:
        print(f"\n🔁 Event: {event}")
        for page in range(1, num_pages + 1):
            url = f"{base_url}/{event}/{gender}?regionType=world&rankDate={rank_date}&limitByCountry=3&page={page}"
            print(f"🌍 Page {page}: {url}")
            try:
                driver.get(url)
                time.sleep(2)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.records-table")))
                table = driver.find_element(By.CSS_SELECTOR, "table.records-table")
                rows = table.find_elements(By.XPATH, ".//tbody/tr")
            except Exception as e:
                print(f"⚠️ Failed to load page: {e}")
                continue

            print(f"✅ Found {len(rows)} rows.")
            for i, row in enumerate(rows, start=1):
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 6:
                    continue
                try:
                    profile_url = row.get_attribute("data-athlete-url")
                    full_url = f"https://worldathletics.org{profile_url}" if profile_url else ""
                    row_data = {
                        "Event Type": event,
                        "Rank": cells[0].text.strip(),
                        "Name": cells[1].text.strip(),
                        "DOB": cells[2].text.strip(),
                        "Country": cells[3].text.strip(),
                        "Score": cells[4].text.strip(),
                        "Event": cells[5].text.strip(),
                        "Profile URL": full_url
                    }
                    all_data.append(row_data)
                    print(f"   ✅ {row_data['Name']} ({row_data['Country']})")
                except Exception as e:
                    print(f"   ❌ Error parsing row {i}: {e}")

    driver.quit()

    # --- SAVE CSV ---
    header = ["Event Type", "Rank", "Name", "DOB", "Country", "Score", "Event", "Profile URL"]
    csv_file = os.path.join(save_dir, f"rankings_{gender}_all_events.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"\n💾 Saved {len(all_data)} rows to {csv_file}")

# --- RUN FOR BOTH GENDERS ---
for gender, events in event_lists.items():
    scrape_gender(gender, events)
