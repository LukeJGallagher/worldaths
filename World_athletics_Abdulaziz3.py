import os
import csv
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# ========= UTILITY FUNCTIONS =========
def debug_print(msg):
    print(msg)

def ensure_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# ========= SETUP =========
debug_print("=== Starting Full Athlete Profile Scraping with Extended Debugging ===")
profiles_folder = ensure_folder("Profiles")
url = "https://worldathletics.org/athletes/saudi-arabia/abdulaziz-abdui-atafi-15017843"
debug_print(f"[DEBUG] Opening URL: {url}")
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)
actions = ActionChains(driver)

try:
    driver.get(url)
    debug_print("[DEBUG] Page loaded successfully.")
except Exception as e:
    debug_print(f"[ERROR] Error loading the page: {e}")

athlete_id = url.split("-")[-1]
debug_print(f"[DEBUG] Extracted AthleteID: {athlete_id}")

# ========= SCRAPE ATHLETE BIO =========
bio = {
    "Name": "",
    "Country": "",
    "Born": "",
    "Abdulaziz Abdui'S Code": "",
    "AthleteID": athlete_id,
    "ProfileURL": url
}

try:
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "athletesBio_athletesBioTitle__3pPRL")))
    bio_elem = driver.find_element(By.CLASS_NAME, "athletesBio_athletesBioTitle__3pPRL")
    bio["Name"] = bio_elem.text.strip()
    debug_print(f"[DEBUG] Extracted Name (Bio): {bio['Name']}")

    bio_items = driver.find_elements(By.CLASS_NAME, "athletesBio_athletesBioDetails__1wgSI")
    for item in bio_items:
        try:
            label = item.find_element(By.CLASS_NAME, "athletesBio_athletesBioTagLabel__3orD4").text.strip()
            value = item.find_element(By.CLASS_NAME, "athletesBio_athletesBioTagValue__oKZC4").text.strip()
            if label in bio:
                bio[label] = value
        except:
            continue
except Exception as e:
    debug_print(f"[ERROR] Error extracting name: {e}")

bio_file = os.path.join(profiles_folder, f"{athlete_id}_bio.csv")
with open(bio_file, "w", newline="", encoding="utf-8") as bfile:
    writer = csv.DictWriter(bfile, fieldnames=["Name", "Country", "Born", "Abdulaziz Abdui'S Code", "AthleteID", "ProfileURL"])
    writer.writeheader()
    writer.writerow(bio)
debug_print(f"[DEBUG] Athlete bio saved to {bio_file}")

# ========= DOWNLOAD PROFILE PICTURE =========
try:
    img_elem = driver.find_element(By.XPATH, "//div[contains(@class,'athletesBio_athletesBioImage')]/img")
    img_url = img_elem.get_attribute("src")
    debug_print(f"[DEBUG] Found profile image URL: {img_url}")
    resp = requests.get(img_url, timeout=10)
    if resp.status_code == 200:
        img_path = os.path.join(profiles_folder, f"{athlete_id}_profile.jpg")
        with open(img_path, "wb") as img_file:
            img_file.write(resp.content)
        debug_print(f"[DEBUG] Profile picture downloaded to {img_path}")
    else:
        debug_print(f"[WARN] Failed to download image, status code {resp.status_code}")
except Exception as e:
    debug_print(f"[ERROR] Downloading profile picture failed: {e}")

# ========= CLICK STATISTICS TAB =========
try:
    statistics_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='STATISTICS']")))
    driver.execute_script("arguments[0].scrollIntoView(true);", statistics_tab)
    statistics_tab.click()
    time.sleep(2)
    debug_print("[DEBUG] Statistics tab clicked.")
except Exception as e:
    debug_print(f"[ERROR] Could not click Statistics tab: {e}")

# ========= CLICK RESULTS TAB =========
try:
    results_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='athletesTabsButton_AthletesTabsButtonItem__1pPWF']/button[@value='Results']")))
    driver.execute_script("arguments[0].scrollIntoView(true);", results_tab)
    results_tab.click()
    time.sleep(2)
    debug_print("[DEBUG] Results tab clicked.")
except Exception as e:
    debug_print(f"[ERROR] Could not click Results tab: {e}")

# ========= RESULTS TABLE =========
results_data = []
processed_years = set()

canonical_fields = [
    "discipline", "performance", "date", "country", "score", "wind",
    "competition", "category", "race", "place", "remark",
    "athlete_id", "name", "profile_url", "year"
]

def extract_table_rows(year):
    row_count_before = len(results_data)
    rows = driver.find_elements(By.XPATH, "//tbody[contains(@class, 'profileStatistics_tableBody__1w5O9')]/tr")
    i = 0
    while i < len(rows):
        row = rows[i]
        if row.get_attribute("class") and "profileStatistics_trDropdown" in row.get_attribute("class"):
            i += 1
            continue
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) < 3:
            i += 1
            continue
        entry = {
            "discipline": cols[0].text.strip(),
            "performance": cols[1].text.strip(),
            "date": cols[2].text.strip(),
            "country": bio.get("Country", ""),
            "score": "", "wind": "", "competition": "", "category": "",
            "race": "", "place": "", "remark": "",
            "athlete_id": athlete_id,
            "name": bio.get("Name"),
            "profile_url": url,
            "year": year
        }
        if i + 1 < len(rows):
            next_row = rows[i + 1]
            if "profileStatistics_trDropdown" in (next_row.get_attribute("class") or ""):
                detail_elems = next_row.find_elements(By.XPATH, ".//div[contains(@class, 'athletesEventsDetails_athletesEventsDetails__hU6mX')]")
                for d in detail_elems:
                    try:
                        label = d.find_element(By.CLASS_NAME, "athletesEventsDetails_athletesEventsDetailsLabel__6KN98").text.strip().lower()
                        value = d.find_element(By.CLASS_NAME, "athletesEventsDetails_athletesEventsDetailsValue__FrHFZ").text.strip()
                        if label in entry:
                            entry[label] = value
                    except:
                        continue
                i += 1
        results_data.append(entry)
        i += 1
    debug_print(f"[DEBUG] Added {len(results_data) - row_count_before} rows for year {year}")

try:
    debug_print("[DEBUG] Scraping default year (likely 2025) before dropdown interaction.")
    expand_buttons = driver.find_elements(By.XPATH, "//button[contains(@class, 'athletesDropdownButton_athletesDropdownButton__3k-Ds')]")
    for btn in expand_buttons:
        try:
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.2)
        except:
            continue
    extract_table_rows("2025")
    processed_years.add("2025")

    dropdown_control = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".athletesSelectInput__control")))
    dropdown_indicator = dropdown_control.find_element(By.CLASS_NAME, "athletesSelectInput__dropdown-indicator")

    dropdown_indicator.click()
    time.sleep(1)

    year_elements = driver.find_elements(By.CSS_SELECTOR, ".athletesSelectInput__menu-list div")
    available_years = [el.text.strip() for el in year_elements if el.text.strip()]
    debug_print(f"[DEBUG] Year options found: {len(available_years)}")

    for year_text in available_years:
        if year_text in processed_years:
            continue
        try:
            dropdown_control = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".athletesSelectInput__control")))
            dropdown_indicator = dropdown_control.find_element(By.CLASS_NAME, "athletesSelectInput__dropdown-indicator")
            dropdown_indicator.click()
            time.sleep(1)

            year_elements = driver.find_elements(By.CSS_SELECTOR, ".athletesSelectInput__menu-list div")
            for el in year_elements:
                if el.text.strip() == year_text:
                    actions.move_to_element(el).click().perform()
                    time.sleep(2)
                    debug_print(f"[DEBUG] Year selected: {year_text}")
                    expand_buttons = driver.find_elements(By.XPATH, "//button[contains(@class, 'athletesDropdownButton_athletesDropdownButton__3k-Ds')]")
                    for btn in expand_buttons:
                        try:
                            driver.execute_script("arguments[0].click();", btn)
                            time.sleep(0.2)
                        except:
                            continue
                    extract_table_rows(year_text)
                    processed_years.add(year_text)
                    break
        except Exception as e:
            debug_print(f"[WARN] Could not process year: {e}")

except Exception as e:
    debug_print(f"[ERROR] Error processing results by year: {e}")

# ========= WRITE TO RESULTS FILE =========
if results_data:
    results_file = os.path.join(profiles_folder, f"{athlete_id}_results.csv")
    with open(results_file, "w", newline="", encoding="utf-8") as dbfile:
        writer = csv.DictWriter(dbfile, fieldnames=canonical_fields)
        writer.writeheader()
        for row in results_data:
            writer.writerow({k: row.get(k, "") for k in canonical_fields})
    debug_print(f"[DEBUG] Athlete results saved to {results_file}")
else:
    debug_print("[INFO] No results data found.")

# ========= CLEANUP =========
driver.quit()
debug_print("[DEBUG] Browser session ended.")
