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

# ========= CLICK STATISTICS THEN RESULTS TAB =========
try:
    statistics_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='STATISTICS']")))
    driver.execute_script("arguments[0].scrollIntoView(true);", statistics_tab)
    statistics_tab.click()
    time.sleep(2)
    debug_print("[DEBUG] Statistics tab clicked.")
except Exception as e:
    debug_print(f"[ERROR] Could not click Statistics tab: {e}")

try:
    results_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='Results']")))
    driver.execute_script("arguments[0].scrollIntoView(true);", results_tab)
    results_tab.click()
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "profileStatistics_results__1xala")))
    time.sleep(2)
    debug_print("[DEBUG] Results tab clicked and content loaded.")
except Exception as e:
    debug_print(f"[ERROR] Could not click Results tab: {e}")

# ========= SCRAPE RESULTS =========
results = []
try:
    expand_buttons = driver.find_elements(By.CLASS_NAME, "athletesDropdownButton_athletesDropdownButton__3k-Ds")
    for btn in expand_buttons:
        try:
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.5)
        except:
            continue

    rows = driver.find_elements(By.XPATH, "//tbody[contains(@class, 'profileStatistics_tableBody__1w5O9')]/tr")
    i = 0
    while i < len(rows):
        try:
            row = rows[i]
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 3:
                i += 1
                continue
            entry = {
                "Date": cols[2].text.strip(),
                "Discipline": cols[0].text.strip(),
                "Performance": cols[1].text.strip(),
                "AthleteID": athlete_id,
                "AthleteName": bio.get("Name", ""),
                "Wind": "", "Competition": "", "Category": "",
                "Race": "", "Place": "", "Score": "", "Remark": "", "Country": ""
            }
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                if "profileStatistics_trDropdown" in (next_row.get_attribute("class") or ""):
                    try:
                        entry["Country"] = next_row.find_element(By.CLASS_NAME, "Flags_name__28uFw").text.strip()
                    except:
                        entry["Country"] = ""
                    details = next_row.find_elements(By.CLASS_NAME, "athletesEventsDetails_athletesEventsDetails__hU6mX")
                    for d in details:
                        try:
                            label = d.find_element(By.CLASS_NAME, "athletesEventsDetails_athletesEventsDetailsLabel__6KN98").text.strip()
                            value = d.find_element(By.CLASS_NAME, "athletesEventsDetails_athletesEventsDetailsValue__FrHFZ").text.strip()
                            entry[label] = value
                        except:
                            continue
                    i += 1
            results.append(entry)
        except Exception as e:
            debug_print(f"[WARN] Could not parse a result row: {e}")
        i += 1
except Exception as e:
    debug_print(f"[ERROR] Could not parse results section: {e}")

if results:
    result_file = os.path.join(profiles_folder, f"{athlete_id}_results.csv")
    keys = [
        "Date", "Discipline", "Performance", "Wind", "Competition",
        "Category", "Race", "Place", "Score", "Remark",
        "Country", "AthleteID", "AthleteName"
    ]
    with open(result_file, "w", newline="", encoding="utf-8") as rfile:
        writer = csv.DictWriter(rfile, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in keys})
    debug_print(f"[DEBUG] Results saved to {result_file}")
else:
    debug_print("[INFO] No results found to save.")

# ========= CLEANUP =========
driver.quit()
debug_print("[DEBUG] Browser session ended.")
