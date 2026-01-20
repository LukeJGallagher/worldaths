#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, csv, time, json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

# Initialize WebDriver
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)

driver.get(url)

# ========= BIO =========
athlete_id = url.rstrip('/').split('-')[-1]
bio = {'AthleteID': athlete_id}
try:
    name_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".athletesBio_athletesBioTitle__3pPRL")))
    bio['Name'] = name_elem.text.strip()
    debug_print(f"[DEBUG] Extracted Name: {bio['Name']}")
except Exception as e:
    debug_print(f"[ERROR] Error extracting Name: {e}")

for item in driver.find_elements(By.CSS_SELECTOR, ".athletesBio_athletesBioDetails__1wgSI"):
    try:
        label = item.find_element(By.CSS_SELECTOR, ".athletesBio_athletesBioTagLabel__3orD4").text.strip()
        val = item.find_element(By.CSS_SELECTOR, ".athletesBio_athletesBioTagValue__oKZC4").text.strip()
        bio[label] = val
    except Exception:
        continue

csv_bio = os.path.join(profiles_folder, "athlete_bio.csv")
with open(csv_bio, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=bio.keys())
    writer.writeheader()
    writer.writerow(bio)
debug_print(f"[DEBUG] Athlete bio saved to {csv_bio}")

# ========= STATISTICS =========
try:
    stats_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='STATISTICS']")))
    driver.execute_script("arguments[0].click();", stats_btn)
    time.sleep(1)
    rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//table[contains(@class,'profileStatistics_table__1o71p')]/tbody/tr")))
    stats_list = []
    i = 0
    while i < len(rows):
        row = rows[i]
        cls = row.get_attribute('class') or ''
        if 'profileStatistics_trDropdown' not in cls:
            cells = row.find_elements(By.TAG_NAME, 'td')
            if len(cells) >= 4:
                rec = {
                    'AthleteID': athlete_id,
                    'Name': bio.get('Name',''),
                    'Event': cells[0].text.strip(),
                    'Result': cells[1].text.strip(),
                    'Date': cells[2].text.strip()
                }
                # details row
                if i+1 < len(rows) and 'profileStatistics_trDropdown' in (rows[i+1].get_attribute('class') or ''):
                    for d in rows[i+1].find_elements(By.CSS_SELECTOR, '.athletesEventsDetails_athletesEventsDetails__hU6mX'):
                        try:
                            k = d.find_element(By.CSS_SELECTOR, '.athletesEventsDetailsLabel').text.strip()
                            v = d.find_element(By.CSS_SELECTOR, '.athletesEventsDetailsValue').text.strip()
                            rec[k] = v
                        except Exception:
                            continue
                    i += 1
                stats_list.append(rec)
        i += 1
    if stats_list:
        csv_stats = os.path.join(profiles_folder, 'statistics_results.csv')
        keys = set().union(*stats_list)
        with open(csv_stats, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            writer.writerows(stats_list)
        debug_print(f"[DEBUG] Statistics saved to {csv_stats}")
except Exception as e:
    debug_print(f"[ERROR] Error scraping Statistics: {e}")

# ========= PROGRESSION =========
try:
    prog_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='Progression']")))
    driver.execute_script("arguments[0].click();", prog_btn)
    time.sleep(1)
    for blk in driver.find_elements(By.CSS_SELECTOR, '.profileStatistics_statsTable__xU9PN > div'):
        title_nodes = blk.find_elements(By.CSS_SELECTOR, '.profileStatistics_tableName .athletesTitle_athletesTitle__388RT')
        event = title_nodes[0].text.strip() if title_nodes else 'Unknown'
        prog_rows = blk.find_elements(By.CSS_SELECTOR, 'tbody tr')
        prog_data = []
        for r in prog_rows:
            cells = r.find_elements(By.TAG_NAME,'td')
            if len(cells)>=4:
                prog_data.append({
                    'AthleteID': athlete_id,
                    'Event': event,
                    'Year': cells[0].text.strip(),
                    'Performance': cells[1].text.strip(),
                    'Competition': cells[2].text.strip()
                })
        if prog_data:
            path = os.path.join(profiles_folder, f'progression_{event.replace(" ","_")}.csv')
            with open(path,'w',newline='',encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=prog_data[0].keys())
                writer.writeheader()
                writer.writerows(prog_data)
            debug_print(f"[DEBUG] Progression for {event} saved to {path}")
except Exception as e:
    debug_print(f"[ERROR] Error scraping Progression: {e}")

# ========= SEASON'S BESTS =========
try:
    # Parse embedded Next.js data for Apollo state
    nd_elem = driver.find_element(By.CSS_SELECTOR, 'script#__NEXT_DATA__')
    nd_json = nd_elem.get_attribute('innerHTML')
    nd_data = json.loads(nd_json)
    state = nd_data.get('props', {}).get('apolloState', {}).get('data', {})
    if not state:
        debug_print("[WARN] No Apollo state found in __NEXT_DATA__.")
    else:
        seasons = state.get('seasonsBests') or state.get(f"singleCompetitor:{athlete_id}.seasonsBests")
        if seasons and 'results' in seasons:
            sb_entries = []
            for e in seasons['results']:
                sb_entries.append({
                    'AthleteID': athlete_id,
                    'Event': e.get('discipline',''),
                    'Result': e.get('mark',''),
                    'Venue': e.get('venue',''),
                    'Date': e.get('date','')
                })
            if sb_entries:
                sb_file = os.path.join(profiles_folder, 'season_bests.csv')
                with open(sb_file,'w',newline='',encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=sb_entries[0].keys())
                    writer.writeheader()
                    writer.writerows(sb_entries)
                debug_print(f"[DEBUG] Season's Bests saved to {sb_file}")
        else:
            debug_print("[WARN] No Season's Bests results in Apollo state.")
except Exception as e:
    debug_print(f"[ERROR] Error extracting Season's Bests: {e}")

# ========= COMPLETE =========
driver.quit()
