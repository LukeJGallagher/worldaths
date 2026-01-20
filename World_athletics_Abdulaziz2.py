#!/usr/bin/env python
# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os, csv

# ========= UTILITY FUNCTIONS =========
def debug_print(msg):
    print(msg)

def ensure_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# ========= SETUP =========
debug_print("=== Starting Full Athlete Profile Scraping with Extended Debugging ===")

# Create folder "Profiles" (all CSVs will be saved there)
profiles_folder = ensure_folder("Profiles")

url = "https://worldathletics.org/athletes/saudi-arabia/abdulaziz-abdui-atafi-15017843"
debug_print(f"[DEBUG] Opening URL: {url}")
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)

try:
    driver.get(url)
    debug_print("[DEBUG] Page loaded successfully.")
except Exception as e:
    debug_print(f"[ERROR] Error loading the page: {e}")

# Extract AthleteID from URL
athlete_id = url.split("-")[-1]  # Assumes URL ends with athleteID
debug_print(f"[DEBUG] Extracted AthleteID: {athlete_id}")

# ========= SCRAPE ATHLETE BIO =========
bio = {}
try:
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "athletesBio_athletesBioTitle__3pPRL")))
    bio_elem = driver.find_element(By.CLASS_NAME, "athletesBio_athletesBioTitle__3pPRL")
    bio["Name"] = bio_elem.text.strip()
    debug_print(f"[DEBUG] Extracted Name (Bio): {bio['Name']}")
except Exception as e:
    debug_print(f"[ERROR] Error extracting athlete name from bio: {e}")

# Extract additional bio details (e.g. Country, Born, Athlete Code)
bio_details = {}
try:
    bio_items = driver.find_elements(By.CLASS_NAME, "athletesBio_athletesBioDetails__1wgSI")
    debug_print(f"[DEBUG] Found {len(bio_items)} bio detail items.")
    for idx, item in enumerate(bio_items, start=1):
        try:
            label = item.find_element(By.CLASS_NAME, "athletesBio_athletesBioTagLabel__3orD4").text.strip()
            value = item.find_element(By.CLASS_NAME, "athletesBio_athletesBioTagValue__oKZC4").text.strip()
            bio_details[label] = value
            debug_print(f"[DEBUG] Bio detail #{idx}: {label} => {value}")
        except Exception as inner_e:
            debug_print(f"[WARN] Could not extract bio detail in item #{idx}: {inner_e}")
    bio.update(bio_details)
except Exception as e:
    debug_print(f"[ERROR] Error extracting additional bio details: {e}")

# Save athlete bio to CSV (include AthleteID and Name)
bio["AthleteID"] = athlete_id
csv_path_bio = os.path.join(profiles_folder, "athlete_bio.csv")
try:
    with open(csv_path_bio, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(bio.keys()))
        writer.writeheader()
        writer.writerow(bio)
    debug_print(f"[DEBUG] Athlete bio saved to {csv_path_bio}")
except Exception as e:
    debug_print(f"[ERROR] Error saving athlete bio: {e}")

# ========= CLICK STATISTICS TAB =========
try:
    stats_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='STATISTICS']")))
    driver.execute_script("arguments[0].click();", stats_button)
    debug_print("[DEBUG] STATISTICS tab clicked.")
    time.sleep(1)
except Exception as e:
    debug_print(f"[ERROR] Error clicking STATISTICS tab: {e}")

# ========= PROCESS RESULTS (Primary + Details) =========
results_data = []
try:
    # If you want the "Top 10" table, you might need to click that tab first:
    # Uncomment the following block if the "Top 10" button is available.
    """
    try:
        top10_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='Top 10']")))
        driver.execute_script("arguments[0].click();", top10_button)
        debug_print("[DEBUG] Top 10 tab clicked.")
        time.sleep(1)
    except Exception as e:
        debug_print(f"[WARN] Could not click Top 10 tab: {e}")
    """
    # Otherwise, assume we want the default "Results" table.
    debug_print("[DEBUG] Applying Results Filter: setting first filter to 'Date' and second to '2024'.")
    try:
        filter_boxes = driver.find_elements(By.CSS_SELECTOR, "div.athletesSelectInput__value-container")
        debug_print(f"[DEBUG] Found {len(filter_boxes)} filter boxes in results filter container.")
        if filter_boxes:
            # For simplicity, assume the first filter is "Date" and we want the second to be "2024".
            first_filter = filter_boxes[0].text.strip()
            debug_print(f"[DEBUG] First filter currently: {first_filter}")
            # For the second filter, try to extract text; if not found, default to "2024"
            second_filter = filter_boxes[1].text.strip() if len(filter_boxes) > 1 else "2024"
            debug_print(f"[DEBUG] Second filter currently: {second_filter if second_filter else 'Not found (defaulting to 2024)'}")
            # (Here, you might add code to set the second filter if needed)
        else:
            debug_print("[WARN] No filter boxes found in results filter container.")
    except Exception as e:
        debug_print(f"[ERROR] Error applying results filter: {e}")

    debug_print("[DEBUG] Looking for Results table...")
    results_table = wait.until(EC.presence_of_element_located((By.XPATH, "//table[contains(@class,'profileStatistics_table__1o71p')]")))
    results_rows = driver.find_elements(By.XPATH, "//table[contains(@class,'profileStatistics_table__1o71p')]/tbody/tr")
    debug_print(f"[DEBUG] Found {len(results_rows)} rows in Results table.")
    
    i = 0
    while i < len(results_rows):
        row = results_rows[i]
        # Skip dropdown rows by checking if class contains "profileStatistics_trDropdown"
        if "profileStatistics_trDropdown" not in (row.get_attribute("class") or ""):
            cells = row.find_elements(By.TAG_NAME, "td")
            # Adjust your expectation: if primary rows have 4 cells
            if len(cells) >= 4:
                entry = {
                    "Event": cells[0].text.strip(),
                    "Result": cells[1].text.strip(),
                    "Date": cells[2].text.strip(),
                    "AthleteID": athlete_id,
                    "AthleteName": bio.get("Name", "")
                }
                # Check if the next row is a detail row (dropdown)
                if i + 1 < len(results_rows):
                    next_row = results_rows[i+1]
                    if "profileStatistics_trDropdown" in (next_row.get_attribute("class") or ""):
                        detail_elems = next_row.find_elements(By.XPATH, ".//div[contains(@class, 'athletesEventsDetails_athletesEventsDetails__hU6mX')]")
                        details = {}
                        for detail in detail_elems:
                            try:
                                label = detail.find_element(By.XPATH, ".//div[contains(@class, 'athletesEventsDetails_athletesEventsDetailsLabel')]").text.strip()
                                value = detail.find_element(By.XPATH, ".//span[contains(@class, 'athletesEventsDetails_athletesEventsDetailsValue')]").text.strip()
                                details[label] = value
                            except Exception as de:
                                debug_print(f"[WARN] Could not extract a detail: {de}")
                        entry.update(details)
                        i += 1  # Skip the detail row
                results_data.append(entry)
                debug_print(f"[DEBUG] Processed primary result row: {entry}")
            else:
                debug_print(f"[WARN] Results row skipped; expected ≥4 cells but found {len(cells)}. HTML snippet: {row.get_attribute('outerHTML')[:250]}...")
        else:
            debug_print("[DEBUG] Skipping a dropdown detail row.")
        i += 1

    # Save results data to CSV
    results_csv_path = os.path.join(profiles_folder, "results.csv")
    if results_data:
        # Collect all keys from the entries to form the header
        fieldnames = set()
        for entry in results_data:
            fieldnames.update(entry.keys())
        fieldnames = list(fieldnames)
        with open(results_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in results_data:
                writer.writerow(entry)
        debug_print(f"[DEBUG] Results data saved to {results_csv_path}")
    else:
        debug_print("[INFO] No Results data found; writing an empty row.")
        with open(results_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])
except Exception as e:
    debug_print(f"[ERROR] Error processing Results table: {e}")

# ========= PROCESS PROGRESSION =========
progression_data = {}
try:
    debug_print("[DEBUG] Clicking the PROGRESSION tab...")
    progression_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='Progression']")))
    driver.execute_script("arguments[0].click();", progression_button)
    debug_print("[DEBUG] PROGRESSION tab clicked.")
    time.sleep(2)
    
    debug_print("[DEBUG] Waiting for PROGRESSION container to load...")
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "profileStatistics_statsContainer__96c3m")))
    debug_print("[DEBUG] PROGRESSION container loaded.")
    
    progression_blocks = driver.find_elements(By.XPATH, "//div[contains(@class, 'profileStatistics_statsTable__xU9PN')]/div[not(contains(@class, 'profileStatistics_HonneurViewMore'))]")
    debug_print(f"[DEBUG] Found {len(progression_blocks)} progression event blocks.")
    
    for block in progression_blocks:
        try:
            event_name_elem = block.find_element(By.XPATH, ".//div[contains(@class, 'profileStatistics_tableName')]//div[contains(@class, 'athletesTitle_athletesTitle__388RT')]")
            event_name = event_name_elem.text.strip()
        except Exception:
            event_name = "Unknown"
        debug_print(f"[DEBUG] Processing progression for event: '{event_name}'")
        try:
            table = block.find_element(By.XPATH, ".//table[contains(@class, 'profileStatistics_table__1o71p')]")
            rows = table.find_elements(By.XPATH, ".//tbody[contains(@class, 'profileStatistics_tableBody__1w5O9')]/tr")
            debug_print(f"[DEBUG] Event '{event_name}' has {len(rows)} progression rows.")
            event_progression = []
            for r in rows:
                cells = r.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 4:
                    entry = {
                        "Year": cells[0].text.strip(),
                        "Performance": cells[1].text.strip(),
                        "Competition": cells[2].text.strip(),
                        "EventDate": cells[3].text.strip(),
                        "AthleteID": athlete_id,
                        "AthleteName": bio.get("Name", "")
                    }
                    event_progression.append(entry)
                    debug_print(f"[DEBUG] {event_name} progression row: {entry}")
                else:
                    debug_print(f"[WARN] Progression row skipped; expected ≥4 cells but found {len(cells)}.")
            if event_progression:
                progression_csv = os.path.join(profiles_folder, f"world_athletics_progression_{event_name.replace(' ', '_')}.csv")
                with open(progression_csv, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = list(event_progression[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in event_progression:
                        writer.writerow(entry)
                debug_print(f"[DEBUG] Progression data for '{event_name}' saved to {progression_csv}")
                progression_data[event_name] = event_progression
        except Exception as pe:
            debug_print(f"[ERROR] Error processing progression block for event '{event_name}': {pe}")
except Exception as e:
    debug_print(f"[ERROR] Error scraping progression data: {e}")

# ========= PROCESS SEASON'S BESTS =========


# --- Define your target season best years ---
target_years = ["2024", "2025"]
season_bests_data = []

# Loop through each target year for Season's Bests
for year in target_years:
    debug_print(f"[DEBUG] Attempting to select season best year: {year}")
    try:
        # Locate the year dropdown container in the Season's Bests section.
        # (This XPath assumes that the dropdown control is inside a container with a common class.)
        year_dropdown = wait.until(EC.element_to_be_clickable((By.XPATH,
            "//div[contains(@class, 'athletesSelectInput__control')]")))
        driver.execute_script("arguments[0].click();", year_dropdown)
        time.sleep(1)  # Wait a bit for options to appear

        # Look for the option element that exactly matches the year text.
        # Adjust the class in the XPath below if necessary.
        year_option_xpath = f"//div[contains(@class, 'athletesSelectInput__option') and normalize-space(.)='{year}']"
        year_options = driver.find_elements(By.XPATH, year_option_xpath)
        if year_options:
            year_options[0].click()
            debug_print(f"[DEBUG] Selected season best year: {year}")
            time.sleep(2)  # Allow the page to update with season bests for the year
        else:
            debug_print(f"[WARN] Season best year {year} option not found; skipping.")
            continue
    except Exception as e:
        debug_print(f"[ERROR] Error selecting season best year {year}: {e}")
        continue

    # Now extract season best cards for the current year.
    try:
        # Use a broader XPath to get all season best cards.
        season_best_cards = driver.find_elements(By.XPATH,
            "//div[contains(@class, 'profileStatistics_personnalBestCardWrapper__-09Nt')]")
        debug_print(f"[DEBUG] Found {len(season_best_cards)} Season's Best cards for year {year}.")

        for idx, card in enumerate(season_best_cards, start=1):
            entry = {
                "SelectedYear": year,
                "AthleteID": athlete_id,                # athlete_id must have been set earlier
                "AthleteName": bio.get("Name", "")
            }

            # Try to extract the event name inside the card (if available)
            try:
                event_elems = card.find_elements(By.XPATH, ".//div[contains(@class, 'athletesTitle_athletesTitle__388RT')]")
                if event_elems:
                    entry["Event"] = event_elems[0].text.strip()
                    debug_print(f"[DEBUG] Season Best Card {idx} Event: {entry['Event']}")
                else:
                    entry["Event"] = "Unknown"
                    debug_print(f"[WARN] Season Best Card {idx} event name element not found.")
            except Exception as ex:
                entry["Event"] = "Unknown"
                debug_print(f"[WARN] Season Best Card {idx} event extraction error: {ex}")

            # Extract the content (e.g. Competition and EventDate)
            try:
                content_elem = card.find_element(By.CLASS_NAME, "profileStatistics_personnalBestCardContent__1GplY")
                inner_divs = content_elem.find_elements(By.TAG_NAME, "div")
                if inner_divs and len(inner_divs) >= 2:
                    # We assume the first div holds the competition/location and the second holds the date
                    entry["Competition"] = inner_divs[0].text.strip()
                    entry["EventDate"] = inner_divs[1].text.strip()
                else:
                    entry["Competition"] = ""
                    entry["EventDate"] = ""
                debug_print(f"[DEBUG] Season Best Card {idx} Competition: {entry['Competition']}, EventDate: {entry['EventDate']}")
            except Exception as ce:
                entry["Competition"] = ""
                entry["EventDate"] = ""
                debug_print(f"[ERROR] Error extracting content from Season Best Card {idx}: {ce}")

            # Extract footer details like Performance and Score
            try:
                footer_elem = card.find_element(By.CLASS_NAME, "profileStatistics_personnalBestCardFooter__1DNo8")
                detail_blocks = footer_elem.find_elements(By.XPATH, ".//div[contains(@class, 'athletesEventsDetails_')]")
                debug_print(f"[DEBUG] Season Best Card {idx} found {len(detail_blocks)} footer detail blocks.")
                for detail in detail_blocks:
                    try:
                        label_elem = detail.find_element(By.XPATH, ".//div[contains(@class, 'athletesEventsDetailsLabel')]")
                        label = label_elem.text.strip()
                    except Exception:
                        label = "Unknown"
                    try:
                        value_elem = detail.find_element(By.XPATH, ".//span[contains(@class, 'athletesEventsDetailsValue')]")
                        value = value_elem.text.strip()
                    except Exception:
                        value = ""
                    entry[label] = value
                debug_print(f"[DEBUG] Processed Season Best Card {idx}: {entry}")
            except Exception as fe:
                debug_print(f"[ERROR] Error extracting footer from Season Best Card {idx}: {fe}")

            season_bests_data.append(entry)
    except Exception as e:
        debug_print(f"[ERROR] Error extracting season best cards for year {year}: {e}")

# Save the combined season bests data
season_bests_csv = os.path.join(profiles_folder, "world_athletics_season_bests.csv")
if season_bests_data:
    # Gather the union of all keys across entries for CSV header
    fieldnames = sorted({key for entry in season_bests_data for key in entry.keys()})
    with open(season_bests_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in season_bests_data:
            writer.writerow(entry)
    debug_print(f"[DEBUG] Combined Season Bests data saved to {season_bests_csv}")
else:
    debug_print("[INFO] No Season Bests data found; writing an empty CSV.")
    with open(season_bests_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])



# ========= PROCESS WORLD RANKINGS =========
world_rankings_data = []
try:
    debug_print("[DEBUG] Clicking the WORLD RANKINGS tab...")
    rankings_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='World rankings']")))
    driver.execute_script("arguments[0].click();", rankings_button)
    debug_print("[DEBUG] WORLD RANKINGS tab clicked.")
    time.sleep(2)
    
    debug_print("[DEBUG] Waiting for world rankings container to load...")
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "profileStatistics_rankingWrapper__1203x")))
    debug_print("[DEBUG] World rankings container loaded.")
    
    ranking_cards = driver.find_elements(By.XPATH, "//div[contains(@class, 'profileStatistics_rankingCard__3vkmx')]")
    debug_print(f"[DEBUG] Found {len(ranking_cards)} ranking cards.")
    for idx, card in enumerate(ranking_cards, start=1):
        entry = {"AthleteID": athlete_id, "AthleteName": bio.get("Name", "")}
        try:
            event_elem = card.find_element(By.XPATH, ".//a[contains(@class, 'profileStatistics_rankingCardTitle')]")
            entry["Event"] = event_elem.text.strip()
        except Exception as e:
            entry["Event"] = "Unknown"
            debug_print(f"[WARN] Ranking Card {idx} missing event: {e}")
        detail_blocks = card.find_elements(By.XPATH, ".//div[contains(@class, 'athletesEventsDetails_athletesEventsDetails__hU6mX')]")
        for detail in detail_blocks:
            try:
                label = detail.find_element(By.XPATH, ".//div[contains(@class, 'athletesEventsDetails_athletesEventsDetailsLabel')]").text.strip()
            except Exception:
                label = "Unknown"
            try:
                value = detail.find_element(By.XPATH, ".//span[contains(@class, 'athletesEventsDetails_athletesEventsDetailsValue')]").text.strip()
            except Exception:
                value = ""
            entry[label] = value
        world_rankings_data.append(entry)
        debug_print(f"[DEBUG] Processed Ranking Card {idx}: {entry}")
    
    rankings_csv = os.path.join(profiles_folder, "world_athletics_world_rankings.csv")
    if world_rankings_data:
        # Build fieldnames as union of all keys
        all_keys = set()
        for entry in world_rankings_data:
            all_keys.update(entry.keys())
        all_keys = list(all_keys)
        with open(rankings_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_keys)
            writer.writeheader()
            for entry in world_rankings_data:
                writer.writerow(entry)
        debug_print(f"[DEBUG] World rankings data saved to {rankings_csv}")
    else:
        debug_print("[INFO] No world rankings data found; writing an empty row.")
        with open(rankings_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])
except Exception as e:
    debug_print(f"[ERROR] Error scraping world rankings data: {e}")

# ========= CLEANUP =========
driver.quit()
debug_print("[DEBUG] Browser session ended.")
