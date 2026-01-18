import os
import csv
import time
import sqlite3
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

# --- CONFIG ---
base_url = "https://worldathletics.org/stats-zone/road-to/7190593"
save_dir = os.path.join("world_athletics", "Data", "athlete_performances")
os.makedirs(save_dir, exist_ok=True)
db_file = os.path.join(save_dir, "athlete_performances.db")

def setup_driver():
    """Setup Chrome driver with options"""
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    # Uncomment the next line to run headless
    # chrome_options.add_argument("--headless")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_user_selections():
    """Get user selections for event and country"""
    print("\n" + "="*60)
    print("🎯 ATHLETE PERFORMANCE DATA SCRAPER")
    print("="*60)
    print("This scraper extracts detailed performance data from athlete modals")
    print("="*60)
    
    # Show available events
    print("\n📋 AVAILABLE EVENTS:")
    print("-" * 60)
    
    # Complete list of events from the dropdown
    available_events = [
        ("10229509", "Women's 100 Metres"),
        ("10229510", "Women's 200 Metres"),
        ("10229511", "Women's 400 Metres"),
        ("10229512", "Women's 800 Metres"),
        ("10229513", "Women's 1500 Metres"),
        ("10229514", "Women's 5000 Metres"),
        ("10229521", "Women's 10,000 Metres"),
        ("10229522", "Women's 100 Metres Hurdles"),
        ("10229523", "Women's 400 Metres Hurdles"),
        ("10229524", "Women's 3000 Metres Steeplechase"),
        ("10229526", "Women's High Jump"),
        ("10229527", "Women's Pole Vault"),
        ("10229528", "Women's Long Jump"),
        ("10229529", "Women's Triple Jump"),
        ("10229530", "Women's Shot Put"),
        ("10229531", "Women's Discus Throw"),
        ("10229532", "Women's Hammer Throw"),
        ("10229533", "Women's Javelin Throw"),
        ("10229534", "Women's Marathon"),
        ("10229535", "Women's 20 Kilometres Race Walk"),
        ("10229989", "Women's 35 Kilometres Race Walk"),
        ("10229536", "Women's Heptathlon"),
        ("204594", "Women's 4x100 Metres Relay"),
        ("204596", "Women's 4x400 Metres Relay"),
        ("10229630", "Men's 100 Metres"),
        ("10229605", "Men's 200 Metres"),
        ("10229631", "Men's 400 Metres"),
        ("10229501", "Men's 800 Metres"),
        ("10229502", "Men's 1500 Metres"),
        ("10229609", "Men's 5000 Metres"),
        ("10229610", "Men's 10,000 Metres"),
        ("10229611", "Men's 110 Metres Hurdles"),
        ("10229612", "Men's 400 Metres Hurdles"),
        ("10229614", "Men's 3000 Metres Steeplechase"),
        ("10229615", "Men's High Jump"),
        ("10229616", "Men's Pole Vault"),
        ("10229617", "Men's Long Jump"),
        ("10229618", "Men's Triple Jump"),
        ("10229619", "Men's Shot Put"),
        ("10229620", "Men's Discus Throw"),
        ("10229621", "Men's Hammer Throw"),
        ("10229636", "Men's Javelin Throw"),
        ("10229634", "Men's Marathon"),
        ("10229508", "Men's 20 Kilometres Race Walk"),
        ("10229627", "Men's 35 Kilometres Race Walk"),
        ("10229629", "Men's Decathlon"),
        ("204593", "Men's 4x100 Metres Relay"),
        ("204595", "Men's 4x400 Metres Relay"),
        ("10229988", "Mixed 4x400 Metres Relay")
    ]
    
    # Display events in organized format
    print("\n🏃‍♀️ WOMEN'S EVENTS:")
    for event_id, event_name in available_events:
        if "Women's" in event_name:
            print(f"   {event_id} - {event_name}")
    
    print("\n🏃‍♂️ MEN'S EVENTS:")
    for event_id, event_name in available_events:
        if "Men's" in event_name:
            print(f"   {event_id} - {event_name}")
    
    print("\n🏃‍♂️🏃‍♀️ MIXED EVENTS:")  
    for event_id, event_name in available_events:
        if "Mixed" in event_name:
            print(f"   {event_id} - {event_name}")
    
    print("\n" + "-" * 60)
    
    # Event selection with validation
    while True:
        event_id = input("\n📋 Enter Event ID (or 'list' to see events again): ").strip()
        
        if event_id.lower() == 'list':
            # Show compact list
            print("\nQUICK REFERENCE:")
            for event_id_ref, event_name in available_events:
                print(f"   {event_id_ref} - {event_name}")
            continue
        
        # Validate event ID
        valid_event_ids = [e[0] for e in available_events]
        if event_id in valid_event_ids:
            selected_event_name = next(e[1] for e in available_events if e[0] == event_id)
            print(f"✅ Selected: {event_id} - {selected_event_name}")
            break
        else:
            print(f"❌ Invalid Event ID: {event_id}")
            print("Please enter a valid Event ID from the list above.")
    
    # Country selection
    print("\n🌍 COUNTRY SELECTION:")
    print("="*40)
    print("POPULAR COUNTRIES:")
    popular_countries = [
        ("usa", "United States"),
        ("gbr", "Great Britain"),
        ("aus", "Australia"), 
        ("can", "Canada"),
        ("fra", "France"),
        ("ger", "Germany"),
        ("jpn", "Japan"),
        ("ksa", "Saudi Arabia"),
        ("chn", "China"),
        ("jam", "Jamaica"),
        ("ken", "Kenya"),
        ("eth", "Ethiopia"),
        ("ita", "Italy"),
        ("esp", "Spain"),
        ("ned", "Netherlands")
    ]
    
    for code, name in popular_countries:
        print(f"   {code} - {name}")
    
    print("\nOTHER OPTIONS:")
    print("   all - All countries (comprehensive scraping)")
    print("   Or enter any other 3-letter country code")
    
    while True:
        country_code = input("\n🌍 Enter Country Code (or 'all'): ").strip().lower()
        
        if country_code == 'all':
            print("✅ Selected: All Countries (this may take a long time)")
            break
        elif len(country_code) == 3 and country_code.isalpha():
            # Find country name if it's in popular list
            country_name = next((name for code, name in popular_countries if code == country_code), country_code.upper())
            print(f"✅ Selected: {country_code.upper()} - {country_name}")
            break
        else:
            print("❌ Please enter a valid 3-letter country code or 'all'")
    
    return event_id, country_code

def get_available_events(driver):
    """Get all available events from dropdown for reference"""
    try:
        event_dropdown = driver.find_element(By.CSS_SELECTOR, "select#eventId")
        select_obj = Select(event_dropdown)
        options = select_obj.options
        
        events_info = []
        for option in options:
            value = option.get_attribute("value")
            text = option.text.strip()
            if value and text and text.lower() != "discipline":
                events_info.append(f"{value} - {text}")
        
        return events_info
    except:
        return []

def get_available_countries(driver):
    """Get all available countries from dropdown for reference"""
    try:
        # Look for country dropdown
        country_selectors = [
            "select[id*='country']", "select[id*='federation']", 
            ".country-selector select", ".federation-selector select"
        ]
        
        for selector in country_selectors:
            try:
                country_dropdown = driver.find_element(By.CSS_SELECTOR, selector)
                select_obj = Select(country_dropdown)
                options = select_obj.options
                
                countries_info = []
                for option in options:
                    value = option.get_attribute("value")
                    text = option.text.strip()
                    if value and text and text.lower() not in ["all", "select", ""]:
                        countries_info.append(f"{value} - {text}")
                
                return countries_info
            except:
                continue
        
        return []
    except:
        return []

def scrape_athlete_modal_data(driver, row, athlete_name, event_id, event_name, country_code):
    """Click on a row and extract modal data"""
    modal_data = []
    
    try:
        print(f"   🔍 Clicking modal for: {athlete_name}")
        
        # Click the row to open modal
        ActionChains(driver).click(row).perform()
        time.sleep(2)
        
        # Wait for modal to appear
        wait = WebDriverWait(driver, 10)
        modal = wait.until(EC.presence_of_element_located((
            By.CSS_SELECTOR, 
            ".RankingScoreCalculationModal_container__3BidC, .modal-dialog, [class*='modal']"
        )))
        
        # Extract athlete name from modal header
        try:
            modal_athlete_name = modal.find_element(
                By.CSS_SELECTOR, 
                ".RankingScoreCalculationModal_title__1XN80, .modal-title, h2, h3"
            ).text.strip()
            print(f"   ✅ Modal opened for: {modal_athlete_name}")
        except:
            modal_athlete_name = athlete_name
        
        # Extract profile URL if available
        profile_url = ""
        try:
            profile_link = modal.find_element(By.CSS_SELECTOR, "a[href*='/athletes/']")
            profile_url = profile_link.get_attribute("href")
            
            # Extract athlete ID from profile URL
            athlete_id = ""
            if "/athletes/" in profile_url:
                athlete_id = profile_url.split("/athletes/")[-1].split("-")[-1]
            
            print(f"   🔗 Profile URL: {profile_url}")
        except:
            athlete_id = ""
            print("   ⚠️ No profile URL found")
        
        # Extract performance table data
        try:
            table = modal.find_element(By.CSS_SELECTOR, "table")
            headers = []
            
            # Get headers
            header_elements = table.find_elements(By.CSS_SELECTOR, "thead th, thead td, th")
            headers = [h.text.strip() for h in header_elements]
            print(f"   📊 Table headers: {headers}")
            
            # Get data rows
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr, tr")
            print(f"   📋 Found {len(rows)} performance rows")
            
            for i, perf_row in enumerate(rows):
                try:
                    cells = perf_row.find_elements(By.TAG_NAME, "td")
                    if len(cells) > 0:
                        # Create performance record
                        perf_data = {
                            "Event_ID": event_id,
                            "Event_Name": event_name,
                            "Country_Code": country_code,
                            "Athlete_Name": modal_athlete_name,
                            "Athlete_ID": athlete_id,
                            "Profile_URL": profile_url,
                            "Performance_Index": i + 1,
                            "Scrape_Timestamp": pd.Timestamp.now().isoformat()
                        }
                        
                        # Add cell data with headers
                        for j, cell in enumerate(cells):
                            cell_text = cell.text.strip()
                            if j < len(headers) and headers[j]:
                                col_name = headers[j].replace(".", "").replace(" ", "_")
                            else:
                                col_name = f"Column_{j+1}"
                            
                            perf_data[col_name] = cell_text
                        
                        modal_data.append(perf_data)
                        
                        # Print first few performances
                        if i < 3:
                            key_info = [cell.text.strip()[:15] for cell in cells[:4]]
                            print(f"     📋 Performance {i+1}: {key_info}")
                
                except Exception as e:
                    print(f"     ❌ Error processing performance row {i+1}: {e}")
                    continue
        
        except Exception as e:
            print(f"   ❌ Error extracting table data: {e}")
        
        # Extract ranking scores from footer
        try:
            footer_elements = modal.find_elements(By.CSS_SELECTOR, ".RankingScoreCalculationModal_footer__2xjDo p, .modal-footer p")
            for element in footer_elements:
                text = element.text.strip()
                if "Average of Performance Scores:" in text or "Ranking Score:" in text:
                    print(f"   📊 {text}")
                    
                    # Add ranking scores to all performance records
                    if "Average of Performance Scores:" in text:
                        avg_score = text.split(":")[-1].strip()
                        for record in modal_data:
                            record["Average_Performance_Score"] = avg_score
                    elif "Ranking Score:" in text:
                        ranking_score = text.split(":")[-1].strip()
                        for record in modal_data:
                            record["Ranking_Score"] = ranking_score
        except:
            pass
        
        # Close modal
        try:
            close_button = modal.find_element(By.CSS_SELECTOR, 
                ".RankingScoreCalculationModal_close__3wzIX, .close, [class*='close']")
            ActionChains(driver).click(close_button).perform()
            time.sleep(1)
            print(f"   ✅ Modal closed")
        except:
            # Try pressing Escape key
            try:
                ActionChains(driver).send_keys_to_element(modal, '\ue00c').perform()  # ESC key
                time.sleep(1)
            except:
                print("   ⚠️ Could not close modal")
        
    except Exception as e:
        print(f"   ❌ Error processing modal for {athlete_name}: {e}")
    
    return modal_data

def scrape_event_country_data(driver, event_id, country_code):
    """Scrape modal data for a specific event and country"""
    print(f"\n🎯 Scraping Event {event_id} - Country {country_code.upper()}")
    
    all_modal_data = []
    wait = WebDriverWait(driver, 15)
    
    try:
        # Navigate to the page with event selected
        url = f"{base_url}?eventId={event_id}"
        if country_code != "all":
            url += f"&federationCode={country_code}"
        
        print(f"🌐 Loading: {url}")
        driver.get(url)
        time.sleep(3)
        
        # Get actual event name
        try:
            event_dropdown = driver.find_element(By.CSS_SELECTOR, "select#eventId")
            select_obj = Select(event_dropdown)
            selected_option = select_obj.first_selected_option
            event_name = selected_option.text.strip() if selected_option else f"Event_{event_id}"
        except:
            event_name = f"Event_{event_id}"
        
        print(f"🏃 Event: {event_name}")
        
        # Select all qualification statuses except "Qualified by Entry Standard"
        try:
            qual_dropdown = driver.find_element(By.CSS_SELECTOR, "select#qualificationType")
            select_obj = Select(qual_dropdown)
            
            # Try different qualification statuses
            qual_statuses = ["q4", "q7", "n4", ""]  # Skip q1 (Qualified by Entry Standard)
            
            for qual_status in qual_statuses:
                print(f"\n📊 Processing qualification status: {qual_status or 'All Status'}")
                
                try:
                    select_obj.select_by_value(qual_status)
                    time.sleep(2)
                    
                    # Look for athlete table rows
                    table_selectors = [
                        "tr.QualifiedCompetitors_boldFont__2vgJA",
                        "tr.QualifiedCompetitors_rowClickPointer__1itql",
                        "table tr[class*='row']",
                        "tbody tr"
                    ]
                    
                    clickable_rows = []
                    for selector in table_selectors:
                        try:
                            rows = driver.find_elements(By.CSS_SELECTOR, selector)
                            if rows:
                                clickable_rows = rows
                                print(f"✅ Found {len(rows)} rows using selector: {selector}")
                                break
                        except:
                            continue
                    
                    if not clickable_rows:
                        print("⚠️ No clickable rows found")
                        continue
                    
                    # Process each row
                    for i, row in enumerate(clickable_rows):
                        try:
                            # Extract athlete name from row
                            athlete_name = "Unknown"
                            try:
                                name_cell = row.find_element(By.CSS_SELECTOR, 
                                    ".QualifiedCompetitors_nameWidth___ryQL div, td:nth-child(4) div, td:nth-child(4)")
                                athlete_name = name_cell.text.strip()
                            except:
                                cells = row.find_elements(By.TAG_NAME, "td")
                                if len(cells) > 3:
                                    athlete_name = cells[3].text.strip()
                            
                            print(f"\n[{i+1}/{len(clickable_rows)}] Processing: {athlete_name}")
                            
                            # Scrape modal data for this athlete
                            modal_data = scrape_athlete_modal_data(
                                driver, row, athlete_name, event_id, event_name, country_code
                            )
                            
                            if modal_data:
                                all_modal_data.extend(modal_data)
                                print(f"   ✅ Extracted {len(modal_data)} performance records")
                            
                            # Small delay between athletes
                            time.sleep(1)
                            
                        except Exception as e:
                            print(f"   ❌ Error processing row {i+1}: {e}")
                            continue
                
                except Exception as e:
                    print(f"❌ Error processing qualification status {qual_status}: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ Error with qualification dropdown: {e}")
    
    except Exception as e:
        print(f"❌ Error loading page: {e}")
    
    return all_modal_data

def save_modal_data(all_data, event_id, country_code):
    """Save modal data to files"""
    if not all_data:
        print("⚠️ No modal data to save")
        return
    
    df = pd.DataFrame(all_data)
    
    # Create filename
    country_suffix = country_code if country_code != "all" else "all_countries"
    filename = f"athlete_performances_{event_id}_{country_suffix}"
    
    # Save CSV
    csv_file = os.path.join(save_dir, f"{filename}.csv")
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"💾 Saved {len(df)} performance records to {csv_file}")
    
    # Save to SQLite
    try:
        conn = sqlite3.connect(db_file)
        table_name = filename.replace("-", "_")
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        print(f"💾 Saved to database table: {table_name}")
    except Exception as e:
        print(f"⚠️ Database error: {e}")

def main():
    """Main scraping function"""
    driver = setup_driver()
    
    try:
        # Load the main page to show available options
        print(f"🌐 Loading main page: {base_url}")
        driver.get(base_url)
        time.sleep(3)
        
        # Show available events and countries
        events = get_available_events(driver)
        countries = get_available_countries(driver)
        
        if events:
            print(f"\n📋 Available Events ({len(events)} total):")
            for event in events[:10]:  # Show first 10
                print(f"   {event}")
            if len(events) > 10:
                print(f"   ... and {len(events) - 10} more")
        
        if countries:
            print(f"\n🌍 Available Countries ({len(countries)} total):")
            for country in countries[:15]:  # Show first 15
                print(f"   {country}")
            if len(countries) > 15:
                print(f"   ... and {len(countries) - 15} more")
        
        # Get user selections
        event_id, country_code = get_user_selections()
        
        if not event_id:
            print("❌ Event ID is required")
            return
        
        print(f"\n✅ Selected Event ID: {event_id}")
        print(f"✅ Selected Country: {country_code.upper()}")
        
        # Confirm before starting
        confirm = input("\n🚀 Start scraping modal data? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Scraping cancelled.")
            return
        
        print("\n" + "="*60)
        print("🚀 STARTING MODAL DATA EXTRACTION...")
        print("="*60)
        
        # Scrape the data
        all_modal_data = scrape_event_country_data(driver, event_id, country_code)
        
        # Save the data
        if all_modal_data:
            save_modal_data(all_modal_data, event_id, country_code)
            
            print("\n" + "="*60)
            print("🎉 SCRAPING COMPLETE!")
            print("="*60)
            print(f"📊 Total performance records extracted: {len(all_modal_data)}")
            print(f"📁 Files saved in: {save_dir}")
            print("="*60)
        else:
            print("\n⚠️ No modal data was extracted")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
