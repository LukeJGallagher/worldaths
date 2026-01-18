import os
import csv
import time
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

def setup_driver():
    """Setup Chrome driver"""
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=chrome_options)

def show_events():
    """Display all available events"""
    events = [
        ("10229509", "Women's 100 Metres"), ("10229510", "Women's 200 Metres"),
        ("10229511", "Women's 400 Metres"), ("10229512", "Women's 800 Metres"),
        ("10229513", "Women's 1500 Metres"), ("10229514", "Women's 5000 Metres"),
        ("10229521", "Women's 10,000 Metres"), ("10229522", "Women's 100 Metres Hurdles"),
        ("10229523", "Women's 400 Metres Hurdles"), ("10229524", "Women's 3000 Metres Steeplechase"),
        ("10229526", "Women's High Jump"), ("10229527", "Women's Pole Vault"),
        ("10229528", "Women's Long Jump"), ("10229529", "Women's Triple Jump"),
        ("10229530", "Women's Shot Put"), ("10229531", "Women's Discus Throw"),
        ("10229532", "Women's Hammer Throw"), ("10229533", "Women's Javelin Throw"),
        ("10229534", "Women's Marathon"), ("10229535", "Women's 20 Kilometres Race Walk"),
        ("10229989", "Women's 35 Kilometres Race Walk"), ("10229536", "Women's Heptathlon"),
        ("204594", "Women's 4x100 Metres Relay"), ("204596", "Women's 4x400 Metres Relay"),
        ("10229630", "Men's 100 Metres"), ("10229605", "Men's 200 Metres"),
        ("10229631", "Men's 400 Metres"), ("10229501", "Men's 800 Metres"),
        ("10229502", "Men's 1500 Metres"), ("10229609", "Men's 5000 Metres"),
        ("10229610", "Men's 10,000 Metres"), ("10229611", "Men's 110 Metres Hurdles"),
        ("10229612", "Men's 400 Metres Hurdles"), ("10229614", "Men's 3000 Metres Steeplechase"),
        ("10229615", "Men's High Jump"), ("10229616", "Men's Pole Vault"),
        ("10229617", "Men's Long Jump"), ("10229618", "Men's Triple Jump"),
        ("10229619", "Men's Shot Put"), ("10229620", "Men's Discus Throw"),
        ("10229621", "Men's Hammer Throw"), ("10229636", "Men's Javelin Throw"),
        ("10229634", "Men's Marathon"), ("10229508", "Men's 20 Kilometres Race Walk"),
        ("10229627", "Men's 35 Kilometres Race Walk"), ("10229629", "Men's Decathlon"),
        ("204593", "Men's 4x100 Metres Relay"), ("204595", "Men's 4x400 Metres Relay"),
        ("10229988", "Mixed 4x400 Metres Relay")
    ]
    
    print("\n🏃‍♀️ WOMEN'S EVENTS:")
    for event_id, name in events:
        if "Women's" in name:
            print(f"   {event_id} - {name}")
    
    print("\n🏃‍♂️ MEN'S EVENTS:")
    for event_id, name in events:
        if "Men's" in name:
            print(f"   {event_id} - {name}")
    
    print("\n🏃‍♂️🏃‍♀️ MIXED EVENTS:")
    for event_id, name in events:
        if "Mixed" in name:
            print(f"   {event_id} - {name}")
    
    return {event_id: name for event_id, name in events}

def get_selections():
    """Get user event and country selections"""
    print("🎯 MODAL DATA SCRAPER")
    print("="*50)
    
    # Show events
    event_dict = show_events()
    
    # Get events
    print("\n📋 EVENT SELECTION (comma-separated for multiple):")
    print("Example: 10229630,10229509")
    event_input = input("Enter Event ID(s): ").strip()
    event_ids = [e.strip() for e in event_input.split(',') if e.strip()]
    
    # Validate events
    valid_events = []
    for event_id in event_ids:
        if event_id in event_dict:
            valid_events.append(event_id)
            print(f"✅ {event_id} - {event_dict[event_id]}")
        else:
            print(f"❌ Invalid Event ID: {event_id}")
    
    if not valid_events:
        print("❌ No valid events selected")
        return None, None
    
    # Get countries
    print("\n🌍 COUNTRY SELECTION (comma-separated for multiple):")
    print("Popular: usa,gbr,aus,can,fra,ger,jpn,ksa,chn,jam,ken,eth")
    print("Or 'all' for all countries")
    country_input = input("Enter Country Code(s): ").strip().lower()
    
    if country_input == 'all':
        countries = ['all']
    else:
        countries = [c.strip() for c in country_input.split(',') if c.strip()]
    
    print(f"✅ Selected {len(countries)} countries: {', '.join(countries)}")
    
    return valid_events, countries

def click_athlete_row(driver, row):
    """Try multiple methods to click an athlete row"""
    try:
        # Method 1: Regular click
        row.click()
        return True
    except:
        try:
            # Method 2: JavaScript click
            driver.execute_script("arguments[0].click();", row)
            return True
        except:
            try:
                # Method 3: ActionChains
                ActionChains(driver).move_to_element(row).click().perform()
                return True
            except:
                return False

def extract_modal_data(driver, athlete_name, event_id, event_name, country_code):
    """Extract data from the modal popup"""
    data = []
    
    try:
        # Wait for modal
        wait = WebDriverWait(driver, 10)
        modal = wait.until(EC.presence_of_element_located((
            By.CSS_SELECTOR, ".RankingScoreCalculationModal_container__3BidC"
        )))
        
        # Get athlete name from modal
        try:
            modal_name = modal.find_element(By.CSS_SELECTOR, ".RankingScoreCalculationModal_title__1XN80").text.strip()
        except:
            modal_name = athlete_name
        
        # Get profile URL
        profile_url = ""
        athlete_id = ""
        try:
            profile_link = modal.find_element(By.CSS_SELECTOR, "a[href*='/athletes/']")
            profile_url = profile_link.get_attribute("href")
            if "/athletes/" in profile_url:
                athlete_id = profile_url.split("/athletes/")[-1].split("-")[-1]
        except:
            pass
        
        # Get performance table
        try:
            table = modal.find_element(By.CSS_SELECTOR, "table")
            
            # Get headers
            headers = []
            header_elements = table.find_elements(By.CSS_SELECTOR, "thead th")
            for h in header_elements:
                headers.append(h.text.strip())
            
            # Get performance rows
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            
            for i, row in enumerate(rows):
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) > 0:
                    perf_data = {
                        "Event_ID": event_id,
                        "Event_Name": event_name,
                        "Country_Code": country_code,
                        "Athlete_Name": modal_name,
                        "Athlete_ID": athlete_id,
                        "Profile_URL": profile_url,
                        "Performance_Index": i + 1
                    }
                    
                    # Add cell data
                    for j, cell in enumerate(cells):
                        col_name = headers[j] if j < len(headers) else f"Column_{j+1}"
                        col_name = col_name.replace(".", "").replace(" ", "_")
                        perf_data[col_name] = cell.text.strip()
                    
                    data.append(perf_data)
        
        except Exception as e:
            print(f"   ❌ Error extracting table: {e}")
        
        # Get ranking scores
        try:
            footer_text = modal.find_element(By.CSS_SELECTOR, ".RankingScoreCalculationModal_footer__2xjDo").text
            for record in data:
                if "Average of Performance Scores:" in footer_text:
                    avg_score = footer_text.split("Average of Performance Scores:")[-1].split()[0]
                    record["Average_Performance_Score"] = avg_score
                if "Ranking Score:" in footer_text:
                    rank_score = footer_text.split("Ranking Score:")[-1].split()[0]
                    record["Ranking_Score"] = rank_score
        except:
            pass
        
        # Close modal
        try:
            close_btn = modal.find_element(By.CSS_SELECTOR, ".RankingScoreCalculationModal_close__3wzIX")
            close_btn.click()
        except:
            try:
                ActionChains(driver).send_keys('\ue00c').perform()  # ESC key
            except:
                pass
        
        time.sleep(1)
        
    except Exception as e:
        print(f"   ❌ Modal error: {e}")
    
    return data

def scrape_event_country(driver, event_id, country_code):
    """Scrape one event-country combination"""
    print(f"\n🎯 Event {event_id} | Country {country_code}")
    
    all_data = []
    
    try:
        # Build URL
        url = f"{base_url}?eventId={event_id}"
        if country_code != "all":
            url += f"&federationCode={country_code}"
        
        driver.get(url)
        time.sleep(3)
        
        # Get event name
        try:
            event_dropdown = driver.find_element(By.CSS_SELECTOR, "select#eventId")
            selected_option = Select(event_dropdown).first_selected_option
            event_name = selected_option.text.strip()
        except:
            event_name = f"Event_{event_id}"
        
        print(f"🏃 {event_name}")
        
        # Process qualification statuses (exclude q1 - Qualified by Entry Standard)
        qual_statuses = [("", "All Status"), ("q4", "World Rankings"), ("q7", "Wild Card"), ("n4", "Next Best")]
        
        for qual_value, qual_name in qual_statuses:
            print(f"\n📊 {qual_name}")
            
            try:
                # Select qualification status
                qual_dropdown = driver.find_element(By.CSS_SELECTOR, "select#qualificationType")
                Select(qual_dropdown).select_by_value(qual_value)
                time.sleep(2)
                
                # Find athlete rows
                rows = driver.find_elements(By.CSS_SELECTOR, "tr.QualifiedCompetitors_rowClickPointer__1itql")
                print(f"   Found {len(rows)} athletes")
                
                for i, row in enumerate(rows):
                    try:
                        # Get athlete info from row
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 4:
                            athlete_country = cells[2].text.strip()
                            athlete_name = cells[3].text.strip()
                            
                            # Skip if wrong country
                            if country_code != "all" and athlete_country.upper() != country_code.upper():
                                continue
                            
                            print(f"   [{i+1}] {athlete_name} ({athlete_country})")
                            
                            # Click and extract modal data
                            if click_athlete_row(driver, row):
                                modal_data = extract_modal_data(driver, athlete_name, event_id, event_name, country_code)
                                if modal_data:
                                    all_data.extend(modal_data)
                                    print(f"       ✅ {len(modal_data)} performances")
                            else:
                                print(f"       ❌ Could not click row")
                            
                            time.sleep(1)
                    
                    except Exception as e:
                        print(f"   ❌ Row error: {e}")
                        continue
            
            except Exception as e:
                print(f"❌ Qualification status error: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Page error: {e}")
    
    return all_data

def save_data(data, event_ids, countries):
    """Save data to CSV"""
    if not data:
        print("⚠️ No data to save")
        return
    
    df = pd.DataFrame(data)
    
    # Create filename
    events_str = "_".join(event_ids[:3])  # Limit filename length
    countries_str = "_".join(countries[:3])
    filename = f"performances_{events_str}_{countries_str}.csv"
    
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False, encoding="utf-8")
    
    print(f"\n💾 Saved {len(df)} records to {filename}")

def main():
    """Main function"""
    print("🚀 ATHLETE PERFORMANCE MODAL SCRAPER")
    print("="*50)
    
    # Get selections
    event_ids, countries = get_selections()
    if not event_ids or not countries:
        return
    
    # Confirm
    total = len(event_ids) * len(countries)
    print(f"\n📊 Will process {total} combinations")
    if input("Continue? (y/n): ").lower() != 'y':
        return
    
    # Start scraping
    driver = setup_driver()
    all_data = []
    
    try:
        for event_id in event_ids:
            for country in countries:
                data = scrape_event_country(driver, event_id, country)
                all_data.extend(data)
        
        # Save results
        save_data(all_data, event_ids, countries)
        
        print(f"\n🎉 Complete! Total records: {len(all_data)}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()