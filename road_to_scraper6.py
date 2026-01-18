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

# --- CONFIG ---
base_url = "https://worldathletics.org/stats-zone/road-to/7190593"
# Paris https://worldathletics.org/stats-zone/road-to/7153115?eventId=10229509
# Budapest https://worldathletics.org/stats-zone/road-to/7138987?eventId=10229509
save_dir = os.path.join("world_athletics", "Data", "road_to")
os.makedirs(save_dir, exist_ok=True)
db_file = os.path.join(save_dir, "road_to_all_federations.db")

# Qualification types to scrape
qualification_types = [
    {"value": "", "name": "All_Status"},
    {"value": "q1", "name": "Qualified_by_Entry_Standard"},
    {"value": "q4", "name": "In_World_Rankings_quota"},
    {"value": "q7", "name": "Qualified_by_Wild_Card"},
    {"value": "n4", "name": "Next_best_by_World_Rankings"}
]

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

def get_user_preferences():
    """Get user preferences for federation scraping"""
    print("\n" + "="*50)
    print("🌍 FEDERATION SCRAPING OPTIONS")
    print("="*50)
    print("1. All Federations (comprehensive - takes longer)")
    print("2. Specific Country/Federation (quick update)")
    print("="*50)
    
    while True:
        choice = input("\nSelect option (1 or 2): ").strip()
        if choice == "1":
            print("✅ Selected: All Federations (comprehensive)")
            return "all", None
        elif choice == "2":
            print("\nEnter the country/federation code:")
            print("Examples: usa, ksa, gbr, fra, ger, aus, can, jpn")
            country_code = input("Country code: ").strip().lower()
            if country_code:
                print(f"✅ Selected: {country_code.upper()} (quick update)")
                return "specific", country_code
            else:
                print("❌ Please enter a valid country code.")
        else:
            print("❌ Please enter 1 or 2.")

def get_all_events(driver):
    """Get all available events from the events dropdown"""
    print("🎯 Fetching all available events...")
    events = []
    
    try:
        # Look for the specific event dropdown
        event_dropdown = driver.find_element(By.CSS_SELECTOR, "select#eventId")
        if event_dropdown:
            select_obj = Select(event_dropdown)
            options = select_obj.options
            
            for option in options:
                value = option.get_attribute("value")
                text = option.text.strip()
                
                # Skip empty values and "Discipline" placeholder
                if value and text and text.lower() != "discipline":
                    events.append({
                        "eventId": value,
                        "name": text.replace(" ", "_").replace("'", ""),
                        "display_name": text
                    })
            
            print(f"✅ Found {len(events)} events from dropdown")
            for event in events[:5]:  # Show first 5
                print(f"   - {event['display_name']} (ID: {event['eventId']})")
            if len(events) > 5:
                print(f"   ... and {len(events) - 5} more events")
                
            return events
        else:
            print("⚠️ Event dropdown not found")
            
    except Exception as e:
        print(f"⚠️ Could not fetch events from dropdown: {e}")
    
    # Fallback: try to get current event from URL or page
    try:
        current_url = driver.current_url
        current_event_id = "10229509"  # default
        if "eventId=" in current_url:
            current_event_id = current_url.split("eventId=")[1].split("&")[0]
        
        # Try to get event name from page
        current_event_name = get_actual_event_name(driver)
        if current_event_name == "Unknown_Event":
            current_event_name = "Default_Event"
            
        return [{
            "eventId": current_event_id,
            "name": current_event_name.replace(" ", "_").replace("'", ""),
            "display_name": current_event_name
        }]
        
    except Exception as e:
        print(f"⚠️ Could not get current event: {e}")
        return [{
            "eventId": "10229509",
            "name": "Womens_100_Metres",
            "display_name": "Women's 100 Metres"
        }]

def get_all_federations(driver):
    """Get all available federations from the federation/country dropdown"""
    print("🌍 Fetching all federations...")
    federations = []
    
    try:
        # Always start with "All Federations" to capture the initial landing state
        federations.append({"value": "", "name": "All_Federations"})
        
        # Look for federation/country dropdown
        federation_selectors = [
            "select[id*='country']", "select[id*='federation']", "select[id*='region']",
            ".country-selector select", ".federation-selector select",
            "select option[value*='federation']"
        ]
        
        federation_select = None
        for selector in federation_selectors:
            try:
                federation_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if federation_elements:
                    federation_select = federation_elements[0]
                    break
            except:
                continue
        
        if federation_select:
            select_obj = Select(federation_select)
            options = select_obj.options
            
            for option in options:
                value = option.get_attribute("value")
                text = option.text.strip()
                
                # Skip empty values, "All", "Select" options as we already have "All_Federations"
                if value and text and text.lower() not in ["all", "select", ""] and value != "":
                    federations.append({"value": value, "name": text})
            
            print(f"✅ Found {len(federations)} federations total (including All_Federations)")
        else:
            print("⚠️ No federation dropdown found, will only scrape All_Federations view")
            
    except Exception as e:
        print(f"⚠️ Error fetching federations: {e}")
        # Keep the "All_Federations" option even if there's an error
    
    return federations

def get_federations_based_on_choice(driver, choice_type, specific_code=None):
    """Get federations based on user choice"""
    if choice_type == "all":
        print("🌍 Will scrape ALL federations...")
        return get_all_federations(driver)
    else:
        print(f"🎯 Will scrape specific federation: {specific_code.upper()}")
        # Still get all federations to find the specific one
        all_feds = get_all_federations(driver)
        
        # Find the specific federation
        specific_fed = None
        for fed in all_feds:
            if fed["value"].lower() == specific_code or specific_code in fed["name"].lower():
                specific_fed = fed
                break
        
        if specific_fed:
            # Return "All Federations" view + specific federation
            return [
                {"value": "", "name": "All_Federations"},
                specific_fed
            ]
        else:
            print(f"⚠️ Federation '{specific_code}' not found. Available federations:")
            for fed in all_feds[:10]:  # Show first 10
                print(f"   - {fed['name']} ({fed['value']})")
            print("Defaulting to All Federations only...")
            return [{"value": "", "name": "All_Federations"}]

def extract_event_metadata(driver, event_id, event_name):
    """Extract event description and metadata from the page"""
    metadata = {
        "Event_ID": event_id,
        "Event_Name": event_name,
        "Scrape_Timestamp": pd.Timestamp.now().isoformat()
    }
    
    try:
        # Look for event description/information sections
        info_selectors = [
            ".event-info", ".competition-info", ".description", 
            "[class*='info']", "[class*='description']", "[class*='details']",
            ".qualification-info", ".entry-info"
        ]
        
        info_text = ""
        for selector in info_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if len(text) > 50:  # Only consider substantial text blocks
                        info_text += text + "\n"
            except:
                continue
        
        # If no specific info sections, look for all text content
        if not info_text:
            try:
                # Look for paragraphs or divs with substantial text
                text_elements = driver.find_elements(By.CSS_SELECTOR, "p, div")
                for element in text_elements:
                    text = element.text.strip()
                    if any(keyword in text.lower() for keyword in [
                        "entry number", "qualification period", "entry standard", 
                        "world rankings", "maximum quota", "number of athletes"
                    ]):
                        info_text += text + "\n"
            except:
                pass
        
        # Parse specific metadata from the text
        if info_text:
            metadata["full_description"] = info_text.strip()
            
            # Extract specific fields using regex-like parsing
            lines = info_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Entry number:"):
                    metadata["entry_number"] = line.replace("Entry number:", "").strip()
                elif "qualification period for entry standard:" in line.lower():
                    metadata["qualification_period"] = line.split(":")[-1].strip()
                elif line.startswith("Entry standard:"):
                    metadata["entry_standard"] = line.replace("Entry standard:", "").strip()
                elif "world rankings period" in line.lower():
                    metadata["world_rankings_period"] = line.split(":")[-1].strip()
                elif "maximum quota per" in line.lower():
                    metadata["maximum_quota"] = line.split(":")[-1].strip()
                elif "by entry standard:" in line.lower():
                    try:
                        metadata["athletes_by_entry_standard"] = line.split(":")[-1].strip()
                    except:
                        pass
                elif "by finishing position" in line.lower():
                    try:
                        metadata["athletes_by_finishing_position"] = line.split(":")[-1].strip()
                    except:
                        pass
                elif "by world rankings position:" in line.lower():
                    try:
                        metadata["athletes_by_world_rankings"] = line.split(":")[-1].strip()
                    except:
                        pass
                elif "by top list:" in line.lower():
                    try:
                        metadata["athletes_by_top_list"] = line.split(":")[-1].strip()
                    except:
                        pass
                elif "by universality places:" in line.lower():
                    try:
                        metadata["athletes_by_universality"] = line.split(":")[-1].strip()
                    except:
                        pass
        
        # Also look for any additional notes or important information
        try:
            notes_elements = driver.find_elements(By.CSS_SELECTOR, ".note, .important, [class*='note'], small, .footnote")
            notes = []
            for element in notes_elements:
                text = element.text.strip()
                if len(text) > 10 and any(keyword in text.lower() for keyword in [
                    "nb:", "*", "note", "important", "wild card", "federation", "quota"
                ]):
                    notes.append(text)
            if notes:
                metadata["additional_notes"] = " | ".join(notes)
        except:
            pass
            
    except Exception as e:
        print(f"⚠️ Error extracting metadata: {e}")
    
    return metadata

def get_actual_event_name(driver):
    """Extract the actual event name from the current page"""
    try:
        # First try to get from the event dropdown if it exists
        try:
            event_dropdown = driver.find_element(By.CSS_SELECTOR, "select#eventId")
            if event_dropdown:
                select_obj = Select(event_dropdown)
                selected_option = select_obj.first_selected_option
                if selected_option and selected_option.text.strip().lower() != "discipline":
                    return selected_option.text.strip()
        except:
            pass
        
        # Try to get from URL parameter
        current_url = driver.current_url
        if "eventId=" in current_url:
            event_id = current_url.split("eventId=")[1].split("&")[0]
            
            # Map some common event IDs to names
            event_map = {
                "10229509": "Women's 100 Metres",
                "10229510": "Women's 200 Metres", 
                "10229511": "Women's 400 Metres",
                "10229630": "Men's 100 Metres",
                "10229605": "Men's 200 Metres",
                "10229631": "Men's 400 Metres"
            }
            
            if event_id in event_map:
                return event_map[event_id]
        
        # Look for event name in various page locations
        selectors = [
            "h1", "h2", ".event-title", ".competition-name", 
            "[class*='title']", "[class*='event']", "[class*='competition']"
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    # Look for athletic event patterns
                    if any(pattern in text.lower() for pattern in [
                        "100m", "200m", "400m", "800m", "1500m", "5000m", "10000m",
                        "marathon", "jump", "throw", "hurdles", "steeplechase",
                        "men", "women", "relay", "walk", "metres", "decathlon", "heptathlon"
                    ]):
                        return text
            except:
                continue
        
        return "Unknown_Event"
    except:
        return "Unknown_Event"

def scrape_page_data(driver, event_id, event_name, federation_value, federation_name, qual_type, qual_name):
    """Scrape data from a single page configuration"""
    print(f"\n📊 Scraping: {event_name} | {federation_name} | {qual_name}")
    
    data = []
    wait = WebDriverWait(driver, 15)
    
    try:
        # Construct URL with parameters
        url = f"{base_url}?eventId={event_id}"
        if qual_type:
            url += f"&qualificationType={qual_type}"
        if federation_value and federation_value != "":
            url += f"&federationCode={federation_value}"
            
        print(f"🌐 Loading: {url}")
        driver.get(url)
        time.sleep(3)
        
        # Get the actual event name from the page
        actual_event_name = get_actual_event_name(driver)
        print(f"🏃 Actual event: {actual_event_name}")
        
        # Wait for page content to load
        try:
            wait.until(EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table")),
                EC.presence_of_element_located((By.CSS_SELECTOR, ".results-table")),
                EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='table']")),
                EC.presence_of_element_located((By.CSS_SELECTOR, ".athlete-row")),
                EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='result']"))
            ))
        except:
            print("⚠️ Page content may not have loaded completely")
        
        # Try to select qualification type if dropdown exists
        try:
            qual_dropdown = driver.find_element(By.CSS_SELECTOR, "select#qualificationType, .qualification-selector select")
            if qual_dropdown:
                select_obj = Select(qual_dropdown)
                select_obj.select_by_value(qual_type if qual_type else "")
                time.sleep(2)
                print(f"✅ Selected qualification type: {qual_name}")
        except:
            print("⚠️ Could not find or select qualification dropdown")
        
        # Try to select federation if dropdown exists
        try:
            fed_dropdown = driver.find_element(By.CSS_SELECTOR, "select[id*='country'], select[id*='federation']")
            if fed_dropdown and federation_value and federation_value != "":
                select_obj = Select(fed_dropdown)
                select_obj.select_by_value(federation_value)
                time.sleep(2)
                print(f"✅ Selected federation: {federation_name}")
            elif federation_value == "":
                print(f"✅ Using default 'All Federations' view (no federation selected)")
        except:
            if federation_value == "":
                print("✅ No federation dropdown found, using default 'All Federations' view")
            else:
                print("⚠️ Could not find or select federation dropdown")
        
        # Look for data tables
        table_selectors = [
            "table.records-table",
            "table",
            ".results-table table",
            "[class*='table']",
            ".qualification-results table"
        ]
        
        table = None
        for selector in table_selectors:
            try:
                tables = driver.find_elements(By.CSS_SELECTOR, selector)
                if tables:
                    table = tables[0]
                    print(f"✅ Found table using: {selector}")
                    break
            except:
                continue
        
        if table:
            # Get headers
            headers = []
            try:
                header_elements = table.find_elements(By.CSS_SELECTOR, "thead th, thead td, th")
                headers = [h.text.strip() for h in header_elements if h.text.strip()]
                if not headers:
                    # Try first row as headers
                    first_row = table.find_element(By.CSS_SELECTOR, "tr")
                    headers = [cell.text.strip() for cell in first_row.find_elements(By.TAG_NAME, "td")]
            except:
                headers = []
            
            print(f"📋 Headers: {headers}")
            
            # Get data rows
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr, tr")
            print(f"📊 Found {len(rows)} rows")
            
            for i, row in enumerate(rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 2:  # Skip rows with too few cells
                        continue
                    
                    # Row data structure with actual event name
                    row_data = {
                        "Event_Type": event_name,
                        "Event_ID": event_id,
                        "Actual_Event_Name": actual_event_name,  # NEW COLUMN
                        "Federation": federation_name,
                        "Qualification_Status": qual_name,
                        "Row_Index": i + 1
                    }
                    
                    # Add cell data
                    for j, cell in enumerate(cells):
                        cell_text = cell.text.strip()
                        
                        # Use header if available, otherwise generic name
                        if j < len(headers) and headers[j]:
                            col_name = headers[j].replace(" ", "_").replace(".", "").replace("#", "Rank")
                        else:
                            col_name = f"Column_{j+1}"
                        
                        row_data[col_name] = cell_text
                        
                        # Try to get profile URLs
                        try:
                            profile_url = row.get_attribute("data-athlete-url")
                            if profile_url:
                                row_data["Profile_URL"] = f"https://worldathletics.org{profile_url}"
                        except:
                            pass
                    
                    data.append(row_data)
                    
                    # Print progress for first few rows
                    if i < 5:
                        athlete_info = [cell.text.strip()[:20] for cell in cells[:3]]
                        print(f"   📋 Row {i+1}: {athlete_info}")
                
                except Exception as e:
                    print(f"❌ Error processing row {i+1}: {e}")
                    continue
        else:
            print("⚠️ No table found on page")
            # Save page source for debugging
            debug_file = os.path.join(save_dir, f"debug_{event_name}_{federation_name}_{qual_name}.html")
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print(f"💾 Saved page source to {debug_file}")
    
    except Exception as e:
        print(f"❌ Error scraping page: {e}")
    
    return data

def save_batch_data(all_data, batch_name):
    """Save data batch to files"""
    if not all_data:
        return
    
    df = pd.DataFrame(all_data)
    
    # Save CSV
    csv_file = os.path.join(save_dir, f"{batch_name}.csv")
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"💾 Saved {len(df)} rows to {csv_file}")
    
    # Save to SQLite
    try:
        conn = sqlite3.connect(db_file)
        table_name = batch_name.replace("-", "_").replace(" ", "_")
        df.to_sql(table_name, conn, if_exists="append", index=False)
        conn.close()
        print(f"💾 Added to database table: {table_name}")
    except Exception as e:
        print(f"⚠️ Database error: {e}")

def save_metadata(all_metadata, filename="event_metadata"):
    """Save event metadata to separate files"""
    if not all_metadata:
        return
    
    df = pd.DataFrame(all_metadata)
    
    # Save CSV
    csv_file = os.path.join(save_dir, f"{filename}.csv")
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"📋 Saved event metadata to {csv_file}")
    
    # Save to SQLite
    try:
        conn = sqlite3.connect(db_file)
        df.to_sql(filename, conn, if_exists="replace", index=False)
        conn.close()
        print(f"📋 Saved event metadata to database")
    except Exception as e:
        print(f"⚠️ Database error for metadata: {e}")

def main():
    """Main scraping function"""
    print("🚀 ROAD TO TOKYO FEDERATION SCRAPER")
    print("="*60)
    print("📋 This scraper ALWAYS scans ALL qualification statuses:")
    print("   ✅ All Status")
    print("   ✅ Qualified by Entry Standard")  
    print("   ✅ In World Rankings quota")
    print("   ✅ Qualified by Wild Card")
    print("   ✅ Next best by World Rankings")
    print("="*60)
    
    # Force the prompt to appear before initializing driver
    try:
        # Get user preference for federations FIRST
        choice_type, specific_code = get_user_preferences()
        print(f"\n✅ User selected: {choice_type} - {specific_code or 'all federations'}")
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return
    except Exception as e:
        print(f"❌ Error getting user input: {e}")
        return
    
    # Now initialize driver
    driver = setup_driver()
    
    try:
        # Load the main page first
        print(f"\n🌐 Loading main page: {base_url}")
        driver.get(base_url)
        time.sleep(3)
        
        # Get all available events
        events = get_all_events(driver)
        if not events:
            events = [{"eventId": "10229509", "name": "Men_100m", "url": ""}]
        
        # Get federations based on user choice
        federations = get_federations_based_on_choice(driver, choice_type, specific_code)
        
        print(f"\n📊 SCRAPING PLAN:")
        print(f"   - Events: {len(events)}")
        for event in events[:5]:  # Show first 5 events
            print(f"     • {event['display_name']} (ID: {event['eventId']})")
        if len(events) > 5:
            print(f"     ... and {len(events) - 5} more events")
        print(f"   - Federations: {len(federations)}")
        for fed in federations:
            print(f"     • {fed['name']}")
        print(f"   - Qualification statuses: {len(qualification_types)} (ALL)")
        print(f"   - Total combinations: {len(events) * len(federations) * len(qualification_types)}")
        
        if choice_type == "specific":
            print(f"\n🎯 QUICK UPDATE MODE")
            print(f"   - All Federations view (global context)")
            print(f"   - {specific_code.upper()} specific data")
            estimated_time = f"{len(events) * 2}-{len(events) * 5} minutes"
        else:
            print(f"\n🌍 FULL DATASET MODE") 
            print(f"   - All Federations view")
            print(f"   - All individual federations")
            estimated_time = f"{len(events) * len(federations)}-{len(events) * len(federations) * 2} minutes"
        
        print(f"   - Estimated time: {estimated_time}")
        
        # Final confirmation
        print("\n" + "="*60)
        try:
            confirm = input("🚀 Start scraping? (y/n): ").strip().lower()
            if confirm != 'y':
                print("❌ Scraping cancelled.")
                return
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return
        
        print("="*60)
        print("🚀 STARTING SCRAPE...")
        print("="*60)
        
        all_data = []
        all_metadata = []
        batch_size = 100
        batch_count = 0
        processed_events = set()
        
        # Loop through all combinations
        for event in events:
            event_id = event["eventId"]
            event_name = event["name"]
            
            # Extract metadata once per event
            if event_id not in processed_events:
                try:
                    driver.get(f"{base_url}?eventId={event_id}")
                    time.sleep(2)
                    metadata = extract_event_metadata(driver, event_id, event_name)
                    all_metadata.append(metadata)
                    processed_events.add(event_id)
                    print(f"📋 Extracted metadata for {event_name}")
                except Exception as e:
                    print(f"⚠️ Could not extract metadata for {event_name}: {e}")
            
            for federation in federations:
                fed_value = federation["value"]
                fed_name = federation["name"]
                
                # ALWAYS scan ALL qualification statuses
                for qual_type in qualification_types:
                    qual_value = qual_type["value"]
                    qual_name = qual_type["name"]
                    
                    # Scrape this combination
                    page_data = scrape_page_data(
                        driver, event_id, event_name, 
                        fed_value, fed_name, 
                        qual_value, qual_name
                    )
                    
                    if page_data:
                        all_data.extend(page_data)
                        print(f"✅ Collected {len(page_data)} records")
                        
                        # Save batch if we've collected enough data
                        if len(all_data) >= batch_size:
                            batch_count += 1
                            save_batch_data(all_data, f"road_to_tokyo_batch_{batch_count}")
                            all_data = []  # Reset for next batch
                    
                    # Small delay between requests
                    time.sleep(1)
        
        # Save any remaining data
        if all_data:
            batch_count += 1
            save_batch_data(all_data, f"road_to_tokyo_batch_{batch_count}")
        
        # Save metadata with consistent naming
        save_metadata(all_metadata, "event_metadata")
        
        print("\n" + "="*60)
        print("🎉 SCRAPING COMPLETE!")
        print("="*60)
        if choice_type == "specific":
            print(f"📊 Quick update for {specific_code.upper()}: {batch_count} batches")
        else:
            print(f"📊 Full dataset: {batch_count} batches")
        print(f"📋 Event metadata: {len(all_metadata)} records")
        print(f"📁 Files saved in: {save_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    main()