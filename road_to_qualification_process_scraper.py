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
import re

# --- CONFIG ---
base_url = "https://worldathletics.org/stats-zone/road-to/7190593"
save_dir = os.path.join("world_athletics", "Data", "qualification_processes")
os.makedirs(save_dir, exist_ok=True)

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
            return events
        else:
            print("⚠️ Event dropdown not found")
            return []
            
    except Exception as e:
        print(f"⚠️ Could not fetch events from dropdown: {e}")
        return []

def extract_qualification_details(driver, event_id, event_name, display_name):
    """Extract detailed qualification information for a specific event"""
    print(f"\n📋 Extracting qualification details for: {display_name}")
    
    qualification_data = {
        "Event_ID": event_id,
        "Event_Name": event_name,
        "Display_Name": display_name,
        "Scrape_Timestamp": pd.Timestamp.now().isoformat()
    }
    
    try:
        # Navigate to the event page
        url = f"{base_url}?eventId={event_id}"
        print(f"🌐 Loading: {url}")
        driver.get(url)
        time.sleep(3)
        
        # Get all text content from the page
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        # Extract qualification information using regex patterns
        patterns = {
            "entry_number": r"Entry number:\s*(\d+)",
            "qualification_period": r"Qualification period for entry standard:\s*([^\\n]+)",
            "entry_standard": r"Entry standard:\s*([^\\n]+)",
            "world_rankings_period": r"World Rankings period[^:]*:\s*([^\\n]+)",
            "maximum_quota": r"Maximum quota per [^:]*:\s*(\d+)",
            "athletes_by_entry_standard": r"By entry standard:\s*(\d+)",
            "athletes_by_finishing_position": r"By finishing position[^:]*:\s*(\d+)",
            "athletes_by_world_rankings": r"By world rankings position[^:]*:\s*([^\\n]+)",
            "athletes_by_top_list": r"By top list:\s*(\d+)",
            "athletes_by_universality": r"By universality places:\s*(\d+)"
        }
        
        # Extract data using patterns
        for key, pattern in patterns.items():
            try:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    qualification_data[key] = match.group(1).strip()
                    print(f"   ✅ {key}: {match.group(1).strip()}")
                else:
                    qualification_data[key] = ""
            except Exception as e:
                qualification_data[key] = ""
                print(f"   ⚠️ Could not extract {key}: {e}")
        
        # Extract special notes and additional information
        notes_patterns = [
            r"\\*NB:[^\\n]+",
            r"\\*\\*[^\\n]+",
            r"\\*\\*\\*[^\\n]+",
            r"QP:[^\\n]+",
            r"FP:[^\\n]+",
            r"All events except relays[^\\n]*",
            r"According to the World Rankings criteria[^\\n]+"
        ]
        
        notes = []
        for pattern in notes_patterns:
            try:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                notes.extend(matches)
            except:
                pass
        
        if notes:
            qualification_data["additional_notes"] = " | ".join(notes)
            print(f"   ✅ Additional notes: {len(notes)} notes found")
        else:
            qualification_data["additional_notes"] = ""
        
        # Store the full qualification text for reference
        # Look for text blocks that contain qualification information
        qual_text_blocks = []
        try:
            # Find all divs/paragraphs that contain qualification keywords
            elements = driver.find_elements(By.CSS_SELECTOR, "div, p, section")
            for element in elements:
                text = element.text.strip()
                if any(keyword in text.lower() for keyword in [
                    "entry number", "qualification period", "entry standard",
                    "world rankings", "maximum quota", "number of athletes"
                ]) and len(text) > 50:
                    qual_text_blocks.append(text)
        except:
            pass
        
        if qual_text_blocks:
            qualification_data["full_qualification_text"] = "\\n\\n".join(qual_text_blocks)
        else:
            qualification_data["full_qualification_text"] = ""
        
        print(f"   ✅ Qualification details extracted successfully")
        
    except Exception as e:
        print(f"   ❌ Error extracting qualification details: {e}")
        qualification_data["error"] = str(e)
    
    return qualification_data

def save_qualification_data(all_data, filename="qualification_processes"):
    """Save qualification data to files"""
    if not all_data:
        print("⚠️ No qualification data to save")
        return
    
    df = pd.DataFrame(all_data)
    
    # Save comprehensive CSV
    csv_file = os.path.join(save_dir, f"{filename}.csv")
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"💾 Saved {len(df)} qualification processes to {csv_file}")
    
    # Save summary CSV (key fields only)
    summary_columns = [
        "Event_ID", "Display_Name", "entry_number", "entry_standard", 
        "maximum_quota", "athletes_by_entry_standard", "athletes_by_world_rankings"
    ]
    summary_df = df[summary_columns].copy()
    summary_csv = os.path.join(save_dir, f"{filename}_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"📊 Saved qualification summary to {summary_csv}")
    
    # Save to SQLite
    try:
        db_file = os.path.join(save_dir, "qualification_processes.db")
        conn = sqlite3.connect(db_file)
        df.to_sql("qualification_processes", conn, if_exists="replace", index=False)
        summary_df.to_sql("qualification_summary", conn, if_exists="replace", index=False)
        conn.close()
        print(f"💾 Saved to database: {db_file}")
    except Exception as e:
        print(f"⚠️ Database error: {e}")
    
    # Create a readable text report
    create_readable_report(df, filename)

def create_readable_report(df, filename):
    """Create a human-readable text report of qualification processes"""
    report_file = os.path.join(save_dir, f"{filename}_report.txt")
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ROAD TO TOKYO 2025 - QUALIFICATION PROCESSES\\n")
            f.write("=" * 60 + "\\n\\n")
            
            for _, row in df.iterrows():
                f.write(f"EVENT: {row['Display_Name']}\\n")
                f.write("-" * 40 + "\\n")
                f.write(f"Event ID: {row['Event_ID']}\\n")
                
                if row['entry_number']:
                    f.write(f"Entry Number: {row['entry_number']}\\n")
                if row['entry_standard']:
                    f.write(f"Entry Standard: {row['entry_standard']}\\n")
                if row['qualification_period']:
                    f.write(f"Qualification Period: {row['qualification_period']}\\n")
                if row['world_rankings_period']:
                    f.write(f"World Rankings Period: {row['world_rankings_period']}\\n")
                if row['maximum_quota']:
                    f.write(f"Maximum Quota per Federation: {row['maximum_quota']}\\n")
                
                f.write("\\nNumber of Athletes:\\n")
                if row['athletes_by_entry_standard']:
                    f.write(f"  • By entry standard: {row['athletes_by_entry_standard']}\\n")
                if row['athletes_by_finishing_position']:
                    f.write(f"  • By finishing position: {row['athletes_by_finishing_position']}\\n")
                if row['athletes_by_world_rankings']:
                    f.write(f"  • By world rankings: {row['athletes_by_world_rankings']}\\n")
                if row['athletes_by_top_list']:
                    f.write(f"  • By top list: {row['athletes_by_top_list']}\\n")
                if row['athletes_by_universality']:
                    f.write(f"  • By universality places: {row['athletes_by_universality']}\\n")
                
                if row['additional_notes']:
                    f.write(f"\\nAdditional Notes:\\n{row['additional_notes']}\\n")
                
                f.write("\\n" + "=" * 60 + "\\n\\n")
        
        print(f"📖 Created readable report: {report_file}")
        
    except Exception as e:
        print(f"⚠️ Error creating report: {e}")

def main():
    """Main function to extract qualification processes for all events"""
    print("🚀 ROAD TO TOKYO - QUALIFICATION PROCESS EXTRACTOR")
    print("=" * 60)
    print("📋 This tool extracts qualification details for ALL events")
    print("=" * 60)
    
    # Confirm before starting
    try:
        confirm = input("\\n🚀 Start extracting qualification processes? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Extraction cancelled.")
            return
    except KeyboardInterrupt:
        print("\\n❌ Cancelled by user")
        return
    
    driver = setup_driver()
    all_qualification_data = []
    
    try:
        # Load the main page
        print(f"\\n🌐 Loading main page: {base_url}")
        driver.get(base_url)
        time.sleep(3)
        
        # Get all events
        events = get_all_events(driver)
        if not events:
            print("❌ No events found. Exiting.")
            return
        
        print(f"\\n📊 Will extract qualification processes for {len(events)} events")
        print("=" * 60)
        
        # Process each event
        for i, event in enumerate(events, 1):
            print(f"\\n[{i}/{len(events)}] Processing: {event['display_name']}")
            
            qualification_data = extract_qualification_details(
                driver, 
                event['eventId'], 
                event['name'], 
                event['display_name']
            )
            
            all_qualification_data.append(qualification_data)
            
            # Small delay between events
            time.sleep(2)
        
        # Save all data
        if all_qualification_data:
            save_qualification_data(all_qualification_data)
            
            print("\\n" + "=" * 60)
            print("🎉 EXTRACTION COMPLETE!")
            print("=" * 60)
            print(f"📊 Processed {len(all_qualification_data)} events")
            print(f"📁 Files saved in: {save_dir}")
            print("\\nFiles created:")
            print("  • qualification_processes.csv - Complete data")
            print("  • qualification_processes_summary.csv - Key fields only")
            print("  • qualification_processes_report.txt - Human-readable")
            print("  • qualification_processes.db - SQLite database")
            print("=" * 60)
        else:
            print("⚠️ No qualification data was extracted")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
