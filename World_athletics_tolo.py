from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Setup
url = "https://worldathletics.org/athletes/saudi-arabia/abdulaziz-abdui-atafi-15017843"
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)
driver.get(url)

# Click STATISTICS
wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='STATISTICS']"))).click()
time.sleep(1)

# Click RESULTS
wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@value='Results']"))).click()
time.sleep(1)

# Wait for results table
wait.until(EC.presence_of_element_located((By.XPATH, "//table[contains(@class,'profileStatistics_table__1o71p')]")))
print("✅ Table loaded.")

# Loop through result rows (excluding dropdowns)
row_index = 0
while True:
    rows = driver.find_elements(By.XPATH, "//table[contains(@class,'profileStatistics_table__1o71p')]//tbody/tr[not(contains(@class,'trDropdown'))]")
    
    if row_index >= len(rows):
        break

    try:
        print(f"\n🔍 Expanding row {row_index + 1}...")
        row = rows[row_index]
        button = row.find_element(By.TAG_NAME, "button")
        driver.execute_script("arguments[0].click();", button)
        time.sleep(1)

        # Find next sibling row as dropdown
        dropdown = row.find_element(By.XPATH, "./following-sibling::tr[contains(@class,'trDropdown')]")
        labels = dropdown.find_elements(By.CLASS_NAME, "athletesEventsDetails_athletesEventsDetailsLabel__6KN98")
        values = dropdown.find_elements(By.CLASS_NAME, "dropdown-item")

        for label, value in zip(labels, values):
            print(f"{label.text.strip()}: {value.text.strip()}")

    except Exception as e:
        print(f"⚠️ Row {row_index + 1} failed or no dropdown. ({e})")

    row_index += 1

driver.quit()
