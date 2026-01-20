from seleniumwire import webdriver
import json
import time

# Launch browser
driver = webdriver.Chrome()
driver.get("https://worldathletics.org/athletes/saudi-arabia/abdulaziz-abdui-atafi-15017843")

# Let the page fully load
time.sleep(10)

# Scan for the real GraphQL request
for request in driver.requests:
    if request.method == "POST" and "graphql" in request.url:
        print("✅ Found GraphQL request")
        print("\n--- HEADERS ---")
        print(json.dumps(dict(request.headers), indent=2))

        print("\n--- COOKIES ---")
        print(json.dumps(driver.get_cookies(), indent=2))

        print("\n--- BODY (query) ---")
        print(request.body.decode())

        break

driver.quit()
