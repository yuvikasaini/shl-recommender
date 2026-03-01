import time
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def scrape_shl_catalog():
    output_file = 'data/shl_catalog.csv'
    if not os.path.exists('data'): os.makedirs('data')

    options = Options()
    options.add_argument("--start-maximized")
    # This prevents the "Automation" detection that causes the redirect
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    all_assessments = []
    seen_urls = set()

    print("🚀 Opening SHL Catalog...")
    driver.get("https://www.shl.com/solutions/products/product-catalog/?solutions=individual-test-solutions")
    
    try:
        while True:
            print(f"\n Scanning page for new items...")
            
            # Scroll to the bottom to ensure all cards load
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/view/']")
            new_found = 0
            
            for link in links:
                href = link.get_attribute("href")
                name = link.text.strip()
                if href and href not in seen_urls:
                    all_assessments.append({
                        "url": href,
                        "name": name.split('\n')[0],
                        "description": f"SHL Individual Assessment: {name}",
                        "adaptive_support": "No", "duration": 15, "remote_support": "Yes", "test_type": "['Individual Test']"
                    })
                    seen_urls.add(href)
                    new_found += 1

            if new_found > 0:
                print(f" Found {new_found} new items. (Total Unique: {len(all_assessments)})")
                pd.DataFrame(all_assessments).to_csv(output_file, index=False)
            else:
                print(" No new items detected. (Waiting for you to click 'Next'...)")

            print(" CLICK 'NEXT' ON THE WEBSITE NOW (or type 'ctrl+c' in terminal to stop)")
            
            # The script will wait 8 seconds for you to click and for the page to load
            # You can increase this if you need more time to click
            time.sleep(8) 

    except KeyboardInterrupt:
        print("\n Scrape stopped by user.")
    finally:
        driver.quit()
        print(f" FINAL SAVED: {len(all_assessments)} items.")

if __name__ == "__main__":
    scrape_shl_catalog()