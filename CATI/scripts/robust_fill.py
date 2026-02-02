
import os
import sys
import django
import time
from datetime import datetime, timedelta

# Setup Django Environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CATI.settings')
django.setup()

from app.models import StockData
from app.scraper import ShareSansarScraper

def run():
    print("--- Starting Robust Gap Fill ---")
    scraper = ShareSansarScraper()
    
    # 1. Check Last NEPSE Date
    try:
        last_entry = StockData.objects.filter(symbol='NEPSE').latest('date')
        last_date = last_entry.date
        print(f"Last NEPSE Date: {last_date}")
    except StockData.DoesNotExist:
        last_date = datetime.now().date() - timedelta(days=60)
        print(f"No NEPSE data. Starting from {last_date}")

    today = datetime.now().date()
    current_date = last_date + timedelta(days=1)
    
    # 2. Loop
    while current_date <= today:
        # Skip Weekends (Fri/Sat or Sat/Sun? In scraper it says 5,6)
        if current_date.weekday() in [5, 6]:
            print(f"Skipping Weekend: {current_date}")
            current_date += timedelta(days=1)
            continue
            
        print(f"Fetching for {current_date}...")
        try:
            data = scraper.fetch_date(current_date)
            if data:
                created, skipped = scraper.save_to_db(data)
                print(f"  Saved: {created} new, {skipped} skipped.")
            else:
                print("  No data found (Holiday?)")
        except Exception as e:
            print(f"  ERROR: {e}")
        
        current_date += timedelta(days=1)
        time.sleep(1.5) # Polite delay

    print("Gap Fill Complete.")

if __name__ == '__main__':
    run()
