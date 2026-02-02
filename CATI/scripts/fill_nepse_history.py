
import os
import sys
import django
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Setup Django
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CATI.settings')
django.setup()

from app.models import StockData

def fill_nepse_history(start_date, end_date):
    print(f"--- Filling NEPSE History ({start_date} to {end_date}) ---")
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    current = start_date
    while current <= end_date:
        url = f"https://www.sharesansar.com/datewise-indices?date={current.strftime('%Y-%m-%d')}"
        print(f"Fetching {current}...", end=" ", flush=True)
        
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"❌ HTTP {resp.status_code}")
                current += timedelta(days=1)
                continue
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table')
            
            if not table:
                print("⚠️ No table (Holiday?)")
                current += timedelta(days=1)
                continue
            
            # Datewise Indices usually has:
            # 0: Index Name (NEPSE Index)
            # 1: Current (Close)
            # 2: Points Change
            # 3: % Change
            # 4: Turnover
            
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if not cols: continue
                
                name = cols[0].get_text(strip=True).upper()
                if 'NEPSE INDEX' in name or name == 'NEPSE':
                    try:
                        # Only Close price is reliably available here as 'Current'
                        # We use Close for Open/High/Low to ensure DB integrity
                        val_text = cols[1].get_text().strip().replace(',', '')
                        _close = float(val_text)
                        
                        StockData.objects.update_or_create(
                            symbol='NEPSE',
                            date=current,
                            defaults={
                                'open': _close,
                                'high': _close,
                                'low': _close,
                                'close': _close,
                                'volume': 0, 
                                'category': 'index'
                            }
                        )
                        print(f"✅ Saved: {_close}")
                        found = True
                        break
                    except Exception as e:
                        print(f"❌ Parse Error: {e} | Text: {cols[1].get_text()}")
            
            if not found:
                print("⚠️ NEPSE row not found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
        current += timedelta(days=1)
        time.sleep(0.5)

if __name__ == '__main__':
    # Define range relative to missing data
    # Last known was Dec 11, 2025.
    # We fill up to today.
    start = datetime(2025, 12, 11).date() 
    end = datetime.now().date()
    fill_nepse_history(start, end)
