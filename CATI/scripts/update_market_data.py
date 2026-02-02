
import os
import sys
import django
import time
import pandas as pd
from datetime import datetime, timedelta

# Setup Django Environment
# This allows the script to be run from the root directory or scripts/ directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CATI.settings')
django.setup()

from app.models import StockData
from app.scraper import ShareSansarScraper

# Top 20 companies by market capitalization (Nepal Stock Exchange) + NEPSE Index
TOP_21_SYMBOLS = [
    'NEPSE',  # Market Index
    'NABIL', 'NICA', 'GBIME', 'NTC', 'ADBL', 'SBL', 'CZBIL', 'EBL', 'SANIMA', 'NLIC',
    'NBL', 'HBL', 'SCB', 'SHIVM', 'NMB', 'CHCL', 'CIT', 'HIDCL', 'API', 'PRVU'
]

def export_to_files():
    """
    Exports Top 20 companies + NEPSE from DB to CSV and XLSX files in saved_states/data/.
    """
    print("\n[EXPORT] Syncing DB to CSV/XLSX files (Top 21 only)...")
    
    data_dir = os.path.join(parent_dir, 'saved_states', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Only export TOP_21_SYMBOLS
    for symbol in TOP_21_SYMBOLS:
        try:
            qs = StockData.objects.filter(symbol=symbol).order_by('date').values(
                'date', 'open', 'high', 'low', 'close', 'volume'
            )
            if not qs.exists():
                continue
                
            df = pd.DataFrame(list(qs))
            df.rename(columns={
                'date': 'time', 'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'volume': 'volume'
            }, inplace=True)
            
            # Sanitize symbol for filename (remove special chars)
            safe_symbol = "".join(c for c in symbol if c.isalnum() or c in ('_', '-'))
            
            # Save CSV
            csv_path = os.path.join(data_dir, f'{safe_symbol}.csv')
            df.to_csv(csv_path, index=False)
            
            # Save XLSX
            xlsx_path = os.path.join(data_dir, f'{safe_symbol}.xlsx')
            df.to_excel(xlsx_path, index=False)
            
            print(f"   ✅ {symbol}: {len(df)} records -> CSV & XLSX")
        except Exception as e:
            print(f"   ❌ {symbol}: Export failed - {e}")
    
    print("[EXPORT] Complete.\n")

def update_market_data(days_back=30):
    """
    Updates market data (All Stocks + NEPSE) from ShareSansar.
    Checks for missing dates in the DB and fetches them.
    """
    print("="*60)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Market Data Update...")
    print("="*60)

    scraper = ShareSansarScraper()
    today = datetime.now().date()
    
    # 1. Determine Start Date
    # We check the latest date for 'NEPSE' as a proxy for the system status
    try:
        last_entry = StockData.objects.filter(symbol='NEPSE').latest('date')
        last_date = last_entry.date
        print(f"Latest Data in DB (NEPSE): {last_date}")
    except StockData.DoesNotExist:
        # If DB is empty, go back 'days_back' days
        last_date = today - timedelta(days=days_back)
        print(f"No Data Found. Starting fresh from {last_date}")

    if last_date >= today:
        print("Database is already up to date!")
        # Still export to keep files in sync
        export_to_files()
        return

    # 2. Iterate from Last Date + 1 to Today
    current_date = last_date + timedelta(days=1)
    
    while current_date <= today:
        print(f"Fetching data for {current_date}...", end=" ", flush=True)
        
        try:
            data = scraper.fetch_date(current_date)
            if data:
                created, skipped = scraper.save_to_db(data)
                print(f"✅ Success! ({created} new, {skipped} existing)")
            else:
                print(f"⚠️  No Data (Holiday/Weekend)")
        except Exception as e:
            print(f"❌ Error: {e}")
            # Identify connection issues
            if "connection" in str(e).lower():
                print("   Connection lost. Waiting 5s...")
                time.sleep(5)

        # Rate Limiting
        time.sleep(1)
        current_date += timedelta(days=1)

    print("\n" + "="*60)
    print("Scraping Complete. Now Exporting to Files...")
    print("="*60)
    
    # 3. Export DB to CSV/XLSX
    export_to_files()
    
    print("="*60)
    print("All Done!")
    print("="*60)

if __name__ == '__main__':
    # You can pass a clearer 'days_back' if DB is empty
    update_market_data(days_back=60)
