"""
ShareSansar Scraper Module
Scrapes daily stock data from sharesansar.com and saves to database.
"""
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from django.utils import timezone

logger = logging.getLogger('pipeline')

SHARESANSAR_URL = "https://www.sharesansar.com/today-share-price"

class ShareSansarScraper:
    """
    Scrapes stock market data from ShareSansar.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_today_data(self):
        """
        Fetches the latest stock data from ShareSansar.
        Extracts the actual date from the page (handles weekends/holidays).
        Returns a list of dictionaries with stock data.
        """
        logger.info(f"[SCRAPER] Fetching data from {SHARESANSAR_URL}")
        
        try:
            response = self.session.get(SHARESANSAR_URL, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"[SCRAPER] Request failed: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the actual date from "As of : YYYY-MM-DD" text on the page
        data_date = self._extract_date_from_page(soup)
        if not data_date:
            logger.warning("[SCRAPER] Could not extract date from page, using today")
            data_date = datetime.now().date()
        else:
            logger.info(f"[SCRAPER] Data date extracted from page: {data_date}")
        
        table = soup.find('table', {'class': 'table'})
        
        if not table:
            logger.warning("[SCRAPER] Could not find data table on page")
            return []
        
        rows = table.find('tbody').find_all('tr') if table.find('tbody') else []
        data = []
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 12:
                try:
                    # Column mapping based on ShareSansar table:
                    # 0: S.No, 1: Symbol, 2: Conf, 3: Open, 4: High, 5: Low, 
                    # 6: Close(LTP), 7: Close-LTP, 8: %, 9: %, 10: VWAP, 11: Vol
                    symbol = cols[1].get_text(strip=True)
                    open_price = self._parse_float(cols[3].get_text(strip=True))
                    high_price = self._parse_float(cols[4].get_text(strip=True))
                    low_price = self._parse_float(cols[5].get_text(strip=True))
                    close_price = self._parse_float(cols[6].get_text(strip=True))
                    volume = self._parse_int(cols[11].get_text(strip=True))
                    
                    if symbol and close_price:
                        data.append({
                            'symbol': symbol,
                            'date': data_date,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume,
                            'category': 'stock'
                        })
                except (ValueError, IndexError) as e:
                    logger.debug(f"[SCRAPER] Skipping row due to error: {e}")
                    continue
        
        logger.info(f"[SCRAPER] Scraped {len(data)} records for {data_date}")
        return data
    
    def _extract_date_from_page(self, soup):
        """
        Extracts the data date from the 'As of : YYYY-MM-DD' text on the page.
        Returns a date object or None if not found.
        """
        import re
        
        # Look for text like "As of : 2026-02-01" or "As of: 2026-02-01"
        page_text = soup.get_text()
        date_pattern = r'As\s*of\s*:\s*(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, page_text)
        
        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return None
        return None

    
    def _parse_float(self, value):
        """Parse string to float, handling commas."""
        if not value or value == '-':
            return 0.0
        return float(value.replace(',', ''))
    
    def _parse_int(self, value):
        """Parse string to int, handling commas and decimals."""
        if not value or value == '-':
            return 0
        # Handle values like '4169.00' by parsing as float first
        return int(float(value.replace(',', '')))
    
    def save_to_db(self, data):
        """
        Saves scraped data to the database.
        Returns tuple (created_count, skipped_count).
        """
        from app.models import StockData
        
        created = 0
        skipped = 0
        
        for record in data:
            obj, was_created = StockData.objects.get_or_create(
                symbol=record['symbol'],
                date=record['date'],
                defaults={
                    'open': record['open'],
                    'high': record['high'],
                    'low': record['low'],
                    'close': record['close'],
                    'volume': record['volume'],
                    'category': record['category'],
                }
            )
            if was_created:
                created += 1
            else:
                skipped += 1
        
        logger.info(f"[SCRAPER] DB Update: {created} created, {skipped} already existed")
        return created, skipped
    
    def fetch_date(self, target_date):
        """
        Fetches stock data for a specific date from ShareSansar.
        Uses the date filter feature on their website.
        Returns a list of dictionaries with stock data.
        """
        # ShareSansar accepts date as query parameter
        url = f"https://www.sharesansar.com/today-share-price?date={target_date.strftime('%Y-%m-%d')}"
        logger.info(f"[SCRAPER] Fetching historical data for {target_date}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"[SCRAPER] Request failed for {target_date}: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Verify the page is showing data for our requested date
        extracted_date = self._extract_date_from_page(soup)
        if extracted_date and extracted_date != target_date:
            logger.warning(f"[SCRAPER] Requested {target_date} but got {extracted_date}")
            # Still use extracted date as it's the actual data date
            data_date = extracted_date
        else:
            data_date = target_date
        
        table = soup.find('table', {'class': 'table'})
        
        if not table:
            logger.warning(f"[SCRAPER] No data table for {target_date} (holiday?)")
            return []
        
        rows = table.find('tbody').find_all('tr') if table.find('tbody') else []
        data = []
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 12:
                try:
                    symbol = cols[1].get_text(strip=True)
                    open_price = self._parse_float(cols[3].get_text(strip=True))
                    high_price = self._parse_float(cols[4].get_text(strip=True))
                    low_price = self._parse_float(cols[5].get_text(strip=True))
                    close_price = self._parse_float(cols[6].get_text(strip=True))
                    volume = self._parse_int(cols[11].get_text(strip=True))
                    
                    if symbol and close_price:
                        data.append({
                            'symbol': symbol,
                            'date': data_date,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume,
                            'category': 'stock'
                        })
                except (ValueError, IndexError) as e:
                    continue
        
        logger.info(f"[SCRAPER] Scraped {len(data)} records for {data_date}")
        return data


def run_scraper():
    """
    Main function to run the scraper and save data.
    Called by management commands or scheduled tasks.
    """
    logger.info("=" * 50)
    logger.info("[SCRAPER] Starting daily scrape job")
    
    scraper = ShareSansarScraper()
    data = scraper.fetch_today_data()
    
    if data:
        created, skipped = scraper.save_to_db(data)
        logger.info(f"[SCRAPER] Job completed: {created} new records, {skipped} duplicates")
        return {'status': 'success', 'created': created, 'skipped': skipped}
    else:
        logger.warning("[SCRAPER] No data scraped (market may be closed)")
        return {'status': 'no_data', 'created': 0, 'skipped': 0}


def run_scraper_with_gap_fill(max_days_back=30):
    """
    Runs the scraper and automatically fills any gaps in the database.
    Checks from the last recorded date up to today and fills missing dates.
    
    Args:
        max_days_back: Maximum number of days to look back for gaps
    """
    from app.models import StockData
    
    logger.info("=" * 50)
    logger.info("[SCRAPER] Starting scrape job with gap-fill")
    
    scraper = ShareSansarScraper()
    total_created = 0
    total_skipped = 0
    
    # First, scrape today's data
    today_data = scraper.fetch_today_data()
    if today_data:
        created, skipped = scraper.save_to_db(today_data)
        total_created += created
        total_skipped += skipped
        logger.info(f"[SCRAPER] Today's data: {created} new, {skipped} existing")
    
    # Check for gaps
    today = datetime.now().date()
    
    # Find the earliest date we should check
    try:
        latest_entry = StockData.objects.order_by('-date').first()
        if latest_entry:
            last_date = latest_entry.date
            logger.info(f"[SCRAPER] Last recorded date in DB: {last_date}")
        else:
            # No data at all, start from max_days_back
            last_date = today - timedelta(days=max_days_back)
            logger.info(f"[SCRAPER] No data in DB, starting from {last_date}")
    except Exception as e:
        logger.error(f"[SCRAPER] Error checking DB: {e}")
        last_date = today - timedelta(days=7)
    
    # Fill gaps (skip weekends - Saturday=5, Sunday=6 in Nepal)
    current_date = last_date + timedelta(days=1)
    gaps_filled = 0
    
    while current_date <= today and gaps_filled < max_days_back:
        # Skip weekends (Nepal: Friday is working, Sat-Sun are holidays)
        if current_date.weekday() in [5, 6]:  # Saturday, Sunday
            logger.debug(f"[SCRAPER] Skipping weekend: {current_date}")
            current_date += timedelta(days=1)
            continue
        
        # Check if we have data for this date
        has_data = StockData.objects.filter(date=current_date).exists()
        
        if not has_data:
            logger.info(f"[SCRAPER] Filling gap for {current_date}")
            data = scraper.fetch_date(current_date)
            if data:
                created, skipped = scraper.save_to_db(data)
                total_created += created
                total_skipped += skipped
                gaps_filled += 1
            else:
                logger.info(f"[SCRAPER] No data for {current_date} (holiday?)")
        
        current_date += timedelta(days=1)
    
    logger.info(f"[SCRAPER] Gap-fill complete: {gaps_filled} gaps processed")
    logger.info(f"[SCRAPER] Total: {total_created} new records, {total_skipped} duplicates")
    
    return {
        'status': 'success',
        'created': total_created,
        'skipped': total_skipped,
        'gaps_filled': gaps_filled
    }
