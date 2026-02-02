"""
Management command to import historical CSV data into the database.
Run with: python manage.py import_history
"""
import os
import logging
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from app.models import StockData

logger = logging.getLogger('pipeline')


class Command(BaseCommand):
    help = 'Classes management command to import historical stock data from CSV files'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, help='Specific symbol to import')
        parser.add_argument('--top20', action='store_true', help='Import only Top 20 stocks')
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be imported without actually importing',
        )

    def handle(self, *args, **options):
        data_dir = os.path.join(settings.BASE_DIR, 'saved_states', 'data')
        
        if not os.path.exists(data_dir):
            self.stdout.write(self.style.ERROR(f"Data directory not found: {data_dir}"))
            return
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        target_symbol = options['symbol']
        if target_symbol:
            csv_files = [f"{target_symbol}.csv"]
        
        if options['top20'] and hasattr(settings, 'TOP_20_STOCKS'):
            top_files = [f"{s}.csv" for s in settings.TOP_20_STOCKS]
            # Filter only those that exist
            csv_files = [f for f in csv_files if f in top_files]
            self.stdout.write(f"Filtered to Top 20 stocks ({len(csv_files)} found)")

        logger.info(f"[CMD] import_history started for {len(csv_files)} files...")
        self.stdout.write(self.style.NOTICE(f"Importing from: {data_dir}"))
        
        total_imported = 0
        total_skipped = 0
        
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            imported, skipped = self._import_csv(file_path, options['dry_run'])
            total_imported += imported
            total_skipped += skipped
            
            self.stdout.write(f"  {csv_file}: {imported} imported, {skipped} skipped")
        
        logger.info(f"[CMD] import_history completed. Total: {total_imported} imported, {total_skipped} skipped")
        self.stdout.write(
            self.style.SUCCESS(f"\nTotal: {total_imported} imported, {total_skipped} skipped")
        )
    
    def _import_csv(self, file_path, dry_run=False):
        """Import a single CSV file into the database."""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"[CMD] Error reading {file_path}: {e}")
            return 0, 0
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        
        column_mapping = {
            'time': 'date',
            'date': 'date',
        }
        df = df.rename(columns=column_mapping)
        
        if 'symbol' not in df.columns:
            # Infer symbol from filename
            symbol_from_file = os.path.splitext(os.path.basename(file_path))[0]
            df['symbol'] = symbol_from_file

        required_cols = ['date', 'symbol', 'open', 'close', 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"[CMD] Skipping {file_path}: missing required columns. Found: {df.columns.tolist()}")
            return 0, 0
        
        batch_size = 1000
        objects = []
        
        imported = 0
        skipped = 0 # With ignore_conflicts, we don't know exact skipped count easily, but that's fine
        
        for _, row in df.iterrows():
            try:
                date_val = pd.to_datetime(row['date']).date()
                
                obj = StockData(
                    symbol=row['symbol'],
                    date=date_val,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']),
                    category=row.get('category', 'stock')
                )
                objects.append(obj)
                
                if len(objects) >= batch_size:
                    if not dry_run:
                        StockData.objects.bulk_create(objects, ignore_conflicts=True)
                    imported += len(objects)
                    objects = []
            except Exception as e:
                logger.debug(f"[CMD] Error parsing row: {e}")
                continue
        
        # Insert remaining
        if objects:
            if not dry_run:
                StockData.objects.bulk_create(objects, ignore_conflicts=True)
            imported += len(objects)
            
        return imported, 0 # We assume conflicts are skipped silently
