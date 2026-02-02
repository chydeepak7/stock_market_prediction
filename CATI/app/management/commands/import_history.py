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
    help = 'Imports historical stock data from CSV files into the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbol',
            type=str,
            help='Import only a specific symbol (e.g., NABIL)',
        )
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
        
        logger.info(f"[CMD] import_history started. Directory: {data_dir}")
        self.stdout.write(self.style.NOTICE(f"Importing from: {data_dir}"))
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if options['symbol']:
            csv_files = [f for f in csv_files if f.startswith(options['symbol'])]
        
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
        column_mapping = {
            'time': 'date',
            'Time': 'date',
            'Date': 'date',
        }
        df = df.rename(columns=column_mapping)
        
        required_cols = ['date', 'symbol', 'open', 'close', 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"[CMD] Skipping {file_path}: missing required columns")
            return 0, 0
        
        imported = 0
        skipped = 0
        
        for _, row in df.iterrows():
            try:
                date_val = pd.to_datetime(row['date']).date()
                
                if dry_run:
                    imported += 1
                    continue
                
                obj, created = StockData.objects.get_or_create(
                    symbol=row['symbol'],
                    date=date_val,
                    defaults={
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume']),
                        'category': row.get('category', 'stock'),
                    }
                )
                if created:
                    imported += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.debug(f"[CMD] Error importing row: {e}")
                skipped += 1
                continue
        
        return imported, skipped
