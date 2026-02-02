"""
Management command to sync data between cloud (Neon) and local SQLite database.
Enables offline mode by pulling cloud data to local storage.

Usage:
    python manage.py sync_db --direction=pull   # Cloud → Local SQLite
    python manage.py sync_db --direction=push   # Local SQLite → Cloud
    python manage.py sync_db --export-csv       # Export DB to CSV files
"""
import os
import logging
from django.core.management.base import BaseCommand
from django.conf import settings
from app.models import StockData, ForecastResult, ModelMetaData
import pandas as pd

logger = logging.getLogger('pipeline')


class Command(BaseCommand):
    help = 'Sync database between cloud and local, or export to CSV'

    def add_arguments(self, parser):
        parser.add_argument(
            '--direction',
            type=str,
            choices=['pull', 'push'],
            help='Sync direction: pull (cloud→local) or push (local→cloud)',
        )
        parser.add_argument(
            '--export-csv',
            action='store_true',
            help='Export current database to CSV files for offline use',
        )
        parser.add_argument(
            '--symbol',
            type=str,
            help='Specific symbol to sync (optional)',
        )

    def handle(self, *args, **options):
        if options['export_csv']:
            self._export_to_csv(options.get('symbol'))
        elif options['direction'] == 'pull':
            self._pull_from_cloud(options.get('symbol'))
        elif options['direction'] == 'push':
            self.stdout.write(self.style.WARNING('Push not implemented yet'))
        else:
            self.stdout.write(self.style.ERROR('Please specify --direction or --export-csv'))

    def _export_to_csv(self, symbol=None):
        """Export database data to CSV files in saved_states/data/"""
        data_dir = os.path.join(settings.BASE_DIR, 'saved_states', 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Get all unique symbols
        if symbol:
            symbols = [symbol]
        else:
            symbols = StockData.objects.values_list('symbol', flat=True).distinct()

        self.stdout.write(f'Exporting {len(list(symbols))} symbols to CSV...')

        for sym in symbols:
            # Sanitize symbol for filename
            safe_sym = "".join([c for c in sym if c.isalnum() or c in (' ', '-', '_')]).strip()
            
            qs = StockData.objects.filter(symbol=sym).order_by('date')
            if not qs.exists():
                continue

            # Convert to DataFrame
            data = list(qs.values('date', 'open', 'high', 'low', 'close', 'volume', 'category'))
            df = pd.DataFrame(data)
            
            # Rename columns to match expected format
            df.rename(columns={
                'date': 'time',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'category': 'category'
            }, inplace=True)

            # Save to CSV
            csv_path = os.path.join(data_dir, f'{safe_sym}.csv')
            df.to_csv(csv_path, index=False)
            self.stdout.write(f'  ✓ {sym}: {len(df)} records → {csv_path}')

        self.stdout.write(self.style.SUCCESS('CSV export complete!'))
        logger.info(f"[SYNC] Exported {len(list(symbols))} symbols to CSV")

    def _pull_from_cloud(self, symbol=None):
        """
        Pull data from cloud DB and save to local SQLite.
        This requires switching DATABASE_URL temporarily.
        """
        # For now, just export to CSV which can be used offline
        self.stdout.write('Pulling data from current database to CSV...')
        self._export_to_csv(symbol)
        
        self.stdout.write(self.style.SUCCESS(
            '\nTo use offline:\n'
            '1. Remove DATABASE_URL from .env (or set USE_LOCAL_DB=true)\n'
            '2. Run: python manage.py migrate\n'
            '3. Run: python manage.py import_history\n'
            '4. App will now use local SQLite with exported data'
        ))
