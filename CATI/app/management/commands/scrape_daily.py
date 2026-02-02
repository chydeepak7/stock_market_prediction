"""
Management command to scrape daily stock data from ShareSansar.
Run with: python manage.py scrape_daily
         python manage.py scrape_daily --fill-gaps  (to also fill missing dates)
"""
import logging
from django.core.management.base import BaseCommand
from app.scraper import run_scraper, run_scraper_with_gap_fill
from app.models import StockData
from datetime import datetime, timedelta

logger = logging.getLogger('pipeline')


class Command(BaseCommand):
    help = 'Scrapes stock data from ShareSansar and saves to database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--fill-gaps',
            action='store_true',
            help='Automatically fill missing dates between last entry and today',
        )
        parser.add_argument(
            '--max-days',
            type=int,
            default=30,
            help='Maximum number of days to look back when filling gaps (default: 30)',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE('Starting daily scrape...'))
        logger.info("[CMD] scrape_daily command started")
        
        if options['fill_gaps']:
            # Run with gap-filling
            self.stdout.write('Gap-fill mode enabled. Will scrape missing dates...')
            result = run_scraper_with_gap_fill(max_days_back=options['max_days'])
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"Scrape completed: {result['created']} new records, "
                    f"{result['skipped']} duplicates, {result['gaps_filled']} gaps filled"
                )
            )
        else:
            # Run normal daily scrape
            result = run_scraper()
            
            if result['status'] == 'success':
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Scrape completed: {result['created']} new records, {result['skipped']} duplicates"
                    )
                )
            elif result['status'] == 'no_data':
                self.stdout.write(
                    self.style.WARNING('No data scraped today (market may be closed)')
                )
            else:
                self.stdout.write(self.style.ERROR('Scrape failed'))
        
        logger.info("[CMD] scrape_daily command completed")
