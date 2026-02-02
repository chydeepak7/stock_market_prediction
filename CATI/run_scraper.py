#!/usr/bin/env python
"""
Standalone scraper script for Railway/Render cron jobs.
This script can be run independently to scrape stock data.
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CATI.settings')
django.setup()

from app.scraper import run_scraper_with_gap_fill
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pipeline')

if __name__ == '__main__':
    print("=" * 50)
    print("Stock Market Scraper - Starting...")
    print("=" * 50)
    
    result = run_scraper_with_gap_fill(max_days_back=30)
    
    print(f"\nResult: {result}")
    print("Scraper completed successfully!")
