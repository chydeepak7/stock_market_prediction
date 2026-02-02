import os
import django
import sys

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CATI.settings')
django.setup()

from django.core.management import call_command
from app.models import StockData
from app.services import run_prediction_pipeline

# Target stocks + NEPSE
stocks = ['NABIL', 'NTC', 'CIT', 'GBIME', 'HRL', 'CHCL', 'NEPSE']

print("="*60)
print("       VERIFICATION & BENCHMARKING REPORT")
print("="*60)

print("\n--- PHASE 1: DATA IMPORT ---")
for s in stocks:
    print(f"Processing {s}...")
    # Run import (bulk_create handles speed, ignore_conflicts handles dupes)
    try:
        call_command('import_history', symbol=s)
    except Exception as e:
        print(f"  Import Command Error: {e}")
        
    count = StockData.objects.filter(symbol=s).count()
    print(f"  -> DB Count: {count}")

print("\n--- PHASE 2: MODEL TRAINING & EVALUATION ---")
results = {}

for s in stocks:
    if s == 'NEPSE': 
        continue # Don't train on index
        
    print(f"\nTraining Model for {s}...")
    try:
        res = run_prediction_pipeline(stock_symbol=s)
        
        if res['status'] == 'success':
            acc = res.get('ensemble_accuracy', 'N/A')
            bal = res.get('ensemble_balanced_accuracy', 'N/A')
            results[s] = {'acc': acc, 'bal': bal}
            print(f"  [SUCCESS] Accuracy: {acc}% | Balanced: {bal}%")
        else:
            msg = res.get('message', 'Unknown error')
            print(f"  [FAILED] {msg}")
            results[s] = {'error': msg}
            
    except Exception as e:
        print(f"  [CRASH] {str(e)}")
        results[s] = {'error': str(e)}

print("\n" + "="*60)
print("             FINAL SUMMARY")
print("="*60)
print(f"{'STOCK':<10} | {'ACCURACY':<10} | {'BALANCED':<10} | {'STATUS'}")
print("-" * 50)

for s in stocks:
    if s == 'NEPSE': continue
    stats = results.get(s, {})
    if 'error' in stats:
        print(f"{s:<10} | {'N/A':<10} | {'N/A':<10} | ERROR: {stats['error']}")
    else:
        print(f"{s:<10} | {str(stats.get('acc')) + '%':<10} | {str(stats.get('bal')) + '%':<10} | OK")
