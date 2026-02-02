from django.core.management.base import BaseCommand
from app.services import run_prediction_pipeline

class Command(BaseCommand):
    help = 'Train ML model and generate forecasts for a specific stock'

    def add_arguments(self, parser):
        parser.add_argument(
            '--stock',
            type=str,
            default='NABIL',
            help='Stock symbol to train (default: NABIL)',
        )

    def handle(self, *args, **options):
        stock = options['stock']
        self.stdout.write(f"Starting training pipeline for {stock}...")
        
        try:
            result = run_prediction_pipeline(stock_symbol=stock)
            
            if result['status'] == 'success':
                self.stdout.write(self.style.SUCCESS(f"Training successful for {stock}"))
                self.stdout.write(f"Accuracy: {result.get('ensemble_accuracy', 'N/A')}")
                self.stdout.write(f"Balanced Accuracy: {result.get('ensemble_balanced_accuracy', 'N/A')}")
            elif result['status'] == 'skip':
                self.stdout.write(self.style.WARNING(f"Skipped: {result['message']}"))
            else:
                self.stdout.write(self.style.ERROR(f"Failed: {result['message']}"))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error executing pipeline: {str(e)}"))
