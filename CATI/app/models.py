from django.db import models

class StockData(models.Model):
    CATEGORY_CHOICES = [
        ('stock', 'Stock'),
        ('index', 'Index'),
    ]

    symbol = models.CharField(max_length=20)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()
    category = models.CharField(max_length=10, choices=CATEGORY_CHOICES, default='stock')

    class Meta:
        unique_together = ('symbol', 'date')
        indexes = [
            models.Index(fields=['symbol', 'date']),
        ]
        ordering = ['-date']

    def __str__(self):
        return f"{self.symbol} - {self.date}"


class ForecastResult(models.Model):
    symbol = models.CharField(max_length=20)
    forecast_date = models.DateField()
    predicted_price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'forecast_date']),
        ]
        ordering = ['-forecast_date']

    def __str__(self):
        return f"{self.symbol} Forecast for {self.forecast_date}: {self.predicted_price}"


class ModelMetaData(models.Model):
    symbol = models.CharField(max_length=20)
    last_trained_at = models.DateTimeField(auto_now=True)
    accuracy = models.FloatField(null=True, blank=True, help_text="Accuracy score of the model")
    classification_report = models.TextField(null=True, blank=True, help_text="JSON or text representation of usage metrics")
    
    def __str__(self):
        return f"{self.symbol} - Last Train: {self.last_trained_at}"
