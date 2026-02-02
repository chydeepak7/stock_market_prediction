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
    SIGNAL_CHOICES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
        ('HOLD', 'Hold'),
    ]
    
    symbol = models.CharField(max_length=20)
    forecast_date = models.DateField()
    predicted_signal = models.CharField(max_length=10, choices=SIGNAL_CHOICES, default='HOLD')
    confidence_score = models.FloatField(default=0.5, help_text="Model confidence (0-1)")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'forecast_date']),
        ]
        ordering = ['forecast_date']

    def __str__(self):
        return f"{self.symbol} {self.forecast_date}: {self.predicted_signal} ({self.confidence_score:.2f})"


class ModelMetaData(models.Model):
    symbol = models.CharField(max_length=20, unique=True)
    last_trained = models.DateTimeField(auto_now=True)
    accuracy = models.FloatField(null=True, blank=True, help_text="Accuracy score (0-1)")
    balanced_accuracy = models.FloatField(null=True, blank=True, help_text="Balanced accuracy score (0-1)")
    classification_report = models.TextField(null=True, blank=True, help_text="JSON classification report")
    
    class Meta:
        verbose_name_plural = "Model Metadata"
    
    def __str__(self):
        return f"{self.symbol} - Acc: {self.accuracy:.2%} - {self.last_trained}"
