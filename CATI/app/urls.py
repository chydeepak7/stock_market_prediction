from django.urls import path
from . import views

urlpatterns = [
    
    path('backtest/', views.backtest_view, name='backtest'),
    path('forecast/', views.forecast_view, name='forecast'),
    path('model/', views.trigger_training_view, name='model'),
]