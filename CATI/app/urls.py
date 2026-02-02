from django.urls import path
from . import views

urlpatterns = [
    
    # path('model/', views.trigger_training_view, name='model'),
    
    path('backtest/', views.backtest_view, name='backtest'),
    path('forecast/', views.forecast_view, name='forecast'),
    path('logs/', views.log_view, name='logs'),
]
