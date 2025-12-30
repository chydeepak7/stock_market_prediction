import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (perfect for servers)

from django.shortcuts import render, redirect
import os
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from django.conf import settings
from django.http import HttpResponse 
from django.template.loader import render_to_string
from django import forms
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Create your views here.



def backtest_view(request):
    context = {}
    # Path to the Excel file
    file_path = os.path.join(settings.BASE_DIR, 'saved_states', 'backtest_results.xlsx')
    df = pd.read_excel(file_path)
    df.set_index('Date', inplace=True)
    plt.plot(df.index, df['Real_Market_Price'], label='Real Market Price', color='black', alpha=0.7)
    plt.plot(df.index, df['Hybrid_Prediction'], label='Hybrid Prediction', color='orange', linewidth=2)
    plt.title('Backtest Results: Real vs Hybrid', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    context = {
        'chart_image': f"data:image/png;base64,{image_base64}"
    }
            
            
  
    return render(request, 'app/app.html', context)





def forecast_view(request):
    context = {}
    # Path to the Excel file
    file_path = os.path.join(settings.BASE_DIR, 'saved_states', 'future_30_day_forecast.xlsx')
    df = pd.read_excel(file_path)
    df.set_index('Forecast_Date', inplace=True)
    plt.plot(df.index, df['Predicted_Close_Price'], label='Prediction', color='orange', linewidth=2)
    plt.title('Backtest Results: Real vs Hybrid', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    context = {
        'chart_image': f"data:image/png;base64,{image_base64}"
    }
            
            
  
    return render(request, 'app/app.html', context)


from django.shortcuts import render
from django.contrib import messages
from .services import run_prediction_pipeline # Import the function we just made

def home_view(request):
    stock = request.GET.get('stock', 'CHCL')
    context = {
        'forecast_signals': [],
        'stock_symbol': stock,
        'overall_summary': "No forecast data available yet.",
        'summary_class': 'neutral',
        'ensemble_accuracy': None,
        'ensemble_balanced_accuracy': None,
        'confusion_image': None,
        'indicators_comparison_image': None,
        'signals_test_image': None,
        'equity_image': None,
        'backtest_metrics': {},
    }
    
    # Paths
    backtest_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock}_backtest_results.xlsx')
    forecast_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock}_future_30_day_forecast.xlsx')
    
    cm_path = os.path.join(settings.BASE_DIR, 'saved_states/images', f'{stock}_confusion.png')
    fi_path = os.path.join(settings.BASE_DIR, 'saved_states/images', f'{stock}_feature_importance.png')
    signals_path = os.path.join(settings.BASE_DIR, 'saved_states/images', f'{stock}_signals_test.png')
    equity_path = os.path.join(settings.BASE_DIR, 'saved_states/images', f'{stock}_equity_curve.png')
    
    # Load Images (simple & safe)
    context['confusion_image'] = f"/media/{stock}_confusion.png" if os.path.exists(cm_path) else None
    context['indicators_comparison_image'] = f"/media/{stock}_feature_importance.png" if os.path.exists(fi_path) else None
    context['signals_test_image'] = f"/media/{stock}_signals_test.png" if os.path.exists(signals_path) else None
    context['equity_image'] = f"/media/{stock}_equity_curve.png" if os.path.exists(equity_path) else None
    
    # Load Backtest Data
    if os.path.exists(backtest_path):
        try:
            # Metrics
            metrics_df = pd.read_excel(backtest_path, sheet_name='Metrics')
            context['ensemble_accuracy'] = dict(metrics_df.set_index('Metric')['Value']).get('Accuracy')
            context['ensemble_balanced_accuracy'] = dict(metrics_df.set_index('Metric')['Value']).get('Balanced Accuracy')
            
            # Backtest Metrics
            backtest_metrics_df = pd.read_excel(backtest_path, sheet_name='Backtest_Metrics')
            context['backtest_metrics'] = dict(zip(backtest_metrics_df['Metric'], backtest_metrics_df['Value']))
        except Exception as e:
            messages.error(request, f"Error loading backtest data: {str(e)}")
    
    # Load Forecast Data
    if os.path.exists(forecast_path):
        try:
            df_forecast = pd.read_excel(forecast_path)
            df_forecast['Forecast_Date'] = pd.to_datetime(df_forecast['Forecast_Date'])
            context['forecast_signals'] = [
                {
                    'date': row['Forecast_Date'].strftime('%Y-%m-%d'),
                    'signal': row['Signal'],
                    'score': round(row['Score'], 4)
                } for _, row in df_forecast.iterrows()
            ]
            # Overall Summary
            buy_count = sum(1 for s in context['forecast_signals'] if s['signal'] == 'BUY')
            sell_count = sum(1 for s in context['forecast_signals'] if s['signal'] == 'SELL')
            if buy_count > sell_count + 3:
                summary = "Bullish – Recommended to Buy"
                context['summary_class'] = 'buy'
            elif sell_count > buy_count + 3:
                summary = "Bearish – Consider Selling"
                context['summary_class'] = 'sell'
            else:
                summary = "Neutral – Hold Position"
                context['summary_class'] = 'neutral'
            context['overall_summary'] = f"Overall 30-Day Outlook: {summary}"
        except Exception as e:
            messages.error(request, f"Error loading forecast data: {str(e)}")
    
    # Handle Model Training
    if request.method == "POST" and request.POST.get('train_model'):
        try:
            messages.info(request, f"Training started for {stock}... This may take 30–90 seconds.")
            result = run_prediction_pipeline(stock_symbol=stock)
            
            if result.get('status') == 'skip':
                messages.info(request, result['message'])
            elif result['status'] == 'success':
                # Update from new training
                context.update({
                    'confusion_image': result['confusion_image'],
                    'indicators_comparison_image': result['indicators_comparison_image'],
                    'signals_test_image': result['signals_test_image'],
                    'equity_image': result['equity_image'],
                    'ensemble_accuracy': result.get('ensemble_accuracy'),
                    'ensemble_balanced_accuracy': result.get('ensemble_balanced_accuracy'),
                    'backtest_metrics': result.get('backtest_metrics', {}),
                })
                messages.success(request, "New best model saved — accuracy improved!")
            else:
                messages.error(request, result.get('message', 'Training failed.'))
        except Exception as e:
            messages.error(request, f"Training error: {str(e)}")
        
        return redirect(f'/predict/?stock={stock}')
    
    return render(request, 'app/app.html', context)

def welcome_view(request):
    """
    Welcome page with stock selection dropdown.
    """
    stocks = [
        ('NABIL', 'Nabil Bank (NABIL)'),
        ('NTC', 'Nepal Telecom (NTC)'),
        ('CIT', 'Citizen Investment Trust (CIT)'),
        ('HRL', 'Himalayan Reinsurance Limited  (HRL)'),
        ('GBIME', 'GBIME Bank (GBIME)'),
        ('NRIC', 'Nepal Reinsurance Company Limited (NRIC)'),
        # Add more stocks here as needed
    ]

    context = {
        'stocks': stocks,
        'selected_stock': request.GET.get('stock', 'NABIL'),  # Default to CHCL
    }

    if request.method == "POST":
        selected_stock = request.POST.get('stock')
        if selected_stock:
            # Redirect to the prediction page with stock as query param
            return redirect(f'/predict/?stock={selected_stock}')

    return render(request, 'app/welcome.html', context)

def generate_chart_image(df, chart_type='backtest'):
    """
    Generates a base64 PNG chart from a DataFrame.
    chart_type: 'backtest' or 'forecast'
    """
    plt.figure(figsize=(12, 6))
    
    if chart_type == 'backtest':
        plt.plot(df.index, df['Real_Market_Price'], 
                 label='Real Market Price', color='black', alpha=0.8, linewidth=2)
        plt.plot(df.index, df['Hybrid_Prediction'], 
                 label='Hybrid Prediction', color='orange', linewidth=3)
        plt.title('Backtest: Real vs Hybrid Prediction', fontsize=16)
    
    elif chart_type == 'forecast':
        plt.plot(df.index, df['Predicted_Close_Price'], 
                 label='Predicted Close Price', color='green', linewidth=3)
        plt.title('30-Day Future Price Forecast', fontsize=16)
    
    else:
        plt.close()
        return None

    plt.xlabel('Date')
    plt.ylabel('Price (NPR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return f"data:image/png;base64,{image_base64}"
