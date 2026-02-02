import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (perfect for servers)

from django.shortcuts import render, redirect, HttpResponse
from django.conf import settings
from django.contrib import messages
from django.template.loader import render_to_string
from matplotlib.figure import Figure
import os
import io
import base64
import pandas as pd
import numpy as np
from .services import run_prediction_pipeline

# --- Helper Functions ---

def generate_chart_image(df, chart_type='backtest'):
    """
    Generates a base64 PNG chart from a DataFrame using Object-Oriented Matplotlib.
    chart_type: 'backtest' or 'forecast'
    """
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()
    
    if chart_type == 'backtest':
        ax.plot(df.index, df['Real_Market_Price'], 
                 label='Real Market Price', color='black', alpha=0.8, linewidth=2)
        ax.plot(df.index, df['Hybrid_Prediction'], 
                 label='Hybrid Prediction', color='orange', linewidth=3)
        ax.set_title('Backtest: Real vs Hybrid Prediction', fontsize=16)
    
    elif chart_type == 'forecast':
        ax.plot(df.index, df['Predicted_Close_Price'], 
                 label='Predicted Close Price', color='green', linewidth=3)
        ax.set_title('30-Day Future Price Forecast', fontsize=16)
    
    else:
        return None

    ax.set_xlabel('Date')
    ax.set_ylabel('Price (NPR)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return f"data:image/png;base64,{image_base64}"

# --- Views ---

def backtest_view(request):
    stock = request.GET.get('stock', 'NABIL') # Default or get from request if needed
    # Note: The original code hardcoded the filename. 
    # Logic suggests we might want to respect the stock symbol if possible, 
    # but sticking to previous behavior for the specific 'backtest_results.xlsx' file 
    # if it was intended to be global, OR improving it to be dynamic.
    # Given the other views use f'{stock}_...', let's assume valid paths are mostly dynamic 
    # but the original code pointed to a specific file. 
    # I will keep the original file path behavior but cleaner.
    
    # Original behavior was: file_path = os.path.join(settings.BASE_DIR, 'saved_states', 'backtest_results.xlsx')
    # If the user wants specific stock backtests, they should probably be routed differently, 
    # but for this specific view function refactor, I will maintain the file path logic 
    # OR update it to be generic if that file strictly exists.
    
    # Let's use the generic file path from the original code for safety, 
    # but ideally this should be dynamic like home_view.
    file_path = os.path.join(settings.BASE_DIR, 'saved_states', 'backtest_results.xlsx')
    
    context = {}
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            
            chart_image = generate_chart_image(df, chart_type='backtest')
            context['chart_image'] = chart_image
        except Exception as e:
            messages.error(request, f"Error generating backtest chart: {str(e)}")
    else:
        # Fallback or empty state
        pass

    return render(request, 'app/app.html', context)


def forecast_view(request):
    # Similar logic for forecast
    file_path = os.path.join(settings.BASE_DIR, 'saved_states', 'future_30_day_forecast.xlsx')
    
    context = {}
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            if 'Forecast_Date' in df.columns:
                df.set_index('Forecast_Date', inplace=True)
            
            chart_image = generate_chart_image(df, chart_type='forecast')
            context['chart_image'] = chart_image
        except Exception as e:
            messages.error(request, f"Error generating forecast chart: {str(e)}")
            
    return render(request, 'app/app.html', context)


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


def log_view(request):
    """
    View to display pipeline logs for developers.
    Shows the last N lines of the pipeline.log file.
    """
    log_file_path = os.path.join(settings.BASE_DIR, 'saved_states', 'pipeline.log')
    lines_to_show = int(request.GET.get('lines', 100))
    
    logs = []
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                logs = all_lines[-lines_to_show:] if len(all_lines) > lines_to_show else all_lines
        except Exception as e:
            logs = [f"Error reading log file: {str(e)}"]
    else:
        logs = ["Log file not found. Run the scraper or training to generate logs."]
    
    context = {
        'logs': logs,
        'lines_shown': len(logs),
        'lines_requested': lines_to_show,
    }
    
    return render(request, 'app/log_viewer.html', context)
