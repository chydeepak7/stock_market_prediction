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
from .models import ForecastResult, ModelMetaData, StockData

# --- Helper Functions ---

def generate_chart_image(df, chart_type='backtest'):
    """
    Generates a base64 PNG chart from a DataFrame using Object-Oriented Matplotlib.
    chart_type: 'backtest' or 'forecast'
    """
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()
    
    if chart_type == 'backtest':
        ax.plot(df.index, df['Close'], 
                 label='Real Market Price', color='black', alpha=0.8, linewidth=2)
        # Assuming we have predictions in df, but backtest view logic is tricky without full data
        # For now, let's just plot Close if that's all we have from DB
        ax.set_title('Stock Price History', fontsize=16)
    
    elif chart_type == 'forecast':
        ax.plot(df.index, df['confidence_score'], 
                 label='Confidence Score', color='green', linewidth=3, marker='o')
        ax.set_title('30-Day Forecast Confidence', fontsize=16)
        ax.set_ylim(0, 1)
    
    else:
        return None

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
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
    """
    View to show historical data/backtest.
    For now, we'll show the generic backtest result file if it exists,
    or better: redirect to the main dashboard which has all the charts.
    """
    return redirect('home')


def forecast_view(request):
    """
    Shows forecast for a specific stock using DB data.
    """
    stock = request.GET.get('stock', 'NABIL')
    
    context = {'stock_symbol': stock}
    
    # Fetch from DB
    forecasts = ForecastResult.objects.filter(symbol=stock).order_by('forecast_date')
    
    if forecasts.exists():
        data = list(forecasts.values('forecast_date', 'predicted_signal', 'confidence_score'))
        df = pd.DataFrame(data)
        df.rename(columns={'forecast_date': 'Date'}, inplace=True)
        df.set_index('Date', inplace=True)
        
        chart_image = generate_chart_image(df, chart_type='forecast')
        context['chart_image'] = chart_image
        context['forecasts'] = forecasts
    else:
        messages.warning(request, f"No forecast data found for {stock}. Please train the model first.")
            
    return render(request, 'app/app.html', context) # Reusing app.html or create specific template


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
    
    # 1. Load Forecasts from DB
    forecasts = ForecastResult.objects.filter(symbol=stock).order_by('forecast_date')
    if forecasts.exists():
        context['forecast_signals'] = [
            {
                'date': f.forecast_date.strftime('%Y-%m-%d'),
                'signal': f.predicted_signal,
                'score': round(f.confidence_score, 4)
            } for f in forecasts
        ]
        
        # Overall Summary
        buy_count = forecasts.filter(predicted_signal='BUY').count()
        sell_count = forecasts.filter(predicted_signal='SELL').count()
        
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

    # 2. Load Model Metadata from DB
    meta = ModelMetaData.objects.filter(symbol=stock).first()
    if meta:
        context['ensemble_accuracy'] = meta.accuracy
        context['ensemble_balanced_accuracy'] = meta.balanced_accuracy
        # We could parse classification_report json if needed

    # 3. Load Images (Filesystem)
    # Images are still generated by services.py and saved to disk
    cm_path = os.path.join(settings.BASE_DIR, 'saved_states/images', f'{stock}_confusion.png')
    fi_path = os.path.join(settings.BASE_DIR, 'saved_states/images', f'{stock}_feature_importance.png')
    signals_path = os.path.join(settings.BASE_DIR, 'saved_states/images', f'{stock}_signals_test.png')
    equity_path = os.path.join(settings.BASE_DIR, 'saved_states/images', f'{stock}_equity_curve.png')
    
    context['confusion_image'] = f"/media/{stock}_confusion.png" if os.path.exists(cm_path) else None
    context['indicators_comparison_image'] = f"/media/{stock}_feature_importance.png" if os.path.exists(fi_path) else None
    context['signals_test_image'] = f"/media/{stock}_signals_test.png" if os.path.exists(signals_path) else None
    context['equity_image'] = f"/media/{stock}_equity_curve.png" if os.path.exists(equity_path) else None
    
    # 4. Load Backtest Metrics (Excel fallback for now, or could store in ModelMetaData)
    backtest_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock}_backtest_results.xlsx')
    if os.path.exists(backtest_path):
        try:
            backtest_metrics_df = pd.read_excel(backtest_path, sheet_name='Backtest_Metrics')
            context['backtest_metrics'] = dict(zip(backtest_metrics_df['Metric'], backtest_metrics_df['Value']))
        except Exception:
            pass
    
    # Handle Model Training
    if request.method == "POST" and request.POST.get('train_model'):
        try:
            messages.info(request, f"Training started for {stock}... This may take 30–90 seconds.")
            result = run_prediction_pipeline(stock_symbol=stock)
            
            if result.get('status') == 'skip':
                messages.info(request, result['message'])
            elif result['status'] == 'success':
                # Update context from result immediately
                context.update({
                    'confusion_image': result['confusion_image'],
                    'indicators_comparison_image': result['indicators_comparison_image'],
                    'signals_test_image': result['signals_test_image'],
                    'equity_image': result['equity_image'],
                    'ensemble_accuracy': result.get('ensemble_accuracy'),
                    'ensemble_balanced_accuracy': result.get('ensemble_balanced_accuracy'),
                    'backtest_metrics': result.get('backtest_metrics', {}),
                })
                # Reload forecasts from DB
                forecasts = ForecastResult.objects.filter(symbol=stock).order_by('forecast_date')
                context['forecast_signals'] = [
                    {
                        'date': f.forecast_date.strftime('%Y-%m-%d'),
                        'signal': f.predicted_signal,
                        'score': round(f.confidence_score, 4)
                    } for f in forecasts
                ]
                messages.success(request, "New best model saved — accuracy improved!")
            else:
                messages.error(request, result.get('message', 'Training failed.'))
        except Exception as e:
            messages.error(request, f"Training error: {str(e)}")
        
        return redirect(f'/predict/?stock={stock}')
    
    return render(request, 'app/app.html', context)

def welcome_view(request):
    """
    Welcome page with stock selection dropdown (Top 20 restricted).
    """
    stocks_list = settings.TOP_20_STOCKS
    stocks = [(s, f"{s}") for s in stocks_list]

    context = {
        'stocks': stocks,
        'selected_stock': request.GET.get('stock', 'NABIL'),
    }

    if request.method == "POST":
        selected_stock = request.POST.get('stock')
        if selected_stock:
            return redirect(f'/predict/?stock={selected_stock}')

    return render(request, 'app/welcome.html', context)


def log_view(request):
    """
    View to display pipeline logs.
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
