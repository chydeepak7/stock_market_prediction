from django.shortcuts import render, redirect
import os
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from django.conf import settings
from django.http import HttpResponse 
from django.template.loader import render_to_string
from django import forms
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

# def trigger_training_view(request):
#     """
#     Call this view (e.g., via a button) to run the ML training.
#     """
#     context = {}
#     if request.method == "POST":
#         try:
#             # Run the heavy logic
#             result = run_prediction_pipeline()
            
#             if result['status'] == 'success':
#                 messages.success(request, "Model trained and Excel files updated successfully!")
#             else:
#                 messages.error(request, result['message'])
                
#         except Exception as e:
#             messages.error(request, f"Error running pipeline: {str(e)}")
            
#     return render(request, 'app/app.html', context)

def home_view(request):
    context = {
        'chart_image': None,
        'forecast_chart': None,
    }

    # Correct paths inside saved_states folder
    backtest_path = os.path.join(settings.BASE_DIR, 'saved_states', 'backtest_results.xlsx')
    forecast_path = os.path.join(settings.BASE_DIR, 'saved_states', 'future_30_day_forecast.xlsx')

    # Load Backtest Chart
    if os.path.exists(backtest_path):
        try:
            df_backtest = pd.read_excel(backtest_path)
            df_backtest['Date'] = pd.to_datetime(df_backtest['Date'])
            df_backtest.set_index('Date', inplace=True)
            context['chart_image'] = generate_chart_image(df_backtest, 'backtest')
        except Exception as e:
            messages.error(request, f"Error loading backtest chart: {str(e)}")

    # Load Forecast Chart
    if os.path.exists(forecast_path):
        try:
            df_forecast = pd.read_excel(forecast_path)
            df_forecast['Forecast_Date'] = pd.to_datetime(df_forecast['Forecast_Date'])
            df_forecast.set_index('Forecast_Date', inplace=True)
            context['forecast_chart'] = generate_chart_image(df_forecast, 'forecast')
        except Exception as e:
            messages.error(request, f"Error loading forecast chart: {str(e)}")

    # Handle Model Training (only on valid POST)
    if request.method == "POST" and request.POST.get('train_model'):
        try:
            messages.info(request, "Training started... This may take 30–90 seconds.")
            result = run_prediction_pipeline()

            if result['status'] == 'success':
                messages.success(request, "Model trained successfully! Charts updated below.")
            else:
                messages.error(request, result.get('message', 'Training failed.'))
        except Exception as e:
            messages.error(request, f"Training error: {str(e)}")

        # Prevent re-submission and infinite loop
        return redirect('home')

    return render(request, 'app/app.html', context)



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
