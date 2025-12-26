from django.shortcuts import render
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

def trigger_training_view(request):
    """
    Call this view (e.g., via a button) to run the ML training.
    """
    context = {}
    if request.method == "POST":
        try:
            # Run the heavy logic
            result = run_prediction_pipeline()
            
            if result['status'] == 'success':
                messages.success(request, "Model trained and Excel files updated successfully!")
            else:
                messages.error(request, result['message'])
                
        except Exception as e:
            messages.error(request, f"Error running pipeline: {str(e)}")
            
    return render(request, 'app/app.html', context)





