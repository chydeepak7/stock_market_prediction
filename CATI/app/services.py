import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from django.conf import settings

# --- 1. CONFIGURATION ---
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD',
            'MACD_EMA', 'EMA_20', 'BB_Upper', 'BB_Lower', 'ADX',
            'Stoch_K', 'ATR', 'OBV', 'VWAP']

def add_technical_indicators(df):
    """Calculates all technical indicators optimized for speed."""
    df = df.copy() # Prevent SettingWithCopyWarning
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_EMA'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # EMA & BB
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)

    # ATR & ADX
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    plus_dm = (df['High'].diff()).clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    pdi = 100 * (plus_dm.ewm(alpha=1/14).mean() / df['ATR'])
    mdi = 100 * (minus_dm.ewm(alpha=1/14).mean() / df['ATR'])
    dx = 100 * abs(pdi - mdi) / (pdi + mdi)
    df['ADX'] = dx.ewm(alpha=1/14).mean()

    # Stochastic & Others
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    return df.dropna()

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 3]) # Index 3 is Close price
    return np.array(X), np.array(y)

def run_prediction_pipeline():
    """
    Main function to be called from Django View.
    Runs the entire training and prediction process.
    """
    print("--- Starting Prediction Pipeline ---")
    
    # 1. Load Data
    file_path = os.path.join(settings.BASE_DIR, 'saved_states', 'CHCL.xlsx')
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found at {file_path}"}
        
    df = pd.read_excel(file_path)
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Cleaning
    if df['Percent Change'].dtype == object:
        df['Percent Change'] = df['Percent Change'].str.replace('%', '').astype(float) / 100
        
    df_final = add_technical_indicators(df)
    print("running")
    # Date Indexing
    df_final = df_final.reset_index(drop=True)
    if 'Date' in df_final.columns:
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        df_final.set_index('Date', inplace=True)
        df_final.sort_index(inplace=True)

    # 2. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_final[FEATURES])
    
    X_seq, y_seq = create_sequences(scaled_data)
    
    # 3. Train LSTM
    split_index = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    print("running")
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    
    # Fast training settings for web (epochs reduced, early stopping active)
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    model_lstm.fit(X_train, y_train, batch_size=32, epochs=30, verbose=0, callbacks=[early_stopping])

    # 4. Train XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=5)
    model_xgb.fit(X_train_flat, y_train)
    print("running")

    # 5. Hybrid Backtesting
    lstm_preds = model_lstm.predict(X_test, verbose=0)
    xgb_preds = model_xgb.predict(X_test_flat)
    hybrid_preds = (lstm_preds.flatten() + xgb_preds) / 2
    
    # Inverse Transform Helpers
    def get_real_prices(scaled_vector):
        dummy = np.zeros((len(scaled_vector), len(FEATURES)))
        dummy[:, 3] = scaled_vector
        return scaler.inverse_transform(dummy)[:, 3]

    backtest_actual = get_real_prices(y_test)
    backtest_hybrid = get_real_prices(hybrid_preds)

    # --- SAVE BACKTEST RESULTS (As requested) ---
    backtest_df = pd.DataFrame({
        'Date': df_final.index[-len(backtest_actual):],
        'Real_Market_Price': backtest_actual,
        'Hybrid_Prediction': backtest_hybrid
    })
    print("running")
    
    backtest_path = os.path.join(settings.BASE_DIR, 'backtest_results.xlsx')
    backtest_df.to_excel(backtest_path, index=False, sheet_name='Backtest_Results')

    # 6. Future Forecasting (Next 30 Days)
    last_60_days = scaled_data[-60:]
    current_batch = last_60_days.reshape((1, 60, len(FEATURES)))
    future_predictions = []
    
    # Calculate volatility for noise
    volatility = df_final['Close'].pct_change().std()

    for i in range(30):
        lstm_p = model_lstm.predict(current_batch, verbose=0)[0][0]
        xgb_p = model_xgb.predict(current_batch.reshape(1, -1))[0]
        
        # Hybrid + Noise
        noise = np.random.normal(0, volatility * 0.2)
        hybrid_p = ((lstm_p + xgb_p) / 2) + noise
        future_predictions.append(hybrid_p)
        
        # Update Batch
        new_row = np.copy(current_batch[0, -1, :])
        new_row[3] = hybrid_p
        new_row = new_row.reshape((1, 1, len(FEATURES)))
        current_batch = np.append(current_batch[:, 1:, :], new_row, axis=1)

    # Inverse Scale Future
    res_dummy = np.zeros((30, len(FEATURES)))
    res_dummy[:, 3] = np.array(future_predictions)
    unscaled_forecast = scaler.inverse_transform(res_dummy)[:, 3]

    last_date = df_final.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    print("running")

    # --- SAVE FORECAST RESULTS (As requested) ---
    forecast_results_df = pd.DataFrame({
        'Forecast_Date': forecast_dates,
        'Predicted_Close_Price': unscaled_forecast
    })
    
    forecast_path = os.path.join(settings.BASE_DIR, 'future_30_day_forecast.xlsx')
    forecast_results_df.to_excel(forecast_path, index=False, sheet_name='Forecast')

    print("Pipeline Complete. Files Saved.")
    return {"status": "success", "backtest_file": backtest_path, "forecast_file": forecast_path}