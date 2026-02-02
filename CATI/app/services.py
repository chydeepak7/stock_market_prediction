# # services.py
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    balanced_accuracy_score
)
from sklearn.utils import resample
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger('pipeline')

# --- CONFIGURATION ---
FEATURES = ['RSI', 'MACD', 'MACD_EMA', 'EMA_20', 'BB_Upper', 'BB_Lower', 'ATR','OBV','VWAP', 'Beta']
SEQ_LEN = 30


# ==================== DATABASE HELPER FUNCTIONS ====================

def get_stock_data_from_db(symbol, use_csv_fallback=True):
    """
    Fetch stock data from database. Falls back to CSV if DB is empty.
    Returns a DataFrame with standard column names.
    """
    from app.models import StockData
    
    qs = StockData.objects.filter(symbol=symbol).order_by('date')
    
    if qs.exists():
        logger.info(f"[SERVICES] Loading {symbol} from database ({qs.count()} records)")
        data = list(qs.values('date', 'open', 'high', 'low', 'close', 'volume'))
        df = pd.DataFrame(data)
        df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 
                          'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    elif use_csv_fallback:
        logger.warning(f"[SERVICES] No DB data for {symbol}, falling back to CSV")
        return _load_from_csv(symbol)
    
    else:
        logger.error(f"[SERVICES] No data found for {symbol}")
        return None


def get_market_data_from_db(use_csv_fallback=True):
    """Fetch NEPSE market data from database."""
    return get_stock_data_from_db('NEPSE', use_csv_fallback)


def _load_from_csv(symbol):
    """Load stock data from CSV file (fallback method)."""
    file_path = os.path.join(settings.BASE_DIR, 'saved_states/data', f'{symbol}.csv')
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df = df.iloc[::-1].reset_index(drop=True)
    
    rename_dict = {
        'time': 'Date', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }
    df.rename(columns=rename_dict, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df


def save_forecast_to_db(symbol, forecast_df):
    """Save forecast results to ForecastResult model."""
    from app.models import ForecastResult
    
    # Clear old forecasts for this symbol
    ForecastResult.objects.filter(symbol=symbol).delete()
    
    # Create new forecast records
    forecasts = []
    for _, row in forecast_df.iterrows():
        forecasts.append(ForecastResult(
            symbol=symbol,
            forecast_date=row['Forecast_Date'],
            predicted_signal=row['Signal'],
            confidence_score=float(row['Score'])
        ))
    
    ForecastResult.objects.bulk_create(forecasts)
    logger.info(f"[SERVICES] Saved {len(forecasts)} forecasts for {symbol} to database")
    return len(forecasts)


def save_model_metadata_to_db(symbol, accuracy, balanced_accuracy, classification_report_dict):
    """Save model training metadata to ModelMetaData model."""
    from app.models import ModelMetaData
    
    # Update or create metadata for this symbol
    metadata, created = ModelMetaData.objects.update_or_create(
        symbol=symbol,
        defaults={
            'last_trained': timezone.now(),
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'classification_report': json.dumps(classification_report_dict),
        }
    )
    
    action = "Created" if created else "Updated"
    logger.info(f"[SERVICES] {action} model metadata for {symbol}: acc={accuracy:.2%}")
    return metadata


def get_available_symbols():
    """Get list of symbols available in the database."""
    from app.models import StockData
    return list(StockData.objects.values_list('symbol', flat=True).distinct())



def add_all_indicators(df_stock, df_market):
    df = df_stock.copy()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_EMA'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    mid = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = mid + 2 * std
    df['BB_Lower'] = mid - 2 * std

    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)

    df['ATR'] = tr.rolling(14).mean()

    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    # Beta
    if not df_market.empty:
        returns_stock = df['Close'].pct_change()
        returns_market = df_market['Close'].pct_change()
        returns = pd.DataFrame({'stock': returns_stock, 'market': returns_market}).dropna()
        window = 60
        cov = returns['stock'].rolling(window).cov(returns['market'])
        var = returns['market'].rolling(window).var()
        beta = cov / var
        df['Beta'] = beta.reindex(df.index)
    else:
        df['Beta'] = 1.0
        
    # Fill any remaining NaNs in Beta (e.g. non-overlapping dates) with 1.0
    df['Beta'] = df['Beta'].fillna(1.0)

    return df.dropna()

def add_binary_barrier_labels(df, horizon=40, atr_mult=2.0, deadzone=1.1):
    labels = []

    for i in range(len(df)):
        if i + horizon >= len(df):
            labels.append(np.nan)
            continue

        price_now = df['Close'].iloc[i]
        price_future = df['Close'].iloc[i + horizon]
        atr = df['ATR'].iloc[i]

        threshold = atr_mult * atr
        diff = price_future - price_now

        if diff > threshold * deadzone:
            labels.append(1)
        elif diff < -threshold * deadzone:
            labels.append(0)
        else:
            labels.append(np.nan)

    df = df.copy()
    df['Label'] = labels
    df = df.dropna()
    df['Label'] = df['Label'].astype(int)

    print("Label distribution:")
    print(df['Label'].value_counts())

    # FALLBACK FOR LOW-VOL STOCKS (like CHCL)
    if df['Label'].nunique() < 2 or df['Label'].value_counts().min() < 20:
        print("Low volatility detected → fallback to next-day direction label")
        df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(subset=['Label'], inplace=True)

    print("Final label distribution:", df['Label'].value_counts().to_dict())

    return df

# def build_sequences(X_data, y_data, seq_len):
#     X, y = [], []
#     for i in range(seq_len, len(X_data)):
#         X.append(X_data[i-seq_len:i])
#         y.append(y_data[i])
#     return np.array(X), np.array(y)

def build_future_tabular_inputs(df, feature_cols, feature_scaler, seq_length=30, future_days=30):
    df_temp = df.copy()
    X_future = []
    for _ in range(future_days):
        last_window = df_temp[feature_cols].tail(seq_length)
        last_scaled = feature_scaler.transform(last_window.values)
        X_future.append(last_scaled)
        # Synthetic next close
        recent_ret = df_temp['Close'].pct_change().tail(20).mean()
        recent_vol = df_temp['Close'].pct_change().tail(20).std()
        recent_ret = 0 if np.isnan(recent_ret) else recent_ret
        recent_vol = 0.005 if np.isnan(recent_vol) or recent_vol == 0 else recent_vol
        noise = np.random.normal(recent_ret, recent_vol)
        new_close = df_temp['Close'].iloc[-1] * (1 + noise)
        new_row = df_temp.iloc[-1].copy()
        new_row['Close'] = new_close
        new_row['Open'] = new_close
        new_row['High'] = new_close * (1 + abs(noise))
        new_row['Low'] = new_close * (1 - abs(noise))
        new_row['Volume'] = df_temp['Volume'].rolling(10).mean().iloc[-1]
        df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)
    return np.array(X_future)

def backtest_long_only(prices, signals):
    prices = np.asarray(prices)
    position = np.roll((signals == 1).astype(int), 1)
    position[0] = 0
    returns = np.diff(prices) / prices[:-1]
    strat_returns = returns * position[:-1]
    equity = np.cumprod(1 + np.nan_to_num(strat_returns, 0))
    sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-6) * np.sqrt(252)
    dd = equity / np.maximum.accumulate(equity) - 1
    exposure = position.mean()
    return {
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(dd.min() * 100, 2),   # in percent
        'exposure': round(exposure * 100, 2),
    }, equity

def run_prediction_pipeline(stock_symbol='NABIL'):
    print("--- Starting Prediction Pipeline ---")
    
    # 1. Load Stock Data (from DB)
    df_stock = get_stock_data_from_db(stock_symbol)
    if df_stock is None or df_stock.empty:
        return {"status": "error", "message": f"No data found for {stock_symbol} in DB or CSV"}
    
    # 2. Load Market Data (from DB)
    df_market = get_market_data_from_db()
    if df_market is None or df_market.empty:
        print("Warning: NEPSE data not found in DB, indicators checking beta will fail")
        df_market = pd.DataFrame()
        
    print("Fetched dataset from Database")
    
    # 3. Add Indicators
    df_final = add_all_indicators(df_stock, df_market)
    
    # 4. Add Labels
    df_final = add_binary_barrier_labels(df_final, atr_mult=1.0, deadzone=1.0)
    
    # 5. Scale Features
    scaler = MinMaxScaler()
    X_raw = scaler.fit_transform(df_final[FEATURES])
    y_raw = df_final['Label'].values
    
    # 6. Balance classes
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_raw, y_raw)
    print("Indicators added and labels generated")
    
    # 7. Build Sequences
    X, y = [], []
    for i in range(SEQ_LEN, len(X_bal)):
        X.append(X_bal[i-SEQ_LEN:i])
        y.append(y_bal[i])

    X = np.array(X)
    y = np.array(y)
    
    # 8. Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print("Data split into training and testing sets")
    
    # 9. Train LSTM
    model = Sequential([
        Input(shape=(SEQ_LEN, len(FEATURES))),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=35,
        batch_size=64,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
        verbose=1
    )
    print("LSTM model trained")
    
    # 10. Train XGB and RF
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_flat, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_flat, y_train)

    print("XGBoost and Random Forest models trained")
    
    # 11. Ensemble
    lstm_p = model.predict(X_test, verbose=0).flatten()
    xgb_p = xgb_model.predict_proba(X_test_flat)[:, 1]
    rf_p = rf_model.predict_proba(X_test_flat)[:, 1]
    
    acc_lstm = balanced_accuracy_score(y_test, (lstm_p > 0.5).astype(int))
    acc_xgb = balanced_accuracy_score(y_test, (xgb_p > 0.5).astype(int))
    acc_rf = balanced_accuracy_score(y_test, (rf_p > 0.5).astype(int))
    
    w_sum = acc_lstm + acc_xgb + acc_rf
    if w_sum == 0:
        w_lstm = w_xgb = w_rf = 1/3
    else:
        w_lstm, w_xgb, w_rf = acc_lstm/w_sum, acc_xgb/w_sum, acc_rf/w_sum
    
    ensemble_score = w_lstm * lstm_p + w_xgb * xgb_p + w_rf * rf_p
    ensemble_pred = (ensemble_score > 0.6).astype(int)
    
    print("Ensemble predictions generated")
    
    # Metrics
    acc = accuracy_score(y_test, ensemble_pred) 
    bal_acc = balanced_accuracy_score(y_test, ensemble_pred) 
    cm = confusion_matrix(y_test, ensemble_pred, labels=[0, 1])
    
    # Save Model Metadata to DB
    cls_report = classification_report(y_test, ensemble_pred, target_names=["SELL","BUY"], output_dict=True)
    save_model_metadata_to_db(stock_symbol, acc, bal_acc, cls_report)
    
    conf_df = pd.DataFrame(cm, index=['SELL', 'BUY'], columns=['Pred SELL', 'Pred BUY'])
    
    # Feature Importance
    importances = rf_model.feature_importances_.reshape(SEQ_LEN, len(FEATURES)).mean(axis=0)
    fi_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    # Backtest
    prices_test = df_final['Close'].values[-len(y_test):]
    backtest_metrics, equity = backtest_long_only(prices_test, ensemble_pred)
    
    # Save Backtest Results (Excel backup)
    backtest_df = pd.DataFrame({
        'Date': df_final.index[-len(y_test):],
        'Close': prices_test,
        'Ensemble_Pred': ensemble_pred
    })
    
    backtest_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock_symbol}_backtest_results.xlsx')
    with pd.ExcelWriter(backtest_path, engine='openpyxl') as writer:
        backtest_df.to_excel(writer, index=False, sheet_name='Backtest_Results')
        pd.DataFrame({
            'Metric': ['Accuracy', 'Balanced Accuracy'],
            'Value': [acc, bal_acc]
        }).to_excel(writer, index=False, sheet_name='Metrics')
        
    print("Generating future predictions")
    
    # 12. Future Predictions
    X_future = build_future_tabular_inputs(df_final, FEATURES, scaler, seq_length=SEQ_LEN, future_days=30)
    lstm_future_p = model.predict(X_future, verbose=0).flatten()
    xgb_future_p = xgb_model.predict_proba(X_future.reshape(len(X_future), -1))[:, 1]
    rf_future_p = rf_model.predict_proba(X_future.reshape(len(X_future), -1))[:, 1]
    
    future_score = w_lstm * lstm_future_p + w_xgb * xgb_future_p + w_rf * rf_future_p
    future_signal = np.where(future_score > 0.55, 'BUY', 'SELL')
    future_dates = pd.date_range(start=df_final.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
    
    forecast_df = pd.DataFrame({
        'Forecast_Date': future_dates,
        'Signal': future_signal,
        'Score': future_score
    })
    
    # Save Forecasts to DB
    saved_count = save_forecast_to_db(stock_symbol, forecast_df)
    
    # Also save to Excel as artifact
    forecast_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock_symbol}_future_30_day_forecast.xlsx')
    forecast_df.to_excel(forecast_path, index=False, sheet_name='Forecast')
    
    # === Image saving ===
    images_dir = os.path.join(settings.BASE_DIR, 'saved_states', 'images')
    os.makedirs(images_dir, exist_ok=True)

    def save_plot(fig, filename):
        path = os.path.join(images_dir, f"{stock_symbol}_{filename}")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return f"/media/{stock_symbol}_{filename}"

    # 1. Confusion Matrix
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, linewidths=0.5)
    ax.set_title('Ensemble Confusion Matrix')
    confusion_url = save_plot(fig, 'confusion.png')

    # 2. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    fi_df.plot(kind='barh', x='Feature', y='Importance', ax=ax, color='skyblue', legend=False)
    ax.set_title('Technical Indicators Importance')
    ax.invert_yaxis()
    fi_url = save_plot(fig, 'feature_importance.png')

    # 3. Buy/Sell Signals
    fig, ax = plt.subplots(figsize=(12, 6))
    prices_test = df_final['Close'].values[-len(y_test):]
    ax.plot(df_final.index[-len(y_test):], prices_test, label='Price (Test Period)', color='blue', alpha=0.8)
    buy_idx = np.where(ensemble_pred == 1)[0]
    sell_idx = np.where(ensemble_pred == 0)[0]
    ax.scatter(df_final.index[-len(y_test):][buy_idx], prices_test[buy_idx], marker='^', s=80, color='green', label='BUY Signal', zorder=5)
    ax.scatter(df_final.index[-len(y_test):][sell_idx], prices_test[sell_idx], marker='v', s=80, color='red', label='SELL Signal', zorder=5)
    ax.set_title('Ensemble Buy/Sell Signals (Test Period)')
    ax.legend()
    ax.grid(alpha=0.3)
    signals_url = save_plot(fig, 'signals_test.png')

    # 4. Equity Curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.concatenate(([1], equity)), color='purple', linewidth=2)
    ax.set_title('Backtest Equity Curve (Long-Only)')
    ax.grid(alpha=0.3)
    ax.set_ylabel('Equity Growth')
    equity_url = save_plot(fig, 'equity_curve.png')

    print(f"Pipeline Complete for {stock_symbol}. Saved {saved_count} forecasts to DB.")
    return {
        "status": "success",
        "confusion_image": confusion_url,
        "indicators_comparison_image": fi_url,
        "signals_test_image": signals_url,
        "equity_image": equity_url,
        "ensemble_accuracy": round(acc *100, 4),
        "ensemble_balanced_accuracy": round(bal_acc *100, 4),
        "backtest_metrics": backtest_metrics
    }










# def run_prediction_pipeline(stock_symbol='NABIL'):
#     print("--- Starting Prediction Pipeline ---")
    
#     # 1. Load Stock Data
#     file_path = os.path.join(settings.BASE_DIR, 'saved_states/data', f'{stock_symbol}.csv')
#     if not os.path.exists(file_path):
#         return {"status": "error", "message": f"File {stock_symbol}.csv not found at {file_path}"}
    
#     df_stock = pd.read_csv(file_path)
#     df_stock = df_stock.iloc[::-1].reset_index(drop=True)
    
#     # Rename columns if necessary
#     rename_dict = {
#         'time': 'Date',
#         'open': 'Open',
#         'high': 'High',
#         'low': 'Low',
#         'close': 'Close',
#         'volume': 'Volume',
#         'category': 'Category'
#     }
#     df_stock.rename(columns=rename_dict, inplace=True)
    
#     df_stock['Date'] = pd.to_datetime(df_stock['Date'])
#     df_stock.set_index('Date', inplace=True)
#     df_stock.sort_index(inplace=True)
    
#     # Load Market Data
#     market_path = os.path.join(settings.BASE_DIR, 'saved_states/data', 'NEPSE.csv')
#     if not os.path.exists(market_path):
#         return {"status": "error", "message": "NEPSE.csv not found"}
    
#     df_market = pd.read_csv(market_path)
#     df_market.rename(columns=rename_dict, inplace=True)
#     df_market = df_market.iloc[::-1].reset_index(drop=True)
#     df_market['Date'] = pd.to_datetime(df_market['Date'])
#     df_market.set_index('Date', inplace=True)
#     df_market.sort_index(inplace=True)
#     print("Fetched dataset ")
#     # 2. Add Indicators
#     df_final = add_all_indicators(df_stock, df_market)
    
#     # 3. Add Labels
#     df_final = add_binary_barrier_labels(df_final, atr_mult=1.0, deadzone=1.0)
    
#     # 4. Scale Features
#     scaler = MinMaxScaler()
#     X_raw = scaler.fit_transform(df_final[FEATURES])
#     y_raw = df_final['Label'].values
    
#     # 5. Balance classes
#     smote = SMOTE(random_state=42)
#     X_bal, y_bal = smote.fit_resample(X_raw, y_raw)
#     print("Indicators added and labels generated ")
#     # df_bal = pd.DataFrame(X_bal, columns=FEATURES)
#     # df_bal['Label'] = y_bal
#     # class_counts = df_bal['Label'].value_counts()
#     # if len(class_counts) < 2:
#     #     return {"status": "error", "message": "Only one class in labels - cannot train classifier"}
    
#     # minority_class = class_counts.idxmin()
#     # majority_class = class_counts.idxmax()
#     # minority = df_bal[df_bal['Label'] == minority_class]
#     # majority = df_bal[df_bal['Label'] == majority_class]
#     # minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
#     # df_bal = pd.concat([majority, minority_up])
#     # X_bal = df_bal[FEATURES].values
#     # y_bal = df_bal['Label'].values
    
#     # 6. Build Sequences
#     X, y = [], []
#     for i in range(SEQ_LEN, len(X_bal)):
#         X.append(X_bal[i-SEQ_LEN:i])
#         y.append(y_bal[i])

#     X = np.array(X)
#     y = np.array(y)
#     # X, y = build_sequences(X_bal, y_bal, SEQ_LEN)
    
#     # 7. Split
#     split = int(0.8 * len(X))
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]
#     print("Data split into training and testing sets ")
#     # 8. Train LSTM
#     model = Sequential([
#     Input(shape=(SEQ_LEN, len(FEATURES))),
#     LSTM(64),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     model.fit(
#         X_train, y_train,
#         validation_data=(X_test, y_test),
#         epochs=35,
#         batch_size=64,
#         callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
#         verbose=1
#     )
#     print("LSTM model trained ")
    
#     # 9. Train XGB and RF
#     X_train_flat = X_train.reshape(len(X_train), -1)
#     X_test_flat  = X_test.reshape(len(X_test), -1)

#     xgb_model = xgb.XGBClassifier(
#         n_estimators=300,
#         max_depth=5,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         eval_metric='logloss'
#     )
#     xgb_model.fit(X_train_flat, y_train)

#     rf_model = RandomForestClassifier(
#         n_estimators=300,
#         max_depth=8,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
#     )
#     rf_model.fit(X_train_flat, y_train)

#     print("XGBoost and Random Forest models trained ")
    
#     # 10. Ensemble
#     lstm_p = model.predict(X_test, verbose=0).flatten()
#     xgb_p = xgb_model.predict_proba(X_test_flat)[:, 1]
#     rf_p = rf_model.predict_proba(X_test_flat)[:, 1]
    
#     acc_lstm = balanced_accuracy_score(y_test, (lstm_p > 0.5).astype(int))
#     acc_xgb = balanced_accuracy_score(y_test, (xgb_p > 0.5).astype(int))
#     acc_rf = balanced_accuracy_score(y_test, (rf_p > 0.5).astype(int))
    
#     w_sum = acc_lstm + acc_xgb + acc_rf
#     if w_sum == 0:
#         w_lstm = w_xgb = w_rf = 1/3
#     else:
#         w_lstm, w_xgb, w_rf = acc_lstm/w_sum, acc_xgb/w_sum, acc_rf/w_sum
    
#     ensemble_score = w_lstm * lstm_p + w_xgb * xgb_p + w_rf * rf_p
#     ensemble_pred = (ensemble_score > 0.6).astype(int)
    
#     print("Ensemble predictions generated ")
    
#     print("Weights:", w_lstm, w_xgb, w_rf)
#     print("Accuracy:", accuracy_score(y_test, ensemble_pred))
#     print("Balanced Accuracy:", balanced_accuracy_score(y_test, ensemble_pred))
#     print("Classification Report:")
#     print(classification_report(y_test, ensemble_pred, target_names=["SELL","BUY"]))
    
#     # 11. Metrics
#     acc = accuracy_score(y_test, ensemble_pred)
#     bal_acc = balanced_accuracy_score(y_test, ensemble_pred)
#     cm = confusion_matrix(y_test, ensemble_pred,labels=[0, 1])
    
#     conf_df = pd.DataFrame(cm, index=['SELL', 'BUY'], columns=['Pred SELL', 'Pred BUY'])
#     try:
#         report = classification_report(y_test, ensemble_pred, target_names=["SELL", "BUY"], output_dict=True)
#     except ValueError:
#         report = {"accuracy": acc}
    

#     # 12. Feature Importance
#     importances = rf_model.feature_importances_.reshape(SEQ_LEN, len(FEATURES)).mean(axis=0)
#     fi_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances}).sort_values('Importance', ascending=False)
    
#     # 13. Backtest
#     prices_test = df_final['Close'].values[-len(y_test):]
#     backtest_metrics, equity = backtest_long_only(prices_test, ensemble_pred)
    
#     # 14. Save Backtest Results
#     backtest_df = pd.DataFrame({
#         'Date': df_final.index[-len(y_test):],
#         'Close': prices_test,
#         'Ensemble_Pred': ensemble_pred
#     })
    
#     print("Generating backtest results ")

#     backtest_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock_symbol}_backtest_results.xlsx')
#     with pd.ExcelWriter(backtest_path, engine='openpyxl') as writer:
#         backtest_df.to_excel(writer, index=False, sheet_name='Backtest_Results')
#         pd.DataFrame({
#             'Metric': ['Accuracy', 'Balanced Accuracy'],
#             'Value': [acc, bal_acc]
#         }).to_excel(writer, index=False, sheet_name='Metrics')
#         pd.DataFrame(cm, index=['SELL', 'BUY'], columns=['Pred SELL', 'Pred BUY']).to_excel(writer, sheet_name='Confusion_Matrix')
#         fi_df.to_excel(writer, index=False, sheet_name='Feature_Importance')
#         pd.DataFrame({
#             'Metric': list(backtest_metrics.keys()),
#             'Value': list(backtest_metrics.values())
#         }).to_excel(writer, index=False, sheet_name='Backtest_Metrics')
#         pd.DataFrame({
#             'Date': backtest_df['Date'].values,
#             'Equity': np.concatenate(([1], equity))
#         }).to_excel(writer, index=False, sheet_name='Equity_Curve')
    

#     print("Generating future predictions ")
#     # 15. Future Predictions
#     X_future = build_future_tabular_inputs(df_final, FEATURES, scaler, seq_length=SEQ_LEN, future_days=30)
#     lstm_future_p = model.predict(X_future, verbose=0).flatten()
#     xgb_future_p = xgb_model.predict_proba(X_future.reshape(len(X_future), -1))[:, 1]
#     rf_future_p = rf_model.predict_proba(X_future.reshape(len(X_future), -1))[:, 1]
#     future_score = w_lstm * lstm_future_p + w_xgb * xgb_future_p + w_rf * rf_future_p
#     future_signal = np.where(future_score > 0.55, 'BUY', 'SELL')
#     future_dates = pd.date_range(start=df_final.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
#     forecast_df = pd.DataFrame({
#         'Forecast_Date': future_dates,
#         'Signal': future_signal,
#         'Score': future_score
#     })
#     forecast_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock_symbol}_future_30_day_forecast.xlsx')
#     forecast_df.to_excel(forecast_path, index=False, sheet_name='Forecast')
    
#     # === Image saving ===
#     images_dir = os.path.join(settings.BASE_DIR, 'saved_states', 'images')
#     os.makedirs(images_dir, exist_ok=True)

#     def save_plot(fig, filename):
#         path = os.path.join(images_dir, f"{stock_symbol}_{filename}")
#         fig.savefig(path, dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         return f"/media/{stock_symbol}_{filename}"

#     # 1. Confusion Matrix
#     fig, ax = plt.subplots(figsize=(6, 5))
#     sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, linewidths=0.5)
#     ax.set_title('Ensemble Confusion Matrix')
#     confusion_url = save_plot(fig, 'confusion.png')

#     # 2. Feature Importance
#     fig, ax = plt.subplots(figsize=(10, 6))
#     fi_df.plot(kind='barh', x='Feature', y='Importance', ax=ax, color='skyblue', legend=False)
#     ax.set_title('Technical Indicators Importance')
#     ax.invert_yaxis()
#     fi_url = save_plot(fig, 'feature_importance.png')

#     # 3. Buy/Sell Signals on Test Period
#     fig, ax = plt.subplots(figsize=(12, 6))
#     prices_test = df_final['Close'].values[-len(y_test):]
#     ax.plot(df_final.index[-len(y_test):], prices_test, label='Price (Test Period)', color='blue', alpha=0.8)
#     buy_idx = np.where(ensemble_pred == 1)[0]
#     sell_idx = np.where(ensemble_pred == 0)[0]
#     ax.scatter(df_final.index[-len(y_test):][buy_idx], prices_test[buy_idx], marker='^', s=80, color='green', label='BUY Signal', zorder=5)
#     ax.scatter(df_final.index[-len(y_test):][sell_idx], prices_test[sell_idx], marker='v', s=80, color='red', label='SELL Signal', zorder=5)
#     ax.set_title('Ensemble Buy/Sell Signals (Test Period)')
#     ax.legend()
#     ax.grid(alpha=0.3)
#     signals_url = save_plot(fig, 'signals_test.png')

#     # 4. Equity Curve
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.plot(np.concatenate(([1], equity)), color='purple', linewidth=2)
#     ax.set_title('Backtest Equity Curve (Long-Only)')
#     ax.grid(alpha=0.3)
#     ax.set_ylabel('Equity Growth')
#     equity_url = save_plot(fig, 'equity_curve.png')

#     print(f"Pipeline Complete for {stock_symbol}. Files and images saved.")
#     return {
#         "status": "success",
#         "confusion_image": confusion_url,
#         "indicators_comparison_image": fi_url,
#         "signals_test_image": signals_url,
#         "equity_image": equity_url,
#         "ensemble_accuracy": round(acc, 4),
#         "ensemble_balanced_accuracy": round(bal_acc, 4),
#     }

def plot_test_prices(prices,prices_test,ensemble_pred):
    # Convert ensemble prediction to direction
    # 1 = BUY, 0 = SELL
    ensemble_dir = ensemble_pred.copy()

    plt.figure(figsize=(12,6))  
    plt.plot(prices_test, label="Price (test period)", alpha=0.8)

    buy_idx  = np.where(ensemble_dir == 1)[0]
    sell_idx = np.where(ensemble_dir == 0)[0]

    plt.scatter(buy_idx,  prices_test[buy_idx], marker="^", s=70, label="BUY")
    plt.scatter(sell_idx, prices_test[sell_idx], marker="v", s=70, label="SELL")

    plt.legend()
    plt.title("Ensemble Buy / Sell Signals (Test Period)")
    plt.grid(alpha=0.3)
    plt.show()


# import os
# import io
# import base64
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score, confusion_matrix
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from xgboost import XGBRegressor
# from django.conf import settings



# # --- 1. CONFIGURATION ---
# FEATURES = [
#     'RSI', 'MACD', 'MACD_EMA', 'EMA_20',
#     'BB_Upper', 'BB_Lower', 'ADX',
#     'Stoch_K', 'ATR', 'OBV', 'VWAP', 'Beta'
# ]

# # ================= HELPERS =================
# def save_plot(fig, filename):
#     path = os.path.join(settings.BASE_DIR, 'saved_states', filename)
#     fig.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     return path


# def create_buy_sell_labels(prices, threshold=0.002):
#     labels = []
#     for i in range(1, len(prices)):
#         change = (prices[i] - prices[i-1]) / prices[i-1]
#         labels.append(1 if change > threshold else 0)  # 1=BUY, 0=SELL
#     return np.array(labels)


# def add_technical_indicators(df):
#     """Calculates all technical indicators optimized for speed."""
#     df = df.copy() # Prevent SettingWithCopyWarning
    
#     # RSI (14)
#     delta = df['Close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['RSI'] = 100 - (100 / (1 + rs))

#     # MACD (12, 26, 9)
#     exp12 = df['Close'].ewm(span=12, adjust=False).mean()
#     exp26 = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = exp12 - exp26
#     df['MACD_EMA'] = df['MACD'].ewm(span=9, adjust=False).mean()

#     # EMA & BB
#     df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
#     df['BB_Middle'] = df['Close'].rolling(window=20).mean()
#     std = df['Close'].rolling(window=20).std()
#     df['BB_Upper'] = df['BB_Middle'] + (std * 2)
#     df['BB_Lower'] = df['BB_Middle'] - (std * 2)

#     # ATR & ADX
#     tr1 = df['High'] - df['Low']
#     tr2 = abs(df['High'] - df['Close'].shift())
#     tr3 = abs(df['Low'] - df['Close'].shift())
#     tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#     df['ATR'] = tr.rolling(window=14).mean()

#     plus_dm = (df['High'].diff()).clip(lower=0)
#     minus_dm = (-df['Low'].diff()).clip(lower=0)
#     pdi = 100 * (plus_dm.ewm(alpha=1/14).mean() / df['ATR'])
#     mdi = 100 * (minus_dm.ewm(alpha=1/14).mean() / df['ATR'])
#     dx = 100 * abs(pdi - mdi) / (pdi + mdi)
#     df['ADX'] = dx.ewm(alpha=1/14).mean()

#     # Stochastic & Others
#     low_14 = df['Low'].rolling(window=14).min()
#     high_14 = df['High'].rolling(window=14).max()
#     df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
#     df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
#     df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

#     return df.dropna()

# def create_sequences(data, seq_length=60):
#     X, y = [], []
#     for i in range(seq_length, len(data)):
#         X.append(data[i-seq_length:i])
#         y.append(data[i, 3]) # Index 3 is Close price
#     return np.array(X), np.array(y)

# def run_prediction_pipeline(stock_symbol='CHCL'):
#     """
#     Main function to be called from Django View.
#     Runs the entire training and prediction process.
#     """
#     print("--- Starting Prediction Pipeline ---")
    
#     # 1. Load Data
#     file_path = os.path.join(settings.BASE_DIR, 'saved_states/data', f'{stock_symbol}.xlsx')
#     if not os.path.exists(file_path):
#         return {"status": "error", "message": f"File {stock_symbol}.xlsx not found at {file_path}"}
        
#     df = pd.read_excel(file_path)
#     df = df.iloc[::-1].reset_index(drop=True)
    
#     # Cleaning
#     if df['Percent Change'].dtype == object:
#         df['Percent Change'] = df['Percent Change'].str.replace('%', '').astype(float) / 100
        
#     df_final = add_technical_indicators(df)
#     print("running")
#     # Date Indexing
#     df_final = df_final.reset_index(drop=True)
#     if 'Date' in df_final.columns:
#         df_final['Date'] = pd.to_datetime(df_final['Date'])
#         df_final.set_index('Date', inplace=True)
#         df_final.sort_index(inplace=True)

#     # 2. Scaling
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(df_final[FEATURES])
    
#     X_seq, y_seq = create_sequences(scaled_data)
    
#     # 3. Train LSTM
#     split_index = int(len(X_seq) * 0.8)
#     X_train, X_test = X_seq[:split_index], X_seq[split_index:]
#     y_train, y_test = y_seq[:split_index], y_seq[split_index:]

#     model_lstm = Sequential([
#         LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(1)
#     ])
#     print("running")
#     model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    
#     # Fast training settings for web (epochs reduced, early stopping active)
#     early_stopping = EarlyStopping(monitor='loss', patience=3)
#     model_lstm.fit(X_train, y_train, batch_size=32, epochs=30, verbose=0, callbacks=[early_stopping])

#     # 4. Train XGBoost
#     X_train_flat = X_train.reshape(X_train.shape[0], -1)
#     X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
#     model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=5)
#     model_xgb.fit(X_train_flat, y_train)
#     print("running")

#     # 5. Hybrid Backtesting
#     lstm_preds = model_lstm.predict(X_test, verbose=0)
#     xgb_preds = model_xgb.predict(X_test_flat)
#     hybrid_preds = (lstm_preds.flatten() + xgb_preds) / 2
    
#     # Inverse Transform Helpers
#     def get_real_prices(scaled_vector):
#         dummy = np.zeros((len(scaled_vector), len(FEATURES)))
#         dummy[:, 3] = scaled_vector
#         return scaler.inverse_transform(dummy)[:, 3]

#     backtest_actual = get_real_prices(y_test)
#     backtest_hybrid = get_real_prices(hybrid_preds)

#     # --- SAVE BACKTEST RESULTS (As requested) ---
#     backtest_df = pd.DataFrame({
#         'Date': df_final.index[-len(backtest_actual):],
#         'Real_Market_Price': backtest_actual,
#         'Hybrid_Prediction': backtest_hybrid
#     })
#     print("running")
    
#     backtest_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock_symbol}_backtest_results.xlsx')
#     backtest_df.to_excel(backtest_path, index=False, sheet_name='Backtest_Results')

#     # 6. Future Forecasting (Next 30 Days)
#     last_60_days = scaled_data[-60:]
#     current_batch = last_60_days.reshape((1, 60, len(FEATURES)))
#     future_predictions = []
    
#     # Calculate volatility for noise
#     volatility = df_final['Close'].pct_change().std()

#     for i in range(30):
#         lstm_p = model_lstm.predict(current_batch, verbose=0)[0][0]
#         xgb_p = model_xgb.predict(current_batch.reshape(1, -1))[0]
        
#         # Hybrid + Noise
#         noise = np.random.normal(0, volatility * 0.2)
#         hybrid_p = ((lstm_p + xgb_p) / 2) + noise
#         future_predictions.append(hybrid_p)
        
#         # Update Batch
#         new_row = np.copy(current_batch[0, -1, :])
#         new_row[3] = hybrid_p
#         new_row = new_row.reshape((1, 1, len(FEATURES)))
#         current_batch = np.append(current_batch[:, 1:, :], new_row, axis=1)

#     # Inverse Scale Future
#     res_dummy = np.zeros((30, len(FEATURES)))
#     res_dummy[:, 3] = np.array(future_predictions)
#     unscaled_forecast = scaler.inverse_transform(res_dummy)[:, 3]

#     last_date = df_final.index[-1]
#     forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
#     print("running")

#     # --- SAVE FORECAST RESULTS (As requested) ---
#     forecast_results_df = pd.DataFrame({
#         'Forecast_Date': forecast_dates,
#         'Predicted_Close_Price': unscaled_forecast
#     })
    
#     forecast_path = os.path.join(settings.BASE_DIR, 'saved_states', f'{stock_symbol}_future_30_day_forecast.xlsx')
#     forecast_results_df.to_excel(forecast_path, index=False, sheet_name='Forecast')

#     print(f"Pipeline Complete for {stock_symbol}. Files Saved.")
#     return {"status": "success", "backtest_file": backtest_path, "forecast_file": forecast_path}