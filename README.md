# Comparative Analysis of Technical Indicators (CATI)

A Django-based Stock Market Prediction platform for **NEPSE (Nepal Stock Exchange)**. Uses a Hybrid ML Ensemble (LSTM + XGBoost + Random Forest) to forecast stock trends.

---

## Features

- **Automated Data Pipeline**: Scrapes daily prices from ShareSansar (Top 20 companies + NEPSE Index)
- **ML Models**: Hybrid Ensemble (LSTM, XGBoost, Random Forest) for Buy/Sell signal prediction
- **Technical Indicators**: RSI, MACD, Bollinger Bands, EMA, ATR, OBV, VWAP, Beta
- **Interactive Dashboard**: Candlestick charts, forecasts, and trading signals
- **Cloud Database**: PostgreSQL (Neon) for shared data storage

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- Git

### 1. Clone & Install
```bash
git clone <repo-url>
cd CATI
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```
Edit `.env` and set your `DATABASE_URL` (or leave blank for SQLite).

### 3. Initialize Database
```bash
python manage.py migrate
```

### 4. Load Data
```bash
# Option A: Scrape fresh data (Internet required)
python scripts/update_market_data.py

# Option B: Import from CSV
python manage.py import_history --symbol NABIL
```

### 5. Train Models
```bash
python manage.py train_model --stock NABIL
```

### 6. Run Server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/`

---

## Production Deployment (Railway)

For automatic daily data updates, deploy the scraper to Railway:

1. Push code to GitHub
2. Create Railway project from GitHub repo
3. Configure Cron Job: `15 17 * * *` (11 PM Nepal Time)
4. Set environment variables (`DATABASE_URL`, `SECRET_KEY`, `DEBUG=False`)

**See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.**

---

## Data Sources

| Data | Source | Update Frequency |
|------|--------|------------------|
| Stock Prices | [ShareSansar](https://www.sharesansar.com/today-share-price) | Daily |
| NEPSE Index | [ShareSansar Datewise](https://www.sharesansar.com/datewise-indices) | Daily |

### Top 20 Companies Tracked
```
NABIL, NICA, GBIME, NTC, ADBL, SBL, CZBIL, EBL, SANIMA, NLIC,
NBL, HBL, SCB, SHIVM, NMB, CHCL, CIT, HIDCL, API, PRVU
```

---

## Project Structure

```
CATI/
├── app/                    # Django app
│   ├── models.py           # StockData, ForecastResult models
│   ├── views.py            # Dashboard views
│   ├── services.py         # ML Pipeline (LSTM, XGBoost, RF)
│   ├── scraper.py          # ShareSansar scraper
│   └── templates/          # HTML templates
├── scripts/
│   ├── update_market_data.py   # Main scraper script (Cron Job)
│   ├── fill_nepse_history.py   # One-time NEPSE backfill
│   └── export_data.py          # Export DB to CSV/XLSX
├── saved_states/
│   ├── data/               # CSV/XLSX exports (Top 21)
│   └── images/             # Generated charts
├── CATI/                   # Django settings
├── requirements.txt        # Python dependencies
├── DEPLOYMENT.md           # Railway deployment guide
└── .env.example            # Environment template
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes (Production) | PostgreSQL connection string |
| `SECRET_KEY` | Yes | Django secret key |
| `DEBUG` | No | Set to `False` in production |

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `update_market_data.py` | Daily scraper + export | `python scripts/update_market_data.py` |
| `fill_nepse_history.py` | Backfill NEPSE history | `python scripts/fill_nepse_history.py` |
| `export_data.py` | Export all DB to files | `python scripts/export_data.py` |
| `verify_all_models.py` | Validate trained models | `python scripts/verify_all_models.py` |

---

## License

This project is for educational purposes (Bachelor's Thesis).
