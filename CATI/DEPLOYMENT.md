# Stock Market Prediction - Production Deployment Guide

This guide covers deploying the **Data Scraper** on Railway to automatically update the database daily at 11 PM Nepal Time.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      RAILWAY CLOUD                          │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │  Cron Job       │───▶│  PostgreSQL (Neon DB)        │    │
│  │  (11 PM NPT)    │    │  - Stock prices              │    │
│  │                 │    │  - NEPSE Index               │    │
│  │  Runs:          │    │  - Top 20 Companies          │    │
│  │  update_market_ │    └──────────────────────────────┘    │
│  │  data.py        │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- GitHub account with this repository pushed
- [Railway.app](https://railway.app) account (free tier works)
- Neon PostgreSQL database (already configured)

---

## Step 1: Push to GitHub

```bash
cd D:\Home\BE\projects\stock_market_prediction\CATI
git add .
git commit -m "Production ready: Scraper with 11 PM cron job"
git push origin main
```

---

## Step 2: Create Railway Project

1. Go to [railway.app](https://railway.app) → Sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your `stock_market_prediction` repository
4. Railway will detect the project (but we'll configure it manually)

---

## Step 3: Configure Cron Job Service

In Railway dashboard:

1. Go to your service → **Settings**
2. Under **Deploy** section:
   - **Start Command**: `python scripts/update_market_data.py`
3. Under **Cron** section:
   - **Schedule**: `15 17 * * *`
   - This runs at **11:00 PM Nepal Time** daily (5:15 PM UTC)

### Cron Schedule Reference (Nepal Time)

| Nepal Time | UTC Time | Cron Expression |
|------------|----------|-----------------|
| 10:00 PM   | 4:15 PM  | `15 16 * * *`   |
| **11:00 PM** | **5:15 PM** | **`15 17 * * *`** |
| 11:30 PM   | 5:45 PM  | `45 17 * * *`   |

---

## Step 4: Configure Environment Variables

In Railway → **Variables**, add:

| Variable | Value | Description |
|----------|-------|-------------|
| `DATABASE_URL` | `postgresql://...` | Your Neon PostgreSQL URL |
| `SECRET_KEY` | `your-secret-key` | Django secret (generate a new one) |
| `DEBUG` | `False` | Always False in production |

---

## Step 5: Verify Deployment

After the first cron run (or manual trigger), check Railway logs:

```
============================================================
[23:00:00] Starting Market Data Update...
============================================================
Latest Data in DB (NEPSE): 2026-02-02
Database is already up to date!

[EXPORT] Syncing DB to CSV/XLSX files (Top 21 only)...
   ✅ NEPSE: 6467 records -> CSV & XLSX
   ✅ NABIL: 3177 records -> CSV & XLSX
   ...
[EXPORT] Complete.

============================================================
All Done!
============================================================
```

---

## What Gets Updated Daily

| Data | Source | Records |
|------|--------|---------|
| NEPSE Index | ShareSansar | ~6,500+ |
| Top 20 Stocks | ShareSansar | Variable |
| CSV/XLSX Files | Database | 21 each |

### Top 20 Companies List
```
NABIL, NICA, GBIME, NTC, ADBL, SBL, CZBIL, EBL, SANIMA, NLIC,
NBL, HBL, SCB, SHIVM, NMB, CHCL, CIT, HIDCL, API, PRVU
```

---

## Files Required for Deployment

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `scripts/update_market_data.py` | Main scraper script |
| `app/scraper.py` | ShareSansar scraping logic |
| `CATI/settings.py` | Django configuration |
| `.env.example` | Environment template |

---

## Manual Trigger (Testing)

You can manually run the scraper on Railway:

```bash
railway run python scripts/update_market_data.py
```

Or from Railway dashboard: **Deploy** → **Trigger Build**

---

## Troubleshooting

### Database Connection Timeout
```
django.db.utils.OperationalError: connection failed
```
**Solution**: Wait a few minutes and retry. Neon DB sometimes has cold-start delays.

### No Data for Today
```
⚠️ No Data (Holiday/Weekend)
```
**Normal behavior**: NEPSE is closed on Fridays and Saturdays.

### Missing Symbol Data
Some symbols may have empty data if they are newly listed or suspended. The export skips them gracefully.

---

## Local Development

Test the scraper locally before pushing:

```bash
cd CATI
python scripts/update_market_data.py
```

This will:
1. Scrape any missing data
2. Export Top 21 symbols to `saved_states/data/`
