# Stock Market Scraper - Cloud Deployment Guide

## Prerequisites
- GitHub account
- Railway.app account (free tier available)
- The Neon DB is already configured in `.env`

## Step 1: Push to GitHub

```bash
cd D:\Home\BE\projects\stock_market_prediction\CATI
git init
git add .
git commit -m "Initial commit: Stock scraper with Neon DB"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/stock-scraper.git
git push -u origin main
```

## Step 2: Deploy on Railway

1. Go to [Railway.app](https://railway.app) and sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repository
4. Railway will auto-detect the Django project

## Step 3: Configure Environment Variables

In Railway dashboard, go to your project → **Variables** and add:

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | `postgresql://neondb_owner:npg_HRSmd04DflxJ@ep-square-lake-a1f208fr-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require` |
| `SECRET_KEY` | `your-production-secret-key` |
| `DEBUG` | `False` |

## Step 4: Set Up Cron Job (Daily at 5 PM Nepal Time)

In Railway:
1. Go to your project → **Settings**
2. Under **Cron**, add:
   - **Schedule**: `15 11 * * *` (5:00 PM Nepal = 11:15 UTC)
   - **Command**: `python run_scraper.py`

### Cron Explanation:
- `15 11 * * *` = 11:15 AM UTC = 5:00 PM Nepal Time (UTC+5:45)

## Step 5: Verify Deployment

After deployment, check Railway logs to see:
```
[SCRAPER] Starting scrape job with gap-fill
[SCRAPER] Scraped 329 records for 2026-02-01
[SCRAPER] DB Update: 329 created, 0 already existed
```

## How It Works

1. **On every git push**: Railway automatically redeploys
2. **Every day at 5 PM Nepal time**: Railway runs the scraper via cron
3. **Scraper checks for gaps**: If any dates are missing, it fills them
4. **Data goes to Neon DB**: All stock data is stored in the cloud PostgreSQL

## Local Development

Test locally before pushing:
```bash
cd D:\Home\BE\projects\stock_market_prediction\CATI
..\myvenv\Scripts\python.exe manage.py scrape_daily --fill-gaps
```

## Files Created for Deployment

| File | Purpose |
|------|---------|
| `railway.json` | Railway deployment config |
| `Procfile` | Process definition |
| `runtime.txt` | Python version |
| `requirements.txt` | Python dependencies |
| `.env` | Local environment variables (NOT committed) |
| `.gitignore` | Excludes sensitive files |
| `run_scraper.py` | Standalone script for cron |
