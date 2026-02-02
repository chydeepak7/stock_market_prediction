# Stock Market Scraper - Scheduled Task Setup (Windows)

## Option 1: Using Windows Task Scheduler (GUI)

1. Press `Win + R`, type `taskschd.msc`, press Enter
2. Click **Create Basic Task** in the right panel
3. **Name**: `Stock Market Scraper`
4. **Description**: `Scrapes stock data from ShareSansar daily at 5 PM`
5. **Trigger**: Daily
6. **Start time**: `5:00:00 PM` (17:00)
7. **Action**: Start a program
8. **Program/script**: 
   ```
   D:\Home\BE\projects\stock_market_prediction\CATI\run_scraper.bat
   ```
9. **Start in**: 
   ```
   D:\Home\BE\projects\stock_market_prediction\CATI
   ```
10. Click **Finish**

### Important Settings:
- Right-click the task → Properties → Settings
- ✅ Check "Run task as soon as possible after a scheduled start is missed"
- ✅ Check "If the task fails, restart every 5 minutes" (up to 3 attempts)

---

## Option 2: Using PowerShell (One-liner)

Run this in **Administrator PowerShell**:

```powershell
$action = New-ScheduledTaskAction -Execute "D:\Home\BE\projects\stock_market_prediction\CATI\run_scraper.bat" -WorkingDirectory "D:\Home\BE\projects\stock_market_prediction\CATI"
$trigger = New-ScheduledTaskTrigger -Daily -At 5:00PM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "StockMarketScraper" -Action $action -Trigger $trigger -Settings $settings -Description "Daily stock data scraper at 5PM"
```

---

## Option 3: Using schtasks command

Run in **Administrator Command Prompt**:

```cmd
schtasks /Create /SC DAILY /TN "StockMarketScraper" /TR "D:\Home\BE\projects\stock_market_prediction\CATI\run_scraper.bat" /ST 17:00
```

---

## Verify Setup

Check if the task is registered:
```cmd
schtasks /Query /TN "StockMarketScraper"
```

Run manually to test:
```cmd
schtasks /Run /TN "StockMarketScraper"
```

Delete the task if needed:
```cmd
schtasks /Delete /TN "StockMarketScraper" /F
```

---

## What Happens at 11 PM Daily

1. Windows runs `run_scraper.bat`
2. The script runs `python manage.py scrape_daily --fill-gaps`
3. This:
   - Scrapes today's stock data from ShareSansar
   - Checks for any missing dates since the last recorded entry
   - Fills any gaps automatically (up to 30 days back)
4. All activity is logged to:
   - `saved_states/pipeline.log` (detailed pipeline logs)
   - `saved_states/scheduler.log` (task scheduler output)
