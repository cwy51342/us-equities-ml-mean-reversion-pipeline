import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
from supabase import create_client, Client
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# BATCH SETTINGS
BATCH_SIZE = 50
CHECK_WORKERS = 20
DEFAULT_PERIOD = "1y"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
NY_TZ = pytz.timezone('America/New_York')


def get_active_tickers():
    """Fetch all tickers that are Active."""
    all_tickers = []
    start = 0
    while True:
        try:
            response = supabase.table('tickers').select("symbol") \
                .eq('is_active', True) \
                .range(start, start + 999).execute()
            data = response.data
            if not data: break
            all_tickers.extend([row['symbol'] for row in data])
            start += 1000
        except Exception:
            time.sleep(1)
    return all_tickers


def check_stock_status(ticker):
    """
    Checks DB to see if we ACTUALLY need to download this stock.
    Returns: (ticker, needs_update_bool)
    """
    try:
        latest = supabase.table('candles') \
            .select('date') \
            .eq('symbol', ticker) \
            .order('date', desc=True) \
            .limit(1) \
            .execute()

        if not latest.data:
            return ticker, True

        last_date_str = latest.data[0]['date']
        last_date = pd.to_datetime(last_date_str)
        if last_date.tz is None: last_date = last_date.tz_localize('UTC')

        if last_date < datetime.now(pytz.utc) - timedelta(hours=1):
            return ticker, True

        return ticker, False

    except Exception:
        return ticker, True


def safe_upsert(data, retries=3):
    """Uploads to Supabase in chunks."""
    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        for attempt in range(retries):
            try:
                supabase.table('candles').upsert(chunk).execute()
                break
            except Exception:
                time.sleep(2)


def process_batch(tickers_batch):
    try:
        df = yf.download(tickers_batch, period=DEFAULT_PERIOD, interval="1h", group_by='ticker', progress=False,
                         threads=True)

        if df.empty: return

        all_records = []

        for ticker in tickers_batch:
            try:
                if len(tickers_batch) > 1:
                    if ticker not in df.columns: continue
                    stock_df = df[ticker].copy()
                else:
                    stock_df = df.copy()

                stock_df = stock_df.reset_index()
                stock_df.columns = [str(c).lower().strip() for c in stock_df.columns]
                rename_map = {'datetime': 'date', 'index': 'date', 'adj close': 'close'}
                stock_df = stock_df.rename(columns=rename_map)

                if 'close' not in stock_df.columns: continue

                for _, row in stock_df.iterrows():
                    ts = row.get('date')
                    if pd.isna(ts): continue
                    if ts > datetime.now(ts.tz): continue

                    close_val = row.get('close', 0)
                    if pd.isna(close_val) or close_val == 0: continue

                    all_records.append({
                        "symbol": ticker,
                        "date": ts.isoformat(),
                        "open": float(row.get('open', 0)),
                        "high": float(row.get('high', 0)),
                        "low": float(row.get('low', 0)),
                        "close": float(row.get('close', 0)),
                        "volume": int(row.get('volume', 0))
                    })
            except:
                continue

        if all_records:
            safe_upsert(all_records)
            print(f"📥 Saved batch of {len(tickers_batch)} stocks ({len(all_records)} candles).")
        else:
            print(f"⚠️ Batch of {len(tickers_batch)} returned no data.")

    except Exception as e:
        print(f"🔥 Batch Error: {e}")


def main():
    print("--- 🧠 STARTING SMART WAREHOUSE ---")
    start_time = datetime.now()

    all_tickers = get_active_tickers()
    print(f"Scanning {len(all_tickers)} tickers for stale data...")

    download_list = []

    with ThreadPoolExecutor(max_workers=CHECK_WORKERS) as executor:
        future_to_ticker = {executor.submit(check_stock_status, t): t for t in all_tickers}

        checked = 0
        for future in as_completed(future_to_ticker):
            ticker, needs_update = future.result()
            checked += 1
            if needs_update:
                download_list.append(ticker)

            if checked % 200 == 0:
                print(f"   Checked {checked}/{len(all_tickers)}...")

    print(f"\n✅ Scan Complete. Found {len(download_list)} stocks needing updates.")
    print(f"   Skipped {len(all_tickers) - len(download_list)} fresh stocks.")

    if not download_list:
        print("🎉 Everything is up to date!")
        return

    print(f"\n--- 🚀 DOWNLOADING {len(download_list)} STOCKS ---")

    for i in range(0, len(download_list), BATCH_SIZE):
        batch = download_list[i:i + BATCH_SIZE]
        print(f"Processing Batch {i // BATCH_SIZE + 1} ({len(batch)} stocks)...")

        process_batch(batch)
        time.sleep(1)

    duration = datetime.now() - start_time
    print(f"\n--- 🏁 FINISHED in {duration} ---")


if __name__ == "__main__":
    main()
