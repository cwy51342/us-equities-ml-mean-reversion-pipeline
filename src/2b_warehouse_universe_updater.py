import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
from supabase import create_client, Client
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MAX_WORKERS = 20

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
NY_TZ = pytz.timezone('America/New_York')


def get_universe_tickers():
    """Fetch ONLY the stocks currently in your Universe (Top 153)."""
    try:
        response = supabase.table('tickers').select("symbol") \
            .eq('is_universe', True) \
            .execute()
        data = response.data
        return [row['symbol'] for row in data]
    except Exception as e:
        print(f"Error fetching universe: {e}")
        return []


def safe_upsert(data, retries=3):
    for attempt in range(retries):
        try:
            supabase.table('candles').upsert(data).execute()
            return True
        except Exception:
            time.sleep(1)
    return False


def process_stock(ticker):
    try:
        start_date = None
        try:
            latest_query = supabase.table('candles') \
                .select('date') \
                .eq('symbol', ticker) \
                .order('date', desc=True) \
                .limit(1) \
                .execute()

            if latest_query.data:
                last_date_str = latest_query.data[0]['date']
                last_date = pd.to_datetime(last_date_str)
                if last_date.tz is None: last_date = last_date.tz_localize('UTC')
                last_date_ny = last_date.astimezone(NY_TZ)
                now_ny = datetime.now(NY_TZ)

                if last_date_ny > now_ny - timedelta(minutes=60):
                    return f"✅ {ticker}: Up to date"

                if last_date_ny.date() == now_ny.date() and last_date_ny.hour >= 15:
                    return f"✅ {ticker}: Market Closed"

                start_date = last_date + timedelta(hours=1)
            else:
                start_date = datetime.now() - timedelta(days=365)
        except:
            start_date = datetime.now() - timedelta(days=365)

        try:
            df = yf.download(ticker, start=start_date, interval="1h", progress=False, threads=False)
        except:
            return f"⚠️ {ticker}: Download Error"

        if df.empty: return f"✅ {ticker}: No new data"

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        clean_cols = {}
        for col in df.columns:
            c = col.lower()
            if 'open' in c:
                clean_cols[col] = 'open'
            elif 'high' in c:
                clean_cols[col] = 'high'
            elif 'low' in c:
                clean_cols[col] = 'low'
            elif 'close' in c:
                clean_cols[col] = 'close'
            elif 'volume' in c:
                clean_cols[col] = 'volume'

        df = df.rename(columns=clean_cols)
        df = df.reset_index()

        records = []
        for _, row in df.iterrows():
            try:
                ts = None
                for col in ['Datetime', 'Date', 'index']:
                    if col in row:
                        ts = row[col]
                        break
                if not isinstance(ts, (pd.Timestamp, datetime)): continue
                if ts > datetime.now(ts.tz): continue

                records.append({
                    "symbol": ticker,
                    "date": ts.isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": int(row['volume'])
                })
            except:
                continue

        if records:
            if not safe_upsert(records): return f"🔥 {ticker}: DB Error"
            return f"📥 {ticker}: Saved {len(records)} candles"

        return f"✅ {ticker}: Up to date"

    except Exception as e:
        return f"🔥 Error {ticker}: {str(e)[:20]}"


def main():
    print("--- 🚀 FAST UNIVERSE UPDATE 🚀 ---")
    tickers = get_universe_tickers()
    print(f"Updating ONLY your {len(tickers)} Universe stocks...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(process_stock, t): t for t in tickers}
        for i, future in enumerate(as_completed(future_to_ticker)):
            print(f"[{i + 1}/{len(tickers)}] {future.result()}")


if __name__ == "__main__":
    main()
