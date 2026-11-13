import os
from dotenv import load_dotenv
import pandas as pd
import pandas_ta as ta
import joblib
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LOOKAHEAD_HOURS = 72

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_data():
    print("Fetching data for training...")
    try:
        tickers = supabase.table('tickers').select("symbol").eq('is_universe', True).execute()
        ticker_list = [x['symbol'] for x in tickers.data]

        all_dfs = []

        print(f"Downloading history for {len(ticker_list)} stocks...")
        for i, t in enumerate(ticker_list):
            res = supabase.table('candles').select('*').eq('symbol', t).order('date', desc=False).execute()

            if res.data:
                df = pd.DataFrame(res.data)
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = df[c].astype(float)
                all_dfs.append(df)

            if (i + 1) % 50 == 0:
                print(f"  ... fetched {i + 1}/{len(ticker_list)}")

        if not all_dfs: return None
        return pd.concat(all_dfs)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def train():
    df = get_data()
    if df is None or df.empty:
        print("❌ No data found. Aborting training.")
        return

    print("Calculating Indicators...")
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.ema(length=200, append=True)

    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['STD_20'] = df['close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA_20'] + (2.0 * df['STD_20'])
    df['Lower_BB'] = df['SMA_20'] - (2.0 * df['STD_20'])

    df['feat_dist_ema'] = (df['close'] - df['EMA_200']) / df['close']
    df['feat_rsi'] = df['RSI_14'] / 100.0
    df['feat_atr_pct'] = df['ATRr_14'] / df['close']
    df['feat_bb_width'] = (df['Upper_BB'] - df['Lower_BB']) / df['close']

    print("Generating Target Labels...")
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOKAHEAD_HOURS)
    df['future_high'] = df['high'].rolling(window=indexer).max()
    df['future_low'] = df['low'].rolling(window=indexer).min()
    df['label_gain'] = (df['future_high'] - df['close']) / df['close']
    df['label_loss'] = (df['close'] - df['future_low']) / df['close']

    df.dropna(inplace=True)

    df = df[df['RSI_14'] < 30]

    features = ['feat_rsi', 'feat_dist_ema', 'feat_atr_pct', 'feat_bb_width']

    print(f"🧠 Training Models on {len(df)} historical dip scenarios...")

    profit_model = RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=42)
    profit_model.fit(df[features], df['label_gain'])

    risk_model = RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=42)
    risk_model.fit(df[features], df['label_loss'])

    joblib.dump(profit_model, 'profit_model.pkl')
    joblib.dump(risk_model, 'risk_model.pkl')
    print("✅ Brains saved: profit_model.pkl & risk_model.pkl")


if __name__ == "__main__":
    train()
