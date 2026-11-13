import os
from dotenv import load_dotenv
import pandas as pd
import pandas_ta as ta
import joblib
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LOOKAHEAD_HOURS = 72

# ⚖️ RISK/REWARD SETTINGS (1:1 Ratio)
TARGET_PERCENT = 0.05  # +5% Gain
STOP_PERCENT = 0.05    # -5% Loss

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
                for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = df[c].astype(float)
                all_dfs.append(df)

            if (i + 1) % 50 == 0: print(f"  ... fetched {i + 1}/{len(ticker_list)}")

        if not all_dfs: return None
        return pd.concat(all_dfs)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def train():
    df = get_data()
    if df is None or df.empty: return

    df = df.reset_index(drop=True)

    print("Calculating Advanced Indicators...")

    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.ema(length=200, append=True)

    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.roc(length=10, append=True)

    df['SMA_Volume_20'] = df['volume'].rolling(window=20).mean()
    df['feat_vol_surge'] = df['volume'] / df['SMA_Volume_20']

    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['STD_20'] = df['close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA_20'] + (2.0 * df['STD_20'])
    df['Lower_BB'] = df['SMA_20'] - (2.0 * df['STD_20'])

    df['feat_dist_ema'] = (df['close'] - df['EMA_200']) / df['close']
    df['feat_rsi'] = df['RSI_14'] / 100.0
    df['feat_atr_pct'] = df['ATRr_14'] / df['close']
    df['feat_bb_width'] = (df['Upper_BB'] - df['Lower_BB']) / df['close']
    df['feat_macd_hist'] = df['MACDh_12_26_9'] / df['close']
    df['feat_roc'] = df['ROC_10'] / 100.0

    print("Generating Classification Labels (1:1 R:R)...")

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOKAHEAD_HOURS)
    df['future_high'] = df['high'].rolling(window=indexer).max()
    df['future_low'] = df['low'].rolling(window=indexer).min()

    df['target_hit'] = df['future_high'] >= df['close'] * (1 + TARGET_PERCENT)
    df['stop_hit'] = df['future_low'] <= df['close'] * (1 - STOP_PERCENT)
    df['label_success'] = ((df['target_hit'] == True) & (df['stop_hit'] == False)).astype(int)

    df.dropna(inplace=True)
    df = df[df['RSI_14'] < 30]

    features = [
        'feat_rsi', 'feat_dist_ema', 'feat_atr_pct', 'feat_bb_width',
        'feat_macd_hist', 'feat_roc', 'feat_vol_surge'
    ]

    print(f"Splitting {len(df)} historical setups into Training and Testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['label_success'], test_size=0.2, random_state=42
    )

    print(f"🧠 Training Probability Model on {len(X_train)} setups...")
    prob_model = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=42)
    prob_model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, prob_model.predict(X_train))
    test_acc = accuracy_score(y_test, prob_model.predict(X_test))
    test_precision = precision_score(y_test, prob_model.predict(X_test), zero_division=0)

    print("\n--- 📈 MODEL DIAGNOSTICS ---")
    print(f"Training Accuracy : {train_acc * 100:.1f}% (How well it memorized the past)")
    print(f"Testing Accuracy  : {test_acc * 100:.1f}% (How well it predicts UNSEEN future data)")
    print(f"True Win Rate     : {test_precision * 100:.1f}% (When it signals a trade, % that hit +5%)")

    if (train_acc - test_acc) > 0.15:
        print("⚠️ WARNING: Your model is Overfitting! It memorized the training data but fails on new data.")
    elif test_precision < 0.50:
        print("⚠️ WARNING: The True Win Rate is below 50%. The model is basically guessing coin flips.")
    else:
        print("✅ SUCCESS: Model looks healthy and profitable on unseen data!")
    print("----------------------------\n")

    joblib.dump(prob_model, 'probability_model.pkl')
    print("✅ Brain saved: probability_model.pkl")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train()
