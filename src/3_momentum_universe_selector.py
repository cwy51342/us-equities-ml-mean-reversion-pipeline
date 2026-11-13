import os
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client, Client

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# MOMENTUM TARGETS
TARGET_3M = 0.45
TARGET_6M = 0.55
TARGET_12M = 0.75
UNIVERSE_SIZE = 153

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_all_active_tickers():
    """Fetches ALL active tickers using pagination to bypass 1000 limit."""
    all_tickers = []
    start = 0
    while True:
        response = supabase.table('tickers').select('symbol') \
            .eq('is_active', True) \
            .range(start, start + 999) \
            .execute()
        data = response.data
        if not data:
            break
        all_tickers.extend([row['symbol'] for row in data])
        start += 1000
    return all_tickers


def get_candles(ticker):
    response = supabase.table('candles') \
        .select('close, date') \
        .eq('symbol', ticker) \
        .order('date', desc=True) \
        .limit(2000) \
        .execute()

    if not response.data: return None
    df = pd.DataFrame(response.data)
    df['close'] = df['close'].astype(float)
    return df


def analyze_stock(ticker, debug=False):
    df = get_candles(ticker)

    if df is None or len(df) < 1000:
        if debug: print(f"   [DEBUG] {ticker}: Insufficient data ({len(df) if df is not None else 0} candles)")
        return None

    current_price = df.iloc[0]['close']

    idx_3m = min(441, len(df) - 1)
    idx_6m = min(882, len(df) - 1)
    idx_12m = min(1764, len(df) - 1)

    price_3m = df.iloc[idx_3m]['close']
    price_6m = df.iloc[idx_6m]['close']
    price_12m = df.iloc[idx_12m]['close']

    if price_3m == 0 or price_6m == 0 or price_12m == 0: return None

    ret_3m = (current_price - price_3m) / price_3m
    ret_6m = (current_price - price_6m) / price_6m
    ret_12m = (current_price - price_12m) / price_12m

    if debug:
        print(f"   [DEBUG] {ticker}: 3m={ret_3m:.1%} | 6m={ret_6m:.1%} | 12m={ret_12m:.1%}")

    pass_3m = ret_3m >= TARGET_3M
    pass_6m = ret_6m >= TARGET_6M
    pass_12m = ret_12m >= TARGET_12M

    if pass_3m or pass_6m or pass_12m:
        score = (ret_3m + ret_6m + ret_12m) / 3
        return {
            "symbol": ticker, "score": score,
            "3m": ret_3m, "6m": ret_6m, "12m": ret_12m
        }
    return None


def main():
    print("--- Starting Hourly Analyst Engine ---")

    print("Resetting old universe flags...")
    supabase.table('tickers').update({'is_universe': False}).neq('symbol', '0000').execute()

    candidates = get_all_active_tickers()
    print(f"Analyzing {len(candidates)} active stocks...")

    qualifiers = []

    for i, ticker in enumerate(candidates):
        show_debug = (i < 5)

        if i % 100 == 0:
            print(f"Processing... {i}/{len(candidates)}")

        result = analyze_stock(ticker, debug=show_debug)
        if result:
            qualifiers.append(result)

    print(f"\nFound {len(qualifiers)} stocks matching targets.")

    if not qualifiers:
        print("❌ Still 0 found. Consider lowering TARGET variables temporarily.")
        return

    qualifiers.sort(key=lambda x: x['score'], reverse=True)
    winners = qualifiers[:UNIVERSE_SIZE]

    print(f"Marking top {len(winners)} as 'Universe'...")
    for winner in winners:
        supabase.table('tickers').update({
            'is_universe': True,
            'momentum_score': winner['score']
        }).eq('symbol', winner['symbol']).execute()

    print("\n✅ New Universe Selected!")
    print("Top 5 Stocks:")
    for w in winners[:5]:
        print(f"{w['symbol']}: Score {w['score']:.2f} (12m: {w['12m']:.0%})")


if __name__ == "__main__":
    main()
