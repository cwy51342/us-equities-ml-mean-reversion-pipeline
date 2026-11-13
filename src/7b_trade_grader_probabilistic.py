import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from supabase import create_client, Client
from datetime import datetime, timedelta
import pytz

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ⚖️ MUST MATCH TRADER SETTINGS
TARGET_PERCENT = 0.05
STOP_PERCENT = 0.05

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
NY_TZ = pytz.timezone('America/New_York')


def grade_predictions():
    print("--- 👨‍🏫 THE GRADER: EVALUATING PROBABILITIES ---")

    # 1. Fetch ungraded predictions older than 5 days (Let the 72-hour window fully close)
    cutoff_time = datetime.now(pytz.utc) - timedelta(days=5)

    try:
        res = supabase.table('probability_logs') \
            .select('*') \
            .is_('actual_success', 'null') \
            .lte('created_at', cutoff_time.isoformat()) \
            .execute()

        ungraded = res.data
    except Exception as e:
        print(f"❌ Failed to fetch logs: {e}")
        return

    if not ungraded:
        print("✅ All previous predictions have been graded.")
        return

    print(f"Found {len(ungraded)} predictions to evaluate...")

    for log in ungraded:
        symbol = log['symbol']
        pred_date = pd.to_datetime(log['created_at'])
        raw_prob = log['raw_win_prob']

        try:
            # Fetch 1-hour interval data for the next few days to track the path accurately
            start_str = pred_date.strftime('%Y-%m-%d')
            df = yf.download(symbol, start=start_str, period="7d", interval="1h", progress=False)

            if df.empty or len(df) < 2:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            entry_price = float(df['Close'].iloc[0])

            target_price = entry_price * (1 + TARGET_PERCENT)
            stop_price = entry_price * (1 - STOP_PERCENT)

            actual_success = 0  # Default to failure (hit stop or time ran out)

            # Walk forward in time hour by hour to see what hit first
            for i in range(1, len(df)):
                hour_high = float(df['High'].iloc[i])
                hour_low = float(df['Low'].iloc[i])

                if hour_high >= target_price:
                    actual_success = 1
                    break

                if hour_low <= stop_price:
                    actual_success = 0
                    break

            # --- PROBABILITY CALIBRATION MATH ---
            error_rate = actual_success - raw_prob

            # Update the database
            supabase.table('probability_logs').update({
                "actual_success": actual_success,
                "error_rate": error_rate
            }).eq('id', log['id']).execute()

            result_str = "WON 🏆" if actual_success == 1 else "LOST ❌"
            print(
                f"📝 Graded {symbol}: Pred {raw_prob * 100:.1f}% | Result: {result_str} | Calibration Adjustment: {error_rate * 100:+.1f}%")

        except Exception as e:
            print(f"⚠️ Could not grade {symbol}: {e}")

    print("--- 🏁 GRADING COMPLETE ---")


if __name__ == "__main__":
    grade_predictions()
