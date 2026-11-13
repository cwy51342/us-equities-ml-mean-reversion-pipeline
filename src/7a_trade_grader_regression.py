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

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
NY_TZ = pytz.timezone('America/New_York')


def grade_predictions():
    print("--- 👨‍🏫 THE GRADER: EVALUATING AI PERFORMANCE ---")

    # 1. Fetch ungraded predictions older than 5 days (Let the window close)
    cutoff_time = datetime.now(pytz.utc) - timedelta(days=5)

    try:
        res = supabase.table('prediction_logs') \
            .select('*') \
            .is_('actual_gain', 'null') \
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
        raw_pred = log['raw_predicted_gain']

        try:
            # Fetch the next 5 days of data starting from the prediction date
            start_str = pred_date.strftime('%Y-%m-%d')
            df = yf.download(symbol, start=start_str, period="5d", progress=False)

            if df.empty or len(df) < 2:
                continue  # Not enough future data yet to grade fairly

            # Flatten columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # The entry price is roughly the close of the first day (when prediction happened)
            entry_price = float(df['Close'].iloc[0])

            # Find the highest peak in the following days
            max_future_price = float(df['High'].iloc[1:].max())

            # Calculate what actually happened
            actual_gain = (max_future_price - entry_price) / entry_price

            # Calculate the Error (Actual minus Predicted)
            error_rate = actual_gain - raw_pred

            # Update the database
            supabase.table('prediction_logs').update({
                "actual_gain": actual_gain,
                "error_rate": error_rate
            }).eq('id', log['id']).execute()

            print(
                f"📝 Graded {symbol}: Pred {raw_pred * 100:.1f}% | Actual {actual_gain * 100:.1f}% | Error {error_rate * 100:.1f}%")

        except Exception as e:
            print(f"⚠️ Could not grade {symbol}: {e}")

    print("--- 🏁 GRADING COMPLETE ---")


if __name__ == "__main__":
    grade_predictions()
