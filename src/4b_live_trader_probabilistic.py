import os
from dotenv import load_dotenv
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import joblib
import smtplib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from supabase import create_client, Client
from datetime import datetime, timedelta, time as dtime
import pytz
import numpy as np

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# TRADING SETTINGS
TARGET_PERCENT = 0.05  # 5% Target
STOP_PERCENT = 0.05    # 5% Stop (1:1 R:R)
COOLDOWN_HOURS = 4
MIN_PROBABILITY = 55.0

# 🎛️ FILTER SETTINGS
USE_HOURLY_TREND_FILTER = False
MIN_HOURLY_RSI = 35

# 💰 POSITION SIZING
ACCOUNT_SIZE = 10000
RISK_PER_TRADE = 0.02
MAX_POSITION_SIZE = 500

# SETUP LOCAL SENTIMENT
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_market_status():
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    current_time = now_ny.time()
    if now_ny.weekday() >= 5: return "Weekend (Closed)"
    morning_start, morning_end = dtime(10, 0), dtime(11, 30)
    afternoon_start, afternoon_end = dtime(13, 30), dtime(15, 0)
    if current_time < dtime(9, 30): return "😴 Pre-Market"
    if dtime(9, 30) <= current_time < morning_start: return "🔴 Market Open (Volatile)"
    if morning_start <= current_time <= morning_end: return "🟢 Morning Trend (Best)"
    if morning_end < current_time < afternoon_start: return "🟡 Lunch Chop (Caution)"
    if afternoon_start <= current_time <= afternoon_end: return "🟢 Afternoon Volume (Good)"
    if current_time > afternoon_end: return "🔴 Power Hour / Closed"
    return "Unknown"


def get_universe():
    try:
        return supabase.table('tickers').select('symbol').eq('is_universe', True).execute().data
    except:
        return []


def is_on_cooldown(symbol):
    try:
        cutoff_time = datetime.now(pytz.utc) - timedelta(hours=COOLDOWN_HOURS)
        res = supabase.table('probability_logs').select('id').eq('symbol', symbol).gte('created_at',
                                                                                       cutoff_time.isoformat()).limit(
            1).execute()
        return len(res.data) > 0
    except:
        return False


def check_hourly_trend(symbol):
    try:
        df_hourly = yf.Ticker(symbol).history(period="5d", interval="1h")
        if df_hourly.empty: return True, 50
        df_hourly.ta.rsi(length=14, append=True)
        current_hourly_rsi = df_hourly.iloc[-1]['RSI_14']
        if current_hourly_rsi < MIN_HOURLY_RSI: return False, current_hourly_rsi
        return True, current_hourly_rsi
    except:
        return True, 50


def get_local_sentiment(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news: return "😐 Neutral", ""
        scores = []
        headlines = []
        for article in news[:3]:
            title = article.get('title', '')
            link = article.get('link', '#')
            score = sia.polarity_scores(title)['compound']
            scores.append(score)
            headlines.append(f"<li><a href='{link}'>{title}</a> ({score:.2f})</li>")
        if not scores: return "😐 Neutral", ""
        avg_score = sum(scores) / len(scores)
        if avg_score >= 0.15:
            return f"🔥 Bullish ({avg_score:.2f})", "".join(headlines)
        elif avg_score <= -0.15:
            return f"❄️ Bearish ({avg_score:.2f})", "".join(headlines)
        return f"😐 Neutral ({avg_score:.2f})", "".join(headlines)
    except:
        return "ERROR", ""


def get_historical_error(symbol):
    try:
        res = supabase.table('probability_logs').select('error_rate').eq('symbol', symbol).not_.is_('error_rate',
                                                                                                    'null').order(
            'created_at', desc=True).limit(10).execute()
        if not res.data: return 0.0
        errors = [row['error_rate'] for row in res.data]
        return sum(errors) / len(errors)
    except:
        return 0.0


def analyze_stock(ticker, prob_model):
    symbol = ticker['symbol']
    try:
        res = supabase.table('candles').select('close, high, low, volume, date').eq('symbol', symbol).order('date',
                                                                                                            desc=True).limit(
            300).execute()
        if not res.data: return None
        df = pd.DataFrame(res.data)
        df = df.iloc[::-1].reset_index(drop=True)

        for c in ['close', 'high', 'low', 'volume']: df[c] = df[c].astype(float)

        rows_added = 0
        try:
            live = yf.Ticker(symbol).history(period="1d", interval="5m")

            if live.empty:
                deep_check = yf.Ticker(symbol).history(period="5d")
                is_dead = False

                if deep_check.empty:
                    is_dead = True
                else:
                    if deep_check['Volume'].max() == 0:
                        is_dead = True

                if is_dead:
                    print(f"💀 {symbol} is officially dead/halted. Purging from database...")
                    try:
                        supabase.table('tickers').update({
                            'is_universe': False,
                            'is_active': False
                        }).eq('symbol', symbol).execute()
                    except Exception:
                        pass
                    return None

                current_price = float(df.iloc[-1]['close'])
            else:
                curr_row = live.iloc[-1]
                current_price = float(curr_row['Close'])
                current_time = live.index[-1]
                if not df.empty:
                    last_db_time = pd.to_datetime(df.iloc[-1]['date'])
                    if last_db_time.date() == current_time.date() and last_db_time.hour == current_time.hour:
                        df.iloc[-1, df.columns.get_loc('close')] = current_price
                        df.iloc[-1, df.columns.get_loc('high')] = float(curr_row['High'])
                        df.iloc[-1, df.columns.get_loc('low')] = float(curr_row['Low'])
                        if 'volume' in df.columns:
                            df.iloc[-1, df.columns.get_loc('volume')] = float(curr_row['Volume'])
                    else:
                        new_row = pd.DataFrame({'close': [current_price], 'high': [float(curr_row['High'])],
                                                'low': [float(curr_row['Low'])], 'volume': [float(curr_row['Volume'])],
                                                'date': [current_time]})
                        df = pd.concat([df, new_row], ignore_index=True)
                        rows_added = 1
                else:
                    current_price = float(curr_row['Close'])
                    df = pd.DataFrame(
                        [{'close': current_price, 'high': float(curr_row['High']), 'low': float(curr_row['Low']),
                          'volume': float(curr_row['Volume']), 'date': current_time}])
        except Exception as e:
            print(f"⚠️ Live Fetch Error for {symbol}: {e}")
            return None

        target_len = 50 + rows_added
        df_short = df.tail(target_len).copy()
        df_short.ta.rsi(length=14, append=True)
        if 'RSI_14' not in df_short.columns: return None
        rsi_val = df_short.iloc[-1]['RSI_14']

        if rsi_val > 30: return None

        is_trend_safe, hourly_rsi = check_hourly_trend(symbol)
        if USE_HOURLY_TREND_FILTER and not is_trend_safe: return None

        df.ta.atr(length=14, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.roc(length=10, append=True)

        df['SMA_Volume_20'] = df['volume'].rolling(window=20).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['STD_20'] = df['close'].rolling(window=20).std()
        df['Upper_BB'] = df['SMA_20'] + (2.0 * df['STD_20'])
        df['Lower_BB'] = df['SMA_20'] - (2.0 * df['STD_20'])

        last = df.iloc[-1]

        ema_val = last.get('EMA_200', np.nan)
        dist_ema = 0.0 if pd.isna(ema_val) else (last['close'] - ema_val) / last['close']
        atr_val = last.get('ATRr_14', np.nan)
        feat_atr = 0.0 if (pd.isna(atr_val) or last['close'] == 0) else atr_val / last['close']
        upper = last.get('Upper_BB', np.nan)
        lower = last.get('Lower_BB', np.nan)
        bb_width = 0.0 if (pd.isna(upper) or pd.isna(lower)) else (upper - lower) / last['close']
        macd_h = last.get('MACDh_12_26_9', np.nan)
        feat_macd = 0.0 if pd.isna(macd_h) else macd_h / last['close']
        roc_val = last.get('ROC_10', np.nan)
        feat_roc = 0.0 if pd.isna(roc_val) else roc_val / 100.0

        vol_avg = last.get('SMA_Volume_20', np.nan)
        feat_vol = 1.0 if (pd.isna(vol_avg) or vol_avg == 0) else last.get('volume', 0) / vol_avg

        features = pd.DataFrame([{
            'feat_rsi': rsi_val / 100.0,
            'feat_dist_ema': dist_ema,
            'feat_atr_pct': feat_atr,
            'feat_bb_width': bb_width,
            'feat_macd_hist': feat_macd,
            'feat_roc': feat_roc,
            'feat_vol_surge': feat_vol
        }]).fillna(0)

        raw_win_prob = prob_model.predict_proba(features)[0][1]
        avg_error = get_historical_error(symbol)
        adjusted_win_prob = raw_win_prob + avg_error
        final_probability = max(0.0, min(100.0, adjusted_win_prob * 100))

        if final_probability < MIN_PROBABILITY: return None

        target_price = current_price * (1 + TARGET_PERCENT)
        stop_price = current_price * (1 - STOP_PERCENT)

        try:
            supabase.table('probability_logs').insert({
                "symbol": symbol,
                "raw_win_prob": float(raw_win_prob),
                "calibrated_win_prob": float(final_probability)
            }).execute()
        except Exception:
            pass

        risk_amt = ACCOUNT_SIZE * RISK_PER_TRADE
        stop_dist = current_price - stop_price
        if stop_dist <= 0: stop_dist = 0.01

        rec_shares = int(risk_amt / stop_dist)
        total_cost = rec_shares * current_price

        if total_cost > MAX_POSITION_SIZE:
            rec_shares = int(MAX_POSITION_SIZE / current_price)
            total_cost = rec_shares * current_price

        if total_cost < 10: return None

        sentiment, headlines = get_local_sentiment(symbol)

        return {
            "symbol": symbol,
            "rsi": round(rsi_val, 2),
            "price": round(current_price, 2),
            "target": round(target_price, 2),
            "stop": round(stop_price, 2),
            "win_prob": round(final_probability, 1),
            "shares": rec_shares,
            "cost": round(total_cost, 2),
            "calibration": round(avg_error * 100, 2),
            "sentiment": sentiment,
            "headlines": headlines
        }

    except Exception as e:
        print(f"❌ Crash on {symbol}: {e}")
        return None


def send_email(alerts, status_msg):
    if not alerts: return
    html = f"""
    <h3>🤖 AI Probabilistic Trader ({status_msg})</h3>
    <table border="1" cellpadding="5" style="border-collapse: collapse; text-align: center; width: 100%;">
        <tr style="background-color: #e0e0e0;">
            <th>Ticker</th><th>Price</th><th>Target (+5%)</th><th>Stop (-5%)</th><th>Win Odds</th><th>Shares (Bet)</th><th>Adjust</th><th>Mood</th>
        </tr>
    """
    for a in alerts:
        mood_color = "green" if "Bullish" in a['sentiment'] else "red" if "Bearish" in a['sentiment'] else "gray"
        prob_color = "green" if a['win_prob'] > 60 else "orange" if a['win_prob'] >= 50 else "red"

        html += f"""
        <tr>
            <td><b>{a['symbol']}</b></td>
            <td>${a['price']}</td>
            <td style="color: green;"><b>${a['target']}</b></td>
            <td style="color: red;"><b>${a['stop']}</b></td>
            <td style="color: {prob_color}; font-weight: bold; background-color: #f8f9fa;">{a['win_prob']}%</td>
            <td><b>{a['shares']}</b> <br><span style="color:blue; font-size:0.8em;">(${a['cost']})</span></td>
            <td style="font-size:0.9em; color: {'red' if a['calibration'] < 0 else 'green'};">{a['calibration']}%</td>
            <td style="color: {mood_color}; font-weight: bold;">{a['sentiment']}</td>
        </tr>"""
    html += "</table><p><i>Predictions reflect the calibrated probability of hitting the +5% target before hitting the -5% stop loss.</i></p>"

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"🤖 {len(alerts)} High-Probability Setups: {alerts[0]['symbol']} ({alerts[0]['win_prob']}% Win Odds)"
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("✅ Email sent.")
    except:
        print("❌ Email failed.")


def main():
    print("--- 🧠 AI PROBABILITY TRADER RUNNING ---")
    status_msg = get_market_status()
    print(f"🕒 Market Condition: {status_msg}")

    try:
        prob_model = joblib.load('probability_model.pkl')
    except:
        print("❌ Model not found! Run '6b_trainer_probabilistic.py' first.")
        return

    universe = get_universe()
    alerts = []

    print(f"Scanning {len(universe)} stocks...")
    for t in universe:
        symbol = t['symbol']
        if is_on_cooldown(symbol): continue

        res = analyze_stock(t, prob_model)
        if res:
            alerts.append(res)

    if alerts and len(alerts) < 20:
        send_email(alerts, status_msg)
    elif alerts and len(alerts) >= 20:
        print(f"\n🛑 Scan complete. Found {len(alerts)} items above {MIN_PROBABILITY}%. Skipping email to prevent spam.")
    else:
        print(f"No setups found with > {MIN_PROBABILITY}% win probability.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
