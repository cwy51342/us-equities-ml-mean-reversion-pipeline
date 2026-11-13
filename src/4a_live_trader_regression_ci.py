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
TARGET_MULT = 0.4
STOP_MULT = 1.35
COOLDOWN_HOURS = 4

# 🎛️ FILTER SETTINGS
USE_HOURLY_TREND_FILTER = False
MIN_HOURLY_RSI = 35
CONFIDENCE_TOLERANCE = 0.20

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

    morning_start = dtime(10, 0)
    morning_end = dtime(11, 30)
    afternoon_start = dtime(13, 30)
    afternoon_end = dtime(15, 0)

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
        res = supabase.table('prediction_logs') \
            .select('id') \
            .eq('symbol', symbol) \
            .gte('created_at', cutoff_time.isoformat()) \
            .limit(1) \
            .execute()
        return len(res.data) > 0
    except Exception:
        return False


def check_hourly_trend(symbol):
    try:
        df_hourly = yf.Ticker(symbol).history(period="5d", interval="1h")
        if df_hourly.empty: return True, 50
        df_hourly.ta.rsi(length=14, append=True)
        current_hourly_rsi = df_hourly.iloc[-1]['RSI_14']

        if current_hourly_rsi < MIN_HOURLY_RSI:
            return False, current_hourly_rsi

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
            label = f"🔥 Bullish ({avg_score:.2f})"
        elif avg_score <= -0.15:
            label = f"❄️ Bearish ({avg_score:.2f})"
        else:
            label = f"😐 Neutral ({avg_score:.2f})"
        return label, "".join(headlines)
    except:
        return "ERROR", ""


def get_historical_error(symbol):
    try:
        res = supabase.table('prediction_logs') \
            .select('error_rate') \
            .eq('symbol', symbol) \
            .not_.is_('error_rate', 'null') \
            .order('created_at', desc=True) \
            .limit(10) \
            .execute()

        if not res.data: return 0.0

        errors = [row['error_rate'] for row in res.data]
        return sum(errors) / len(errors)
    except:
        return 0.0


def analyze_stock(ticker, profit_model, risk_model):
    symbol = ticker['symbol']
    try:
        res = supabase.table('candles').select('close, high, low, date').eq('symbol', symbol).order('date',
                                                                                                    desc=True).limit(
            300).execute()
        if not res.data: return None
        df = pd.DataFrame(res.data)
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['close', 'high', 'low']: df[c] = df[c].astype(float)

        rows_added = 0
        try:
            live = yf.Ticker(symbol).history(period="1d", interval="5m")
            if not live.empty:
                curr_row = live.iloc[-1]
                current_price = float(curr_row['Close'])
                current_time = live.index[-1]
                if not df.empty:
                    last_db_time = pd.to_datetime(df.iloc[-1]['date'])
                    if last_db_time.date() == current_time.date() and last_db_time.hour == current_time.hour:
                        df.iloc[-1, df.columns.get_loc('close')] = current_price
                        df.iloc[-1, df.columns.get_loc('high')] = float(curr_row['High'])
                        df.iloc[-1, df.columns.get_loc('low')] = float(curr_row['Low'])
                    else:
                        new_row = pd.DataFrame({'close': [current_price], 'high': [float(curr_row['High'])],
                                                'low': [float(curr_row['Low'])], 'date': [current_time]})
                        df = pd.concat([df, new_row], ignore_index=True)
                        rows_added = 1
                else:
                    current_price = float(curr_row['Close'])
                    df = pd.DataFrame(
                        [{'close': current_price, 'high': float(curr_row['High']), 'low': float(curr_row['Low']),
                          'date': current_time}])
            else:
                current_price = df.iloc[-1]['close']
        except:
            return None

        target_len = 50 + rows_added
        df_short = df.tail(target_len).copy()
        df_short.ta.rsi(length=14, append=True)
        if 'RSI_14' not in df_short.columns: return None
        rsi_val = df_short.iloc[-1]['RSI_14']

        if rsi_val > 30: return None

        is_trend_safe, hourly_rsi = check_hourly_trend(symbol)
        if USE_HOURLY_TREND_FILTER and not is_trend_safe:
            return None

        df.ta.atr(length=14, append=True)
        df.ta.ema(length=200, append=True)
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

        features = pd.DataFrame([{'feat_rsi': rsi_val / 100.0, 'feat_dist_ema': dist_ema, 'feat_atr_pct': feat_atr,
                                  'feat_bb_width': bb_width}]).fillna(0)

        raw_pred_gain = profit_model.predict(features)[0]
        pred_risk = risk_model.predict(features)[0]
        tree_predictions = [tree.predict(features.values)[0] for tree in profit_model.estimators_]
        std_dev = np.std(tree_predictions)
        confidence_score = max(0.0, min(100.0, 100.0 - (std_dev / CONFIDENCE_TOLERANCE * 100)))

        avg_error = get_historical_error(symbol)
        adjusted_pred_gain = raw_pred_gain + avg_error
        if adjusted_pred_gain <= 0 and raw_pred_gain > 0:
            adjusted_pred_gain = raw_pred_gain * 0.1

        target_price = current_price * (1 + (adjusted_pred_gain * TARGET_MULT))
        stop_price = current_price * (1 - (pred_risk * STOP_MULT))

        est_profit = ((target_price - current_price) / current_price) * 100

        try:
            supabase.table('prediction_logs').insert({
                "symbol": symbol,
                "raw_predicted_gain": float(raw_pred_gain),
                "confidence_score": float(confidence_score)
            }).execute()
        except Exception as e:
            print(f"Warning: Failed to log prediction for {symbol}: {e}")

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
            "trend_1h": round(hourly_rsi, 2),
            "price": round(current_price, 2),
            "target": round(target_price, 2),
            "stop": round(stop_price, 2),
            "est_profit": round(est_profit, 2),
            "shares": rec_shares,
            "cost": round(total_cost, 2),
            "confidence": round(confidence_score, 1),
            "calibration": round(avg_error * 100, 2),
            "sentiment": sentiment,
            "headlines": headlines
        }
    except Exception:
        return None


def send_email(alerts, status_msg):
    if not alerts: return
    html = f"""
    <h3>🤖 AI Smart Alerts ({status_msg})</h3>
    <table border="1" cellpadding="5" style="border-collapse: collapse; text-align: center; width: 100%;">
        <tr style="background-color: #e0e0e0;">
            <th>Ticker</th><th>Price</th><th>Target</th><th>Stop</th><th>Est. Profit</th><th>Shares (Bet)</th><th>Conf.</th><th>Adjust</th><th>Mood</th>
        </tr>
    """
    for a in alerts:
        mood_color = "green" if "Bullish" in a['sentiment'] else "red" if "Bearish" in a['sentiment'] else "gray"
        conf_color = "green" if a['confidence'] > 75 else "orange" if a['confidence'] > 40 else "red"

        html += f"""
        <tr>
            <td><b>{a['symbol']}</b></td>
            <td>${a['price']}</td>
            <td style="color: green;"><b>${a['target']}</b></td>
            <td style="color: red;">${a['stop']}</td>
            <td style="background-color: #d4edda; color: #155724;"><b>+{a['est_profit']}%</b></td>
            <td><b>{a['shares']}</b> <br><span style="color:blue; font-size:0.8em;">(${a['cost']})</span></td>
            <td style="color: {conf_color}; font-weight: bold;">{a['confidence']}%</td>
            <td style="font-size:0.9em; color: {'red' if a['calibration'] < 0 else 'green'};">{a['calibration']}%</td>
            <td style="color: {mood_color}; font-weight: bold;">{a['sentiment']}</td>
        </tr>"""
    html += "</table><p><i>Predictions automatically calibrated based on historical error rates.</i></p>"

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"🤖 {len(alerts)} Calibrated Trades: {alerts[0]['symbol']} ({alerts[0]['confidence']}% Conf)"
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Email sent with {len(alerts)} alerts.")
    except:
        print("❌ Email failed.")


def main():
    print("--- 🧠 AI TRADER + CONTINUOUS CALIBRATION ---")

    status_msg = get_market_status()
    print(f"🕒 Market Condition: {status_msg}")

    if not USE_HOURLY_TREND_FILTER:
        print("⚠️ 1-Hour Trend Filter is OFF. Expect more alerts, but higher risk of falling knives.")

    print("🚀 Proceeding with scan...")

    try:
        p_model = joblib.load('profit_model.pkl')
        r_model = joblib.load('risk_model.pkl')
    except:
        print("❌ Models not found! Run '6a_trainer_regression_ci.py' first.")
        return

    universe = get_universe()
    alerts = []

    print(f"Scanning {len(universe)} stocks...")
    for t in universe:
        symbol = t['symbol']

        if is_on_cooldown(symbol):
            continue

        res = analyze_stock(t, p_model, r_model)
        if res:
            print(f"🚀 SIGNAL: {res['symbol']} (Conf: {res['confidence']}% | Adjust: {res['calibration']}% | AI: +{res['est_profit']}% | News: {res['sentiment']})")
            alerts.append(res)

    if alerts:
        send_email(alerts, status_msg)
    else:
        print("No AI-approved trades right now.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
