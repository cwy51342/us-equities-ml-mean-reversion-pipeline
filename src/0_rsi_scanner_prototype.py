import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time

load_dotenv()

# --- CONFIGURATION (Loaded securely from environment variables) ---
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your_app_password")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "your_email@gmail.com")

# Trading Settings
RSI_PERIOD = 14
RSI_THRESHOLD = 30
UNIVERSE_FILE = "universe.csv"

def get_stock_data(ticker):
    """
    Downloads daily data for the ticker and calculates RSI.
    """
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            close_series = df[('Close', ticker)]
        else:
            close_series = df['Close']
            
        rsi_series = ta.rsi(close_series, length=RSI_PERIOD)
        
        if rsi_series is None or rsi_series.empty:
            return None

        latest_rsi = rsi_series.iloc[-1]
        latest_price = close_series.iloc[-1]
        
        return {
            "ticker": ticker,
            "rsi": round(latest_rsi, 2),
            "price": round(latest_price, 2)
        }

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def send_email_alert(opportunities):
    """
    Sends an email with the list of oversold stocks.
    """
    if not opportunities:
        print("No opportunities found today. No email sent.")
        return

    subject = f"🔔 ALGO ALERT: {len(opportunities)} Stocks Oversold (RSI < {RSI_THRESHOLD})"
    
    body_html = """
    <h3>Oversold High-Performance Stocks</h3>
    <p>The following stocks from your universe have an RSI below 30:</p>
    <table border="1" style="border-collapse: collapse; cellpadding: 5;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 8px;">Ticker</th>
            <th style="padding: 8px;">Price</th>
            <th style="padding: 8px;">RSI (14)</th>
        </tr>
    """
    
    for stock in opportunities:
        body_html += f"""
        <tr>
            <td style="padding: 8px;"><b>{stock['ticker']}</b></td>
            <td style="padding: 8px;">${stock['price']}</td>
            <td style="padding: 8px; color: red;"><b>{stock['rsi']}</b></td>
        </tr>
        """
    
    body_html += """
    </table>
    <p><i>Check VIX, News, and Analyst Ratings before entry.</i></p>
    """

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body_html, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Alert email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def main():
    print(f"--- Starting RSI Scan for {datetime.now().strftime('%Y-%m-%d')} ---")
    
    try:
        tickers = pd.read_csv(UNIVERSE_FILE, header=None)[0].tolist()
        print(f"Loaded {len(tickers)} stocks from {UNIVERSE_FILE}")
    except FileNotFoundError:
        print(f"Error: Could not find {UNIVERSE_FILE}. Please create it.")
        return

    oversold_stocks = []

    for i, ticker in enumerate(tickers):
        ticker = ticker.strip().upper()
        print(f"Scanning {ticker} ({i+1}/{len(tickers)})...", end="\r")
        
        data = get_stock_data(ticker)
        
        if data:
            if data['rsi'] < RSI_THRESHOLD:
                print(f"\n[!] MATCH FOUND: {ticker} | RSI: {data['rsi']}")
                oversold_stocks.append(data)
            
        time.sleep(0.2)

    print("\nScan complete.")
    if oversold_stocks:
        send_email_alert(oversold_stocks)
    else:
        print("No stocks meet the oversold criteria.")

if __name__ == "__main__":
    main()
