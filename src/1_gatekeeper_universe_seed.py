import os
from dotenv import load_dotenv
import requests
import pandas as pd
import io
from supabase import create_client, Client

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")

# Keywords to EXCLUDE junk stocks
BLACKLIST_KEYWORDS = [
    "WARRANT", "UNIT", "ACQUISITION", "SPAC", "RIGHT", "PF", "DEBENTURE"
]

# Initialize Database Connection
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_sec_tickers():
    """
    Fetches the official list of tickers from the SEC.
    Good for verifying a company is legitimate and reporting to the gov.
    """
    print("Fetching SEC data...")
    headers = {'User-Agent': "QuantResearchBot (research@example.com)"}
    url = "https://www.sec.gov/files/company_tickers.json"

    try:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        sec_df = pd.DataFrame(data.values())
        sec_df = sec_df.rename(columns={'ticker': 'symbol', 'title': 'name'})
        print(f"SEC Source: {len(sec_df)} tickers found.")
        return sec_df
    except Exception as e:
        print(f"Error fetching SEC data: {e}")
        return pd.DataFrame()


def get_alphavantage_listing():
    """
    Fetches all active listings from AlphaVantage.
    Good for filtering out delisted stocks and ETFs.
    """
    print("Fetching AlphaVantage listing status...")
    url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={ALPHAVANTAGE_KEY}"

    try:
        s = requests.get(url).content
        av_df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        av_df = av_df[av_df['assetType'] == 'Stock']
        av_df = av_df[av_df['status'] == 'Active']
        print(f"AlphaVantage Source: {len(av_df)} active stocks found.")
        return av_df
    except Exception as e:
        print(f"Error fetching AlphaVantage data: {e}")
        return pd.DataFrame()


def clean_and_merge(sec_df, av_df):
    """
    Intersects the two lists and removes 'junk' based on name keywords.
    """
    print("Merging and Cleaning lists...")

    sec_df['symbol'] = sec_df['symbol'].str.replace('-', '.')
    av_df['symbol'] = av_df['symbol'].str.replace('-', '.')

    merged = pd.merge(sec_df, av_df, on='symbol', how='inner')

    mask = merged['name_x'].str.upper().str.contains('|'.join(BLACKLIST_KEYWORDS))
    clean_df = merged[~mask]

    print(f"After filtering Junk/SPACs: {len(clean_df)} candidates remain.")
    return clean_df


def save_candidates_to_supabase(df):
    """
    Upserts the cleaned list into the 'tickers' table.
    """
    print("Saving to Supabase...")

    records = []
    for index, row in df.iterrows():
        records.append({
            "symbol": row['symbol'],
            "name": row['name_x'],
            "sector": row.get('sector', None),
            "is_active": True
        })

    batch_size = 1000
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        data, count = supabase.table('tickers').upsert(batch).execute()
        print(f"Upserted batch {i} - {i + batch_size}")


def main():
    sec_data = get_sec_tickers()
    av_data = get_alphavantage_listing()

    if sec_data.empty or av_data.empty:
        print("Failed to fetch one of the sources.")
        return

    candidates = clean_and_merge(sec_data, av_data)
    save_candidates_to_supabase(candidates)

    print("✅ Gatekeeper run complete. Ticker database updated.")


if __name__ == "__main__":
    main()
