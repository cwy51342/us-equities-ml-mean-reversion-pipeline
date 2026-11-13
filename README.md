# Cloud-Backed US Equities ML Mean-Reversion & Momentum Pipeline

An end-to-end quantitative equity research, data warehousing, and automated signal generation system backed by **Supabase (PostgreSQL)**. The pipeline filters the entire US equity market down to a high-momentum institutional universe, maintains an hourly OHLCV data warehouse, trains **Random Forest** regression and probabilistic classification ensembles on oversold intraday dips, and dispatches real-time position-sized trade alerts complete with **VADER NLP news sentiment** and **online error auto-calibration**.

---

## Pipeline Architecture

The system is modularized into five sequential stages:

### Stage 1: Universe Gatekeeper (`1_gatekeeper_universe_seed.py`)
* Cross-references official **SEC EDGAR** reporting company filings (`company_tickers.json`) with **AlphaVantage** active stock listings via an inner join to eliminate OTC/unverified tickers and ETFs.
* Applies regex keyword blacklisting (`WARRANT`, `UNIT`, `ACQUISITION`, `SPAC`, `RIGHT`, `PF`, `DEBENTURE`) to strip derivative securities and shell vehicles, reducing ~12,000 raw listings to ~5,000 clean operating companies in Supabase (`tickers` table).

### Stage 2: Concurrent Hourly Data Warehouse (`2a_warehouse_bulk_ingestor.py` & `2b_warehouse_universe_updater.py`)
* **Multi-Threaded Staleness Check:** Uses `ThreadPoolExecutor` (20 workers) to query the latest candle timestamp per symbol in Supabase and filter out tickers updated within the last hour or after the 16:00 EST market close.
* **Vectorized Batch Ingestion:** Downloads 1-year of 1-hour OHLCV bars in bulk batches of 50 symbols via `yfinance` and performs chunked idempotent upserts into the `candles` table.
* **Fast Universe Updater (`2b`):** Dedicated low-latency updater targeting only the active 153-stock trading universe during market hours.

### Stage 3: Cross-Sectional Momentum Selector (`3_momentum_universe_selector.py`)
* Evaluates rolling hourly price histories (~2,000 hourly bars per stock) across 3-month (441 trading hours), 6-month (882 hours), and 12-month (1,764 hours) horizons.
* Screens for structural trend leaders exceeding multi-horizon return thresholds ($R_{3M} \ge 45\%$, $R_{6M} \ge 55\%$, or $R_{12M} \ge 75\%$) and ranks candidates by composite momentum:
  $$\text{Momentum Score} = \frac{R_{3M} + R_{6M} + R_{12M}}{3}$$
* Flags the **Top 153** highest-scoring equities (`is_universe = True`) for intraday mean-reversion harvesting ("buying dips in structural uptrends").

---

## Machine Learning Engines (Stage 4 & Stage 6)

The repository implements two distinct supervised learning frameworks trained on historical oversold regimes ($\text{RSI}_{14} < 30$) over a **72-hour forward horizon** (`FixedForwardWindowIndexer`):

### Architecture A: Dual-Model Regression + Tree-Variance Confidence Intervals (`6a` & `4a`)
* **Feature Vector:** Normalized 14-period RSI ($\text{RSI}_{14}/100$), relative distance from the 200-period EMA, normalized Average True Range ($\text{ATR}_{14}/\text{Close}$), and normalized 20-period Bollinger Band width ($4\sigma_{20}/\text{Close}$).
* **Dual Random Forest Regressors:**
  * `profit_model.pkl`: Predicts maximum 72-hour forward excursion $(\max(H_{t+1 \dots t+72}) - C_t)/C_t$.
  * `risk_model.pkl`: Predicts maximum 72-hour forward drawdown $(C_t - \min(L_{t+1 \dots t+72}))/C_t$.
* **Ensemble Tree-Variance Confidence Scoring:** Queries all 100 individual decision trees (`profit_model.estimators_`) to compute cross-tree prediction standard deviation ($\sigma_{\text{trees}}$) and maps ensemble agreement to a $[0, 100]$ confidence score:
  $$\text{Confidence} = \max\left(0, \min\left(100, 100 - \frac{\sigma_{\text{trees}}}{\tau} \times 100\right)\right)$$

### Architecture B: Probabilistic Triple-Barrier Classifier (`6b` & `4b`)
* **Expanded Feature Vector:** Augments Architecture A with normalized MACD Histogram ($\text{MACDh}_{12,26,9}/\text{Close}$), 10-period Rate of Change ($\text{ROC}_{10}$), and 20-period Volume Surge ratio ($V_t / \text{SMA}_{20}(V)$).
* **1:1 Risk/Reward Labeling:** Labels historical setups as `1` (`label_success`) strictly when price achieves a $+5\%$ target before breaching a $-5\%$ stop-loss within the 72-hour window.
* **Out-of-Sample Diagnostics:** Evaluates train vs. test accuracy gap to guard against overfitting and tracks true precision win-rate on unseen test folds.
* **Dead-Ticker Self-Healing:** Automatically detects halted or delisted equities returning zero volume over a 5-day lookback and purges them from the active Supabase universe.

---

## Live Execution Features

* **Intraday 5-Minute Splice:** Merges historical 1-hour database candles with live 5-minute intraday bars for real-time indicator calculation.
* **Online Auto-Calibration:** Queries the last 10 resolved predictions per ticker from Supabase (`prediction_logs` / `probability_logs`) and adjusts raw model outputs by the rolling empirical error rate.
* **Fixed-Fractional Risk Position Sizing:** Sizes share quantities dynamically based on a 2% account risk budget (`RISK_PER_TRADE = 0.02`) divided by per-share stop distance, capped at `MAX_POSITION_SIZE`.
* **Local NLP News Sentiment:** Fetches the top 3 recent headlines per triggered ticker and computes compound polarity scores using **NLTK VADER** (`SentimentIntensityAnalyzer`) to tag setups as Bullish, Neutral, or Bearish.
* **Cooldown Gate & Automated HTML Dispatch:** Enforces a 4-hour per-ticker alert cooldown via database logs and sends formatted HTML execution tables over SMTP.

---

## Repository Structure

```text
├── notebooks/
│   └── 0_rsi_scanner_prototype.ipynb      # Initial daily RSI<30 screener prototype
├── src/
│   ├── 1_gatekeeper_universe_seed.py      # SEC + AlphaVantage intersection & SPAC filter
│   ├── 2a_warehouse_bulk_ingestor.py      # Multi-threaded staleness check & batch OHLCV loader
│   ├── 2b_warehouse_universe_updater.py   # Fast hourly updater for Top 153 Universe stocks
│   ├── 3_momentum_universe_selector.py    # 3M/6M/12M momentum ranking engine
│   ├── 4a_live_trader_regression_ci.py    # Live scanner using RF Regressors + Tree Variance CI
│   ├── 4b_live_trader_probabilistic.py    # Live scanner using RF Classifier (+5%/-5% odds)
│   ├── 6a_trainer_regression_ci.py        # Offline trainer for profit_model & risk_model
│   └── 6b_trainer_probabilistic.py        # Offline trainer for probability_model with OOS diagnostics
├── .env.example                           # Environment variable template
├── requirements.txt                       # Python dependencies
└── README.md                              # Documentation
```

---

## Setup & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure environment variables:**
   Copy `.env.example` to `.env` and populate your Supabase URL/Key, AlphaVantage API Key, and SMTP credentials.
3. **Run the pipeline sequentially:**
   ```bash
   # 1. Seed clean US equity tickers
   python src/1_gatekeeper_universe_seed.py

   # 2. Populate hourly candle warehouse
   python src/2a_warehouse_bulk_ingestor.py

   # 3. Rank momentum and select Top 153 Universe
   python src/3_momentum_universe_selector.py

   # 4. Train ML models (Regression CI or Probabilistic)
   python src/6a_trainer_regression_ci.py
   python src/6b_trainer_probabilistic.py

   # 5. Run live intraday scanner
   python src/4b_live_trader_probabilistic.py
   ```
