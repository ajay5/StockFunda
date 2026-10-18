# Bharat Market AI Production Research Platform

A deployable Streamlit research platform with NSE/BSE sector selection, Yahoo Finance historical price ingestion, Yahoo Finance and Google News current headlines, timestamped news collection, return forecasting, BUY/HOLD/SELL classification, expanding-window validation, probability calibration, baseline comparisons, transaction-cost checks and permanent live feedback.

## Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # Windows: copy .env.example .env
streamlit run app.py
```

## Docker
```bash
docker compose up --build
```

## Data that improves learning
Yahoo Finance supplies historical OHLCV prices and currently available company headlines; Google News RSS and the free GDELT DOC API add current news coverage. The default model price history is five years (selectable as five or ten years). Model news predictors are separate point-in-time rolling sentiment averages: the latest 365 days of NSE company announcements, 90 days of non-social news, and 30 days of Reddit/Bluesky posts. The app automatically checks the NSE company-announcements archive for the latest 365 days; Data Center can fetch and save a selected range, with a one-year default. NSE RSS is a short latest-items window, not a historical archive. NSE announcements are exchange disclosures rather than general editorial news. Yahoo Finance's `yfinance` news endpoint is unofficial, may return no results or change without notice, and does not provide a complete historical archive. GDELT and social APIs have their own coverage, rate limits, and availability constraints. Collected items are retained locally; Data Center can also import a licensed point-in-time historical news archive. These additional inputs do not guarantee improved predictive accuracy: compare walk-forward results to baselines. Required columns:
`published_at,symbol,headline,source,url,sentiment,relevance`.
Never backdate revised information. Keep symbols identical to the app (`TCS.NS`, `RELIANCE.BO`). Sentiment is -1 to +1 and relevance is 0 to 1. Historical news features use provider timestamps (GDELT's first-seen timestamp for GDELT stories) and assign after-close items to the next Indian trading session to reduce look-ahead leakage. Social posts are assigned lower relevance weights to reduce their influence versus newswire headlines. Model predictors also include rolling stock, market-index and sector-proxy returns/volatility derived from Yahoo historical prices.

To enable Reddit, create an OAuth app with Reddit, then set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and a descriptive `REDDIT_USER_AGENT` in `.env`. These credentials are optional; Yahoo Finance, Google News, GDELT, and public Bluesky search are attempted without them. Never commit real credentials.

Set `ALPHAVANTAGE_API_KEY`, `MARKETAUX_API_TOKEN`, and `FINNHUB_API_KEY` in `.env` to enable the API calls. Alpha Vantage marks `NEWS_SENTIMENT` as premium, so the app skips it for a free-key setup rather than making a request that returns a premium-plan error. A free API key does not grant free access to every endpoint. The **News** tab includes a manual check for `TIME_SERIES_DAILY` with `outputsize=compact`, which is available to free keys and returns up to 100 daily OHLCV rows. Enter the correct Alpha Vantage symbol if the suggested BSE-formatted symbol does not match; responses are cached for six hours to conserve the daily quota. These Alpha prices are raw/as-traded, displayed separately, and are not blended into the model's auto-adjusted Yahoo history. Alpha Vantage data is OHLCV price data, not news.

The app calls Finnhub's free `/news?category=general` endpoint for broad market headlines and tries its company-news endpoint for the selected stock. Broad headlines are assigned lower relevance. The **News** tab's provider table shows whether each key is configured, whether `app.py` called an endpoint, result count, and response details. Click **Refresh news providers now** to bypass the five-minute news cache. Restart Streamlit after changing `.env`. Key values are never displayed. Some APIs may return no articles for an Indian ticker even when authentication succeeds. NSE's RSS feed and company-announcements archive are public endpoints, but remain subject to NSE availability and website terms; the historical fetch applies the selected stock symbol and date range, and duplicate stories are ignored on save.

For production OHLCV, replace Yahoo with a licensed feed and maintain:
`timestamp,symbol,open,high,low,close,volume,adjustment_factor,source,ingested_at`.
Add point-in-time fundamentals, official sector indices, India VIX, FII/DII, USD/INR, crude, overnight global returns and exchange trading calendar.

## Daily operating schedule
- Every 15-30 minutes: collect licensed news/social data with event timestamp, source, relevance and deduplication.
- After market close: ingest finalized bars, validate corporate actions, reconcile matured forecasts, train challengers.
- Weekly: review drift, calibration, live MAE, directional balanced accuracy, BUY/SELL precision and costs.
- Promote only after enough live samples and all gates pass. Do not auto-promote from a single backtest.

## Production migration checklist
Managed PostgreSQL; immutable object storage; remote MLflow registry; OIDC/SSO; secrets manager; encrypted backups; structured logs; alerting; rate limits; WAF; data-provider SLAs; legal/data licensing review; paper-trading period; model-risk approval.

## Limitations
No model can be always accurate, and additional features do not guarantee better out-of-sample results. Compare the walk-forward metrics against their baselines before relying on a model. Social posts may contain spam, promotion, or manipulation; apply provider terms and moderation before production use. Historical news grows only after collection unless you provide a licensed archive. VADER is a lightweight baseline; use a validated finance-domain sentiment/event model for production. Yahoo/yfinance is research-oriented, so commercial production needs licensed market data.
