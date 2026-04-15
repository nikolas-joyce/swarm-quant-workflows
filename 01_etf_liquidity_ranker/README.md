# Workflow 01 — ETF Liquidity Ranker

Ranks the most liquid US ETFs in each category by 30-day average daily dollar
volume (ADV) and delivers a formatted HTML email every Monday morning.

**Schedule:** Every Monday at 9:00 AM ET  
**Data source:** Yahoo Finance (`yfinance`)  
**Coverage:** ~300 ETFs across 28 categories  
**Output:** HTML email with ranked tables per category

---

## What's in the Report

Each category shows the top 5 ETFs ranked by ADV, with:

| Column | Description |
|---|---|
| Rank | Liquidity rank within category |
| ETF | Ticker and full name |
| Price | Last closing price |
| 30d ADV | Average daily dollar volume (30 trading days) |
| 1W | 1-week price return |
| 1M | 1-month price return |

Only ETFs with ADV > $10M are included. This filters out illiquid products
that are unsuitable for active trading.

---

## Categories Covered

- US Large / Mid / Small Cap Equity
- International Developed & Emerging Markets
- US Government, Corporate & High Yield Bonds
- TIPS / Inflation-Linked
- Real Estate (REITs)
- Commodities (Gold, Broad, Energy ETPs)
- Sector ETFs (Energy, Tech, Healthcare, Financials, Consumer, Industrials, Utilities)
- Factor ETFs (Momentum, Value, Quality, Low Volatility)
- Dividend Income
- Volatility Products
- Leveraged Equity
- Thematic & Innovation

---

## Configuration

Edit constants at the top of `etf_liquidity_ranker.py`:

```python
LOOKBACK_DAYS = 30          # Days used for ADV calculation
MIN_ADV_USD   = 10_000_000  # $10M minimum ADV
TOP_N         = 5           # ETFs shown per category
```

To add tickers or categories, edit the `ETF_UNIVERSE` dictionary.

---

## Files

```
01_etf_liquidity_ranker/
├── etf_liquidity_ranker.py        # Main script (run by GitHub Actions)
├── notebooks/
│   └── ETF_Liquidity_Ranker.ipynb # Colab notebook for development
└── README.md
```

See the [root README](../README.md) for setup instructions.
