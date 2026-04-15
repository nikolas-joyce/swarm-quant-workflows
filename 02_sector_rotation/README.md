# Workflow 02 — Sector Rotation Monitor

Tracks relative strength and momentum across all 11 GICS sectors every Monday.
Ranks sectors by a composite momentum score (weighted relative returns vs SPY),
with RSI and moving average trend signals.

**Schedule:** Every Monday at 9:15 AM ET  
**Data source:** Yahoo Finance (`yfinance`)  
**Benchmark:** SPY  
**Output:** HTML email with full sector ranking table

---

## What's in the Report

### Market Pulse Bar
Live snapshot of SPY, QQQ, IWM, and VIX with 1-week returns.

### Top / Lagging Momentum
Quick-scan pills showing the top 3 and bottom 3 sectors by momentum score.

### Sector Rankings Table

| Column | Description |
|---|---|
| Rank | Momentum rank (green top 3, red bottom 3) |
| Sector | Name, ticker, last price |
| 1W / 1M / 3M | Absolute price returns |
| vs SPY 3M | 3-month return relative to S&P 500 |
| RSI | 14-day RSI (red ≥70 overbought, green ≤30 oversold) |
| Trend | MA50/MA200 position: BULL / MIXED / BEAR |
| Score | Composite momentum score |

### Momentum Score Formula
```
Score = (1M relative return × 20%)
      + (3M relative return × 35%)
      + (6M relative return × 45%)
```
Higher = stronger momentum relative to SPY.

---

## Sectors Covered (GICS)

| Sector | ETF |
|---|---|
| Technology | XLK |
| Healthcare | XLV |
| Financials | XLF |
| Energy | XLE |
| Consumer Discretionary | XLY |
| Consumer Staples | XLP |
| Industrials | XLI |
| Materials | XLB |
| Utilities | XLU |
| Real Estate | XLRE |
| Communication Services | XLC |

---

## Files

```
02_sector_rotation/
├── sector_rotation.py     # Main script
├── notebooks/             # Colab notebook (coming soon)
└── README.md
```

See the [root README](../README.md) for setup instructions.
