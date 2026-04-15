#!/usr/bin/env python3
"""
Swarm Quant Workflows — #02: Sector Rotation Monitor
-----------------------------------------------------
Tracks relative strength and momentum across all 11 GICS sectors.
Ranks sectors by a composite momentum score (weighted relative returns
vs SPY over 1M, 3M, and 6M), adds RSI and moving average signals,
and sends a formatted HTML email every Monday.

Run locally:
    python 02_sector_rotation/sector_rotation.py
"""

import base64
import logging
import os
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import yfinance as yf
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
GMAIL_SCOPES    = ["https://www.googleapis.com/auth/gmail.send"]
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "your-email@example.com")
TOKEN_PATH      = Path(os.environ.get("GMAIL_TOKEN_PATH", "token.json"))

# ── Universe ──────────────────────────────────────────────────────────────────
# Primary sector ETF used for calculations + display
SECTORS: dict[str, dict] = {
    "Technology":             {"ticker": "XLK",  "color": "#6366f1"},
    "Healthcare":             {"ticker": "XLV",  "color": "#10b981"},
    "Financials":             {"ticker": "XLF",  "color": "#f59e0b"},
    "Energy":                 {"ticker": "XLE",  "color": "#ef4444"},
    "Consumer Discretionary": {"ticker": "XLY",  "color": "#8b5cf6"},
    "Consumer Staples":       {"ticker": "XLP",  "color": "#06b6d4"},
    "Industrials":            {"ticker": "XLI",  "color": "#f97316"},
    "Materials":              {"ticker": "XLB",  "color": "#84cc16"},
    "Utilities":              {"ticker": "XLU",  "color": "#64748b"},
    "Real Estate":            {"ticker": "XLRE", "color": "#ec4899"},
    "Communication Services": {"ticker": "XLC",  "color": "#14b8a6"},
}

# Market context tickers shown in the pulse bar
MARKET_TICKERS = {
    "S&P 500": "SPY",
    "Nasdaq":  "QQQ",
    "Russell": "IWM",
    "VIX":     "^VIX",
}

BENCHMARK = "SPY"


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        prices   — daily close prices for sectors + benchmark + market tickers
        vix      — VIX close prices (separate as it behaves differently)
    """
    sector_tickers  = [v["ticker"] for v in SECTORS.values()]
    market_tickers  = list(MARKET_TICKERS.values())
    all_tickers     = list(set(sector_tickers + market_tickers))

    end   = datetime.today()
    start = end - timedelta(days=280)   # ~1 year of trading days

    log.info(f"Downloading price data for {len(all_tickers)} tickers...")
    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    prices = {}
    for t in all_tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t in raw.columns.get_level_values(0):
                    s = raw[t]["Close"]
                else:
                    s = raw.xs(t, level=1, axis=1)["Close"]
            else:
                s = raw["Close"]
            prices[t] = s.dropna()
        except Exception as e:
            log.warning(f"Could not load {t}: {e}")

    df = pd.DataFrame(prices).dropna(how="all")
    log.info(f"Price data loaded: {len(df)} trading days.")
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────

def calc_return(series: pd.Series, days: int) -> float:
    """Return over last N trading days."""
    if len(series) < days + 1:
        return float("nan")
    return (series.iloc[-1] / series.iloc[-(days + 1)] - 1) * 100


def calc_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else float("nan")


def ma_signal(series: pd.Series) -> str:
    """Returns signal string based on MA position."""
    if len(series) < 200:
        return "—"
    price = series.iloc[-1]
    ma50  = series.tail(50).mean()
    ma200 = series.tail(200).mean()
    if price > ma50 > ma200:
        return "BULL"
    if price > ma50 and price < ma200:
        return "MIXED"
    if price < ma50 < ma200:
        return "BEAR"
    return "MIXED"


def build_sector_df(prices: pd.DataFrame) -> pd.DataFrame:
    benchmark = prices[BENCHMARK]
    spy_1w  = calc_return(benchmark, 5)
    spy_1m  = calc_return(benchmark, 21)
    spy_3m  = calc_return(benchmark, 63)
    spy_6m  = calc_return(benchmark, 126)

    rows = []
    for name, meta in SECTORS.items():
        t = meta["ticker"]
        if t not in prices.columns:
            continue
        s = prices[t].dropna()

        r1w  = calc_return(s, 5)
        r1m  = calc_return(s, 21)
        r3m  = calc_return(s, 63)
        r6m  = calc_return(s, 126)
        rsi  = calc_rsi(s)
        sig  = ma_signal(s)

        # Relative return vs SPY
        rs1m = r1m - spy_1m
        rs3m = r3m - spy_3m
        rs6m = r6m - spy_6m

        # Composite momentum score (weighted relative strength)
        score = 0.20 * rs1m + 0.35 * rs3m + 0.45 * rs6m

        rows.append({
            "sector":  name,
            "ticker":  t,
            "color":   meta["color"],
            "price":   round(float(s.iloc[-1]), 2),
            "r1w":     r1w,
            "r1m":     r1m,
            "r3m":     r3m,
            "r6m":     r6m,
            "rs1m":    rs1m,
            "rs3m":    rs3m,
            "rs6m":    rs6m,
            "score":   score,
            "rsi":     rsi,
            "signal":  sig,
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def build_market_pulse(prices: pd.DataFrame) -> list[dict]:
    pulse = []
    for label, ticker in MARKET_TICKERS.items():
        if ticker not in prices.columns:
            continue
        s   = prices[ticker].dropna()
        r1w = calc_return(s, 5)
        pulse.append({
            "label":  label,
            "ticker": ticker,
            "price":  round(float(s.iloc[-1]), 2),
            "r1w":    r1w,
        })
    return pulse


# ── Email HTML ────────────────────────────────────────────────────────────────

def _ret(v: float, decimals: int = 2) -> tuple[str, str]:
    """(formatted string, color)"""
    if v != v:  # nan check
        return "—", "#94a3b8"
    color = "#16a34a" if v >= 0 else "#dc2626"
    sign  = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%", color


def _score_color(score: float) -> str:
    """Background color for momentum score cell."""
    if score >= 3:   return "#dcfce7"
    if score >= 1:   return "#f0fdf4"
    if score >= -1:  return "#fefce8"
    if score >= -3:  return "#fff7ed"
    return "#fef2f2"


def _signal_badge(sig: str) -> str:
    styles = {
        "BULL":  ("background:#dcfce7;color:#15803d", "BULL"),
        "MIXED": ("background:#fef9c3;color:#92400e", "MIXED"),
        "BEAR":  ("background:#fee2e2;color:#991b1b", "BEAR"),
        "—":     ("background:#f1f5f9;color:#94a3b8", "—"),
    }
    style, label = styles.get(sig, styles["—"])
    return (
        f'<span style="{style};padding:2px 7px;border-radius:4px;'
        f'font-size:10px;font-weight:700;">{label}</span>'
    )


def build_email_html(sector_df: pd.DataFrame, pulse: list[dict]) -> str:
    today = datetime.now().strftime("%B %d, %Y")

    # ── Market pulse bar ──────────────────────────────────────────────────────
    pulse_cells = ""
    for p in pulse:
        ret_str, ret_color = _ret(p["r1w"])
        is_vix = p["ticker"] == "^VIX"
        # For VIX, rising is bad (red), falling is good (green) — invert color
        if is_vix:
            ret_color = "#dc2626" if p["r1w"] > 0 else "#16a34a"
        pulse_cells += f"""
          <td style="padding:14px 18px;text-align:center;border-right:1px solid #334155;">
            <div style="color:#94a3b8;font-size:10px;font-weight:600;
                text-transform:uppercase;letter-spacing:1px;">{p['label']}</div>
            <div style="color:#f8fafc;font-size:18px;font-weight:800;margin:3px 0;">
              {p['price']:.2f}
            </div>
            <div style="color:{ret_color};font-size:12px;font-weight:600;">{ret_str} 1W</div>
          </td>"""

    # ── Sector table rows ─────────────────────────────────────────────────────
    rows = ""
    for _, row in sector_df.iterrows():
        r1w_s,  r1w_c  = _ret(row["r1w"])
        r1m_s,  r1m_c  = _ret(row["r1m"])
        r3m_s,  r3m_c  = _ret(row["r3m"])
        rs3m_s, rs3m_c = _ret(row["rs3m"])
        sc_s,   _      = _ret(row["score"])
        sc_bg          = _score_color(row["score"])
        badge          = _signal_badge(row["signal"])

        rank_bg = "#1e3a5f"
        if row["rank"] <= 3:   rank_bg = "#15803d"
        if row["rank"] >= 9:   rank_bg = "#991b1b"

        rsi_color = "#94a3b8"
        if row["rsi"] >= 70:   rsi_color = "#dc2626"
        elif row["rsi"] <= 30: rsi_color = "#16a34a"

        rows += f"""
          <tr style="border-bottom:1px solid #f1f5f9;">
            <td style="padding:11px 10px;vertical-align:middle;text-align:center;">
              <span style="display:inline-block;background:{rank_bg};color:#fff;
                border-radius:50%;width:24px;height:24px;text-align:center;
                line-height:24px;font-size:11px;font-weight:700;">{int(row['rank'])}</span>
            </td>
            <td style="padding:11px 10px;vertical-align:middle;">
              <span style="display:inline-block;width:4px;height:32px;
                background:{row['color']};border-radius:2px;
                vertical-align:middle;margin-right:8px;"></span>
              <span style="font-weight:700;color:#0f172a;font-size:13px;
                vertical-align:middle;">{row['sector']}</span><br>
              <span style="color:#64748b;font-size:11px;margin-left:12px;">{row['ticker']} &nbsp; ${row['price']:.2f}</span>
            </td>
            <td style="padding:11px 10px;text-align:right;vertical-align:middle;
                font-weight:600;font-size:13px;color:{r1w_c};">{r1w_s}</td>
            <td style="padding:11px 10px;text-align:right;vertical-align:middle;
                font-weight:600;font-size:13px;color:{r1m_c};">{r1m_s}</td>
            <td style="padding:11px 10px;text-align:right;vertical-align:middle;
                font-weight:600;font-size:13px;color:{r3m_c};">{r3m_s}</td>
            <td style="padding:11px 10px;text-align:right;vertical-align:middle;
                font-weight:600;font-size:13px;color:{rs3m_c};">{rs3m_s}</td>
            <td style="padding:11px 10px;text-align:center;vertical-align:middle;
                font-weight:700;font-size:12px;color:{rsi_color};">{row['rsi']:.0f}</td>
            <td style="padding:11px 10px;text-align:center;vertical-align:middle;">{badge}</td>
            <td style="padding:11px 10px;text-align:right;vertical-align:middle;">
              <span style="background:{sc_bg};padding:4px 8px;border-radius:4px;
                font-weight:700;font-size:13px;">{sc_s}</span>
            </td>
          </tr>"""

    # ── Top / bottom movers ───────────────────────────────────────────────────
    top3    = sector_df.head(3)
    bottom3 = sector_df.tail(3).iloc[::-1]

    def mover_pill(row, good: bool) -> str:
        bg = "#dcfce7" if good else "#fee2e2"
        tc = "#15803d" if good else "#991b1b"
        sc_str, _ = _ret(row["score"])
        return (
            f'<span style="background:{bg};color:{tc};padding:6px 12px;'
            f'border-radius:20px;font-size:12px;font-weight:700;'
            f'display:inline-block;margin:3px;">'
            f'{row["sector"]} ({sc_str})</span>'
        )

    top_pills    = " ".join(mover_pill(r, True)  for _, r in top3.iterrows())
    bottom_pills = " ".join(mover_pill(r, False) for _, r in bottom3.iterrows())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Sector Rotation Report — {today}</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;">
<tr><td align="center" style="padding:32px 16px;">
<table width="760" cellpadding="0" cellspacing="0">

  <!-- Header -->
  <tr>
    <td style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
        padding:36px 40px;border-radius:12px 12px 0 0;text-align:center;">
      <p style="margin:0 0 6px;color:#60a5fa;font-size:11px;font-weight:700;
          letter-spacing:2px;text-transform:uppercase;">Swarm Investments</p>
      <h1 style="margin:0 0 8px;color:#fff;font-size:28px;font-weight:800;
          letter-spacing:-0.5px;">Sector Rotation Monitor</h1>
      <p style="margin:0;color:#94a3b8;font-size:14px;">Weekly Report &mdash; {today}</p>
    </td>
  </tr>

  <!-- Market Pulse -->
  <tr>
    <td style="background:#1e293b;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>{pulse_cells}</tr>
      </table>
    </td>
  </tr>

  <!-- Movers strip -->
  <tr>
    <td style="background:#fff;padding:16px 20px;border-bottom:1px solid #e2e8f0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="width:50%;vertical-align:top;padding-right:12px;">
            <div style="font-size:10px;font-weight:700;color:#64748b;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
              Top Momentum
            </div>
            {top_pills}
          </td>
          <td style="width:50%;vertical-align:top;padding-left:12px;
              border-left:1px solid #f1f5f9;">
            <div style="font-size:10px;font-weight:700;color:#64748b;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
              Lagging Momentum
            </div>
            {bottom_pills}
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Sector Table -->
  <tr>
    <td style="background:#fff;padding:0;border-radius:0 0 12px 12px;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
          <th style="padding:10px;text-align:center;font-size:10px;color:#94a3b8;
              font-weight:600;width:36px;">RK</th>
          <th style="padding:10px;text-align:left;font-size:10px;color:#94a3b8;font-weight:600;">Sector</th>
          <th style="padding:10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">1W</th>
          <th style="padding:10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">1M</th>
          <th style="padding:10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">3M</th>
          <th style="padding:10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">vs SPY 3M</th>
          <th style="padding:10px;text-align:center;font-size:10px;color:#94a3b8;font-weight:600;">RSI</th>
          <th style="padding:10px;text-align:center;font-size:10px;color:#94a3b8;font-weight:600;">Trend</th>
          <th style="padding:10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">Score</th>
        </tr>
        {rows}
      </table>
    </td>
  </tr>

  <!-- Legend -->
  <tr>
    <td style="padding:16px 8px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="background:#fff;padding:12px 16px;border-radius:8px;
              font-size:10px;color:#94a3b8;line-height:1.8;">
            <strong style="color:#64748b;">Score</strong> = weighted relative return vs SPY
            (1M&times;20% + 3M&times;35% + 6M&times;45%) &nbsp;&bull;&nbsp;
            <strong style="color:#64748b;">RSI</strong> = 14-day (
            <span style="color:#dc2626;">&#x2265;70 overbought</span> /
            <span style="color:#16a34a;">&#x2264;30 oversold</span>) &nbsp;&bull;&nbsp;
            <strong style="color:#64748b;">Trend</strong> = MA50/MA200 position
            (<span style="color:#15803d;">BULL</span> = price &gt; MA50 &gt; MA200)
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Footer -->
  <tr>
    <td style="background:#0f172a;padding:20px 32px;border-radius:8px;text-align:center;">
      <p style="margin:0 0 4px;color:#475569;font-size:11px;">
        Generated by <strong style="color:#60a5fa;">Swarm Investments Quant Workflow</strong>
      </p>
      <p style="margin:0;color:#334155;font-size:10px;">
        Data: Yahoo Finance &middot; Benchmark: SPY &middot; {today}
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ── Gmail ─────────────────────────────────────────────────────────────────────

def get_gmail_service():
    if not TOKEN_PATH.exists():
        raise FileNotFoundError(f"token.json not found at '{TOKEN_PATH}'.")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), GMAIL_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def send_email(service, to: str, subject: str, html_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["To"]      = to
    msg["From"]    = "me"
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    log.info(f"Email sent -> {to}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Sector Rotation Monitor -- Weekly Run ===")

    prices    = fetch_data()
    sector_df = build_sector_df(prices)
    pulse     = build_market_pulse(prices)

    log.info(f"Ranked {len(sector_df)} sectors.")

    html    = build_email_html(sector_df, pulse)
    today   = datetime.now().strftime("%B %d, %Y")
    service = get_gmail_service()
    send_email(service, RECIPIENT_EMAIL, f"Sector Rotation Report -- {today}", html)
    log.info("Done.")


if __name__ == "__main__":
    main()
