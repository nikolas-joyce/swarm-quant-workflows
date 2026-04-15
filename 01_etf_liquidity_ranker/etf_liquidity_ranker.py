#!/usr/bin/env python3
"""
Swarm Quant Workflows — #01: ETF Liquidity Ranker
--------------------------------------------------
Fetches OHLCV data via yfinance, ranks ~300 US ETFs by 30-day average
daily dollar volume within each category, and sends a formatted HTML
email via Gmail API.

Run locally:
    python 01_etf_liquidity_ranker/etf_liquidity_ranker.py

GitHub Actions runs this every Monday at 9 AM ET automatically.
Set RECIPIENT_EMAIL and GMAIL_TOKEN_PATH as environment variables or
GitHub Secrets — never hardcode credentials.
"""

import base64
import logging
import os
import sys
import time
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
LOOKBACK_DAYS   = 30
MIN_ADV_USD     = 10_000_000   # $10M floor
TOP_N           = 5
GMAIL_SCOPES    = ["https://www.googleapis.com/auth/gmail.send"]

RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "your-email@example.com")
TOKEN_PATH      = Path(os.environ.get("GMAIL_TOKEN_PATH", "token.json"))

# ── ETF Universe ──────────────────────────────────────────────────────────────
ETF_UNIVERSE: dict[str, list[str]] = {
    "US Large Cap Equity": [
        "SPY", "IVV", "VOO", "QQQ", "VTI", "SPLG", "RSP",
        "SCHB", "ITOT", "IWB", "SPYG", "SPYV", "IWD", "IWF",
    ],
    "US Mid Cap Equity": [
        "MDY", "IJH", "VO", "IWR", "SCHM", "SPMD", "IVOO", "MDYG", "MDYV",
    ],
    "US Small Cap Equity": [
        "IWM", "VB", "IJR", "SCHA", "VTWO", "VIOO", "SLY", "SLYG", "SLYV", "IWC",
    ],
    "International Developed Equity": [
        "EFA", "VEA", "IDEV", "SCHF", "SPDW", "VGK", "EWJ",
        "IEFA", "FEZ", "EWG", "EWU", "EWC", "EWA",
    ],
    "Emerging Markets Equity": [
        "EEM", "VWO", "IEMG", "SCHE", "SPEM", "EWZ", "MCHI",
        "EWY", "EWT", "INDA", "EEMV",
    ],
    "US Government Bonds": [
        "TLT", "IEF", "SHY", "GOVT", "VGIT", "VGLT", "BIL",
        "SHV", "TBT", "EDV", "TMF", "TBF", "SPTS", "SPTL",
    ],
    "US Corporate Bonds": [
        "LQD", "VCIT", "IGIB", "QLTA", "USIG", "VCLT", "IGSB", "FLOT", "JAAA",
    ],
    "High Yield Bonds": [
        "HYG", "JNK", "USHY", "FALN", "ANGL", "SJNK", "SHYG", "HYS",
    ],
    "TIPS / Inflation-Linked": [
        "TIP", "VTIP", "STIP", "SCHP", "LTPZ", "PBTP",
    ],
    "Real Estate (REITs)": [
        "VNQ", "IYR", "SCHH", "RWR", "USRT", "XLRE", "REM", "MORT",
    ],
    "Commodities — Gold": [
        "GLD", "IAU", "GLDM", "SGOL", "BAR", "PHYS",
    ],
    "Commodities — Broad": [
        "DBC", "PDBC", "GSG", "DJP", "USCI", "BCI",
    ],
    "Commodities — Energy (ETPs)": [
        "USO", "UCO", "SCO", "UNG", "BOIL", "KOLD",
    ],
    "Energy Sector": [
        "XLE", "VDE", "FENY", "IYE", "OIH", "XOP", "FCG",
    ],
    "Technology Sector": [
        "XLK", "VGT", "FTEC", "IYW", "SMH", "SOXX", "IGV", "HACK", "CIBR", "WCLD",
    ],
    "Healthcare Sector": [
        "XLV", "VHT", "FHLC", "IYH", "IBB", "XBI", "ARKG", "PPH",
    ],
    "Financials Sector": [
        "XLF", "VFH", "FNCL", "IYF", "KRE", "KBE", "IAI", "KBWB",
    ],
    "Consumer — Staples & Discretionary": [
        "XLP", "VDC", "FSTA", "XLY", "VCR", "FDIS", "IYC",
    ],
    "Industrials & Materials": [
        "XLI", "VIS", "FIDU", "XLB", "VAW", "IYJ", "IYM",
    ],
    "Utilities & Communications": [
        "XLU", "VPU", "FUTY", "IDU", "XLC", "VOX",
    ],
    "Factor — Momentum": [
        "MTUM", "PDP", "QMOM", "IMOM", "MMTM", "FDMO",
    ],
    "Factor — Value": [
        "VTV", "IVE", "RPV", "FVAL", "VLUE", "AVLV", "QVAL",
    ],
    "Factor — Quality": [
        "QUAL", "DGRW", "DGRO", "SPHQ", "JQUA",
    ],
    "Factor — Low Volatility": [
        "USMV", "SPLV", "EFAV", "EEMV", "FDLO",
    ],
    "Dividend Income": [
        "VYM", "SCHD", "DVY", "HDV", "SDY", "VIG", "NOBL",
    ],
    "Volatility Products": [
        "VXX", "UVXY", "SVXY", "VIXY", "VIXM",
    ],
    "Leveraged Equity (US)": [
        "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SPXU", "SSO", "SDS",
    ],
    "Thematic & Innovation": [
        "ARKK", "ARKG", "ARKF", "ARKW", "BOTZ", "ROBO",
        "ICLN", "TAN", "FAN", "DRIV", "LIT", "BATT",
    ],
}


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_etf_data(universe: dict[str, list[str]]) -> pd.DataFrame:
    all_tickers = sorted({t for tickers in universe.values() for t in tickers})
    log.info(f"Downloading OHLCV for {len(all_tickers)} tickers...")

    end   = datetime.today()
    start = end - timedelta(days=90)

    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    records: list[dict] = []
    for ticker in all_tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    df = raw[ticker]
                else:
                    df = raw.xs(ticker, level=1, axis=1)
            else:
                df = raw

            df = df[["Close", "Volume"]].dropna().tail(LOOKBACK_DAYS)
            if len(df) < 10:
                continue

            n          = len(df)
            avg_vol    = float(df["Volume"].mean())
            avg_price  = float(df["Close"].mean())
            last_price = float(df["Close"].iloc[-1])
            adv_usd    = avg_vol * avg_price
            week_ret   = (df["Close"].iloc[-1] / df["Close"].iloc[max(0, n - 5)] - 1) * 100
            month_ret  = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

            records.append({
                "ticker":     ticker,
                "avg_volume": avg_vol,
                "last_price": last_price,
                "adv_usd":    adv_usd,
                "week_ret":   float(week_ret),
                "month_ret":  float(month_ret),
            })
        except Exception as e:
            log.debug(f"Skip {ticker}: {e}")

    df_all = pd.DataFrame(records)
    if df_all.empty:
        return df_all

    qualified = df_all.loc[df_all["adv_usd"] >= MIN_ADV_USD, "ticker"].tolist()
    log.info(f"{len(qualified)}/{len(all_tickers)} tickers pass ${MIN_ADV_USD/1e6:.0f}M ADV floor. Fetching names...")

    name_map: dict[str, str] = {}
    for i, t in enumerate(qualified):
        try:
            info = yf.Ticker(t).info
            name_map[t] = info.get("shortName") or info.get("longName") or t
        except Exception:
            name_map[t] = t
        if i % 25 == 24:
            time.sleep(1)

    df_all["name"] = df_all["ticker"].map(name_map).fillna(df_all["ticker"])
    return df_all


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_by_category(
    df: pd.DataFrame,
    universe: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    ranked: dict[str, pd.DataFrame] = {}
    for category, tickers in universe.items():
        cat = df[df["ticker"].isin(tickers) & (df["adv_usd"] >= MIN_ADV_USD)].copy()
        if cat.empty:
            continue
        cat = cat.sort_values("adv_usd", ascending=False).head(TOP_N).reset_index(drop=True)
        cat["rank"] = cat.index + 1
        ranked[category] = cat
    return ranked


# ── Email HTML ────────────────────────────────────────────────────────────────

def _fmt_adv(v: float) -> str:
    if v >= 1e9:  return f"${v / 1e9:.2f}B"
    if v >= 1e6:  return f"${v / 1e6:.1f}M"
    return f"${v / 1e3:.0f}K"


def _fmt_ret(r: float) -> tuple[str, str]:
    color = "#16a34a" if r >= 0 else "#dc2626"
    sign  = "+" if r >= 0 else ""
    return f"{sign}{r:.2f}%", color


def build_email_html(ranked: dict[str, pd.DataFrame]) -> str:
    today      = datetime.now().strftime("%B %d, %Y")
    total_cats = len(ranked)
    total_etfs = sum(len(v) for v in ranked.values())

    sections = ""
    for category, df in ranked.items():
        rows = ""
        for _, row in df.iterrows():
            wret, wcolor = _fmt_ret(row["week_ret"])
            mret, mcolor = _fmt_ret(row["month_ret"])
            name = str(row.get("name", row["ticker"]))[:48]
            rows += f"""
              <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:11px 10px;vertical-align:middle;">
                  <span style="display:inline-block;background:#1e3a5f;color:#fff;
                    border-radius:50%;width:22px;height:22px;text-align:center;
                    line-height:22px;font-size:11px;font-weight:700;">{int(row['rank'])}</span>
                </td>
                <td style="padding:11px 10px;vertical-align:middle;">
                  <span style="font-weight:700;color:#0f172a;font-size:14px;">{row['ticker']}</span><br>
                  <span style="color:#64748b;font-size:11px;">{name}</span>
                </td>
                <td style="padding:11px 10px;text-align:right;vertical-align:middle;
                    font-size:13px;color:#374151;">${row['last_price']:.2f}</td>
                <td style="padding:11px 10px;text-align:right;vertical-align:middle;">
                  <span style="font-weight:700;font-size:13px;color:#0f172a;">{_fmt_adv(row['adv_usd'])}</span>
                </td>
                <td style="padding:11px 10px;text-align:right;vertical-align:middle;
                    font-weight:600;font-size:13px;color:{wcolor};">{wret}</td>
                <td style="padding:11px 10px;text-align:right;vertical-align:middle;
                    font-weight:600;font-size:13px;color:{mcolor};">{mret}</td>
              </tr>"""

        sections += f"""
        <div style="margin-bottom:28px;">
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="border-radius:8px;overflow:hidden;box-shadow:0 1px 6px rgba(0,0,0,0.08);">
            <tr>
              <td style="background:linear-gradient(135deg,#1e3a5f,#2563eb);padding:11px 16px;">
                <span style="color:#fff;font-size:12px;font-weight:700;
                  letter-spacing:0.8px;text-transform:uppercase;">{category}</span>
              </td>
            </tr>
            <tr>
              <td style="background:#fff;padding:0;">
                <table width="100%" cellpadding="0" cellspacing="0">
                  <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
                    <th style="padding:8px 10px;text-align:left;font-size:10px;color:#94a3b8;font-weight:600;width:36px;">RK</th>
                    <th style="padding:8px 10px;text-align:left;font-size:10px;color:#94a3b8;font-weight:600;">ETF</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">Price</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">30d ADV</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">1W</th>
                    <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">1M</th>
                  </tr>
                  {rows}
                </table>
              </td>
            </tr>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>ETF Liquidity Report — {today}</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;">
<tr><td align="center" style="padding:32px 16px;">
<table width="680" cellpadding="0" cellspacing="0">

  <tr>
    <td style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
        padding:36px 40px;border-radius:12px 12px 0 0;text-align:center;">
      <p style="margin:0 0 6px;color:#60a5fa;font-size:11px;font-weight:700;
          letter-spacing:2px;text-transform:uppercase;">Swarm Investments</p>
      <h1 style="margin:0 0 8px;color:#fff;font-size:28px;font-weight:800;
          letter-spacing:-0.5px;">ETF Liquidity Ranker</h1>
      <p style="margin:0;color:#94a3b8;font-size:14px;">Weekly Report &mdash; {today}</p>
    </td>
  </tr>

  <tr>
    <td style="background:#1e293b;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="padding:16px;text-align:center;border-right:1px solid #334155;">
            <div style="color:#f8fafc;font-size:24px;font-weight:800;">{total_cats}</div>
            <div style="color:#64748b;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Categories</div>
          </td>
          <td style="padding:16px;text-align:center;border-right:1px solid #334155;">
            <div style="color:#f8fafc;font-size:24px;font-weight:800;">{total_etfs}</div>
            <div style="color:#64748b;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1px;">ETFs Ranked</div>
          </td>
          <td style="padding:16px;text-align:center;border-right:1px solid #334155;">
            <div style="color:#f8fafc;font-size:24px;font-weight:800;">${MIN_ADV_USD//1_000_000}M+</div>
            <div style="color:#64748b;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1px;">ADV Floor</div>
          </td>
          <td style="padding:16px;text-align:center;">
            <div style="color:#f8fafc;font-size:24px;font-weight:800;">Top {TOP_N}</div>
            <div style="color:#64748b;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Per Category</div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <tr>
    <td style="background:#f1f5f9;padding:28px 8px;">
      {sections}
    </td>
  </tr>

  <tr>
    <td style="background:#0f172a;padding:24px 32px;border-radius:0 0 12px 12px;text-align:center;">
      <p style="margin:0 0 4px;color:#475569;font-size:11px;">
        Generated by <strong style="color:#60a5fa;">Swarm Investments Quant Workflow</strong>
      </p>
      <p style="margin:0;color:#334155;font-size:10px;">
        Data: Yahoo Finance &middot; ADV = 30-day Avg Daily Dollar Volume
        &middot; Tickers below ${MIN_ADV_USD//1_000_000}M excluded
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
        raise FileNotFoundError(
            f"token.json not found at '{TOKEN_PATH}'. "
            "Run setup/setup_gmail_oauth.py first."
        )
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), GMAIL_SCOPES)
    if creds.expired and creds.refresh_token:
        log.info("Refreshing Gmail token...")
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
    log.info(f"Email sent → {to}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== ETF Liquidity Ranker — Weekly Run ===")

    df = fetch_etf_data(ETF_UNIVERSE)
    if df.empty:
        log.error("No data fetched. Aborting.")
        sys.exit(1)

    ranked = rank_by_category(df, ETF_UNIVERSE)
    log.info(f"Ranked across {len(ranked)} categories.")

    html    = build_email_html(ranked)
    today   = datetime.now().strftime("%B %d, %Y")
    service = get_gmail_service()
    send_email(service, RECIPIENT_EMAIL, f"ETF Liquidity Report — {today}", html)
    log.info("Done.")


if __name__ == "__main__":
    main()
