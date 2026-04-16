#!/usr/bin/env python3
"""
Swarm Quant Workflows — #06: Alternative Signals & Strategy Monitor
--------------------------------------------------------------------
Combines four alternative data lenses into one weekly email:
  1. HF Strategy ETF Scorecard   — yfinance (DBMF, CTA, ISMF, BTAL, HFGM, +)
  2. CFTC Speculative Positioning — CFTC.gov public zip files (no API key)
  3. Sentiment Dashboard         — FRED (AAII) + yfinance (VIX, VVIX, SKEW)
  4. Factor ETF Performance      — yfinance (MTUM, USMV, VLUE, QUAL, MOAT)

Run locally:
    FRED_API_KEY=xxx python 06_alt_signals/alt_signals.py
"""

import base64
import io
import logging
import os
import zipfile
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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
FRED_API_KEY    = os.environ.get("FRED_API_KEY", "")

# ── Universe ──────────────────────────────────────────────────────────────────

# HF strategy proxy ETFs (ticker, display label)
HF_ETFS = [
    ("DBMF", "Managed Futures (DBi)"),
    ("CTA",  "Managed Futures (Simplify)"),
    ("ISMF", "Managed Futures (ISMF)"),
    ("BTAL", "Market Neutral Anti-Beta"),
    ("HFGM", "HF Replication (GS)"),
    ("QAI",  "HF Multi-Strategy (IQ)"),
    ("MNA",  "Merger Arbitrage"),
    ("TAIL", "Tail Risk"),
]

# Smart-factor ETFs
FACTOR_ETFS = [
    ("MTUM", "Momentum"),
    ("USMV", "Min Volatility"),
    ("VLUE", "Value"),
    ("QUAL", "Quality"),
    ("MOAT", "Wide Moat"),
]

BENCHMARK = "SPY"

# Sentiment tickers from yfinance
SENTIMENT_TICKERS = ["^VIX", "^VVIX", "^SKEW"]

# CFTC base URL
CFTC_BASE = "https://www.cftc.gov/files/dea/history"

# COT contracts: (name_fragment, display_label, chart_group, report_type)
#   report_type "tff"    → Traders in Financial Futures, column = Lev_Money (hedge funds/CTAs)
#   report_type "disagg" → Disaggregated Futures,        column = M_Money   (money managers)
COT_CONTRACTS = [
    ("E-MINI S&P 500",         "S&P 500 E-mini",   "equity",    "tff"),
    ("NASDAQ-100 STOCK INDEX", "Nasdaq 100 E-mini", "equity",    "tff"),
    ("CBOE VIX",               "VIX Futures",       "equity",    "tff"),
    ("10-YEAR U.S. TREASURY",  "10Y T-Note",        "rates",     "tff"),
    ("GOLD - COMMODITY",       "Gold",              "commodity", "disagg"),
    ("CRUDE OIL, LIGHT SWEET", "WTI Crude",         "commodity", "disagg"),
]

# Long/short column names keyed by report type
COT_COLS = {
    "tff":    ("Lev_Money_Positions_Long_All",  "Lev_Money_Positions_Short_All"),
    "disagg": ("M_Money_Positions_Long_All",    "M_Money_Positions_Short_All"),
}

# Chart colour palette
CS = {
    "bg":     "#0f172a",
    "panel":  "#1e293b",
    "grid":   "#334155",
    "text":   "#94a3b8",
    "green":  "#22c55e",
    "red":    "#ef4444",
    "blue":   "#60a5fa",
    "amber":  "#f59e0b",
    "purple": "#a78bfa",
    "teal":   "#2dd4bf",
    "pink":   "#f472b6",
}

CONTRACT_COLORS = {
    "S&P 500 E-mini":   CS["blue"],
    "Nasdaq 100 E-mini":CS["purple"],
    "VIX Futures":      CS["pink"],
    "10Y T-Note":       CS["amber"],
    "Gold":             CS["amber"],
    "WTI Crude":        CS["teal"],
}


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_price_data(tickers: list[str], days: int = 400) -> pd.DataFrame:
    """Batch-download adjusted close prices for given tickers."""
    end   = datetime.today()
    start = end - timedelta(days=days)
    log.info(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t in raw.columns.get_level_values(0):
                    s = raw[t]["Close"].dropna()
                elif t in raw.columns.get_level_values(1):
                    s = raw.xs(t, level=1, axis=1)["Close"].dropna()
                else:
                    continue
            else:
                s = raw["Close"].dropna()
            if not s.empty:
                out[t] = s
        except Exception:
            pass
    df = pd.DataFrame(out)
    log.info(f"Got data for {len(df.columns)}/{len(tickers)} tickers")
    return df


def calc_returns(prices: pd.DataFrame, ticker: str) -> dict:
    """Compute 1W / 1M / 3M / YTD returns for a single ticker."""
    if ticker not in prices.columns:
        return {}
    s = prices[ticker].dropna()
    if s.empty:
        return {}
    p = float(s.iloc[-1])

    def ret(n: int) -> float | None:
        return (p / float(s.iloc[-n]) - 1) * 100 if len(s) > n else None

    jan1  = datetime(datetime.today().year, 1, 1)
    s_ytd = s[s.index >= jan1]
    ytd   = (p / float(s_ytd.iloc[0]) - 1) * 100 if not s_ytd.empty else None

    return {"price": p, "1w": ret(5), "1m": ret(21), "3m": ret(63), "ytd": ytd}


# ── CFTC COT data ─────────────────────────────────────────────────────────────

def _load_cot_zip(report_type: str, years: list[int]) -> pd.DataFrame | None:
    """
    Download a CFTC COT zip for the given report type and try each year in order.
    report_type: "tff"    → fut_fin_txt_{year}.zip    (Traders in Financial Futures)
                 "disagg" → fut_disagg_txt_{year}.zip (Disaggregated Futures)
    Returns the first successful DataFrame, or None.
    """
    prefix = {"tff": "fut_fin_txt", "disagg": "fut_disagg_txt"}[report_type]
    for year in years:
        url = f"{CFTC_BASE}/{prefix}_{year}.zip"
        try:
            log.info(f"Downloading CFTC {report_type} {year}: {url}")
            resp = requests.get(url, timeout=45)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                data_files = [n for n in z.namelist()
                              if n.lower().endswith((".csv", ".txt"))]
                if not data_files:
                    log.warning(f"No data file in {report_type} {year} zip")
                    continue
                with z.open(data_files[0]) as f:
                    df = pd.read_csv(f, encoding="latin-1", low_memory=False)
                    df.columns = [c.strip() for c in df.columns]
                    log.info(f"CFTC {report_type} {year}: {len(df)} rows, "
                             f"sample cols: {list(df.columns[:6])}")
                    return df
        except Exception as e:
            log.warning(f"CFTC {report_type} {year} failed: {e}")
    return None


def _parse_cot_df(
    df: pd.DataFrame,
    contracts: list[tuple],   # [(frag, label, group), ...]
    long_col: str,
    short_col: str,
) -> list[dict]:
    """Parse one CFTC report DataFrame into a list of row dicts."""

    def find(frags: list[str]) -> str | None:
        for f in frags:
            m = next((c for c in df.columns if f.lower() in c.lower()), None)
            if m:
                return m
        return None

    name_col = find(["Market_and_Exchange_Names", "Market and Exchange"])
    date_col = find(["Report_Date_as_MM_DD_YYYY", "As_of_Date_In_Form_YYMMDD",
                      "As of Date"])
    lc       = find([long_col])
    sc       = find([short_col])
    oi_col   = find(["Open_Interest_All", "Open Interest"])

    if not all([name_col, date_col, lc, sc]):
        log.warning(f"Missing columns. long='{long_col}' short='{short_col}' "
                    f"found cols: {list(df.columns[:15])}")
        return []

    # Parse dates — try MM/DD/YYYY then YYMMDD
    ds = df[date_col].astype(str).str.strip()
    df = df.copy()
    df["_date"] = pd.to_datetime(ds, format="%m/%d/%Y", errors="coerce")
    bad = df["_date"].isna()
    df.loc[bad, "_date"] = pd.to_datetime(
        ds[bad].str.zfill(6), format="%y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["_date"])

    rows: list[dict] = []
    for frag, label, group in contracts:
        mask   = df[name_col].str.upper().str.contains(frag.upper(), na=False)
        subset = (df[mask].sort_values("_date")
                          .drop_duplicates("_date", keep="last"))
        if subset.empty:
            log.warning(f"No COT rows for '{frag}'")
            continue
        log.info(f"  '{label}': {len(subset)} weekly rows")
        for _, row in subset.iterrows():
            try:
                longs  = float(row[lc])
                shorts = float(row[sc])
                oi     = float(row[oi_col]) if oi_col else np.nan
                net    = longs - shorts
                pct_l  = longs / (longs + shorts) * 100 if (longs + shorts) > 0 else 50.0
                rows.append({
                    "date":     row["_date"],
                    "market":   label,
                    "group":    group,
                    "longs":    longs,
                    "shorts":   shorts,
                    "net":      net,
                    "pct_long": pct_l,
                    "oi":       oi,
                })
            except Exception:
                pass
    return rows


def fetch_cot_data() -> pd.DataFrame:
    """
    Download two CFTC COT reports and parse speculative (hedge fund/CTA) positioning:
      - Traders in Financial Futures (TFF): S&P 500, Nasdaq, VIX, 10Y T-Note
        → uses Leveraged Funds columns (explicitly labeled hedge funds/CTAs)
      - Disaggregated Futures: Gold, WTI Crude
        → uses Money Manager columns (same hedge fund category for commodities)
    Returns tidy DataFrame: date, market, group, net, pct_long, oi
    """
    year  = datetime.today().year
    years = [year, year - 1]
    all_rows: list[dict] = []

    for report_type in ("tff", "disagg"):
        df = _load_cot_zip(report_type, years)
        if df is None:
            continue
        long_col, short_col = COT_COLS[report_type]
        contracts = [
            (frag, label, group)
            for frag, label, group, rt in COT_CONTRACTS
            if rt == report_type
        ]
        all_rows.extend(_parse_cot_df(df, contracts, long_col, short_col))

    if not all_rows:
        log.warning("No CFTC COT data parsed")
        return pd.DataFrame()

    out = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
    log.info(f"COT ready: {out['market'].nunique()} contracts, "
             f"latest {out['date'].max().date()}")
    return out


# ── FRED / AAII sentiment ─────────────────────────────────────────────────────

def fetch_aaii_sentiment() -> pd.DataFrame:
    """Fetch weekly AAII bull/bear survey data from FRED."""
    if not FRED_API_KEY:
        log.warning("No FRED_API_KEY — AAII sentiment skipped")
        return pd.DataFrame()

    cutoff = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    series = {"AAIIBULL": "Bull", "AAIIBEAR": "Bear"}
    cols: dict[str, pd.Series] = {}

    for sid, label in series.items():
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={sid}&api_key={FRED_API_KEY}&file_type=json"
            f"&observation_start={cutoff}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            s = pd.Series(
                {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
                name=label,
            )
            s.index = pd.to_datetime(s.index)
            cols[label] = s
            log.info(f"AAII {sid}: {len(s)} observations")
        except Exception as e:
            log.warning(f"FRED {sid} failed: {e}")

    if not cols:
        return pd.DataFrame()

    df = pd.DataFrame(cols).sort_index()
    if "Bull" in df and "Bear" in df:
        df["Spread"] = df["Bull"] - df["Bear"]
    return df


# ── Chart ─────────────────────────────────────────────────────────────────────

def _dark_ax(ax, title: str) -> None:
    ax.set_facecolor(CS["panel"])
    ax.set_title(title, color="#f1f5f9", fontsize=9, fontweight="bold", pad=8)
    ax.tick_params(colors=CS["text"], labelsize=7)
    ax.xaxis.label.set_color(CS["text"])
    ax.yaxis.label.set_color(CS["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(CS["grid"])
    ax.grid(True, color=CS["grid"], linewidth=0.5, alpha=0.7)
    ax.axhline(0, color=CS["text"], linewidth=0.8, alpha=0.4)


def _plot_cot_panel(ax, cot_df: pd.DataFrame, contracts: list[str], title: str) -> None:
    _dark_ax(ax, title)
    cutoff = datetime.today() - timedelta(weeks=52)
    plotted = False
    for contract in contracts:
        sub = cot_df[cot_df["market"] == contract]
        sub = sub[sub["date"] >= cutoff].sort_values("date")
        if sub.empty:
            continue
        net_k = sub["net"] / 1_000
        color = CONTRACT_COLORS.get(contract, CS["blue"])
        ax.plot(sub["date"], net_k, color=color, linewidth=1.5, label=contract)
        ax.fill_between(sub["date"], net_k, 0,
                        where=(net_k >= 0), alpha=0.12, color=CS["green"])
        ax.fill_between(sub["date"], net_k, 0,
                        where=(net_k < 0),  alpha=0.12, color=CS["red"])
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=CS["text"], fontsize=9)
        return
    ax.set_ylabel("Net Contracts (000s)", fontsize=7)
    ax.legend(fontsize=6, facecolor=CS["panel"], labelcolor=CS["text"], framealpha=0.8)


def build_chart(cot_df: pd.DataFrame, aaii_df: pd.DataFrame) -> str:
    """2×2 chart: equity/commodity/rates COT + AAII bull-bear spread."""
    fig = plt.figure(figsize=(14, 10), facecolor=CS["bg"])
    fig.suptitle(
        f"Alternative Signals Monitor — {datetime.now().strftime('%B %d, %Y')}",
        color="#f1f5f9", fontsize=13, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_eq  = fig.add_subplot(gs[0, 0])
    ax_cm  = fig.add_subplot(gs[0, 1])
    ax_rt  = fig.add_subplot(gs[1, 0])
    ax_aai = fig.add_subplot(gs[1, 1])

    if not cot_df.empty:
        _plot_cot_panel(ax_eq, cot_df,
                        ["S&P 500 E-mini", "Nasdaq 100 E-mini", "VIX Futures"],
                        "Equity & VIX Futures — Leveraged Fund Net Position")
        _plot_cot_panel(ax_cm, cot_df,
                        ["Gold", "WTI Crude"],
                        "Commodity Futures — Money Manager Net Position")
        _plot_cot_panel(ax_rt, cot_df,
                        ["10Y T-Note"],
                        "Rates Futures — Leveraged Fund Net Position")
    else:
        for ax, t in [(ax_eq, "Equity COT"), (ax_cm, "Commodity COT"), (ax_rt, "Rates COT")]:
            _dark_ax(ax, t)
            ax.text(0.5, 0.5, "COT data unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    color=CS["text"], fontsize=9)

    # AAII bull-bear spread bar chart
    _dark_ax(ax_aai, "AAII Investor Sentiment — Bull minus Bear (%)")
    if not aaii_df.empty and "Spread" in aaii_df.columns:
        cutoff = datetime.today() - timedelta(weeks=52)
        a = aaii_df[aaii_df.index >= cutoff].copy()
        if not a.empty:
            colors = [CS["green"] if v >= 0 else CS["red"] for v in a["Spread"]]
            ax_aai.bar(a.index, a["Spread"], color=colors, alpha=0.65, width=5)
            ma4 = a["Spread"].rolling(4).mean()
            ax_aai.plot(a.index, ma4, color=CS["amber"], linewidth=1.8,
                        label="4W MA", zorder=3)
            ax_aai.set_ylabel("Bull − Bear (%)", fontsize=7)
            ax_aai.legend(fontsize=6, facecolor=CS["panel"],
                          labelcolor=CS["text"], framealpha=0.8)
    else:
        ax_aai.text(0.5, 0.5, "No AAII data\n(FRED_API_KEY required)",
                    transform=ax_aai.transAxes, ha="center", va="center",
                    color=CS["text"], fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# ── Email HTML ────────────────────────────────────────────────────────────────

def _ret_cell(v: float | None) -> str:
    if v is None:
        return '<td style="padding:8px 10px;text-align:right;font-size:12px;color:#94a3b8;">—</td>'
    color = "#16a34a" if v >= 0 else "#dc2626"
    sign  = "+" if v >= 0 else ""
    return (
        f'<td style="padding:8px 10px;text-align:right;font-size:12px;'
        f'font-weight:700;color:{color};">{sign}{v:.1f}%</td>'
    )


def _etf_row(ticker: str, label: str, r: dict, highlight: bool = False) -> str:
    if not r:
        return ""
    bg    = "background:#f0f9ff;" if highlight else ""
    fw    = "800" if highlight else "600"
    price = f'${r["price"]:.2f}' if r.get("price") else "—"
    return f"""
      <tr style="border-bottom:1px solid #f1f5f9;{bg}">
        <td style="padding:9px 12px;font-size:12px;color:#0f172a;font-weight:{fw};">
          <span style="color:#64748b;font-size:11px;margin-right:6px;">{ticker}</span>{label}
        </td>
        <td style="padding:9px 10px;text-align:right;font-size:12px;color:#475569;">{price}</td>
        {_ret_cell(r.get("1w"))}{_ret_cell(r.get("1m"))}{_ret_cell(r.get("3m"))}{_ret_cell(r.get("ytd"))}
      </tr>"""


_TABLE_HEADER = """
      <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
        <th style="padding:8px 12px;text-align:left;font-size:10px;color:#94a3b8;font-weight:600;">ETF / Strategy</th>
        <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">Price</th>
        <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">1W</th>
        <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">1M</th>
        <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">3M</th>
        <th style="padding:8px 10px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">YTD</th>
      </tr>"""


def _fmt_k(v: float) -> str:
    """Format a contract count as ±NNNk or ±N.Nm."""
    sign = "+" if v >= 0 else ""
    if abs(v) >= 1_000_000:
        return f"{sign}{v/1_000_000:.2f}m"
    return f"{sign}{v/1_000:.1f}k"


def _cot_cards_html(cot_df: pd.DataFrame) -> str:
    """
    Render one detail card per COT contract showing:
      - Gross longs / shorts / net / week-over-week change
      - Open interest + net as % of OI
      - 52-week percentile rank of net position
      - 52-week range bar with current position marker
    """
    if cot_df.empty:
        return '<p style="padding:16px;color:#94a3b8;font-size:12px;">COT data unavailable</p>'

    cutoff_52w = cot_df["date"].max() - timedelta(weeks=52)
    cards = ""

    report_labels = {"tff": "TFF · Leveraged Funds", "disagg": "Disagg · Money Manager"}

    for frag, label, group, report_type in COT_CONTRACTS:
        sub = cot_df[cot_df["market"] == label].sort_values("date")
        if sub.empty:
            continue

        cur  = sub.tail(1).iloc[0]
        prev = sub.iloc[-2] if len(sub) >= 2 else None

        net    = float(cur["net"])
        longs  = float(cur["longs"])
        shorts = float(cur["shorts"])
        oi     = float(cur["oi"]) if not np.isnan(cur["oi"]) else None
        pct_l  = float(cur["pct_long"])

        # Week-over-week delta
        if prev is not None:
            delta   = net - float(prev["net"])
            dc      = "#16a34a" if delta >= 0 else "#dc2626"
            delta_s = _fmt_k(delta)
        else:
            delta, dc, delta_s = 0.0, "#94a3b8", "—"

        # Net as % of open interest
        net_oi_s = f"{net / oi * 100:+.1f}%" if oi and oi > 0 else "—"

        # 52-week percentile and range
        hist = sub[sub["date"] >= cutoff_52w]["net"]
        if len(hist) >= 4:
            pct_rank  = int((hist <= net).mean() * 100)
            low_52w   = float(hist.min())
            high_52w  = float(hist.max())
            rng       = high_52w - low_52w
            # Position of current value in the range (0–200 px bar)
            bar_pos   = int(max(0, min(200, (net - low_52w) / rng * 200)) if rng > 0 else 100)
            bar_left  = bar_pos
            bar_right = 200 - bar_pos
        else:
            pct_rank = low_52w = high_52w = None
            bar_left = bar_right = 100

        # Bias
        bias   = "LONG" if pct_l > 55 else ("SHORT" if pct_l < 45 else "NEUTRAL")
        bias_c = "#16a34a" if bias == "LONG" else ("#dc2626" if bias == "SHORT" else "#64748b")
        bias_bg = "#dcfce7" if bias == "LONG" else ("#fee2e2" if bias == "SHORT" else "#f1f5f9")

        # Percentile badge colour: extremes are informative (contrarian signals)
        if pct_rank is not None:
            if pct_rank >= 80:
                pct_c = "#dc2626"; pct_bg = "#fee2e2"  # extreme long = crowded
            elif pct_rank <= 20:
                pct_c = "#16a34a"; pct_bg = "#dcfce7"  # extreme short = potential squeeze
            else:
                pct_c = "#475569"; pct_bg = "#f1f5f9"
            pct_s = f"{pct_rank}th pct"
        else:
            pct_c = "#94a3b8"; pct_bg = "#f8fafc"; pct_s = "—"

        oi_s = _fmt_k(oi) if oi else "—"
        report_lbl = report_labels.get(report_type, "")
        as_of = cur["date"].strftime("%b %d")

        cards += f"""
    <div style="border:1px solid #e2e8f0;border-radius:8px;margin:8px 16px;overflow:hidden;">

      <!-- Card header -->
      <div style="background:#f8fafc;padding:10px 16px;
                  border-bottom:1px solid #e2e8f0;
                  display:table;width:100%;box-sizing:border-box;">
        <div style="display:table-cell;vertical-align:middle;">
          <span style="font-size:13px;font-weight:700;color:#0f172a;">{label}</span>
          <span style="font-size:10px;color:#94a3b8;margin-left:8px;">{report_lbl} &middot; as of {as_of}</span>
        </div>
        <div style="display:table-cell;text-align:right;vertical-align:middle;">
          <span style="background:{bias_bg};color:{bias_c};padding:3px 10px;
                       border-radius:4px;font-size:11px;font-weight:700;">{bias}</span>
          &nbsp;
          <span style="background:{pct_bg};color:{pct_c};padding:3px 10px;
                       border-radius:4px;font-size:11px;font-weight:700;">{pct_s}</span>
        </div>
      </div>

      <!-- Stats grid -->
      <div style="padding:12px 16px;">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td style="padding:4px 8px 4px 0;width:25%;">
              <div style="font-size:10px;color:#94a3b8;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.5px;">Net</div>
              <div style="font-size:14px;font-weight:800;
                          color:{'#16a34a' if net >= 0 else '#dc2626'};">{_fmt_k(net)}</div>
            </td>
            <td style="padding:4px 8px;width:25%;">
              <div style="font-size:10px;color:#94a3b8;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.5px;">Wk Change</div>
              <div style="font-size:14px;font-weight:800;color:{dc};">{delta_s}</div>
            </td>
            <td style="padding:4px 8px;width:25%;">
              <div style="font-size:10px;color:#94a3b8;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.5px;">Gross Long</div>
              <div style="font-size:14px;font-weight:700;color:#0f172a;">{_fmt_k(longs)}</div>
            </td>
            <td style="padding:4px 0 4px 8px;width:25%;">
              <div style="font-size:10px;color:#94a3b8;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.5px;">Gross Short</div>
              <div style="font-size:14px;font-weight:700;color:#0f172a;">{_fmt_k(shorts)}</div>
            </td>
          </tr>
          <tr>
            <td style="padding:8px 8px 4px 0;">
              <div style="font-size:10px;color:#94a3b8;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.5px;">Open Interest</div>
              <div style="font-size:12px;font-weight:600;color:#475569;">{oi_s}</div>
            </td>
            <td style="padding:8px 8px 4px 8px;">
              <div style="font-size:10px;color:#94a3b8;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.5px;">Net / OI</div>
              <div style="font-size:12px;font-weight:600;color:#475569;">{net_oi_s}</div>
            </td>
            <td colspan="2" style="padding:8px 0 4px 8px;">
              <div style="font-size:10px;color:#94a3b8;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.5px;">
                % Long of Gross (L+S)</div>
              <div style="font-size:12px;font-weight:600;color:#475569;">{pct_l:.1f}%</div>
            </td>
          </tr>
        </table>

        <!-- 52-week range bar -->
        {''.join([f"""
        <div style="margin-top:10px;">
          <div style="font-size:10px;color:#94a3b8;margin-bottom:4px;font-weight:600;">
            52-WEEK RANGE &nbsp;&nbsp;
            <span style="color:#64748b;">Low: {_fmt_k(low_52w)}</span>
            &nbsp;&nbsp;
            <span style="color:#64748b;">High: {_fmt_k(high_52w)}</span>
          </div>
          <table cellpadding="0" cellspacing="0" style="width:100%;">
            <tr>
              <td style="width:{bar_left/2}%;background:#e2e8f0;height:10px;
                         border-radius:5px 0 0 5px;"></td>
              <td style="width:2px;background:{bias_c};height:14px;
                         vertical-align:middle;"></td>
              <td style="background:#e2e8f0;height:10px;
                         border-radius:0 5px 5px 0;"></td>
            </tr>
          </table>
        </div>"""])
        if pct_rank is not None else ""}
      </div>
    </div>"""

    return cards


def build_email_html(
    hf_returns:     dict[str, dict],
    factor_returns: dict[str, dict],
    cot_df:         pd.DataFrame,
    aaii_df:        pd.DataFrame,
    sentiment:      dict,
    chart_b64:      str,
) -> str:
    today = datetime.now().strftime("%B %d, %Y")

    # ── HF rows ───────────────────────────────────────────────────────────────
    spy_r   = hf_returns.get("SPY", {})
    hf_rows = _etf_row("SPY", "S&P 500 (Benchmark)", spy_r, highlight=True)
    for ticker, label in HF_ETFS:
        hf_rows += _etf_row(ticker, label, hf_returns.get(ticker, {}))

    # ── Factor rows ───────────────────────────────────────────────────────────
    fct_rows = _etf_row("SPY", "S&P 500 (Benchmark)", spy_r, highlight=True)
    for ticker, label in FACTOR_ETFS:
        fct_rows += _etf_row(ticker, label, factor_returns.get(ticker, {}))

    # ── Sentiment pill values ─────────────────────────────────────────────────
    def fmt(key: str, decimals: int = 1) -> str:
        v = sentiment.get(key)
        return f"{v:.{decimals}f}" if isinstance(v, float) else "—"

    vix_s  = fmt("vix")
    vvix_s = fmt("vvix")
    skew_s = fmt("skew", 0)

    aaii_bull_s = aaii_bear_s = aaii_sprd_s = "—"
    aaii_sprd_c = "#94a3b8"
    if not aaii_df.empty:
        last = aaii_df.iloc[-1]
        aaii_bull_s = f'{last.get("Bull", 0):.1f}%'
        aaii_bear_s = f'{last.get("Bear", 0):.1f}%'
        sprd = last.get("Spread", 0)
        aaii_sprd_s = f'{"+" if sprd >= 0 else ""}{sprd:.1f}%'
        aaii_sprd_c = "#16a34a" if sprd > 0 else "#dc2626"

    # VVIX/VIX ratio note
    vix_v  = sentiment.get("vix")
    vvix_v = sentiment.get("vvix")
    vvix_note = ""
    if isinstance(vix_v, float) and isinstance(vvix_v, float) and vix_v > 0:
        if (vvix_v / vix_v) > 6:
            vvix_note = ' <span style="color:#ef4444;font-size:10px;font-weight:700;">ELEVATED</span>'

    cot_cards     = _cot_cards_html(cot_df)
    cot_date_str  = ""
    if not cot_df.empty:
        cot_date_str = f" — as of {cot_df['date'].max().strftime('%b %d, %Y')}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Alternative Signals Monitor — {today}</title>
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
      <h1 style="margin:0 0 8px;color:#fff;font-size:26px;font-weight:800;">
        Alternative Signals &amp; Strategy Monitor</h1>
      <p style="margin:0;color:#94a3b8;font-size:14px;">Weekly Report &mdash; {today}</p>
    </td>
  </tr>

  <!-- Sentiment pill bar -->
  <tr>
    <td style="background:#1e293b;padding:18px 32px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="text-align:center;padding:0 10px;">
            <div style="color:#94a3b8;font-size:10px;font-weight:600;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">VIX</div>
            <div style="color:#f1f5f9;font-size:22px;font-weight:800;">{vix_s}</div>
          </td>
          <td style="text-align:center;padding:0 10px;border-left:1px solid #334155;">
            <div style="color:#94a3b8;font-size:10px;font-weight:600;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
              VVIX{vvix_note}</div>
            <div style="color:#f1f5f9;font-size:22px;font-weight:800;">{vvix_s}</div>
          </td>
          <td style="text-align:center;padding:0 10px;border-left:1px solid #334155;">
            <div style="color:#94a3b8;font-size:10px;font-weight:600;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">SKEW</div>
            <div style="color:#f1f5f9;font-size:22px;font-weight:800;">{skew_s}</div>
          </td>
          <td style="text-align:center;padding:0 10px;border-left:1px solid #334155;">
            <div style="color:#94a3b8;font-size:10px;font-weight:600;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">AAII Bull</div>
            <div style="color:#22c55e;font-size:22px;font-weight:800;">{aaii_bull_s}</div>
          </td>
          <td style="text-align:center;padding:0 10px;border-left:1px solid #334155;">
            <div style="color:#94a3b8;font-size:10px;font-weight:600;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">AAII Bear</div>
            <div style="color:#ef4444;font-size:22px;font-weight:800;">{aaii_bear_s}</div>
          </td>
          <td style="text-align:center;padding:0 10px;border-left:1px solid #334155;">
            <div style="color:#94a3b8;font-size:10px;font-weight:600;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Bull−Bear</div>
            <div style="color:{aaii_sprd_c};font-size:22px;font-weight:800;">{aaii_sprd_s}</div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- HF Strategy Scorecard -->
  <tr>
    <td style="background:#fff;padding:0;border-top:1px solid #e2e8f0;">
      <div style="padding:16px 20px 8px;font-size:10px;font-weight:700;
          color:#64748b;text-transform:uppercase;letter-spacing:1px;">
        HF Strategy ETF Scorecard</div>
      <table width="100%" cellpadding="0" cellspacing="0">
        {_TABLE_HEADER}
        {hf_rows}
      </table>
    </td>
  </tr>

  <!-- CFTC Positioning -->
  <tr>
    <td style="background:#fff;padding:0;border-top:1px solid #e2e8f0;">
      <div style="padding:16px 20px 8px;font-size:10px;font-weight:700;
          color:#64748b;text-transform:uppercase;letter-spacing:1px;">
        CFTC Speculative Positioning{cot_date_str}</div>
      {cot_cards}
      <div style="padding:4px 24px 14px;font-size:10px;color:#94a3b8;">
        Leveraged Funds (TFF) = hedge funds &amp; CTAs &middot;
        Money Manager (Disagg) = hedge funds &amp; CTAs in commodity markets &middot;
        Percentile rank over trailing 52 weeks: &ge;80th = crowded long, &le;20th = crowded short.
      </div>
    </td>
  </tr>

  <!-- Factor Performance -->
  <tr>
    <td style="background:#fff;padding:0;border-top:1px solid #e2e8f0;">
      <div style="padding:16px 20px 8px;font-size:10px;font-weight:700;
          color:#64748b;text-transform:uppercase;letter-spacing:1px;">
        Smart Factor ETF Performance</div>
      <table width="100%" cellpadding="0" cellspacing="0">
        {_TABLE_HEADER}
        {fct_rows}
      </table>
    </td>
  </tr>

  <!-- Chart -->
  <tr>
    <td style="background:#fff;padding:20px;border-top:1px solid #e2e8f0;
        border-radius:0 0 12px 12px;">
      <div style="font-size:10px;font-weight:700;color:#64748b;
          text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
        CFTC Positioning History (52 Weeks) &amp; AAII Sentiment</div>
      <img src="data:image/png;base64,{chart_b64}"
           style="width:100%;max-width:720px;border-radius:8px;"
           alt="Positioning & Sentiment Charts">
    </td>
  </tr>

  <!-- Footer -->
  <tr>
    <td style="background:#0f172a;padding:20px 32px;border-radius:8px;
        margin-top:8px;text-align:center;">
      <p style="margin:0 0 4px;color:#475569;font-size:11px;">
        Generated by <strong style="color:#60a5fa;">Swarm Investments Quant Workflow</strong>
      </p>
      <p style="margin:0;color:#334155;font-size:10px;">
        Data: Yahoo Finance &middot; CFTC.gov (COT) &middot; FRED/AAII &middot; {today}
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
    log.info("=== Alternative Signals & Strategy Monitor — Weekly Run ===")

    # 1. Download all ETF price data in one batch
    all_tickers = (
        [t for t, _ in HF_ETFS]
        + [t for t, _ in FACTOR_ETFS]
        + [BENCHMARK]
        + SENTIMENT_TICKERS
    )
    prices = fetch_price_data(all_tickers)

    # 2. Compute returns
    hf_returns     = {t: calc_returns(prices, t) for t, _ in HF_ETFS}
    hf_returns["SPY"] = calc_returns(prices, "SPY")
    factor_returns = {t: calc_returns(prices, t) for t, _ in FACTOR_ETFS}

    # 3. Sentiment snapshot
    sentiment = {}
    for key, ticker in [("vix", "^VIX"), ("vvix", "^VVIX"), ("skew", "^SKEW")]:
        if ticker in prices.columns:
            s = prices[ticker].dropna()
            if not s.empty:
                sentiment[key] = float(s.iloc[-1])
    log.info(f"Sentiment: {sentiment}")

    # 4. CFTC COT positioning
    cot_df = fetch_cot_data()

    # 5. AAII sentiment (FRED)
    aaii_df = fetch_aaii_sentiment()

    # 6. Build chart
    log.info("Rendering charts...")
    chart_b64 = build_chart(cot_df, aaii_df)

    # 7. Build & send email
    html = build_email_html(
        hf_returns=hf_returns,
        factor_returns=factor_returns,
        cot_df=cot_df,
        aaii_df=aaii_df,
        sentiment=sentiment,
        chart_b64=chart_b64,
    )
    log.info("Sending email...")
    service = get_gmail_service()
    send_email(
        service,
        to=RECIPIENT_EMAIL,
        subject=f"Alt Signals & Strategy Monitor — {datetime.now().strftime('%b %d, %Y')}",
        html_body=html,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
