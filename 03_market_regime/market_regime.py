#!/usr/bin/env python3
"""
Swarm Quant Workflows — #03: Market Regime Classifier & Term Structure Monitor
------------------------------------------------------------------------------
Classifies the current market regime (BULL / CAUTION / RISK-OFF / BEAR) from
multiple signals, and generates term structure charts for:
  - US Treasury Yield Curve (full 13-point curve)
  - VIX Implied Volatility Term Structure
  - WTI Crude Futures Curve
  - Brent Crude Futures Curve

Each chart shows three snapshots: current, 1 month ago, 1 year ago.
Charts are embedded as a PNG image in the weekly HTML email.

Run locally:
    python 03_market_regime/market_regime.py
"""

import base64
import io
import logging
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# ── Futures month codes ───────────────────────────────────────────────────────
MONTH_CODES = {
    1:"F", 2:"G", 3:"H", 4:"J", 5:"K", 6:"M",
    7:"N", 8:"Q", 9:"U", 10:"V", 11:"X", 12:"Z",
}

# ── Treasury maturity labels ──────────────────────────────────────────────────
TREASURY_FIELDS = [
    ("BC_1MONTH",  "1M",   1/12),
    ("BC_2MONTH",  "2M",   2/12),
    ("BC_3MONTH",  "3M",   3/12),
    ("BC_6MONTH",  "6M",   6/12),
    ("BC_1YEAR",   "1Y",   1),
    ("BC_2YEAR",   "2Y",   2),
    ("BC_3YEAR",   "3Y",   3),
    ("BC_5YEAR",   "5Y",   5),
    ("BC_7YEAR",   "7Y",   7),
    ("BC_10YEAR",  "10Y",  10),
    ("BC_20YEAR",  "20Y",  20),
    ("BC_30YEAR",  "30Y",  30),
]

# ── Market tickers for regime signals ─────────────────────────────────────────
REGIME_TICKERS = ["SPY", "HYG", "IEF", "^VIX", "^VIX3M"]


# ── Treasury Yield Curve ──────────────────────────────────────────────────────

def _strip_ns(tag: str) -> str:
    """Strip XML namespace from a tag string."""
    return tag.split("}")[-1] if "}" in tag else tag


def fetch_treasury_curve(year: int, month: int) -> dict[str, float]:
    """
    Fetch the most recent daily treasury yield curve for the given year/month
    from the US Treasury's public XML API. Returns {field_name: yield_pct}.
    Uses namespace-stripped parsing for robustness against API changes.
    """
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/pages/xml?data=daily_treasury_yield_curve"
        f"&field_tdate_id={year}{month:02d}"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Treasury API failed for {year}-{month:02d}: {e}")
        return {}

    valid_fields = {f for f, _, _ in TREASURY_FIELDS}

    try:
        root    = ET.fromstring(resp.content)
        entries = []

        # Walk entire tree looking for 'properties' elements (namespace-agnostic)
        for elem in root.iter():
            if _strip_ns(elem.tag) != "properties":
                continue
            row: dict[str, float] = {}
            for child in elem:
                tag = _strip_ns(child.tag)
                text = (child.text or "").strip()
                if not text or text in (".", "null", ""):
                    continue
                if tag == "NEW_DATE":
                    row["_date"] = text[:10]
                elif tag in valid_fields:
                    try:
                        row[tag] = float(text)
                    except ValueError:
                        pass
            if len(row) > 2:          # date + at least 2 yield points
                entries.append(row)

        if not entries:
            log.warning(f"No treasury data parsed for {year}-{month:02d}")
            return {}

        entries.sort(key=lambda x: x.get("_date", ""), reverse=True)
        result = entries[0]
        log.info(f"Treasury {year}-{month:02d}: {len(result)-1} points, date={result.get('_date')}")
        return result

    except Exception as e:
        log.warning(f"Treasury XML parse error for {year}-{month:02d}: {e}")
        return {}


def get_treasury_curves() -> dict[str, dict]:
    """Fetch treasury curves for today, 1M ago, and 1Y ago."""
    now    = datetime.today()
    ago1m  = now - timedelta(days=32)
    ago1y  = now - timedelta(days=370)

    log.info("Fetching treasury yield curves...")
    return {
        "current": fetch_treasury_curve(now.year,   now.month),
        "1m_ago":  fetch_treasury_curve(ago1m.year, ago1m.month),
        "1y_ago":  fetch_treasury_curve(ago1y.year, ago1y.month),
    }


# ── Futures Term Structures ───────────────────────────────────────────────────

def fetch_futures_curve(prefix: str, ref_date: datetime) -> dict[str, dict]:
    """
    Build futures term structure curves (current, 1M ago, 1Y ago).
    Generates candidate symbols in two formats, batch-downloads all at once,
    and uses the first format that returns data for each month.
    """
    exchange = {"CL": "NYM", "BZ": "NYM"}.get(prefix, "")

    month_info  = []   # [(label, [sym_fmt1, sym_fmt2])]
    all_symbols = []

    for i in range(1, 13):
        month = ref_date.month + i
        year  = ref_date.year
        while month > 12:
            month -= 12
            year  += 1
        code  = MONTH_CODES[month]
        yr2   = str(year)[-2:]
        label = datetime(year, month, 1).strftime("%b %y")

        candidates = [
            f"{prefix}{code}{yr2}=F",          # CLK26=F
            f"{prefix}{code}{yr2}.{exchange}",  # CLK26.NYM
        ]
        month_info.append((label, candidates))
        all_symbols.extend(candidates)

    end_str   = ref_date.strftime("%Y-%m-%d")
    start_str = (ref_date - timedelta(days=400)).strftime("%Y-%m-%d")

    log.info(f"Batch-downloading {len(all_symbols)} {prefix} futures symbols...")
    try:
        raw = yf.download(
            all_symbols,
            start=start_str,
            end=end_str,
            group_by="ticker",
            auto_adjust=False,   # futures don't need adjustment
            progress=False,
            threads=True,
        )
    except Exception as e:
        log.error(f"{prefix} batch download failed: {e}")
        return {}

    # Build a price cache: symbol -> pd.Series of Close prices
    cache: dict[str, pd.Series] = {}
    for sym in all_symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if sym in raw.columns.get_level_values(0):
                    s = raw[sym]["Close"].dropna()
                elif sym in raw.columns.get_level_values(1):
                    s = raw.xs(sym, level=1, axis=1)["Close"].dropna()
                else:
                    continue
            else:
                s = raw["Close"].dropna()
            if not s.empty:
                cache[sym] = s
        except Exception:
            pass

    log.info(f"{prefix}: {len(cache)}/{len(all_symbols)} symbols returned data")

    t_now = ref_date
    t_1m  = ref_date - timedelta(days=21)
    t_1y  = ref_date - timedelta(days=252)

    def asof(s: pd.Series, dt: datetime) -> float | None:
        try:
            v = s.asof(dt)
            return float(v) if v == v else None
        except Exception:
            return None

    curves: dict[str, dict] = {"current": {}, "1m_ago": {}, "1y_ago": {}}
    for label, candidates in month_info:
        series = next((cache[s] for s in candidates if s in cache), None)
        if series is None:
            continue
        p_now = asof(series, t_now)
        p_1m  = asof(series, t_1m)
        p_1y  = asof(series, t_1y)
        if p_now: curves["current"][label] = p_now
        if p_1m:  curves["1m_ago"][label]  = p_1m
        if p_1y:  curves["1y_ago"][label]  = p_1y

    log.info(f"{prefix} curve — current:{len(curves['current'])} "
             f"1m:{len(curves['1m_ago'])} 1y:{len(curves['1y_ago'])}")
    return curves


# ── VIX Term Structure ────────────────────────────────────────────────────────

# Tenor labels and corresponding CBOE index symbols (shortest → longest)
_VIX_TENORS = [
    ("9D",   "VIX9D"),
    ("30D",  "VIX"),
    ("93D",  "VIX3M"),
    ("186D", "VIX6M"),
]


def get_vix_term_structure() -> tuple[dict[str, dict], pd.Series | None]:
    """
    Build VIX implied-volatility term structure using vix_utils, which pulls
    VIX9D / VIX / VIX3M / VIX6M directly from CBOE's CDN — more reliable
    than yfinance for these indices.

    Returns
    -------
    curves : dict  {current/1m_ago/1y_ago -> {tenor_label -> float}}
    vix3m  : pd.Series or None  — daily VIX3M close history, used to
             supplement ^VIX3M for the regime signal if yfinance missed it.
    """
    try:
        from vix_utils import get_vix_index_histories

        log.info("Downloading VIX index histories from CBOE via vix_utils...")
        df = get_vix_index_histories()
        df["Trade Date"] = pd.to_datetime(df["Trade Date"])

        want = {sym for _, sym in _VIX_TENORS}
        pivot = (
            df[df["Symbol"].isin(want)]
            .pivot_table(index="Trade Date", columns="Symbol", values="Close")
            .sort_index()
        )
        pivot.index = pivot.index.tz_localize(None)

        today = datetime.today()
        snapshots = {
            "current": today,
            "1m_ago":  today - timedelta(days=21),
            "1y_ago":  today - timedelta(days=252),
        }

        curves: dict[str, dict] = {"current": {}, "1m_ago": {}, "1y_ago": {}}
        for snap_key, snap_dt in snapshots.items():
            for label, sym in _VIX_TENORS:
                if sym not in pivot.columns:
                    continue
                val = pivot[sym].dropna().asof(snap_dt)
                if val is not None and not np.isnan(float(val)):
                    curves[snap_key][label] = float(val)

        log.info(f"VIX term structure (CBOE): {list(curves['current'].keys())}")

        vix3m = pivot["VIX3M"].dropna() if "VIX3M" in pivot.columns else None
        return curves, vix3m

    except Exception as e:
        log.warning(f"vix_utils failed ({e}) — VIX chart will be empty")
        return {"current": {}, "1m_ago": {}, "1y_ago": {}}, None


# ── Regime Signals ────────────────────────────────────────────────────────────

def calc_regime(prices: pd.DataFrame, treasury: dict) -> dict:
    """Derive regime signals and composite verdict."""
    signals = {}

    # Equity trend (SPY vs 50/200d MA)
    if "SPY" in prices.columns:
        spy = prices["SPY"].dropna()
        p   = float(spy.iloc[-1])
        ma50  = float(spy.tail(50).mean())  if len(spy) >= 50  else None
        ma200 = float(spy.tail(200).mean()) if len(spy) >= 200 else None
        if ma50 and ma200:
            if p > ma50 > ma200:   signals["equity"] = ("BULL",     3)
            elif p > ma50:         signals["equity"] = ("MIXED",    1)
            elif p < ma50 < ma200: signals["equity"] = ("BEAR",    -3)
            else:                  signals["equity"] = ("CAUTION", -1)
        signals["spy_price"] = p
        signals["spy_ma50"]  = ma50
        signals["spy_ma200"] = ma200

    # VIX level
    if "^VIX" in prices.columns:
        vix = float(prices["^VIX"].dropna().iloc[-1])
        signals["vix"] = vix
        if   vix < 15:  signals["vix_signal"] = ("LOW",      2)
        elif vix < 20:  signals["vix_signal"] = ("NORMAL",   1)
        elif vix < 30:  signals["vix_signal"] = ("ELEVATED", -1)
        else:           signals["vix_signal"] = ("HIGH",     -3)

    # VIX term structure (contango = calm, backwardation = stress)
    if "^VIX" in prices.columns and "^VIX3M" in prices.columns:
        vix    = float(prices["^VIX"].dropna().iloc[-1])
        vix3m  = float(prices["^VIX3M"].dropna().iloc[-1])
        spread = vix3m - vix
        signals["vix_spread"] = spread
        if   spread > 3:   signals["vix_ts"] = ("CONTANGO",       2)
        elif spread > 0:   signals["vix_ts"] = ("MILD CONTANGO",  1)
        elif spread > -3:  signals["vix_ts"] = ("FLAT",          -1)
        else:              signals["vix_ts"] = ("BACKWARDATION",  -3)

    # Yield curve (10Y - 2Y spread)
    cur = treasury.get("current", {})
    y10 = cur.get("BC_10YEAR")
    y2  = cur.get("BC_2YEAR")
    y3m = cur.get("BC_3MONTH")
    if y10 and y2:
        spread_10_2 = y10 - y2
        signals["yield_spread_10_2"] = spread_10_2
        if   spread_10_2 > 1.0:  signals["yield_curve"] = ("STEEP",    2)
        elif spread_10_2 > 0:    signals["yield_curve"] = ("NORMAL",   1)
        elif spread_10_2 > -0.5: signals["yield_curve"] = ("FLAT",    -1)
        else:                    signals["yield_curve"] = ("INVERTED", -3)

    # Credit risk appetite (HYG vs IEF)
    if "HYG" in prices.columns and "IEF" in prices.columns:
        hyg_ret = float(prices["HYG"].dropna().pct_change(21).iloc[-1] * 100)
        ief_ret = float(prices["IEF"].dropna().pct_change(21).iloc[-1] * 100)
        rel     = hyg_ret - ief_ret
        signals["credit_rel"] = rel
        if   rel > 1:   signals["credit"] = ("RISK-ON",   2)
        elif rel > -1:  signals["credit"] = ("NEUTRAL",   0)
        else:           signals["credit"] = ("RISK-OFF",  -2)

    # Composite score
    score_keys = ["equity", "vix_signal", "vix_ts", "yield_curve", "credit"]
    scores = [signals[k][1] for k in score_keys if k in signals]
    avg = sum(scores) / len(scores) if scores else 0
    signals["score"] = avg

    if   avg >= 1.5:   signals["regime"] = "BULL"
    elif avg >= 0.0:   signals["regime"] = "CAUTION"
    elif avg >= -1.5:  signals["regime"] = "RISK-OFF"
    else:              signals["regime"] = "BEAR"

    return signals


# ── Charts ────────────────────────────────────────────────────────────────────

CHART_STYLE = {
    "bg":        "#0f172a",
    "panel":     "#1e293b",
    "grid":      "#334155",
    "text":      "#94a3b8",
    "current":   "#60a5fa",
    "one_month": "#f59e0b",
    "one_year":  "#64748b",
}

def _set_dark_axes(ax, title: str) -> None:
    ax.set_facecolor(CHART_STYLE["panel"])
    ax.tick_params(colors=CHART_STYLE["text"], labelsize=8)
    ax.xaxis.label.set_color(CHART_STYLE["text"])
    ax.yaxis.label.set_color(CHART_STYLE["text"])
    ax.set_title(title, color="#f1f5f9", fontsize=10, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_STYLE["grid"])
    ax.grid(True, color=CHART_STYLE["grid"], linewidth=0.5, alpha=0.7)


def plot_treasury(ax, curves: dict[str, dict]) -> None:
    maturities = [(f, lbl, mat) for f, lbl, mat in TREASURY_FIELDS]
    for key, color, lbl, ls in [
        ("current", CHART_STYLE["current"],   "Current",   "-"),
        ("1m_ago",  CHART_STYLE["one_month"], "1M Ago",    "--"),
        ("1y_ago",  CHART_STYLE["one_year"],  "1Y Ago",    ":"),
    ]:
        curve = curves.get(key, {})
        xs = [mat  for f, _l, mat in maturities if f in curve]
        ys = [curve[f] for f, _l, mat in maturities if f in curve]
        xlabels = [lbl for f, lbl, mat in maturities if f in curve]
        if xs:
            ax.plot(range(len(xs)), ys, color=color, linestyle=ls,
                    linewidth=2, marker="o", markersize=4, label=lbl)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)

    _set_dark_axes(ax, "US Treasury Yield Curve")
    ax.set_ylabel("Yield (%)", fontsize=8)
    ax.legend(fontsize=7, facecolor=CHART_STYLE["panel"],
              labelcolor=CHART_STYLE["text"], framealpha=0.8)


def plot_vix(ax, curves: dict[str, dict]) -> None:
    labels_order = [lbl for lbl, _ in _VIX_TENORS]  # ["9D", "30D", "93D", "186D"]
    for key, color, lbl, ls in [
        ("current", CHART_STYLE["current"],   "Current", "-"),
        ("1m_ago",  CHART_STYLE["one_month"], "1M Ago",  "--"),
        ("1y_ago",  CHART_STYLE["one_year"],  "1Y Ago",  ":"),
    ]:
        curve = curves.get(key, {})
        xs  = [i for i, l in enumerate(labels_order) if l in curve]
        ys  = [curve[l] for l in labels_order if l in curve]
        xls = [l for l in labels_order if l in curve]
        if xs:
            ax.plot(xs, ys, color=color, linestyle=ls,
                    linewidth=2, marker="o", markersize=6, label=lbl)
            ax.set_xticks(xs)
            ax.set_xticklabels(xls, fontsize=8)

    _set_dark_axes(ax, "VIX Implied Volatility Term Structure")
    ax.set_ylabel("VIX Level", fontsize=8)
    ax.legend(fontsize=7, facecolor=CHART_STYLE["panel"],
              labelcolor=CHART_STYLE["text"], framealpha=0.8)


def plot_futures(ax, curves: dict[str, dict], title: str, ylabel: str) -> None:
    # Use current curve labels as the x-axis reference
    cur_labels = list(curves.get("current", {}).keys())
    if not cur_labels:
        _set_dark_axes(ax, title)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=CHART_STYLE["text"])
        return

    for key, color, lbl, ls in [
        ("current", CHART_STYLE["current"],   "Current", "-"),
        ("1m_ago",  CHART_STYLE["one_month"], "1M Ago",  "--"),
        ("1y_ago",  CHART_STYLE["one_year"],  "1Y Ago",  ":"),
    ]:
        curve = curves.get(key, {})
        ys = []
        xs = []
        for i, label in enumerate(cur_labels):
            if label in curve:
                xs.append(i)
                ys.append(curve[label])
        if xs:
            ax.plot(xs, ys, color=color, linestyle=ls,
                    linewidth=2, marker="o", markersize=3, label=lbl)

    ax.set_xticks(range(len(cur_labels)))
    ax.set_xticklabels(cur_labels, rotation=45, ha="right", fontsize=7)
    _set_dark_axes(ax, title)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, facecolor=CHART_STYLE["panel"],
              labelcolor=CHART_STYLE["text"], framealpha=0.8)


def build_chart(
    treasury:  dict[str, dict],
    vix_curves: dict[str, dict],
    wti_curves: dict[str, dict],
    brent_curves: dict[str, dict],
) -> str:
    """Render 2x2 term structure chart, return as base64 PNG string."""
    fig = plt.figure(figsize=(14, 10), facecolor=CHART_STYLE["bg"])
    fig.suptitle(
        f"Term Structure Monitor — {datetime.now().strftime('%B %d, %Y')}",
        color="#f1f5f9", fontsize=13, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_tsy   = fig.add_subplot(gs[0, 0])
    ax_vix   = fig.add_subplot(gs[0, 1])
    ax_wti   = fig.add_subplot(gs[1, 0])
    ax_brent = fig.add_subplot(gs[1, 1])

    plot_treasury(ax_tsy,   treasury)
    plot_vix(ax_vix,        vix_curves)
    plot_futures(ax_wti,    wti_curves,   "WTI Crude Futures Curve",   "Price (USD/bbl)")
    plot_futures(ax_brent,  brent_curves, "Brent Crude Futures Curve", "Price (USD/bbl)")

    # Legend note
    fig.text(0.5, 0.01,
             "Blue = Current  |  Amber = 1 Month Ago  |  Gray = 1 Year Ago",
             ha="center", fontsize=8, color=CHART_STYLE["text"])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# ── Email HTML ────────────────────────────────────────────────────────────────

REGIME_STYLES = {
    "BULL":     ("#15803d", "#dcfce7"),
    "CAUTION":  ("#92400e", "#fef9c3"),
    "RISK-OFF": ("#c2410c", "#fff7ed"),
    "BEAR":     ("#991b1b", "#fee2e2"),
}

def _sig_badge(label: str, good: bool) -> str:
    bg  = "#dcfce7" if good else "#fee2e2"
    tc  = "#15803d" if good else "#991b1b"
    return (
        f'<span style="background:{bg};color:{tc};padding:3px 8px;'
        f'border-radius:4px;font-size:11px;font-weight:700;">{label}</span>'
    )


def build_email_html(signals: dict, chart_b64: str, treasury: dict) -> str:
    today   = datetime.now().strftime("%B %d, %Y")
    regime  = signals.get("regime", "—")
    score   = signals.get("score", 0)
    tc, bgc = REGIME_STYLES.get(regime, ("#1e3a5f", "#dbeafe"))

    def row(label: str, value: str, badge: str) -> str:
        return f"""
          <tr style="border-bottom:1px solid #f1f5f9;">
            <td style="padding:10px 14px;font-size:12px;color:#64748b;font-weight:600;">
              {label}</td>
            <td style="padding:10px 14px;font-size:13px;color:#0f172a;font-weight:700;">
              {value}</td>
            <td style="padding:10px 14px;">{badge}</td>
          </tr>"""

    # Build signal rows
    eq   = signals.get("equity",      ("—", 0))
    vs   = signals.get("vix_signal",  ("—", 0))
    vts  = signals.get("vix_ts",      ("—", 0))
    yc   = signals.get("yield_curve", ("—", 0))
    cr   = signals.get("credit",      ("—", 0))

    cur  = treasury.get("current", {})
    y10  = cur.get("BC_10YEAR", 0)
    y2   = cur.get("BC_2YEAR",  0)
    y3m  = cur.get("BC_3MONTH", 0)
    spr  = signals.get("yield_spread_10_2", 0)

    vix_val   = signals.get("vix",    0)
    vix_spr   = signals.get("vix_spread", 0)
    credit_rel = signals.get("credit_rel", 0)

    rows_html = ""
    rows_html += row("Equity Trend (SPY)", f"${signals.get('spy_price',0):.2f}",
                     _sig_badge(eq[0], eq[1] > 0))
    rows_html += row("VIX Level", f"{vix_val:.1f}",
                     _sig_badge(vs[0], vs[1] > 0))
    rows_html += row("VIX Term Structure", f"Spread: {vix_spr:+.1f}",
                     _sig_badge(vts[0], vts[1] > 0))
    rows_html += row("Yield Curve (10Y-2Y)", f"{spr:+.2f}% ({y2:.2f}% → {y10:.2f}%)",
                     _sig_badge(yc[0], yc[1] > 0))
    rows_html += row("Credit Risk Appetite", f"HYG vs IEF: {credit_rel:+.2f}%",
                     _sig_badge(cr[0], cr[1] >= 0))

    # Treasury key rates table
    tsy_rows = ""
    key_rates = [
        ("BC_3MONTH", "3M T-Bill"), ("BC_2YEAR",  "2Y T-Note"),
        ("BC_5YEAR",  "5Y T-Note"), ("BC_10YEAR", "10Y T-Note"),
        ("BC_30YEAR", "30Y T-Bond"),
    ]
    cur_tsy  = treasury.get("current", {})
    ago1m    = treasury.get("1m_ago",  {})
    ago1y    = treasury.get("1y_ago",  {})

    for field, label in key_rates:
        c  = cur_tsy.get(field, 0)
        m  = ago1m.get(field,   0)
        y  = ago1y.get(field,   0)
        dc = c - m  # change vs 1M ago
        dc_color = "#16a34a" if dc < 0 else "#dc2626"  # rates falling = good for bonds
        sign = "+" if dc >= 0 else ""
        tsy_rows += f"""
          <tr style="border-bottom:1px solid #f1f5f9;">
            <td style="padding:9px 12px;font-size:12px;color:#0f172a;font-weight:600;">{label}</td>
            <td style="padding:9px 12px;text-align:right;font-size:13px;font-weight:700;color:#0f172a;">{c:.2f}%</td>
            <td style="padding:9px 12px;text-align:right;font-size:12px;color:{dc_color};font-weight:600;">{sign}{dc:.2f}%</td>
            <td style="padding:9px 12px;text-align:right;font-size:12px;color:#64748b;">{m:.2f}%</td>
            <td style="padding:9px 12px;text-align:right;font-size:12px;color:#94a3b8;">{y:.2f}%</td>
          </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Market Regime — {today}</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;">
<tr><td align="center" style="padding:32px 16px;">
<table width="720" cellpadding="0" cellspacing="0">

  <!-- Header -->
  <tr>
    <td style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
        padding:36px 40px;border-radius:12px 12px 0 0;text-align:center;">
      <p style="margin:0 0 6px;color:#60a5fa;font-size:11px;font-weight:700;
          letter-spacing:2px;text-transform:uppercase;">Swarm Investments</p>
      <h1 style="margin:0 0 8px;color:#fff;font-size:28px;font-weight:800;">
        Market Regime Classifier</h1>
      <p style="margin:0;color:#94a3b8;font-size:14px;">Weekly Report &mdash; {today}</p>
    </td>
  </tr>

  <!-- Regime verdict -->
  <tr>
    <td style="background:{bgc};padding:24px 40px;text-align:center;
        border-bottom:3px solid {tc};">
      <div style="color:{tc};font-size:11px;font-weight:700;
          letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
        Current Market Regime
      </div>
      <div style="color:{tc};font-size:48px;font-weight:900;
          letter-spacing:-1px;line-height:1;">{regime}</div>
      <div style="color:{tc};font-size:13px;margin-top:6px;opacity:0.8;">
        Composite Score: {score:+.2f}
        &nbsp;&bull;&nbsp; Scale: -3 (Bear) to +3 (Bull)
      </div>
    </td>
  </tr>

  <!-- Signal table -->
  <tr>
    <td style="background:#fff;padding:0;">
      <div style="padding:16px 20px 8px;font-size:10px;font-weight:700;
          color:#64748b;text-transform:uppercase;letter-spacing:1px;">
        Regime Signals
      </div>
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
          <th style="padding:8px 14px;text-align:left;font-size:10px;color:#94a3b8;font-weight:600;">Signal</th>
          <th style="padding:8px 14px;text-align:left;font-size:10px;color:#94a3b8;font-weight:600;">Value</th>
          <th style="padding:8px 14px;text-align:left;font-size:10px;color:#94a3b8;font-weight:600;">Status</th>
        </tr>
        {rows_html}
      </table>
    </td>
  </tr>

  <!-- Treasury rates table -->
  <tr>
    <td style="background:#fff;padding:0;border-top:1px solid #e2e8f0;">
      <div style="padding:16px 20px 8px;font-size:10px;font-weight:700;
          color:#64748b;text-transform:uppercase;letter-spacing:1px;">
        US Treasury Key Rates
      </div>
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
          <th style="padding:8px 12px;text-align:left;font-size:10px;color:#94a3b8;font-weight:600;">Tenor</th>
          <th style="padding:8px 12px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">Current</th>
          <th style="padding:8px 12px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">vs 1M Ago</th>
          <th style="padding:8px 12px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">1M Ago</th>
          <th style="padding:8px 12px;text-align:right;font-size:10px;color:#94a3b8;font-weight:600;">1Y Ago</th>
        </tr>
        {tsy_rows}
      </table>
    </td>
  </tr>

  <!-- Term structure chart -->
  <tr>
    <td style="background:#fff;padding:20px;border-top:1px solid #e2e8f0;
        border-radius:0 0 12px 12px;">
      <div style="font-size:10px;font-weight:700;color:#64748b;
          text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
        Term Structure Charts — Current / 1M Ago / 1Y Ago
      </div>
      <img src="data:image/png;base64,{chart_b64}"
           style="width:100%;max-width:680px;border-radius:8px;"
           alt="Term Structure Charts">
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
        Data: Yahoo Finance, CBOE (vix_utils), US Treasury &middot; {today}
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
    log.info("=== Market Regime Classifier -- Weekly Run ===")
    today = datetime.today()

    # 1. Treasury yield curves
    treasury = get_treasury_curves()
    log.info(f"Treasury current: {len(treasury.get('current', {}))} points")

    # 2. Market prices (SPY, VIX, HYG, IEF)
    log.info("Downloading regime signal data...")
    end   = today
    start = today - timedelta(days=400)
    raw_prices = yf.download(
        REGIME_TICKERS,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    prices = {}
    for t in REGIME_TICKERS:
        try:
            if isinstance(raw_prices.columns, pd.MultiIndex):
                if t in raw_prices.columns.get_level_values(0):
                    prices[t] = raw_prices[t]["Close"].dropna()
                else:
                    prices[t] = raw_prices.xs(t, level=1, axis=1)["Close"].dropna()
            else:
                prices[t] = raw_prices["Close"].dropna()
        except Exception:
            pass
    prices_df = pd.DataFrame(prices)

    # 3. VIX term structure (CBOE via vix_utils)
    #    Also returns VIX3M history used to patch prices_df if yfinance missed it
    vix_curves, cboe_vix3m = get_vix_term_structure()

    # Supplement ^VIX3M in prices_df from CBOE data if yfinance didn't return it
    if cboe_vix3m is not None and (
        "^VIX3M" not in prices_df.columns or prices_df["^VIX3M"].dropna().empty
    ):
        try:
            prices_df["^VIX3M"] = cboe_vix3m.reindex(prices_df.index, method="ffill")
            log.info("^VIX3M supplemented from CBOE data for regime signal")
        except Exception as e:
            log.warning(f"Could not reindex CBOE VIX3M: {e}")

    # 4. Regime signals
    signals = calc_regime(prices_df, treasury)
    log.info(f"Regime: {signals.get('regime')} (score: {signals.get('score', 0):+.2f})")

    # 5. Crude futures curves
    log.info("Fetching WTI futures curve...")
    wti_curves   = fetch_futures_curve("CL", today)

    log.info("Fetching Brent futures curve...")
    brent_curves = fetch_futures_curve("BZ", today)

    # 6. Build chart
    log.info("Rendering term structure charts...")
    chart_b64 = build_chart(treasury, vix_curves, wti_curves, brent_curves)

    # 7. Build and send email
    html    = build_email_html(signals, chart_b64, treasury)
    subject = f"Market Regime: {signals.get('regime','?')} -- {today.strftime('%B %d, %Y')}"
    service = get_gmail_service()
    send_email(service, RECIPIENT_EMAIL, subject, html)
    log.info("Done.")


if __name__ == "__main__":
    main()
