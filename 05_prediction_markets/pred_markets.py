#!/usr/bin/env python3
"""
Workflow #05 — Prediction Market & Macro Probability Monitor
Weekly snapshot of market-implied probabilities for key macro events.
Sources: Kalshi (FOMC, CPI, PCE) · Polymarket (recession, macro)
"""

import base64
import io
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import requests

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")
TOKEN_PATH      = Path(os.environ.get("GMAIL_TOKEN_PATH", "token.json"))
GMAIL_SCOPES    = ["https://www.googleapis.com/auth/gmail.send"]

HEADERS = {"User-Agent": "swarm-quant-workflows/1.0 (research)"}

# ── Kalshi ────────────────────────────────────────────────────────────────────
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Additional series beyond FOMC to surface in diagnostics / future email sections
KALSHI_EXTRA_SERIES = ["KXCPI", "KXPCEINFLATION", "KXUNEMPLOYMENT", "KXGDP"]

# ── Polymarket ────────────────────────────────────────────────────────────────
POLYMARKET_BASE = "https://gamma-api.polymarket.com"

MACRO_KEYWORDS = {
    "recession", "federal reserve", "fed rate", "rate cut", "rate hike",
    "inflation", "cpi", "unemployment", "gdp", "tariff", "debt ceiling",
    "treasury", "fomc", "yield curve", "default", "interest rate",
}

# ── Colors ────────────────────────────────────────────────────────────────────
CS = {
    "bg":    "#0f172a",
    "panel": "#1e293b",
    "blue":  "#3b82f6",
    "green": "#22c55e",
    "red":   "#ef4444",
    "amber": "#f59e0b",
    "text":  "#e2e8f0",
    "muted": "#94a3b8",
}


# ── Kalshi helpers ─────────────────────────────────────────────────────────────

def _market_prob(m: dict) -> float:
    """Midpoint YES probability from a Kalshi market (returns 0–1)."""
    try:
        bid = float(m.get("yes_bid_dollars") or 0)
        ask_raw = m.get("yes_ask_dollars")
        ask = float(ask_raw) if ask_raw else bid
        return max(0.0, min(1.0, (bid + ask) / 2))
    except (TypeError, ValueError):
        return 0.0


def fetch_kalshi_markets(series_ticker: str) -> list[dict]:
    """Fetch all open markets for a Kalshi series ticker."""
    url = f"{KALSHI_BASE}/markets"
    params = {"series_ticker": series_ticker, "status": "open", "limit": 200}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        markets = r.json().get("markets", [])
        log.info(f"Kalshi {series_ticker}: {len(markets)} open markets")
        for m in markets[:2]:
            log.info(f"  {m.get('ticker')} | {m.get('title')} | "
                     f"bid={m.get('yes_bid_dollars')} ask={m.get('yes_ask_dollars')}")
        return markets
    except Exception as exc:
        log.warning(f"Kalshi {series_ticker} fetch failed: {exc}")
        return []


def _extract_rate_level(title: str) -> float | None:
    """Parse a Fed funds rate level (%) from a Kalshi market title."""
    # "4.25-4.50%" → midpoint 4.375
    m = re.search(r'(\d+\.\d+)\s*[-–]\s*(\d+\.\d+)\s*%', title)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    # "4.50%" standalone
    m = re.search(r'(\d+\.\d+)\s*%', title)
    if m:
        return float(m.group(1))
    return None


def build_fomc_events(markets: list[dict]) -> list[dict]:
    """
    Group KXFED markets by event_ticker, build probability distributions,
    compute the implied (probability-weighted) rate for each meeting.
    Returns list sorted by earliest close time, capped at next 4 meetings.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        grouped[m.get("event_ticker") or "unknown"].append(m)

    def _first_close(ev_markets):
        ts_list = [m.get("close_time", 0) for m in ev_markets if m.get("close_time")]
        return min(ts_list) if ts_list else 0

    sorted_events = sorted(grouped.items(), key=lambda kv: _first_close(kv[1]))

    result = []
    for event, ev_markets in sorted_events[:4]:
        dist = []
        for m in ev_markets:
            prob  = _market_prob(m)
            rate  = _extract_rate_level(m.get("title", ""))
            # Shorten title for display: strip trailing "after X meeting"
            short = re.sub(r'(?i)\s*after .+? meeting\.?', '', m.get("title", "")).strip()
            dist.append({"title": short, "rate": rate, "prob": prob,
                         "ticker": m.get("ticker", "")})
        dist.sort(key=lambda d: d["rate"] or 0)

        rated = [(d["rate"], d["prob"]) for d in dist if d["rate"] is not None]
        expected = (
            sum(r * p for r, p in rated) / sum(p for _, p in rated)
            if rated and sum(p for _, p in rated) > 0 else None
        )

        close_ts = _first_close(ev_markets)
        close_date = None
        if close_ts:
            try:
                close_date = datetime.fromtimestamp(int(str(close_ts)[:10]))
            except Exception:
                pass

        result.append({
            "event":        event,
            "close_date":   close_date,
            "distribution": dist,
            "expected_rate": expected,
        })

    return result


# ── Polymarket helpers ────────────────────────────────────────────────────────

def fetch_polymarket_macro(max_fetch: int = 300) -> list[dict]:
    """
    Fetch top Polymarket markets by 24h volume and filter for macro keywords.
    Returns up to 20 markets sorted by 24h volume descending.
    """
    url = f"{POLYMARKET_BASE}/markets"
    params = {
        "limit":  max_fetch,
        "offset": 0,
        "order":  "volume_24hr",
        "closed": "false",
        "active": "true",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        raw = r.json()
        log.info(f"Polymarket: fetched {len(raw)} markets")
    except Exception as exc:
        log.warning(f"Polymarket fetch failed: {exc}")
        return []

    macro = []
    for m in raw:
        title = (m.get("question") or "").lower()
        if not any(kw in title for kw in MACRO_KEYWORDS):
            continue
        try:
            prices   = json.loads(m.get("outcomePrices") or "[]")
            yes_prob = float(prices[0]) if prices else 0.0
        except Exception:
            yes_prob = 0.0
        macro.append({
            "question":   m.get("question", ""),
            "yes_prob":   yes_prob,
            "volume":     float(m.get("volume") or 0),
            "volume_24h": float(m.get("volume_24hr") or 0),
            "end_date":   m.get("endDate"),
        })

    macro.sort(key=lambda x: x["volume_24h"], reverse=True)
    log.info(f"Polymarket: {len(macro)} macro markets after keyword filter")
    return macro[:20]


# ── Chart ─────────────────────────────────────────────────────────────────────

def build_chart(fomc_events: list[dict], poly_markets: list[dict]) -> str:
    """
    Left panel:  FOMC rate probability distribution for the nearest meeting.
    Right panel: Top Polymarket macro markets as horizontal probability bars.
    """
    fig = plt.figure(figsize=(14, 8), facecolor=CS["bg"])

    # ── Left: FOMC distribution ───────────────────────────────────────────────
    ax_fomc = fig.add_subplot(1, 2, 1)
    ax_fomc.set_facecolor(CS["panel"])

    if fomc_events:
        ev   = fomc_events[0]
        dist = [d for d in ev["distribution"] if d["rate"] is not None]

        if dist:
            labels = [f"{d['rate']:.2f}%" for d in dist]
            probs  = [d["prob"] * 100 for d in dist]
            colors = [
                CS["green"] if p >= 50 else CS["amber"] if p >= 20 else CS["muted"]
                for p in probs
            ]

            x    = np.arange(len(labels))
            bars = ax_fomc.bar(x, probs, color=colors, alpha=0.88,
                               width=0.6, zorder=3)
            ax_fomc.set_xticks(x)
            ax_fomc.set_xticklabels(labels, rotation=45, ha="right",
                                     color=CS["text"], fontsize=8)
            ax_fomc.set_ylabel("Probability (%)", color=CS["text"], fontsize=9)
            ax_fomc.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
            ax_fomc.tick_params(colors=CS["text"])
            ax_fomc.grid(axis="y", color=CS["bg"], alpha=0.6, zorder=0)
            for spine in ax_fomc.spines.values():
                spine.set_edgecolor(CS["muted"])

            for bar, prob in zip(bars, probs):
                if prob > 3:
                    ax_fomc.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{prob:.0f}%",
                        ha="center", va="bottom",
                        color=CS["text"], fontsize=8, fontweight="bold",
                    )

            meeting = ev["close_date"].strftime("%b %d, %Y") if ev.get("close_date") else ""
            exp     = ev.get("expected_rate")
            exp_str = f"  ·  Implied: {exp:.2f}%" if exp else ""
            ax_fomc.set_title(
                f"FOMC Rate Distribution\n{meeting}{exp_str}",
                color=CS["text"], fontsize=10, pad=10,
            )
        else:
            ax_fomc.text(0.5, 0.5, "Rate levels not parsed from titles",
                         transform=ax_fomc.transAxes, ha="center", va="center",
                         color=CS["muted"], fontsize=9)
            ax_fomc.axis("off")
    else:
        ax_fomc.text(0.5, 0.5, "No FOMC data available",
                     transform=ax_fomc.transAxes, ha="center", va="center",
                     color=CS["muted"], fontsize=10)
        ax_fomc.set_title("FOMC Rate Distribution", color=CS["text"], fontsize=10)
        ax_fomc.axis("off")

    # ── Right: Polymarket ─────────────────────────────────────────────────────
    ax_poly = fig.add_subplot(1, 2, 2)
    ax_poly.set_facecolor(CS["panel"])

    if poly_markets:
        top    = poly_markets[:12]
        labels = [
            m["question"][:50] + ("…" if len(m["question"]) > 50 else "")
            for m in top
        ]
        probs  = [m["yes_prob"] * 100 for m in top]
        colors = [
            CS["green"] if p >= 60 else CS["amber"] if p >= 35 else CS["red"]
            for p in probs
        ]

        y    = np.arange(len(labels))
        bars = ax_poly.barh(y, probs, color=colors, alpha=0.88,
                            height=0.6, zorder=3)
        ax_poly.set_yticks(y)
        ax_poly.set_yticklabels(labels, color=CS["text"], fontsize=7.5)
        ax_poly.set_xlabel("Probability (%)", color=CS["text"], fontsize=9)
        ax_poly.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
        ax_poly.tick_params(colors=CS["text"])
        ax_poly.grid(axis="x", color=CS["bg"], alpha=0.6, zorder=0)
        ax_poly.set_xlim(0, 110)
        ax_poly.invert_yaxis()
        for spine in ax_poly.spines.values():
            spine.set_edgecolor(CS["muted"])
        for bar, prob in zip(bars, probs):
            ax_poly.text(
                min(bar.get_width() + 1, 108),
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.0f}%",
                va="center", color=CS["text"], fontsize=8,
            )
        ax_poly.set_title("Polymarket — Top Macro Markets",
                          color=CS["text"], fontsize=10, pad=10)
    else:
        ax_poly.text(0.5, 0.5, "No Polymarket data available",
                     transform=ax_poly.transAxes, ha="center", va="center",
                     color=CS["muted"], fontsize=10)
        ax_poly.set_title("Polymarket — Top Macro Markets",
                          color=CS["text"], fontsize=10)
        ax_poly.axis("off")

    plt.tight_layout(pad=2.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=CS["bg"])
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# ── Email HTML ────────────────────────────────────────────────────────────────

_FOMC_HEADER = """
<tr style="background:#f8fafc;">
  <th style="padding:8px 12px;text-align:left;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">Meeting</th>
  <th style="padding:8px 12px;text-align:left;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">Date</th>
  <th style="padding:8px 12px;text-align:left;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">Most Likely Outcome</th>
  <th style="padding:8px 12px;text-align:center;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">Probability</th>
  <th style="padding:8px 12px;text-align:center;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">Implied Rate</th>
</tr>"""

_POLY_HEADER = """
<tr style="background:#f8fafc;">
  <th style="padding:8px 12px;text-align:left;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">Market</th>
  <th style="padding:8px 12px;text-align:center;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">Probability</th>
  <th style="padding:8px 12px;text-align:right;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">24h Volume</th>
  <th style="padding:8px 12px;text-align:right;font-size:11px;color:#64748b;
      font-weight:600;border-bottom:1px solid #e2e8f0;">Resolves</th>
</tr>"""


def _prob_badge(prob: float) -> str:
    pct = prob * 100
    if pct >= 65:
        bg, fg = "#dcfce7", "#15803d"
    elif pct >= 35:
        bg, fg = "#fef9c3", "#854d0e"
    else:
        bg, fg = "#fee2e2", "#991b1b"
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:12px;font-size:11px;font-weight:700;">{pct:.0f}%</span>'
    )


def _prob_bar_html(prob: float) -> str:
    pct = prob * 100
    fill = "#22c55e" if pct >= 60 else "#f59e0b" if pct >= 35 else "#ef4444"
    bar = (
        f'<div style="background:#e2e8f0;border-radius:4px;height:6px;'
        f'width:80px;display:inline-block;vertical-align:middle;">'
        f'<div style="background:{fill};width:{min(pct,100):.0f}%;height:6px;'
        f'border-radius:4px;"></div></div>'
    )
    return (
        f'{bar}&nbsp;<span style="font-size:11px;color:#374151;'
        f'font-weight:600;">{pct:.0f}%</span>'
    )


def _fomc_row(ev: dict, alt: bool) -> str:
    bg       = "#f8fafc" if alt else "#ffffff"
    date_str = ev["close_date"].strftime("%b %d, %Y") if ev.get("close_date") else "—"
    dist     = ev.get("distribution", [])
    if dist:
        best       = max(dist, key=lambda d: d["prob"])
        best_label = best["title"] or "—"
        best_prob  = best["prob"]
    else:
        best_label, best_prob = "—", 0.0

    exp     = ev.get("expected_rate")
    exp_str = f"{exp:.2f}%" if exp else "—"
    # Shorten event ticker for display (e.g. KXFED-26MAY07 → May 7 FOMC)
    event_label = ev["event"].replace("KXFED-", "")

    return (
        f'<tr style="background:{bg};">'
        f'<td style="padding:8px 12px;font-size:11px;color:#64748b;">{event_label}</td>'
        f'<td style="padding:8px 12px;font-size:12px;color:#1e293b;">{date_str}</td>'
        f'<td style="padding:8px 12px;font-size:12px;color:#374151;">{best_label}</td>'
        f'<td style="padding:8px 12px;text-align:center;">{_prob_badge(best_prob)}</td>'
        f'<td style="padding:8px 12px;text-align:center;font-size:12px;'
        f'color:#374151;font-weight:600;">{exp_str}</td>'
        f'</tr>'
    )


def _poly_row(m: dict, alt: bool) -> str:
    bg  = "#f8fafc" if alt else "#ffffff"
    end = "—"
    if m.get("end_date"):
        try:
            ts = m["end_date"]
            end = datetime.fromtimestamp(float(ts)).strftime("%b %d") if str(ts).isdigit() else str(ts)[:10]
        except Exception:
            end = str(m["end_date"])[:10]
    vol_24h = m.get("volume_24h", 0)
    vol_str = (f"${vol_24h/1e6:.1f}M" if vol_24h >= 1e6
               else f"${vol_24h/1e3:.0f}K" if vol_24h >= 1e3
               else f"${vol_24h:.0f}")
    return (
        f'<tr style="background:{bg};">'
        f'<td style="padding:8px 12px;font-size:12px;color:#1e293b;">{m["question"]}</td>'
        f'<td style="padding:8px 12px;text-align:center;">{_prob_bar_html(m["yes_prob"])}</td>'
        f'<td style="padding:8px 12px;text-align:right;font-size:11px;color:#64748b;">{vol_str}</td>'
        f'<td style="padding:8px 12px;text-align:right;font-size:11px;color:#64748b;">{end}</td>'
        f'</tr>'
    )


def build_email_html(
    fomc_events:  list[dict],
    poly_markets: list[dict],
    chart_b64:    str,
) -> str:
    today = datetime.now().strftime("%B %d, %Y")

    # FOMC table rows
    if fomc_events:
        fomc_rows = "".join(_fomc_row(ev, i % 2 == 0)
                            for i, ev in enumerate(fomc_events))
    else:
        fomc_rows = ('<tr><td colspan="5" style="padding:16px;text-align:center;'
                     'color:#94a3b8;font-size:12px;">No FOMC markets available</td></tr>')

    # Polymarket rows
    if poly_markets:
        poly_rows = "".join(_poly_row(m, i % 2 == 0)
                            for i, m in enumerate(poly_markets[:15]))
    else:
        poly_rows = ('<tr><td colspan="4" style="padding:16px;text-align:center;'
                     'color:#94a3b8;font-size:12px;">No Polymarket data available</td></tr>')

    # Headline: implied rate at next meeting
    headline = ""
    if fomc_events and fomc_events[0].get("expected_rate"):
        implied     = fomc_events[0]["expected_rate"]
        meeting_str = (fomc_events[0]["close_date"].strftime("%b %d")
                       if fomc_events[0].get("close_date") else "next meeting")
        headline = (
            f'<p style="margin:0;color:#93c5fd;font-size:13px;">'
            f'Markets imply <strong>{implied:.2f}%</strong> target rate '
            f'at {meeting_str} FOMC</p>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Prediction Market Monitor — {today}</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;">
<tr><td align="center" style="padding:32px 16px;">
<table width="760" cellpadding="0" cellspacing="0">

  <!-- Header -->
  <tr>
    <td style="background:linear-gradient(135deg,#0f172a 0%,#1a2e4a 100%);
        padding:36px 40px;border-radius:12px 12px 0 0;text-align:center;">
      <p style="margin:0 0 6px;color:#60a5fa;font-size:11px;font-weight:700;
          letter-spacing:2px;text-transform:uppercase;">Swarm Investments</p>
      <h1 style="margin:0 0 8px;color:#fff;font-size:26px;font-weight:800;">
        Prediction Market Monitor</h1>
      <p style="margin:0 0 12px;color:#94a3b8;font-size:14px;">
        Weekly Report &mdash; {today}</p>
      {headline}
    </td>
  </tr>

  <!-- FOMC Rate Path -->
  <tr>
    <td style="background:#fff;padding:0;border-top:1px solid #e2e8f0;">
      <div style="padding:16px 20px 8px;font-size:10px;font-weight:700;
          color:#64748b;text-transform:uppercase;letter-spacing:1px;">
        FOMC Rate Path &mdash; Market Implied Probabilities (Kalshi)</div>
      <table width="100%" cellpadding="0" cellspacing="0">
        {_FOMC_HEADER}
        {fomc_rows}
      </table>
      <div style="padding:4px 24px 14px;font-size:10px;color:#94a3b8;">
        Probability = midpoint of Kalshi best bid/ask &middot;
        Implied rate = probability-weighted average across all outcome contracts &middot;
        Only open markets shown
      </div>
    </td>
  </tr>

  <!-- Polymarket -->
  <tr>
    <td style="background:#fff;padding:0;border-top:1px solid #e2e8f0;">
      <div style="padding:16px 20px 8px;font-size:10px;font-weight:700;
          color:#64748b;text-transform:uppercase;letter-spacing:1px;">
        Polymarket &mdash; Top Macro &amp; Economic Markets</div>
      <table width="100%" cellpadding="0" cellspacing="0">
        {_POLY_HEADER}
        {poly_rows}
      </table>
      <div style="padding:4px 24px 14px;font-size:10px;color:#94a3b8;">
        Sorted by 24h trading volume &middot;
        Probability reflects current market-clearing YES price &middot;
        Green &ge;60% &middot; Amber 35&ndash;60% &middot; Red &lt;35%
      </div>
    </td>
  </tr>

  <!-- Chart -->
  <tr>
    <td style="background:#fff;padding:20px;border-top:1px solid #e2e8f0;
        border-radius:0 0 12px 12px;">
      <div style="font-size:10px;font-weight:700;color:#64748b;
          text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
        FOMC Distribution &amp; Polymarket Overview</div>
      <img src="data:image/png;base64,{chart_b64}"
           style="width:100%;max-width:720px;border-radius:8px;"
           alt="Prediction Market Charts">
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
        Data: Kalshi &middot; Polymarket &middot; {today}
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
    log.info("=== Prediction Market & Macro Probability Monitor — Weekly Run ===")

    # 1. Kalshi FOMC
    log.info("Fetching Kalshi FOMC (KXFED) markets...")
    fomc_raw    = fetch_kalshi_markets("KXFED")
    fomc_events = build_fomc_events(fomc_raw)
    log.info(f"Parsed {len(fomc_events)} FOMC events")

    # 2. Kalshi extra series (logged for diagnostics, not yet in email)
    for series in KALSHI_EXTRA_SERIES:
        fetch_kalshi_markets(series)

    # 3. Polymarket macro markets
    log.info("Fetching Polymarket macro markets...")
    poly_markets = fetch_polymarket_macro()

    # 4. Chart
    log.info("Rendering chart...")
    chart_b64 = build_chart(fomc_events, poly_markets)

    # 5. Email
    html = build_email_html(
        fomc_events=fomc_events,
        poly_markets=poly_markets,
        chart_b64=chart_b64,
    )
    log.info("Sending email...")
    service = get_gmail_service()
    send_email(
        service,
        to=RECIPIENT_EMAIL,
        subject=f"Prediction Market Monitor — {datetime.now().strftime('%b %d, %Y')}",
        html_body=html,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
