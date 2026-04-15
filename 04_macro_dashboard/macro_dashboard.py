#!/usr/bin/env python3
"""
Swarm Quant Workflows — #04: Macro Dashboard
---------------------------------------------
Pulls key US macroeconomic indicators from the FRED API and delivers
a formatted weekly dashboard covering:
  - Inflation (CPI, Core CPI, Core PCE)
  - Labor Market (Unemployment, Initial Claims, Payrolls)
  - Growth (Industrial Production, Retail Sales, GDP)
  - Housing (Housing Starts, Building Permits)
  - Monetary Policy (Fed Funds Rate, M2, Yield Curve)
  - Sentiment (UMich Consumer Sentiment)

Charts show the last 24 months of key series embedded in the email.

Run locally:
    FRED_API_KEY=your_key python 04_macro_dashboard/macro_dashboard.py

Requires GitHub Secret: FRED_API_KEY (free at fred.stlouisfed.org/docs/api/api_key.html)
"""

import base64
import io
import logging
import os
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import requests
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
FRED_BASE       = "https://api.stlouisfed.org/fred/series/observations"

if not FRED_API_KEY:
    log.error("FRED_API_KEY not set. Get a free key at fred.stlouisfed.org/docs/api/api_key.html")
    sys.exit(1)

# ── Indicator definitions ─────────────────────────────────────────────────────
# (series_id, display_name, category, transform, unit, good_direction)
# transform: "level" | "yoy_pct" | "mom_pct" | "mom_chg"
# good_direction: "up" | "down" | "neutral"
INDICATORS = [
    # Inflation
    ("CPIAUCSL",  "CPI",              "Inflation",        "yoy_pct",  "%",     "down"),
    ("CPILFESL",  "Core CPI",         "Inflation",        "yoy_pct",  "%",     "down"),
    ("PCEPILFE",  "Core PCE",         "Inflation",        "yoy_pct",  "%",     "down"),

    # Labor
    ("UNRATE",    "Unemployment",     "Labor",            "level",    "%",     "down"),
    ("ICSA",      "Initial Claims",   "Labor",            "level",    "K",     "down"),
    ("PAYEMS",    "Nonfarm Payrolls", "Labor",            "mom_chg",  "K",     "up"),

    # Growth
    ("INDPRO",    "Industrial Prod",  "Growth",           "yoy_pct",  "%",     "up"),
    ("RSXFS",     "Retail Sales",     "Growth",           "yoy_pct",  "%",     "up"),
    ("GDP",       "GDP",              "Growth",           "yoy_pct",  "%",     "up"),

    # Housing
    ("HOUST",     "Housing Starts",   "Housing",          "level",    "K",     "up"),
    ("PERMIT",    "Building Permits", "Housing",          "level",    "K",     "up"),

    # Monetary Policy
    ("FEDFUNDS",  "Fed Funds Rate",   "Monetary Policy",  "level",    "%",     "neutral"),
    ("T10Y2Y",    "10Y-2Y Spread",    "Monetary Policy",  "level",    "%",     "up"),
    ("M2SL",      "M2 Money Supply",  "Monetary Policy",  "yoy_pct",  "%",     "neutral"),

    # Sentiment
    ("UMCSENT",   "Consumer Sentiment","Sentiment",       "level",    "pts",   "up"),
]

# Series to chart (series_id, chart_title, reference_line)
CHART_SERIES = [
    ("CPIAUCSL",  "CPI YoY %",              2.0),   # Fed 2% target
    ("UNRATE",    "Unemployment Rate %",     None),
    ("FEDFUNDS",  "Fed Funds Rate %",        None),
    ("ICSA",      "Initial Claims (000s)",   None),
    ("INDPRO",    "Industrial Production YoY %", 0.0),
    ("UMCSENT",   "Consumer Sentiment",      None),
]


# ── FRED Data Fetching ────────────────────────────────────────────────────────

def fetch_fred(series_id: str, limit: int = 60) -> pd.Series:
    """Fetch a FRED series, return as pd.Series with DatetimeIndex."""
    params = {
        "series_id":  series_id,
        "api_key":    FRED_API_KEY,
        "file_type":  "json",
        "limit":      limit,
        "sort_order": "desc",
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=15)
        r.raise_for_status()
    except Exception as e:
        log.warning(f"FRED fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float)

    obs = r.json().get("observations", [])
    records = []
    for o in obs:
        if o.get("value", ".") != ".":
            try:
                records.append({
                    "date":  pd.to_datetime(o["date"]),
                    "value": float(o["value"]),
                })
            except (ValueError, KeyError):
                pass

    if not records:
        return pd.Series(dtype=float)

    df = pd.DataFrame(records).set_index("date").sort_index()
    return df["value"]


def compute_display(series: pd.Series, transform: str, unit: str) -> dict:
    """
    Compute display values for a single indicator.
    Returns dict with: current, prior, change, yoy, label, date
    """
    s = series.dropna()
    if s.empty:
        return {"current": None, "change": None, "yoy": None, "label": "—", "date": "—"}

    latest_date = s.index[-1].strftime("%b %Y")

    if transform == "yoy_pct":
        # Show YoY % change
        if len(s) >= 13:
            cur_yoy  = (s.iloc[-1]  / s.iloc[-13] - 1) * 100
            prev_yoy = (s.iloc[-2]  / s.iloc[-14] - 1) * 100 if len(s) >= 14 else float("nan")
            change   = cur_yoy - prev_yoy
            label    = f"{cur_yoy:.2f}%"
            return {"current": cur_yoy, "change": change, "yoy": None,
                    "label": label, "date": latest_date, "raw": s}
        return {"current": None, "change": None, "yoy": None, "label": "—", "date": latest_date}

    elif transform == "mom_pct":
        cur   = (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) >= 2 else float("nan")
        prior = (s.iloc[-2] / s.iloc[-3] - 1) * 100 if len(s) >= 3 else float("nan")
        yoy   = (s.iloc[-1] / s.iloc[-13] - 1) * 100 if len(s) >= 13 else float("nan")
        return {"current": cur, "change": cur - prior, "yoy": yoy,
                "label": f"{cur:+.2f}%", "date": latest_date, "raw": s}

    elif transform == "mom_chg":
        cur   = s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else float("nan")
        prior = s.iloc[-2] - s.iloc[-3] if len(s) >= 3 else float("nan")
        divisor = 1000 if unit == "K" else 1
        return {"current": cur, "change": cur - prior, "yoy": None,
                "label": f"{cur/divisor:+.0f}K", "date": latest_date, "raw": s}

    else:  # level
        cur   = s.iloc[-1]
        prior = s.iloc[-2] if len(s) >= 2 else float("nan")
        yoy   = s.iloc[-13] if len(s) >= 13 else float("nan")
        change = cur - prior
        divisor = 1000 if unit == "K" else 1
        fmt = f"{cur/divisor:.0f}K" if unit == "K" else f"{cur:.2f}%"
        if unit == "pts":
            fmt = f"{cur:.1f}"
        return {"current": cur, "change": change, "yoy": yoy,
                "label": fmt, "date": latest_date, "raw": s}


def fetch_all(limit: int = 60) -> dict[str, dict]:
    """Fetch and compute all indicators."""
    results = {}
    for sid, name, cat, transform, unit, direction in INDICATORS:
        log.info(f"  Fetching {sid} ({name})...")
        series = fetch_fred(sid, limit=limit)
        data   = compute_display(series, transform, unit)
        data["name"]      = name
        data["category"]  = cat
        data["transform"] = transform
        data["unit"]      = unit
        data["direction"] = direction
        results[sid] = data
    return results


# ── Charts ────────────────────────────────────────────────────────────────────

CHART_BG    = "#0f172a"
CHART_PANEL = "#1e293b"
CHART_GRID  = "#334155"
CHART_TEXT  = "#94a3b8"
CHART_LINE  = "#60a5fa"
CHART_REF   = "#f59e0b"


def _dark_ax(ax, title: str) -> None:
    ax.set_facecolor(CHART_PANEL)
    ax.set_title(title, color="#f1f5f9", fontsize=9, fontweight="bold", pad=8)
    ax.tick_params(colors=CHART_TEXT, labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_GRID)
    ax.grid(True, color=CHART_GRID, linewidth=0.4, alpha=0.6)


def build_chart(data: dict[str, dict]) -> str:
    """Build 3x2 grid of macro charts, return as base64 PNG."""
    cutoff = datetime.today() - timedelta(days=24 * 30)

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8),
        facecolor=CHART_BG,
    )
    fig.suptitle(
        f"US Macro Dashboard — {datetime.now().strftime('%B %d, %Y')}",
        color="#f1f5f9", fontsize=12, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.35,
                        left=0.06, right=0.97, top=0.92, bottom=0.1)

    for ax, (sid, title, ref) in zip(axes.flat, CHART_SERIES):
        entry = data.get(sid, {})
        raw   = entry.get("raw")

        if raw is None or raw.empty:
            _dark_ax(ax, title)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color=CHART_TEXT, fontsize=9)
            continue

        transform = entry.get("transform", "level")

        # Compute plot series
        if transform == "yoy_pct":
            plot_s = raw.pct_change(12) * 100
        elif transform == "mom_pct":
            plot_s = raw.pct_change() * 100
        elif transform == "mom_chg":
            plot_s = raw.diff()
        else:
            plot_s = raw.copy()
            if entry.get("unit") == "K":
                plot_s = plot_s / 1000

        plot_s = plot_s.dropna()
        plot_s = plot_s[plot_s.index >= cutoff]

        if plot_s.empty:
            _dark_ax(ax, title)
            continue

        ax.plot(plot_s.index, plot_s.values,
                color=CHART_LINE, linewidth=1.8, zorder=3)
        ax.fill_between(plot_s.index, plot_s.values,
                        alpha=0.15, color=CHART_LINE)

        if ref is not None:
            ax.axhline(ref, color=CHART_REF, linewidth=1,
                       linestyle="--", alpha=0.8, label=f"Target: {ref}%")
            ax.legend(fontsize=6, facecolor=CHART_PANEL,
                      labelcolor=CHART_TEXT, framealpha=0.8)

        _dark_ax(ax, title)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# ── Email HTML ────────────────────────────────────────────────────────────────

def _change_cell(change, good_dir: str) -> str:
    if change is None or change != change:
        return '<td style="padding:8px 10px;text-align:right;color:#94a3b8;">—</td>'

    if good_dir == "up":
        color = "#16a34a" if change >= 0 else "#dc2626"
    elif good_dir == "down":
        color = "#16a34a" if change <= 0 else "#dc2626"
    else:
        color = "#64748b"

    sign = "+" if change >= 0 else ""
    val  = f"{sign}{change:.2f}"
    return (
        f'<td style="padding:8px 10px;text-align:right;'
        f'font-weight:600;color:{color};font-size:12px;">{val}</td>'
    )


def build_email_html(data: dict[str, dict], chart_b64: str) -> str:
    today = datetime.now().strftime("%B %d, %Y")

    # Group by category
    categories: dict[str, list] = {}
    for sid, *rest in INDICATORS:
        _, name, cat, transform, unit, direction = (sid, *rest)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((sid, direction))

    # Category colors
    cat_colors = {
        "Inflation":       "#ef4444",
        "Labor":           "#10b981",
        "Growth":          "#6366f1",
        "Housing":         "#f59e0b",
        "Monetary Policy": "#3b82f6",
        "Sentiment":       "#ec4899",
    }

    sections = ""
    for cat, items in categories.items():
        color = cat_colors.get(cat, "#64748b")
        rows  = ""
        for sid, direction in items:
            d = data.get(sid, {})
            if d.get("current") is None:
                continue
            rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9;">
              <td style="padding:8px 10px;font-size:12px;color:#0f172a;
                  font-weight:600;white-space:nowrap;">{d['name']}</td>
              <td style="padding:8px 10px;text-align:right;font-size:13px;
                  font-weight:800;color:#0f172a;">{d['label']}</td>
              {_change_cell(d.get('change'), direction)}
              <td style="padding:8px 10px;text-align:right;font-size:11px;
                  color:#94a3b8;">{d['date']}</td>
            </tr>"""

        if not rows:
            continue

        sections += f"""
        <div style="margin-bottom:20px;">
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="border-radius:8px;overflow:hidden;
                        box-shadow:0 1px 6px rgba(0,0,0,0.07);">
            <tr>
              <td style="background:{color};padding:9px 14px;">
                <span style="color:#fff;font-size:11px;font-weight:700;
                  letter-spacing:1px;text-transform:uppercase;">{cat}</span>
              </td>
            </tr>
            <tr>
              <td style="background:#fff;padding:0;">
                <table width="100%" cellpadding="0" cellspacing="0">
                  <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
                    <th style="padding:7px 10px;text-align:left;font-size:10px;
                        color:#94a3b8;font-weight:600;">Indicator</th>
                    <th style="padding:7px 10px;text-align:right;font-size:10px;
                        color:#94a3b8;font-weight:600;">Latest</th>
                    <th style="padding:7px 10px;text-align:right;font-size:10px;
                        color:#94a3b8;font-weight:600;">vs Prior</th>
                    <th style="padding:7px 10px;text-align:right;font-size:10px;
                        color:#94a3b8;font-weight:600;">As Of</th>
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
  <title>Macro Dashboard — {today}</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;">
<tr><td align="center" style="padding:32px 16px;">
<table width="680" cellpadding="0" cellspacing="0">

  <!-- Header -->
  <tr>
    <td style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
        padding:36px 40px;border-radius:12px 12px 0 0;text-align:center;">
      <p style="margin:0 0 6px;color:#60a5fa;font-size:11px;font-weight:700;
          letter-spacing:2px;text-transform:uppercase;">Swarm Investments</p>
      <h1 style="margin:0 0 8px;color:#fff;font-size:28px;font-weight:800;">
        US Macro Dashboard</h1>
      <p style="margin:0;color:#94a3b8;font-size:14px;">Weekly Report &mdash; {today}</p>
    </td>
  </tr>

  <!-- Summary strip -->
  <tr>
    <td style="background:#1e293b;padding:14px 24px;">
      <p style="margin:0;color:#64748b;font-size:10px;font-weight:600;
          text-transform:uppercase;letter-spacing:1px;">Coverage</p>
      <p style="margin:4px 0 0;color:#94a3b8;font-size:12px;">
        Inflation &bull; Labor &bull; Growth &bull; Housing
        &bull; Monetary Policy &bull; Sentiment &nbsp;&mdash;&nbsp;
        Data: FRED (Federal Reserve Bank of St. Louis)
      </p>
    </td>
  </tr>

  <!-- Indicator tables -->
  <tr>
    <td style="background:#f1f5f9;padding:24px 8px;">
      {sections}
    </td>
  </tr>

  <!-- Charts -->
  <tr>
    <td style="background:#fff;padding:20px;border-radius:8px;margin:0 8px;">
      <div style="font-size:10px;font-weight:700;color:#64748b;
          text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
        24-Month Trend Charts
      </div>
      <img src="data:image/png;base64,{chart_b64}"
           style="width:100%;max-width:640px;border-radius:8px;"
           alt="Macro Charts">
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
        Source: FRED (Federal Reserve Bank of St. Louis) &middot; {today}
        &middot; Green/red arrows = better/worse vs prior period
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
    log.info("=== US Macro Dashboard -- Weekly Run ===")

    log.info("Fetching FRED indicators...")
    data = fetch_all(limit=60)
    log.info(f"Fetched {len(data)} indicators.")

    log.info("Building charts...")
    chart_b64 = build_chart(data)

    log.info("Building email...")
    html    = build_email_html(data, chart_b64)
    today   = datetime.now().strftime("%B %d, %Y")
    service = get_gmail_service()
    send_email(service, RECIPIENT_EMAIL, f"US Macro Dashboard -- {today}", html)
    log.info("Done.")


if __name__ == "__main__":
    main()
