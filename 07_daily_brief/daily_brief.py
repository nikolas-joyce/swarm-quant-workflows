#!/usr/bin/env python3
"""
Workflow #07 — Daily Market Brief
Reads financial newsletters from Gmail (Bloomberg, FT), synthesises with
Claude Haiku, sends a lean daily briefing with talking points and strategy ideas.
"""

import base64
import logging
import os
import re
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html.parser import HTMLParser
from pathlib import Path

import anthropic
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RECIPIENT_EMAIL   = os.environ.get("RECIPIENT_EMAIL", "")
TOKEN_PATH        = Path(os.environ.get("GMAIL_TOKEN_PATH", "token.json"))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

# Newsletter sender domains to read
NEWSLETTER_SENDERS = [
    "bloomberg.com",
    "news.bloomberg.com",
    "ft.com",
    "financial-times.com",
]

# Look back this many days — covers weekend gaps on Monday
LOOKBACK_DAYS = 3

# Max chars per email fed to Claude (keeps prompt manageable)
MAX_EMAIL_CHARS = 5000

# Claude model for synthesis
CLAUDE_MODEL = "claude-haiku-4-5-20251001"


# ── Gmail helpers ─────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        s = data.strip()
        if s:
            self._parts.append(s)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    try:
        p = _HTMLStripper()
        p.feed(html)
        return p.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


def _b64_decode(data: str) -> str:
    try:
        padded = data + "=" * (4 - len(data) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_parts(payload: dict, mime: str) -> list[str]:
    """Recursively pull body text of a given MIME type from a Gmail payload."""
    found = []
    if payload.get("mimeType") == mime:
        data = payload.get("body", {}).get("data", "")
        if data:
            found.append(_b64_decode(data))
    for part in payload.get("parts", []):
        found.extend(_extract_parts(part, mime))
    return found


def extract_email_text(message: dict) -> str:
    payload = message.get("payload", {})
    plain = _extract_parts(payload, "text/plain")
    if plain:
        return " ".join(plain)
    html = _extract_parts(payload, "text/html")
    if html:
        return _strip_html(" ".join(html))
    return ""


def get_header(message: dict, name: str) -> str:
    for h in message.get("payload", {}).get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def get_gmail_service():
    if not TOKEN_PATH.exists():
        raise FileNotFoundError(f"token.json not found at '{TOKEN_PATH}'.")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), GMAIL_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ── Newsletter fetch ──────────────────────────────────────────────────────────

def fetch_newsletters(service) -> list[dict]:
    """Fetch newsletter emails from configured senders within the lookback window."""
    sender_q = " OR ".join(f"from:@{d}" for d in NEWSLETTER_SENDERS)
    query    = f"({sender_q}) newer_than:{LOOKBACK_DAYS}d"
    log.info(f"Gmail query: {query}")

    try:
        result = service.users().messages().list(
            userId="me", q=query, maxResults=30
        ).execute()
    except Exception as exc:
        log.warning(f"Gmail list failed: {exc}")
        return []

    refs = result.get("messages", [])
    log.info(f"Found {len(refs)} matching messages")

    emails = []
    seen_subjects: set[str] = set()

    for ref in refs:
        try:
            msg     = service.users().messages().get(
                userId="me", id=ref["id"], format="full"
            ).execute()
            subject = get_header(msg, "Subject")
            sender  = get_header(msg, "From")
            date    = get_header(msg, "Date")
            text    = extract_email_text(msg)

            # De-duplicate by subject (same newsletter re-sent)
            key = subject.strip().lower()
            if key in seen_subjects or not text:
                continue
            seen_subjects.add(key)

            emails.append({
                "subject": subject,
                "sender":  sender,
                "date":    date,
                "text":    text[:MAX_EMAIL_CHARS],
            })
            log.info(f"  Loaded: '{subject[:70]}' ({len(text):,} chars)")
        except Exception as exc:
            log.warning(f"  Failed to fetch {ref['id']}: {exc}")

    log.info(f"Using {len(emails)} newsletters for synthesis")
    return emails


# ── Claude synthesis ──────────────────────────────────────────────────────────

def synthesize_brief(emails: list[dict]) -> str:
    """Call Claude Haiku to extract talking points and strategy ideas."""
    if not emails:
        return "No newsletter content was available for synthesis today."

    sections = []
    for e in emails:
        sections.append(
            f"SOURCE: {e['sender']}\n"
            f"SUBJECT: {e['subject']}\n"
            f"DATE: {e['date']}\n\n"
            f"{e['text']}"
        )
    combined = ("\n\n" + "=" * 60 + "\n\n").join(sections)

    prompt = f"""You are a senior analyst at a quantitative macro hedge fund reviewing today's financial editorials.

Below are newsletters and editorials from Bloomberg and the Financial Times received today.

{combined}

Produce a concise daily market brief with exactly these three sections. Be direct, specific, and actionable — no filler.

## KEY THEMES
5-7 bullet points capturing the most important macro themes, narratives, or market developments from today's editorial flow. Each bullet: 1-2 sentences, specific and punchy.

## STRATEGY IDEAS
For the 3-4 most significant themes, suggest a specific trading or research angle. Format each as:
**[Theme name]**
Idea: [specific instrument, direction, or research angle]
Rationale: [why now, what's the edge — 1-2 sentences]
Key risk: [the main thing that breaks the thesis]

## WATCH LIST
5 specific items to monitor this week: securities, data releases, levels, or events — each with a one-line reason why it matters.

Do not use generic statements. Everything should be specific enough that a portfolio manager could act on it."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        log.info(f"Claude response: {len(text)} chars, "
                 f"input={response.usage.input_tokens} "
                 f"output={response.usage.output_tokens} tokens")
        return text
    except Exception as exc:
        log.warning(f"Claude synthesis failed: {exc}")
        return f"Synthesis unavailable: {exc}"


# ── Email rendering ───────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Convert minimal markdown (##, -, **bold**) to inline-styled HTML."""
    lines     = text.split("\n")
    out       = []
    in_list   = False

    def _bold(s: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

    for line in lines:
        s = line.strip()

        if s.startswith("## "):
            if in_list:
                out.append("</ul>")
                in_list = False
            heading = _bold(s[3:])
            out.append(
                f'<h3 style="margin:20px 0 8px;font-size:11px;font-weight:700;'
                f'color:#64748b;text-transform:uppercase;letter-spacing:1px;'
                f'border-bottom:1px solid #e2e8f0;padding-bottom:6px;">'
                f'{heading}</h3>'
            )

        elif s.startswith("- ") or s.startswith("• "):
            content = _bold(s[2:])
            if not in_list:
                out.append('<ul style="margin:4px 0 8px;padding-left:18px;">')
                in_list = True
            out.append(
                f'<li style="margin:5px 0;font-size:13px;color:#1e293b;'
                f'line-height:1.6;">{content}</li>'
            )

        elif s == "":
            if in_list:
                out.append("</ul>")
                in_list = False

        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            content = _bold(s)
            out.append(
                f'<p style="margin:5px 0;font-size:13px;color:#374151;'
                f'line-height:1.6;">{content}</p>'
            )

    if in_list:
        out.append("</ul>")

    return "\n".join(out)


def build_email_html(emails: list[dict], brief_text: str) -> str:
    today   = datetime.now().strftime("%B %d, %Y")
    weekday = datetime.now().strftime("%A")

    # Source pills
    if emails:
        pills = "".join(
            f'<span style="display:inline-block;background:#1e3a5f;color:#93c5fd;'
            f'padding:3px 10px;border-radius:12px;font-size:10px;margin:2px 3px;">'
            f'{e["subject"][:55]}</span>'
            for e in emails[:10]
        )
    else:
        pills = '<span style="color:#94a3b8;font-size:12px;">No newsletters found in inbox</span>'

    brief_html = _md_to_html(brief_text)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Daily Market Brief — {today}</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;">
<tr><td align="center" style="padding:24px 16px;">
<table width="680" cellpadding="0" cellspacing="0">

  <!-- Header -->
  <tr>
    <td style="background:linear-gradient(135deg,#0f172a 0%,#1a2e4a 100%);
        padding:28px 36px;border-radius:12px 12px 0 0;">
      <p style="margin:0 0 4px;color:#60a5fa;font-size:10px;font-weight:700;
          letter-spacing:2px;text-transform:uppercase;">Swarm Investments</p>
      <h1 style="margin:0 0 4px;color:#fff;font-size:22px;font-weight:800;">
        Daily Market Brief</h1>
      <p style="margin:0;color:#94a3b8;font-size:13px;">{weekday}, {today}</p>
    </td>
  </tr>

  <!-- Sources -->
  <tr>
    <td style="background:#1e293b;padding:12px 36px;">
      <p style="margin:0 0 6px;color:#475569;font-size:10px;font-weight:700;
          text-transform:uppercase;letter-spacing:1px;">Sources ingested</p>
      {pills}
    </td>
  </tr>

  <!-- Brief body -->
  <tr>
    <td style="background:#fff;padding:28px 36px;border-radius:0 0 12px 12px;">
      {brief_html}
    </td>
  </tr>

  <!-- Disclaimer + footer -->
  <tr>
    <td style="padding:16px 0;text-align:center;">
      <p style="margin:0;color:#94a3b8;font-size:10px;">
        Generated by <strong>Swarm Investments Quant Workflow #07</strong>
        &middot; Sources: Bloomberg &middot; Financial Times
        &middot; Synthesis: Claude Haiku (Anthropic) &middot; {today}
      </p>
      <p style="margin:4px 0 0;color:#cbd5e1;font-size:10px;">
        AI-synthesised editorial content for internal research only.
        Not investment advice.
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ── Gmail send ────────────────────────────────────────────────────────────────

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
    log.info("=== Daily Market Brief — Run ===")

    service = get_gmail_service()

    # 1. Read newsletters from inbox
    emails = fetch_newsletters(service)

    # 2. Synthesise with Claude
    log.info("Synthesising brief with Claude...")
    brief_text = synthesize_brief(emails)

    # 3. Build & send email
    html = build_email_html(emails, brief_text)
    send_email(
        service,
        to=RECIPIENT_EMAIL,
        subject=f"Daily Market Brief — {datetime.now().strftime('%A, %b %d')}",
        html_body=html,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
