#!/usr/bin/env python3
"""
Shared one-time Gmail OAuth setup for all Swarm Quant Workflows.

Run this ONCE locally, logged into swarm.quant.reports@gmail.com,
to generate token.json. Then paste token.json contents into GitHub Secrets.

Steps:
  1. Complete the Google Cloud Console setup (see README.md).
  2. Place credentials.json in the project root (this directory).
  3. Run:  python setup/setup_gmail_oauth.py
  4. A browser opens — log in as swarm.quant.reports@gmail.com and click Allow.
  5. token.json is written to the project root.
  6. Add its contents as GitHub Secret GMAIL_TOKEN_JSON (printed below).

IMPORTANT: Never commit token.json or credentials.json (.gitignore covers this).
"""

import sys
from pathlib import Path

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("ERROR: google-auth-oauthlib not installed.")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

SCOPES     = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",   # needed by workflow #07
]
ROOT       = Path(__file__).parent.parent
CREDS_PATH = ROOT / "credentials.json"
TOKEN_PATH = ROOT / "token.json"


def main() -> None:
    if not CREDS_PATH.exists():
        print(f"\nERROR: credentials.json not found at {CREDS_PATH}")
        print("\nTo obtain it:")
        print("  1. Go to https://console.cloud.google.com/")
        print("  2. Create a new project (e.g. 'swarm-quant-reports').")
        print("  3. Enable the Gmail API.")
        print("  4. APIs & Services → Credentials → Create Credentials → OAuth Client ID.")
        print("  5. Application type: Desktop App.")
        print("  6. Download JSON → save as credentials.json in the project root.")
        print("  7. Re-run this script.")
        sys.exit(1)

    print("Opening browser for Google OAuth...")
    print("IMPORTANT: Log in as swarm.quant.reports@gmail.com\n")

    flow  = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
    creds = flow.run_local_server(port=0)

    TOKEN_PATH.write_text(creds.to_json())
    print(f"\n✓  token.json created at {TOKEN_PATH}")

    token_content = TOKEN_PATH.read_text()

    print("\n" + "=" * 65)
    print("NEXT: Add two GitHub Secrets to your repo")
    print("=" * 65)
    print("Go to: GitHub repo → Settings → Secrets and variables → Actions\n")
    print("Secret 1 — Name:  GMAIL_TOKEN_JSON")
    print("           Value: (paste everything between the dashes)\n")
    print("---")
    print(token_content)
    print("---")
    print("\nSecret 2 — Name:  RECIPIENT_EMAIL")
    print("           Value: your-email@example.com")
    print("=" * 65)
    print("\nThis token works for ALL workflows in this repo.")


if __name__ == "__main__":
    main()
