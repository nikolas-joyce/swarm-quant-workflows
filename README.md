# Swarm Quant Workflows

A public collection of automated quantitative trading and research workflows
built with Python, Google Colab, GitHub Actions, and Gmail.

Each workflow runs on a schedule, pulls financial data from free sources, and
delivers a formatted report to your inbox — no infrastructure to manage.

---

## Workflow Library

| # | Workflow | Schedule | Data Source | Status |
|---|---|---|---|---|
| 01 | [ETF Liquidity Ranker](./01_etf_liquidity_ranker/) | Weekly (Mon 9 AM ET) | Yahoo Finance | ✅ Live |
| 02 | Sector Rotation Monitor | Weekly | Yahoo Finance | 🔜 Planned |
| 03 | Market Regime Classifier | Weekly | Yahoo Finance / FRED | 🔜 Planned |
| 04 | Macro Dashboard (FRED) | Weekly | FRED API | 🔜 Planned |
| 05 | ETF Fund Flow Tracker | Weekly | ETF.com | 🔜 Planned |
| 06 | Factor Exposure Scanner | Weekly | Yahoo Finance | 🔜 Planned |
| 07 | Earnings Calendar + IV Ranker | Weekly | Yahoo Finance | 🔜 Planned |
| 08 | Short Interest Monitor | Bi-weekly | FINRA | 🔜 Planned |
| 09 | Insider Transaction Tracker | Weekly | SEC EDGAR | 🔜 Planned |
| 10 | Correlation Regime Monitor | Weekly | Yahoo Finance | 🔜 Planned |

---

## How It Works

```
GitHub Actions (cron)
       │
       ▼
  Python script
       │
       ├── Pulls data (Yahoo Finance / FRED / SEC)
       ├── Computes metrics & rankings
       ├── Builds HTML email
       │
       ▼
  Gmail API (swarm.quant.reports@gmail.com)
       │
       ▼
  Your inbox (formatted report)
```

All workflows share a single Gmail OAuth token stored as a GitHub Secret.
No servers, no databases, no hosting costs.

---

## One-Time Setup (Required for All Workflows)

### 1. Fork or clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/swarm-quant-workflows.git
cd swarm-quant-workflows
pip install -r requirements.txt
```

### 2. Get Gmail API credentials

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Create a new project — name it `swarm-quant-reports`
3. **APIs & Services → Library** → search `Gmail API` → **Enable**
4. **APIs & Services → Credentials → + Create Credentials → OAuth Client ID**
5. Configure consent screen if prompted:
   - User type: External
   - App name: `Swarm Quant Reports`
   - Add `swarm.quant.reports@gmail.com` as a test user
6. Application type: **Desktop App** → Create
7. **Download JSON** → save as `credentials.json` in the project root

> `credentials.json` is in `.gitignore` — it will never be committed.

### 3. Run the OAuth setup script

```bash
python setup/setup_gmail_oauth.py
```

- A browser opens — log in as **`swarm.quant.reports@gmail.com`**
- Click **Allow**
- `token.json` is created and its contents are printed

### 4. Add GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value |
|---|---|
| `GMAIL_TOKEN_JSON` | Full contents of `token.json` (printed by setup script) |
| `RECIPIENT_EMAIL` | Your email address |

This single token and these two secrets power **all workflows** in this repo.

---

## Running a Workflow Manually

Go to **Actions → [workflow name] → Run workflow** to trigger any report
on demand without waiting for the scheduled run.

---

## Adding Your Own Workflows

Each workflow follows the same pattern:

```
NN_workflow_name/
├── notebooks/
│   └── WorkflowName.ipynb      # Colab notebook for development
└── workflow_name.py            # Script executed by GitHub Actions
```

And a corresponding file in `.github/workflows/NN_workflow_name.yml`.

---

## Tech Stack

- **Python 3.11**
- **yfinance** — market data
- **pandas** — data wrangling
- **Google Gmail API** — email delivery
- **GitHub Actions** — scheduling
- **Google Colab** — interactive development

---

## License

MIT — free to use, fork, and adapt.

---

*Built by [Swarm Investments](https://swarminvestments.com)*
