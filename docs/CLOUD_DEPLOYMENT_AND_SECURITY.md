# Cloud Deployment, Security Hardening & Public CV Showcase Guide

## 1. Protecting Your Intellectual Property (Public Showcase vs. Private Bot)

All sensitive credentials and calibrated trading parameters live **outside** Git tracking:
- `.env` — API keys (`ALPACA_*`, `GEMINI_API_KEY`, `FMP_API_KEY`, `FINNHUB_API_KEY`, `TELEGRAM_*`).
- `config/strategy.local.json` — Your private watchlist, ATR multipliers, RSI thresholds, and volume filters.
- `data/trading_audit.db` & `data/candles/*.parquet` — Local SQLite audit logs and cached historical bars.

Only generic baseline defaults and templates (`config/strategy.example.json`, `.env.example`) exist in tracked source files.

### Creating a Clean Public Showcase Repository for Your CV
Keep this repository **Private** for personal trading, and publish a clean **Public** snapshot without prior git history (so old commits never leak `trading_audit.db` or calibrated parameters):

```zsh
# From a clean folder outside this repo
git clone --depth 1 https://github.com/<you>/trading-bot-pro.git trading-bot-showcase
cd trading-bot-showcase
rm -rf .git
git init -b main
git add .
git commit -m "feat: algorithmic trading bot with deterministic risk guardian"
git remote add origin https://github.com/<you>/trading-bot-showcase.git
git push -u origin main
```

---

## 2. Lowest-Cost Reliable Cloud Hosting Options ($0–$5/month)

Because the bot runs as an **outbound-only polling worker** (no exposed HTTP server), memory footprint stays under 250–400 MB:

1. **Oracle Cloud Always Free Tier ($0/month):**
   - `VM.Standard.A1.Flex` (ARM64 Ampere, 1–4 OCPUs, 6–24 GB RAM) or `VM.Standard.E2.1.Micro` (1 GB RAM x86). Zero monthly cost when kept inside Always Free limits.
2. **Low-Cost VPS ($4–$5/month):**
   - Hetzner Cloud `CX22` (~€3.79/mo, 4 GB RAM) or DigitalOcean Basic Droplet (\$4–\$6/mo).

---

## 3. Server Security Hardening Checklist

1. **Zero Inbound Attack Surface (Firewall):**
   ```bash
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   sudo ufw allow OpenSSH
   sudo ufw enable
   ```
   - Never open application ports (the bot needs outbound HTTPS only).
   - Disable SSH password login (`PasswordAuthentication no` in `/etc/ssh/sshd_config`) and use SSH keys only.

2. **Least-Privilege File Permissions:**
   ```bash
   chmod 600 .env config/strategy.local.json
   ```

3. **Container Hardening (already configured in `docker-compose.yml`):**
   - Non-root user (`botuser`) inside container.
   - `read_only: true` root filesystem (only `/app/data` and `/tmp` writable).
   - `cap_drop: [ALL]` and `no-new-privileges:true`.
   - `config/strategy.local.json` mounted read-only (`:ro`).
   - `restart: unless-stopped` + `SIGTERM` graceful shutdown + `ProcessLock` (`data/trading_audit.lock`) preventing duplicate instances.

4. **Crash Recovery & Backups:**
   - Open positions and trailing stop high-watermarks are persisted in SQLite `position_state` and reconciled automatically against Alpaca open positions/orders via `--sync-account`.
   - Schedule a daily cron backup of `data/trading_audit.db` using `sqlite3 data/trading_audit.db ".backup 'data/backup.db'"`.
