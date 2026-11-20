# Cloud Deployment ($0–$5/mo), Gmail Alerts, Micro-Capital Live & Public CV Guide

## 1. Alertas Visuales por Correo en Gmail (`SMTP_APP_PASSWORD`)

El bot utiliza `EmailNotifier` (`src/notifications/email_notifier.py`) para enviar tarjetas HTML ejecutivas que indican de un vistazo si cada evento es:
- **🟢 BUENA NOTICIA (Verde):** Nueva inversión aprobada (pasó los 5 filtros) o **venta con ganancia realizada (`+$XX.XX` / `+X.XX%`)**.
- **🛡️ RIESGO EVITADO (Azul):** El Guardián Algorítmico bloqueó una compra riesgosa (p. ej. *Earnings* en <24h o `VIX > 30`), deduplicado a 1 correo por símbolo al día.
- **🟡 SALIDA PROTECTORA (Ámbar):** Cierre disciplinado por *Trailing Stop* o ventana de tiempo para proteger el capital y rotar a mejores oportunidades.
- **🔴 PROTECCIÓN DE EMERGENCIA (Rojo):** Escudo anti-caída súbita (5 minutos) o *Circuit Breaker* diario (`-3%`).

### Cómo activar tu Gmail en 1 minuto (Costo $0)
1. Entra a tu cuenta de Google (`https://myaccount.google.com/apppasswords` — requiere verificación en 2 pasos activa).
2. Crea una **Contraseña de aplicación** con el nombre `Trading Bot Pro` y copia los 16 caracteres generados.
3. En tu archivo privado `.env` (excluido de Git), completa:
   ```dotenv
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=tu.correo@gmail.com
   SMTP_APP_PASSWORD=xxxx xxxx xxxx xxxx
   EMAIL_RECIPIENT=tu.correo@gmail.com
   ```
4. Prueba el envío inmediatamente con:
   ```bash
   .venv/bin/python main.py --test-email
   ```

---

## 2. Requerimientos para Despliegue en la Nube ($0 a $5 USD/mes)

El bot es un *worker* ligero que solo realiza conexiones HTTPS salientes (no abre puertos web entrantes) y consume **< 250 MB de RAM**.

### Opción A — $0/mes: Oracle Cloud Always Free Tier (Recomendada)
- **Instancia:** `VM.Standard.A1.Flex` (Ubuntu 24.04 ARM64, 1–2 OCPUs, 6 GB RAM) o `VM.Standard.E2.1.Micro` (1 GB RAM x86).
- **Disco:** 47 GB Block Volume (incluido gratis).
- **Red:** Cerrar todos los puertos entrantes en el *Security List* excepto `SSH (22)` desde tu IP.

### Opción B — ~$4/mes: Hetzner Cloud (`CX22`) o DigitalOcean Basic Droplet
- **Especificaciones:** 1–2 vCPU, 1–4 GB RAM, 20–40 GB SSD, Ubuntu 24.04 LTS.

### Pasos de Instalación y Seguridad en el Servidor (VPS)
```bash
# 1. Firewall sin puertos de aplicación abiertos
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw enable

# 2. Clonar repositorio privado y proteger secretos
git clone https://github.com/<tu-usuario>/trading-bot-pro.git
cd trading-bot-pro
cp .env.example .env
cp config/strategy.example.json config/strategy.local.json
chmod 600 .env config/strategy.local.json

# 3. Levantar en segundo plano con Docker endurecido (reinicio automático + apagado <1s)
docker compose up -d --build
docker compose logs -f
```
- **Respaldo diario automático de SQLite (`crontab -e`):**
  ```cron
  5 17 * * 1-5 sqlite3 /ruta/trading-bot-pro/data/trading_audit.db ".backup '/ruta/trading-bot-pro/data/trading_audit_backup.db'"
  ```

---

## 3. Configuración para Iniciar con Micro-Capital Real ($100–$1,000 USD)

Debido a que las órdenes `GTC` (*Good-Til-Canceled*, obligatorias para que los stops protectores permanezcan activos de noche en Alpaca) requieren **acciones enteras (*whole shares*)** y el Guardián limita cada posición al **20% del capital total (`max_asset_exposure_pct = 0.20`)**:

| Capital Inicial | Tope por Posición (20%) | Precio Máximo por Acción (1 acc.) | Plantilla Recomendada |
| :---: | :---: | :---: | :--- |
| **\$150 – \$250 USD** | \$30 – \$50 USD | $\le \$50$ (`PFE`, `NKE`, `VZ`, `CSCO`...) | `config/strategy.micro_live.example.json` |
| **\$500 – \$1,000 USD** | \$100 – \$200 USD | $\le \$175$ (`KO`, `DIS`, `XOM`, `NVDA`, `WMT`, `TGT`...) | `config/strategy.micro_live.example.json` |
| **$\ge \$2,500$ USD** | $\ge \$500$ USD | Cualquier acción de las 31 (`AAPL`, `V`, `JNJ`, `AMD`...) | `config/strategy.local.json` (31 tickers) |

### Paso para activar Micro-Capital Real (solo tras completar Fase 4 en Paper)
1. Copiar la plantilla calibrada para micro-capital:
   ```bash
   cp config/strategy.micro_live.example.json config/strategy.local.json
   ```
2. En `.env`, cambiar a llaves Live y desactivar Paper:
   ```dotenv
   ALPACA_API_KEY=PK_LIVE_...
   ALPACA_SECRET_KEY=...
   ALPACA_PAPER=False
   ```
3. Verificar con una corrida en seco antes de enviar órdenes reales:
   ```bash
   .venv/bin/python main.py --dry-run --sync-account
   ```

---

## 4. Repositorio Público Limpio para tu Currículum (Protección de IP)
Mantén este repositorio en **Privado** y publica una copia limpia sin historial de commits previos:
```zsh
git clone --depth 1 https://github.com/<tu-usuario>/trading-bot-pro.git trading-bot-showcase
cd trading-bot-showcase
rm -rf .git
git init -b main
git add .
git commit -m "feat: algorithmic trading bot with deterministic risk guardian"
git remote add origin https://github.com/<tu-usuario>/trading-bot-showcase.git
git push -u origin main
```
