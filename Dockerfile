FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create non-root runtime user and persistent writable directories
RUN groupadd --system botgroup \
    && useradd --system --gid botgroup --create-home botuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY main.py ./
COPY config/strategy.example.json ./config/strategy.example.json

RUN pip install --upgrade pip \
    && pip install . \
    && mkdir -p /app/data /app/config \
    && chown -R botuser:botgroup /app

USER botuser

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import sqlite3; sqlite3.connect('/app/data/trading_audit.db').execute('SELECT 1')" || exit 1

CMD ["python", "main.py", "--loop", "--interval-seconds", "60", "--sync-account"]
