FROM python:3.11-slim

# ── System dependencies (required for TA-Lib, numpy, torch) ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libta-lib-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ── TA-Lib C library (ta-lib-python requires this) ───────────────────────
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────
COPY . .

# ── Runtime ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kolkata

CMD ["python", "main.py"]
