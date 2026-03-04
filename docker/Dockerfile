# ── Stage 1: Builder ────────────────────────────────────────────────────────────
# Dùng python:3.11-slim thay vì :latest → tránh lỗi #1 (base image không ổn định)
FROM python:3.11-slim AS builder

WORKDIR /build

# Cài system dependencies cho Java (VnCoreNLP cần Java) và build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước, tận dụng Docker layer cache
# (chỉ rebuild layer này khi requirements.txt thay đổi)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download VnCoreNLP (chỉ cần 1 lần, cache vào layer)
RUN git clone --depth 1 https://github.com/vncorenlp/VnCoreNLP.git /build/VnCoreNLP


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Java runtime (cần để chạy VnCoreNLP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — tránh lỗi #6 (chạy container với root)
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy Python packages từ builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy VnCoreNLP
COPY --from=builder /build/VnCoreNLP /app/VnCoreNLP

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Thư mục cho SVM model (mount từ host hoặc COPY thêm)
RUN mkdir -p /app/models

# Chuyển ownership sang non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Health check — Docker sẽ restart container nếu fail
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Dùng exec form (không dùng shell form) để forward signals đúng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
