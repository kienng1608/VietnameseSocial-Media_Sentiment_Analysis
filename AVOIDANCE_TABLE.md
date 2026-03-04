# AVOIDANCE_TABLE.md — Chứng Minh Tránh 8/8 Lỗi Phổ Biến

> Chứng minh đã tránh toàn bộ **8 lỗi phổ biến** được liệt kê trong tài liệu Buổi 1.

---

## Lỗi #1 — Base image `python:latest`

**Vấn đề:** `python:latest` thay đổi theo thời gian → build không reproducible, có thể break.

**Cách xử lý:** Dùng `python:3.11-slim` (pin cả major.minor).

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim AS builder
```

---

## Lỗi #2 — Commit file `.env` lên git

**Vấn đề:** Commit secret (API key, password) lên repo → lộ thông tin.

**Cách xử lý:** `.gitignore` block `.env`, chỉ commit `.env.example`.

```gitignore
# .gitignore
.env
.env.local
.env.*.local
```

---

## Lỗi #3 — Không có restart policy

**Vấn đề:** Container crash không tự khởi động lại → downtime.

**Cách xử lý:** `restart: unless-stopped` trong docker-compose.

```yaml
# docker/docker-compose.yml
services:
  api:
    restart: unless-stopped
```

---

## Lỗi #4 — Hardcode secrets trong code hoặc Dockerfile

**Vấn đề:** API key, password hardcode → lộ trong git history.

**Cách xử lý:** Đọc từ environment variable, load qua `.env` file.

```yaml
# docker-compose.yml
env_file:
  - ../.env
environment:
  - GEMINI_API_KEY=${GEMINI_API_KEY}
```

---

## Lỗi #5 — Không có health check

**Vấn đề:** Docker/orchestrator không biết app đã sẵn sàng hay chưa.

**Cách xử lý:** `HEALTHCHECK` trong Dockerfile + `healthcheck:` trong compose.

```dockerfile
# docker/Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
```

```yaml
# docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
  start_period: 120s  # Cho đủ thời gian load model
```

---

## Lỗi #6 — Chạy container với root user

**Vấn đề:** Root trong container = root trên host nếu escape → security risk.

**Cách xử lý:** Tạo `appuser` non-root.

```dockerfile
# docker/Dockerfile
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser
```

---

## Lỗi #7 — CORS wildcard `*`

**Vấn đề:** Allow tất cả origin → CSRF risk trong môi trường production.

**Cách xử lý:** Chỉ cho phép origin cụ thể, config qua env var.

```python
# app/main.py
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Không dùng ["*"]
    ...
)
```

---

## Lỗi #8 — Floating dependency versions

**Vấn đề:** `pip install fastapi` → version khác nhau mỗi lần build → không reproducible.

**Cách xử lý:** Pin tất cả versions trong `requirements.txt`.

```txt
# requirements.txt
fastapi==0.115.5
uvicorn[standard]==0.32.0
pydantic==2.10.3
sentence-transformers==3.4.1
torch==2.2.2
...
```

---

## Tổng kết

| # | Lỗi | Giải pháp | File |
|---|-----|-----------|------|
| 1 | Base image `latest` | `python:3.11-slim` | `Dockerfile` |
| 2 | Commit `.env` | `.gitignore` | `.gitignore` |
| 3 | Không có restart | `unless-stopped` | `docker-compose.yml` |
| 4 | Hardcode secrets | Env vars + `.env` | `docker-compose.yml` |
| 5 | Không health check | `HEALTHCHECK` + `healthcheck:` | `Dockerfile`, `docker-compose.yml` |
| 6 | Root user | `USER appuser` | `Dockerfile` |
| 7 | CORS wildcard | `ALLOWED_ORIGINS` cụ thể | `app/main.py` |
| 8 | Floating deps | Pinned versions | `requirements.txt` |
