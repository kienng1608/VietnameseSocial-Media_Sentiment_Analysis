# Vietnamese Sentiment Analysis API

**Phân loại cảm xúc bình luận bóng đá tiếng Việt (Ronaldo/Messi)**

## Tính Năng

- 🎯 **POST /api/v1/predict** — nhận text → trả label (-1/0/1) + sentiment
- ❤️ **GET /api/v1/health** — kiểm tra trạng thái model + VnCoreNLP
- 📊 **Swagger UI** tại `/docs` — test API trực tiếp trên browser
- 🐳 **Docker ready** — 1 lệnh deploy toàn bộ

## Pipeline

```
[Raw Text]
    ↓ VnCoreNLP word segmentation
[Preprocessed Text]  
    ↓ dangvantuan/vietnamese-document-embedding
[768-dim Embedding]
    ↓ SVM Classifier
[Label: -1 / 0 / 1]
```

## Cấu Trúc Project

```
fastapi_docker_compose/
├── app/
│   ├── main.py                 # FastAPI entry point (lifespan, CORS, router)
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── routers/
│   │   └── api.py              # Endpoints: /predict, /health
│   └── services/
│       ├── preprocessor.py     # VnCoreNLP wrapper + text functions
│       └── predictor.py        # SentenceTransformer + SVM
├── scripts/
│   └── label_data.py           # Labeling tool (Gemini API)
├── models/                     # Đặt svm_model.pkl ở đây (không commit)
├── data/                       # Raw/processed data (không commit)
├── docker/
│   ├── Dockerfile              # Multi-stage build
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── .dockerignore
```

## Setup Nhanh

### 1. Chuẩn bị model
```bash
# Đặt file svm_model.pkl từ Colab vào thư mục models/
mkdir -p models
cp /path/to/svm_model.pkl models/
```

### 2. Tạo file .env
```bash
cp .env.example .env
# Mở .env và điền GEMINI_API_KEY (nếu dùng labeling script)
```

### 3. Build & Run
```bash
cd docker
docker-compose up --build
```

### 4. Test API
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Messi đá hay quá, fan thực sự tự hào!"}'
```

### 5. Swagger UI
Mở browser: http://localhost:8000/docs

## Labeling Script

```bash
# Label dataset mới bằng Gemini API
python scripts/label_data.py \
  --input data/raw_comments.csv \
  --output data/labeled_data.csv \
  --text-col text
```

## Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/api/v1/predict` | Phân tích cảm xúc 1 câu |
| `GET`  | `/api/v1/health`  | Health check |
| `GET`  | `/docs`           | Swagger UI |

## Labels

| Label | Sentiment | Emoji |
|-------|-----------|-------|
| `1`   | Tích cực  | 😊    |
| `0`   | Trung lập | 😶    |
| `-1`  | Tiêu cực  | 😞    |

## Tránh 8 Lỗi Phổ Biến

Xem chi tiết: [AVOIDANCE_TABLE.md](AVOIDANCE_TABLE.md)

| # | Lỗi | Giải pháp |
|---|-----|-----------|
| 1 | Base image `python:latest` | Dùng `python:3.11-slim` |
| 2 | `.env` commit lên git | `.gitignore` + `.env.example` |
| 3 | Không có restart policy | `restart: unless-stopped` |
| 4 | Hardcode secrets | Đọc từ `.env` file |
| 5 | Không có health check | `HEALTHCHECK` + `healthcheck:` |
| 6 | Run container với root | `USER appuser` (non-root) |
| 7 | CORS wildcard `*` | `ALLOWED_ORIGINS` cụ thể |
| 8 | Floating dependencies | Pinned versions trong `requirements.txt` |
