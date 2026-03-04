"""
FastAPI application entry point.

Chứa:
- Lifespan: load VnCoreNLP + model khi startup, cleanup khi shutdown
- CORS: cho phép origin cụ thể (không dùng wildcard *)
- Router inclusion với prefix /api/v1
- Metadata đầy đủ cho Swagger UI
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.api import router
from app.services.preprocessor import vncore
from app.services.predictor import predictor

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load tất cả heavy resources khi startup, giải phóng khi shutdown."""
    logger.info("=== Startup: loading VnCoreNLP ===")
    vncore_dir = os.getenv("VNCORENLP_DIR", "/app/VnCoreNLP")
    ok_vncore = vncore.load(save_dir=vncore_dir)
    if not ok_vncore:
        logger.warning("VnCoreNLP failed to load — preprocessing will use fallback")

    logger.info("=== Startup: loading SentenceTransformer + SVM ===")
    ok_model = predictor.load()
    if not ok_model:
        logger.warning("Predictor failed to load — /predict will return 503")

    logger.info("=== App ready ===")
    yield

    # Cleanup (nếu cần)
    logger.info("=== Shutdown ===")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vietnamese Sentiment Analysis API",
    description=(
        "API phân loại cảm xúc bình luận bóng đá tiếng Việt.\n\n"
        "**Pipeline:** VnCoreNLP preprocessing → Vietnamese Document Embedding "
        "(dangvantuan) → SVM classifier\n\n"
        "**Labels:** `-1` (tiêu cực) · `0` (trung lập) · `1` (tích cực)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Không dùng wildcard * — chỉ cho phép origin cụ thể (tránh lỗi bảo mật)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost,http://localhost:3000,http://localhost:8080",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1", tags=["Sentiment"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Vietnamese Sentiment Analysis API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
