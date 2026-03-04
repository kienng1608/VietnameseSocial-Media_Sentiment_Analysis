"""
Router layer — định nghĩa tất cả endpoints của Sentiment Analysis API.

Endpoints:
  POST /predict      → phân tích cảm xúc 1 câu
  GET  /health       → check trạng thái app + model
"""
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.predictor import predictor
from app.services.preprocessor import vncore, preprocess

logger = logging.getLogger(__name__)

router = APIRouter()


# ── POST /predict ──────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Phân tích cảm xúc",
    description=(
        "Nhận 1 câu văn bản tiếng Việt, tiền xử lý rồi predict cảm xúc "
        "(-1: tiêu cực, 0: trung lập, 1: tích cực)."
    ),
    responses={
        422: {"description": "Validation error — text rỗng hoặc quá dài"},
        503: {"model": ErrorResponse, "description": "Model chưa sẵn sàng"},
        500: {"model": ErrorResponse, "description": "Lỗi server"},
    },
)
async def predict_sentiment(body: PredictRequest) -> PredictResponse:
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model chưa được load. Vui lòng thử lại sau.",
        )

    try:
        preprocessed = preprocess(body.text)
        label, sentiment, emoji = predictor.predict(preprocessed)

        return PredictResponse(
            text=body.text,
            text_preprocessed=preprocessed,
            label=label,
            sentiment=sentiment,
            emoji=emoji,
            processed_at=datetime.utcnow(),
        )

    except Exception as exc:
        logger.exception("Prediction failed for input: %r", body.text)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình phân tích: {exc}",
        ) from exc


# ── GET /health ────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Kiểm tra trạng thái của app, embedding model và VnCoreNLP.",
)
async def health_check() -> HealthResponse:
    model_ok = predictor.is_loaded
    vncore_ok = vncore.is_loaded

    overall = "ok" if (model_ok and vncore_ok) else "degraded"
    msg = None
    if not model_ok:
        msg = "Embedding model hoặc SVM chưa load xong."
    elif not vncore_ok:
        msg = "VnCoreNLP chưa load — preprocessing sẽ dùng fallback (split by space)."

    return HealthResponse(
        status=overall,
        model_loaded=model_ok,
        vncore_loaded=vncore_ok,
        message=msg,
    )
