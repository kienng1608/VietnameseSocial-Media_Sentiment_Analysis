"""
Pydantic schemas — validate input/output cho tất cả endpoints.
FastAPI tự động trả 422 Unprocessable Entity khi dữ liệu không hợp lệ.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Request ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Văn bản tiếng Việt cần phân tích cảm xúc",
        examples=["Messi đá hay quá, fan thực sự tự hào!"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {"text": "Messi đá hay quá, fan thực sự tự hào!"}
        }
    }


# ── Response ──────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    text: str = Field(..., description="Văn bản gốc")
    text_preprocessed: str = Field(..., description="Văn bản sau tiền xử lý")
    label: int = Field(..., description="Nhãn: -1 (tiêu cực), 0 (trung lập), 1 (tích cực)")
    sentiment: str = Field(..., description="Cảm xúc bằng chữ")
    emoji: str = Field(..., description="Emoji tương ứng cảm xúc")
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vncore_loaded: bool
    version: str = "1.0.0"
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
