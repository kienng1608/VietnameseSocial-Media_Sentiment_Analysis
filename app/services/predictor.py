"""
Predictor service — embedding + SVM classification.

Pipeline:
  1. Nhận text đã preprocessed
  2. Embed bằng SentenceTransformer (dangvantuan/vietnamese-document-embedding)
  3. Predict bằng SVM (đọc từ svm_model.pkl)
  4. Trả về label (-1, 0, 1) và sentiment string
"""
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import torch

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
MODEL_PATH = os.getenv("SVM_MODEL_PATH", "/app/models/svm_model.pkl")

LABEL_MAP = {
    1:  ("Tích cực", "😊"),
    0:  ("Trung lập", "😶"),
    -1: ("Tiêu cực", "😞"),
}


# ── Singleton Predictor ───────────────────────────────────────────────────────

class SentimentPredictor:
    """
    Lazy-load embedding model + SVM classifier.
    Loads một lần duy nhất khi startup.
    """
    _instance = None

    def __init__(self):
        self._embedding_model = None
        self._svm_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load(self) -> bool:
        """Load embedding model và SVM từ disk."""
        try:
            # 1. Load SentenceTransformer
            logger.info("Loading embedding model: %s", MODEL_NAME)
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._embedding_model = SentenceTransformer(
                MODEL_NAME,
                trust_remote_code=True,
                device=str(self._device),
            )
            self._embedding_model.eval()
            logger.info("Embedding model loaded on device: %s", self._device)

            # 2. Load SVM
            logger.info("Loading SVM from: %s", MODEL_PATH)
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(f"svm_model.pkl not found at {MODEL_PATH}")
            self._svm_model = joblib.load(MODEL_PATH)
            logger.info("SVM model loaded successfully")

            self._loaded = True
            return True

        except Exception as exc:
            logger.error("Failed to load predictor: %s", exc)
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _get_embedding(self, texts: list[str]) -> list:
        """Embed danh sách text → list vector."""
        with torch.no_grad():
            embeddings = self._embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return embeddings.tolist()

    def predict(self, preprocessed_text: str) -> tuple[int, str, str]:
        """
        Predict cảm xúc cho 1 câu đã preprocessed.

        Returns:
            (label, sentiment, emoji)
            label: -1 | 0 | 1
        """
        if not self._loaded:
            raise RuntimeError("Predictor not loaded. Call load() first.")

        embeddings = self._get_embedding([preprocessed_text])
        raw = self._svm_model.predict(embeddings)
        label = int(raw[0])

        sentiment, emoji = LABEL_MAP.get(label, (f"Không xác định ({label})", "🤔"))
        return label, sentiment, emoji


# Singleton instance
predictor = SentimentPredictor()
