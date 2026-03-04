"""
Labeling script — dùng Gemini API để tự động label comments tiếng Việt.

Sử dụng:
  python scripts/label_data.py --input data/raw_comments.csv --output data/labeled_data.csv

Labels:
  -1 = Tiêu cực (toxic/toxic)
   0 = Trung lập (neutral)
   1 = Tích cực (positive)
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
BATCH_SIZE = int(os.getenv("LABEL_BATCH_SIZE", "50"))
SLEEP_BETWEEN_BATCHES = float(os.getenv("LABEL_SLEEP", "2.0"))  # seconds


# ── Gemini helper ──────────────────────────────────────────────────────────────

LABEL_PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích cảm xúc bình luận bóng đá tiếng Việt.
Hãy phân loại từng bình luận dưới đây theo 3 nhãn:
  -1 = Tiêu cực (xúc phạm, hate speech, toxic)
   0 = Trung lập (nhận xét bình thường, phân tích)
   1 = Tích cực (khen ngợi, cổ vũ, hâm mộ)

Trả về ĐÚNG định dạng JSON array (không có text khác):
[{{"id": 0, "label": -1}}, {{"id": 1, "label": 0}}, ...]

Danh sách bình luận:
{comments_json}
"""


def call_gemini(comments: list[str]) -> list[int]:
    """Gọi Gemini API để label 1 batch comments."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY chưa được set. Xem file .env.example")

    indexed = [{"id": i, "text": c} for i, c in enumerate(comments)]
    prompt = LABEL_PROMPT_TEMPLATE.format(
        comments_json=json.dumps(indexed, ensure_ascii=False, indent=2)
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }

    resp = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()

    raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    # Clean: bỏ markdown code block nếu có
    raw_text = raw_text.strip().strip("```json").strip("```").strip()

    results = json.loads(raw_text)
    # Sort theo id để đảm bảo thứ tự
    results.sort(key=lambda x: x["id"])
    return [r["label"] for r in results]


# ── Main labeling pipeline ─────────────────────────────────────────────────────

def label_dataset(input_path: str, output_path: str, text_col: str = "text") -> None:
    """
    Đọc CSV input, label từng batch bằng Gemini, lưu CSV output.

    Args:
        input_path: đường dẫn CSV input (phải có cột text_col)
        output_path: đường dẫn CSV output (thêm cột 'label')
        text_col: tên cột chứa text
    """
    logger.info("Reading input: %s", input_path)
    df = pd.read_csv(input_path, encoding="utf-8")

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {df.columns.tolist()}")

    logger.info("Total comments: %d", len(df))

    # Nếu đã có file output (resume sau lỗi)
    output_p = Path(output_path)
    if output_p.exists():
        done_df = pd.read_csv(output_path, encoding="utf-8")
        start_idx = len(done_df)
        logger.info("Resuming from index %d (already labeled: %d)", start_idx, start_idx)
        df_remaining = df.iloc[start_idx:].reset_index(drop=True)
        labels_done = done_df["label"].tolist()
    else:
        df_remaining = df.copy()
        labels_done = []
        start_idx = 0

    all_labels = labels_done.copy()
    total = len(df_remaining)
    errors = 0

    for i in range(0, total, BATCH_SIZE):
        batch = df_remaining[text_col].iloc[i: i + BATCH_SIZE].tolist()
        batch_num = (start_idx + i) // BATCH_SIZE + 1
        logger.info(
            "Batch %d | rows %d-%d / %d",
            batch_num, start_idx + i, start_idx + i + len(batch) - 1, len(df),
        )

        try:
            labels = call_gemini(batch)
            if len(labels) != len(batch):
                raise ValueError(f"Expected {len(batch)} labels, got {len(labels)}")
            all_labels.extend(labels)
            errors = 0  # reset consecutive error count

        except Exception as exc:
            logger.error("Batch %d failed: %s — using label 0 as fallback", batch_num, exc)
            all_labels.extend([0] * len(batch))  # fallback: neutral
            errors += 1
            if errors >= 3:
                logger.error("3 consecutive errors — stopping. Save progress so far.")
                break

        # Lưu progress sau mỗi batch (checkpoint)
        df_partial = df.iloc[: len(all_labels)].copy()
        df_partial["label"] = all_labels
        df_partial.to_csv(output_path, index=False, encoding="utf-8")
        logger.info("Checkpoint saved: %d / %d", len(all_labels), len(df))

        if i + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    logger.info("Labeling done! Output: %s", output_path)
    logger.info(
        "Label distribution: %s",
        pd.Series(all_labels).value_counts().to_dict(),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tự động label dataset cảm xúc bằng Gemini API"
    )
    parser.add_argument(
        "--input", required=True,
        help="Đường dẫn CSV input (cần có cột 'text')",
    )
    parser.add_argument(
        "--output", required=True,
        help="Đường dẫn CSV output",
    )
    parser.add_argument(
        "--text-col", default="text",
        help="Tên cột chứa văn bản (default: 'text')",
    )
    args = parser.parse_args()

    label_dataset(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
    )


if __name__ == "__main__":
    main()
