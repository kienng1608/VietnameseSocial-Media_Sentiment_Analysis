"""
Preprocessing service — tiền xử lý văn bản tiếng Việt.
Tách hoàn toàn khỏi layer business logic.

Pipeline:
  1. lowercase
  2. remove emoji
  3. remove similar repeated letters (haaaa → ha)
  4. remove punctuation
  5. remove whitespace
  6. VnCoreNLP word segmentation (câu lạc bộ → câu_lạc_bộ)
"""
import re
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Emoji pattern ────────────────────────────────────────────────────────────

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u200d"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\u3030"
    "\ufe0f"
    "]+",
    flags=re.UNICODE,
)

# ── Teen-code replacement dict ────────────────────────────────────────────────

TEEN_CODE_MAP = {
    ':v': 'hihi', '<3': 'yêu', '♥️': 'yêu', '❤': 'yêu',
    'ae': 'anh em', 'ah': 'à', 'ak': 'à',
    'bi': 'bị', 'bik': 'biết', 'bn': 'bạn', 'bro': 'anh em',
    'bt': 'bình thường', 'k': 'không', 'ok': 'được',
    'ko': 'không', 'kg': 'không', 'hk': 'không', 'hong': 'không',
    'dc': 'được', 'đc': 'được', 'dk': 'được',
    'e': 'em', 'mn': 'mọi người', 'mk': 'mình',
    'r': 'rồi', 'rui': 'rồi', 'j': 'gì',
    'ntn': 'như thế nào', 'nt': 'nhắn tin',
    'đt': 'điện thoại', 'dt': 'điện thoại',
    'ns': 'nói', 'nc': 'nói chuyện',
    'sp': 'sản phẩm', 'tg': 'thời gian',
}


# ── Core text functions ───────────────────────────────────────────────────────

def text_lowercase(text: str) -> str:
    return text.lower()


def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub(" ", text)


def remove_similar_letters(text: str) -> str:
    """hahaha → ha, aaaaa → a (case-insensitive)"""
    result = re.sub(
        r'([A-Z])\1+',
        lambda m: m.group(1),
        text,
        flags=re.IGNORECASE,
    )
    return result.lower()


def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', ' ', text)


def remove_whitespace(text: str) -> str:
    return " ".join(text.split())


def apply_teen_code(text: str) -> str:
    tokens = text.split()
    tokens = [TEEN_CODE_MAP.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


# ── VnCoreNLP wrapper ─────────────────────────────────────────────────────────

class VnCoreNLPWrapper:
    """
    Lazy-load VnCoreNLP segmenter.
    Chỉ khởi tạo 1 lần (singleton) để tránh tốn RAM.
    """
    _instance = None

    def __init__(self):
        self._segmenter = None
        self._loaded = False

    def load(self, save_dir: str = "/app/VnCoreNLP") -> bool:
        if self._loaded:
            return True
        try:
            import py_vncorenlp  # type: ignore
            self._segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"],
                save_dir=save_dir,
            )
            self._loaded = True
            logger.info("VnCoreNLP loaded successfully from %s", save_dir)
            return True
        except Exception as exc:
            logger.error("Failed to load VnCoreNLP: %s", exc)
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def word_segment(self, text: str) -> list[str]:
        if not self._loaded or self._segmenter is None:
            # Fallback: split by whitespace nếu VnCoreNLP chưa load
            return text.split()
        return self._segmenter.word_segment(text)


# Singleton instance
vncore = VnCoreNLPWrapper()


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess(text: str, use_teen_code: bool = False) -> str:
    """
    Tiền xử lý 1 câu văn bản.

    Args:
        text: câu gốc
        use_teen_code: bật/tắt bước chuyển teen-code (default: tắt)

    Returns:
        Văn bản đã qua preprocessing, dùng underscore cho từ ghép VnCoreNLP.
    """
    text = str(text)
    text = text_lowercase(text)
    text = remove_emoji(text)
    if use_teen_code:
        text = apply_teen_code(text)
    text = remove_similar_letters(text)
    text = remove_punctuation(text)
    text = remove_whitespace(text)

    # Word segmentation
    tokens = vncore.word_segment(text)
    text = " ".join(tokens)

    return text
