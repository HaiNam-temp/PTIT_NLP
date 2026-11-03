"""Tiền xử lý văn bản tiếng Việt: chuẩn hóa, làm sạch, tách từ, stop words, teencode."""
import re
import unicodedata

from pyvi import ViTokenizer

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
PHONE_RE = re.compile(r"(?<!\d)(0\d{9,10})(?!\d)")
NUM_RE = re.compile(r"\d+")
# emoji + ký tự đặc biệt (giữ chữ cái tiếng Việt, số, khoảng trắng và token đặc biệt)
NON_WORD_RE = re.compile(r"[^\w\s_]", re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1{2,}")

TEENCODE = {"ko": "không", "k": "không", "đc": "được", "dc": "được", "bn": "bạn", "vs": "với",
            "mk": "mình", "mik": "mình", "bt": "biết", "mn": "mọi người", "ntn": "như thế nào",
            "j": "gì", "r": "rồi", "ik": "đi", "cx": "cũng", "ok": "ok", "sv": "sinh viên",
            "ctv": "cộng tác viên", "tl": "trả lời", "thik": "thích", "trc": "trước"}

# Stop words tối giản. KHÔNG loại các từ mang tín hiệu spam: ib, inbox, link, giá, bán, mua...
STOPWORDS = set("""và là của có những các một này đó ở thì mà cũng đã sẽ đang bị được cho với
 để trong khi như nhưng nên vì nếu rằng thế vậy ạ nhé nha nhỉ hả ơi à ừ ờ thôi lại rất
 quá cả mỗi từ tại về ra vào lên xuống nào ai gì đâu sao""".split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" mention ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = PHONE_RE.sub(" sdt ", text)
    text = NUM_RE.sub(" num ", text)
    text = REPEAT_RE.sub(r"\1", text)          # "ạạạ" -> "ạ"
    text = NON_WORD_RE.sub(" ", text)           # bỏ emoji, dấu câu
    return re.sub(r"\s+", " ", text).strip()


def expand_teencode(text: str) -> str:
    return " ".join(TEENCODE.get(w, w) for w in text.split())


def segment(text: str) -> str:
    return ViTokenizer.tokenize(text)           # "sinh viên" -> "sinh_viên"


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in STOPWORDS)


def clean_for_bert(text: str) -> str:
    """Cho PhoBERT: chuẩn hóa nhẹ + tách từ, GIỮ stop words (BERT cần ngữ cảnh)."""
    return segment(expand_teencode(normalize(text)))


def preprocess(text: str) -> str:
    """Cho BoW/TF-IDF: chuẩn hóa -> teencode -> tách từ -> bỏ stop words."""
    return remove_stopwords(segment(expand_teencode(normalize(text))))


if __name__ == "__main__":
    s = "MN ơi!!! vào http://bit.ly/abc xem sale 50% ib 0912345678 nha 💥💥"
    print(preprocess(s))
