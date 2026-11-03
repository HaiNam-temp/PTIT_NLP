from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "comments.csv"
SEED = 42


def load_split():
    """Đọc dataset, bỏ rỗng/trùng, chia 80/20 stratified. Dùng chung cho mọi mô hình."""
    df = pd.read_csv(RAW)
    df = df[df.label.isin(["HAM", "SPAM"]) & df.text.notna()]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["y"] = (df.label == "SPAM").astype(int)          # SPAM = lớp dương
    tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=SEED)
    return df, tr.reset_index(drop=True), te.reset_index(drop=True)
