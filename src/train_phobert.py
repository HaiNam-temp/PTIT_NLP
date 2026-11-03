"""Fine-tune PhoBERT-base cho SPAM/HAM (cùng split với baseline)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ROOT, SEED, load_split
from preprocess import clean_for_bert

MODEL = "vinai/phobert-base"
EPOCHS, BS, LR, MAXLEN = 4, 16, 3e-5, 64
torch.manual_seed(SEED); np.random.seed(SEED)

_, tr, te = load_split()
tr, va = train_test_split(tr, test_size=0.1, stratify=tr.y, random_state=SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)


def enc(d):
    e = tok(list(d.text.map(clean_for_bert)), truncation=True, max_length=MAXLEN,
            padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"], torch.tensor(d.y.values)


def predict(d):
    ids, mask, _ = enc(d)
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(ids), 32):
            out.append(model(input_ids=ids[i:i+32], attention_mask=mask[i:i+32]).logits.argmax(-1))
    return torch.cat(out).numpy()


ids, mask, y = enc(tr)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
hist, best, best_state = [], -1, None
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(ids)); tot = 0
    for i in range(0, len(ids), BS):
        b = perm[i:i+BS]
        loss = model(input_ids=ids[b], attention_mask=mask[b], labels=y[b]).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(); tot += loss.item() * len(b)
    vf1 = f1_score(va.y, predict(va))
    hist.append(dict(epoch=ep + 1, train_loss=tot / len(ids), val_f1=vf1)); print(hist[-1]); sys.stdout.flush()
    if vf1 > best:
        best = vf1; best_state = {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
pd.DataFrame(hist).to_csv(ROOT / "results" / "phobert_history.csv", index=False)

pred = predict(te)
tn, fp, fn, tp = confusion_matrix(te.y, pred).ravel()
res = dict(Model="PhoBERT", Feature="Tokenizer/Embedding", Accuracy=accuracy_score(te.y, pred),
           Precision=precision_score(te.y, pred), Recall=recall_score(te.y, pred), F1=f1_score(te.y, pred),
           TN=tn, FP=fp, FN=fn, TP=tp, CV_F1=best)
pd.DataFrame([res]).to_csv(ROOT / "results" / "phobert_metrics.csv", index=False)
te.assign(pred=pred)[te.y != pred].to_csv(ROOT / "results" / "errors_PhoBERT.csv", index=False, encoding="utf-8-sig")
cm = confusion_matrix(te.y, pred)
fig, ax = plt.subplots(figsize=(3.8, 3.6)); ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1], ["HAM", "SPAM"]); ax.set_yticks([0, 1], ["HAM", "SPAM"])
ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title("PhoBERT")
plt.tight_layout(); plt.savefig(ROOT / "figures" / "confusion_phobert.png", dpi=150)
print(res)
