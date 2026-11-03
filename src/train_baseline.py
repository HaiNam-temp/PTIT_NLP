"""NB/SVM x BoW/TF-IDF. Vectorizer nằm TRONG Pipeline -> fit chỉ trên train (không leak)."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import ROOT, SEED, load_split
from preprocess import preprocess

df, tr, te = load_split()
print(f"Tổng {len(df)} | train {len(tr)} | test {len(te)} | SPAM% = {df.y.mean():.2%}")
Xtr = tr.text.map(preprocess)
Xte = te.text.map(preprocess)

VECS = {"BoW": CountVectorizer, "TF-IDF": TfidfVectorizer}
CLFS = {
    "Naive Bayes": (MultinomialNB(), {"clf__alpha": [0.1, 0.5, 1.0]}),
    "SVM": (LinearSVC(random_state=SEED), {"clf__C": [0.1, 1, 10]}),
}
GRID_VEC = {"vec__ngram_range": [(1, 1), (1, 2)], "vec__min_df": [1, 2]}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, params, cms = [], [], {}
for cname, (clf, cgrid) in CLFS.items():
    for vname, V in VECS.items():
        pipe = Pipeline([("vec", V()), ("clf", clf)])
        gs = GridSearchCV(pipe, {**GRID_VEC, **cgrid}, scoring="f1", cv=cv, n_jobs=-1).fit(Xtr, tr.y)
        pred = gs.predict(Xte)
        tn, fp, fn, tp = confusion_matrix(te.y, pred).ravel()
        rows.append(dict(Model=cname, Feature=vname, Accuracy=accuracy_score(te.y, pred),
                         Precision=precision_score(te.y, pred), Recall=recall_score(te.y, pred),
                         F1=f1_score(te.y, pred), TN=tn, FP=fp, FN=fn, TP=tp, CV_F1=gs.best_score_))
        params.append(dict(Model=cname, Feature=vname, **{k: str(v) for k, v in gs.best_params_.items()}))
        cms[(cname, vname)] = confusion_matrix(te.y, pred)
        joblib.dump(gs.best_estimator_, ROOT / "models" / f"{cname.replace(' ', '_')}_{vname}.joblib")
        # lỗi để phân tích
        te.assign(pred=pred, model=f"{cname}+{vname}")[te.y != pred].to_csv(
            ROOT / "results" / f"errors_{cname.replace(' ', '_')}_{vname}.csv", index=False, encoding="utf-8-sig")
        print(rows[-1])

res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "baseline_metrics.csv", index=False)
pd.DataFrame(params).to_csv(ROOT / "results" / "best_params.csv", index=False)
te[["id"]].to_csv(ROOT / "results" / "test_ids.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, ((c, v), cm) in zip(axes, cms.items()):
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
    ax.set_xticks([0, 1], ["HAM", "SPAM"]); ax.set_yticks([0, 1], ["HAM", "SPAM"])
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(f"{c} + {v}")
plt.tight_layout(); plt.savefig(ROOT / "figures" / "confusion_baseline.png", dpi=150)

ax = res.pivot(index="Model", columns="Feature", values="F1").plot.bar(rot=0, figsize=(5, 3.5), ylim=(0, 1.05))
ax.set_ylabel("F1 (SPAM)"); plt.tight_layout(); plt.savefig(ROOT / "figures" / "f1_baseline.png", dpi=150)
print(res.round(4).to_string())
