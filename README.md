# PTIT_NLP – BTL1: Phân loại SPAM/HAM bình luận sinh viên PTIT

Pipeline: tiền xử lý (PyVi) → BoW / TF-IDF → Naive Bayes / SVM → PhoBERT → Accuracy/P/R/F1/Confusion Matrix.
Phân công theo file `BTL NLP nhóm 2.xlsx` (sheet *Phân Công BTL1*).

> **Dữ liệu hiện tại là GIẢ LẬP** (`src/generate_synthetic.py`), kết quả 100% không có giá trị so sánh.
> Thay `data/raw/comments.csv` bằng dữ liệu thật (cột: `id,text,group,timestamp,label,source_url,annotator,review_status`; label ∈ HAM/SPAM) rồi chạy lại.

## Chạy
```
pip install -r requirements.txt
cd src
python generate_synthetic.py   # bỏ qua nếu đã có dữ liệu thật
python train_baseline.py       # 4 cấu hình NB/SVM x BoW/TF-IDF
python train_phobert.py        # fine-tune vinai/phobert-base (CPU ~ vài phút)
```
Kết quả: `results/`, hình: `figures/`, báo cáo: `report/BaoCao.md`.

| Thư mục/file | Nội dung |
|---|---|
| `src/preprocess.py` | chuẩn hóa, teencode, tách từ, stop words |
| `src/train_baseline.py` | pipeline sklearn + GridSearchCV |
| `src/train_phobert.py` | fine-tune PhoBERT |
| `data/raw/comments.csv` | dataset (hiện là synthetic) |
