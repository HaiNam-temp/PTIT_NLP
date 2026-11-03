# PTIT_NLP – BTL1: Phân loại SPAM/HAM bình luận Facebook của sinh viên PTIT

Notebook duy nhất: [`notebooks/BTL1_PTIT_SPAM_HAM_Kaggle.ipynb`](notebooks/BTL1_PTIT_SPAM_HAM_Kaggle.ipynb)
Pipeline: gán nhãn → tiền xử lý (PyVi) → BoW/TF-IDF → Naive Bayes/SVM → PhoBERT → Accuracy/P/R/F1/Confusion Matrix.
Báo cáo: [`report/BaoCao.md`](report/BaoCao.md). Phân công theo `BTL NLP nhóm 2.xlsx`.

## Chạy trên Kaggle
1. Kaggle → *Datasets → New Dataset* → upload `raw_fb.csv` (file **không** đưa lên GitHub vì là bình luận thật).
2. *Code → New Notebook → File → Import Notebook* → chọn file `.ipynb` ở trên.
3. *Add Data* → chọn dataset vừa tạo; *Settings*: **Internet On**, **Accelerator GPU T4**.
4. *Run All*. Kết quả ghi vào `/kaggle/working`: `metrics.csv`, `confusion_matrices.png`, `compare.png`, `errors.csv`, `review_candidates.csv`, `raw_fb_labeled.csv`.

Chạy local: đặt `raw_fb.csv` cạnh notebook (hoặc đặt biến môi trường `RAW_FB_PATH`), `pip install -r requirements.txt`.

## Lưu ý
- `raw_fb.csv` (2948 dòng: 500 bài + 2448 bình luận, 7 group) **chưa có nhãn**. Nhãn trong notebook là **nhãn yếu** (luật regex + duyệt tay): 68 SPAM, 2853 HAM, 27 REVIEW (loại khỏi huấn luyện). Cần 2 thành viên kiểm tra chéo trước khi nộp.
- Test chỉ có 11 SPAM nên điểm số dao động mạnh; xem phân tích trong báo cáo.
- `results/metrics.csv` và `figures/` là kết quả một lần chạy trên CPU (PhoBERT 4 epoch). Trên GPU nên tăng `EPOCHS` lên 8–10 cho PhoBERT.
