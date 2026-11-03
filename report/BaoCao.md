# BÀI TẬP LỚN 1 – PHÂN LOẠI SPAM/HAM BÌNH LUẬN CỦA SINH VIÊN PTIT TRÊN FACEBOOK GROUP

Nhóm 2 – Môn Xử lý ngôn ngữ tự nhiên, PTIT

> **LƯU Ý QUAN TRỌNG:** Số liệu trong bản này được tính trên **dataset giả lập (synthetic)** do `src/generate_synthetic.py` sinh ra, vì dữ liệu thật từ các group Facebook chưa được thu thập. Các kết quả 100% bên dưới **không phản ánh hiệu năng thực tế**. Khi có dữ liệu thật, thay `data/raw/comments.csv`, chạy lại các script trong README và cập nhật mục 3.

## 1. Giới thiệu về bài toán

Các group Facebook của sinh viên PTIT (Cộng đồng sinh viên PTIT, Góc thông tin PTIT, D26 PTIT, PTIT Confession…) thường bị chèn bình luận rác: quảng cáo vay tiền, việc làm online, làm đồ án thuê, link lừa đảo. Bài toán: cho một bình luận tiếng Việt, phân loại thành **SPAM** (lớp dương) hoặc **HAM** (bình thường).

Mục tiêu: xây pipeline thu thập/gán nhãn → tiền xử lý → trích đặc trưng (BoW, TF-IDF) → huấn luyện Naive Bayes, SVM → so sánh với mô hình hiện đại PhoBERT → đánh giá bằng Accuracy, Precision, Recall, F1, Confusion Matrix.

Quy tắc gán nhãn (sheet `Dataset_Label`): SPAM là quảng cáo lặp lại, link không liên quan, bán hàng/dịch vụ trái ngữ cảnh, kéo tương tác, scam. Không gán SPAM chỉ vì bình luận ngắn, nhiều emoji, viết hoa hay teencode; ca khó đánh dấu REVIEW để hai thành viên kiểm tra chéo.

## 2. Cơ sở lý thuyết

**2.1 Tiền xử lý.** Chuẩn hóa Unicode NFC, lowercase; thay URL/số điện thoại/số/mention bằng token (`url`, `sdt`, `num`, `mention`) vì sự xuất hiện của chúng là tín hiệu spam; bỏ emoji, ký tự đặc biệt, rút gọn ký tự lặp; mở rộng teencode (ko→không, đc→được…); tách từ tiếng Việt bằng PyVi (“sinh viên”→“sinh_viên”); loại stop words (giữ lại các từ như *ib, link, giá, bán* vì mang tín hiệu spam).

**2.2 Bag of Words (BoW).** Văn bản biểu diễn bằng vector đếm tần suất từ trong từ điển, bỏ qua thứ tự từ.

**2.3 TF-IDF.** w(t,d) = tf(t,d) · idf(t), với idf(t) = ln((1+N)/(1+df(t))) + 1 (bản scikit-learn, có chuẩn hóa L2). Từ xuất hiện ở ít văn bản có trọng số cao hơn.

**2.4 Naive Bayes (Multinomial).** Giả định các từ độc lập có điều kiện theo lớp: P(c|d) ∝ P(c)·∏ P(t|c)^{n(t,d)}, với làm trơn Laplace/Lidstone tham số α: P(t|c) = (n_tc + α)/(n_c + α|V|).

**2.5 SVM tuyến tính.** Tìm siêu phẳng w·x + b = 0 cực đại hóa lề giữa hai lớp, tối thiểu hóa ½‖w‖² + C·Σ ξ_i (hinge loss). C điều khiển độ đánh đổi giữa lề và lỗi huấn luyện. SVM tuyến tính phù hợp với dữ liệu văn bản thưa, số chiều cao.

**2.6 PhoBERT.** Mô hình BERT (kiến trúc RoBERTa) tiền huấn luyện cho tiếng Việt trên ~20GB văn bản, dùng đầu vào đã tách từ. Fine-tune toàn bộ mô hình kèm một lớp phân loại trên token `<s>`.

**2.7 Chỉ số đánh giá.** Với SPAM là lớp dương: Accuracy = (TP+TN)/Tổng; Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R). Precision thấp nghĩa là chặn nhầm bình luận bình thường; Recall thấp nghĩa là để lọt spam.

## 3. Thực nghiệm và kết quả

**3.1 Dữ liệu.** Dataset giả lập, 762 bình luận sinh ra, còn **664** sau khi loại trùng (SPAM 44,6%). Schema: `id, text, group, timestamp, label, source_url, annotator, review_status`. Chia 80/20 phân tầng (seed 42): train 531, test 133 (74 HAM, 59 SPAM). Dữ liệu sinh từ ~60 mẫu câu nên các câu train/test rất giống nhau (xem hạn chế).

**3.2 Cấu hình.** Vectorizer nằm trong `sklearn.Pipeline` nên chỉ fit trên train (không rò rỉ dữ liệu). Tinh chỉnh bằng GridSearchCV 5-fold, tối ưu F1, chỉ trên tập train: `ngram_range` ∈ {(1,1),(1,2)}, `min_df` ∈ {1,2}, NB `alpha` ∈ {0.1,0.5,1}, SVM `C` ∈ {0.1,1,10}. Tham số tốt nhất ở `results/best_params.csv`. PhoBERT-base: 4 epoch, batch 16, lr 3e-5, max_len 64, chọn epoch theo F1 trên 10% validation tách từ train, chạy CPU.

**3.3 Kết quả trên tập test (133 mẫu, dữ liệu giả lập).**

| Mô hình | Đặc trưng | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP | CV-F1 (train) |
|---|---|---|---|---|---|---|---|---|---|---|
| Naive Bayes | BoW | 1.000 | 1.000 | 1.000 | 1.000 | 74 | 0 | 0 | 59 | 1.000 |
| Naive Bayes | TF-IDF | 1.000 | 1.000 | 1.000 | 1.000 | 74 | 0 | 0 | 59 | 1.000 |
| SVM | BoW | 1.000 | 1.000 | 1.000 | 1.000 | 74 | 0 | 0 | 59 | 0.989 |
| SVM | TF-IDF | 1.000 | 1.000 | 1.000 | 1.000 | 74 | 0 | 0 | 59 | 0.996 |
| PhoBERT | Tokenizer/Embedding | 1.000 | 1.000 | 1.000 | 1.000 | 74 | 0 | 0 | 59 | 1.000 (val) |

Confusion matrix: `figures/confusion_baseline.png`, `figures/confusion_phobert.png`. So sánh F1: `figures/f1_baseline.png`. Lịch sử huấn luyện PhoBERT: `results/phobert_history.csv` (train loss 0,299 → 0,001).

**3.4 Nhận xét và hạn chế.**
- Cả 5 cấu hình đạt 100% trên test, nên **không thể kết luận** BoW hay TF-IDF, NB hay SVM, truyền thống hay PhoBERT tốt hơn. Nguyên nhân: dữ liệu sinh từ mẫu câu lặp lại, tập test gần như trùng phân phối và nội dung với train; các ca “khó” (spam ngắn không link, ham có link/“ib”) vẫn thuộc cùng mẫu với train.
- Điểm duy nhất có chút phân biệt là CV-F1 trên train: SVM+BoW 0,989 < SVM+TF-IDF 0,996 < NB 1,000, chênh lệch quá nhỏ để có ý nghĩa.
- Phân tích lỗi (`results/errors_*.csv`) hiện rỗng vì không có dự đoán sai.
- Việc cần làm với dữ liệu thật: thu thập ≥ vài nghìn bình luận từ các group trong sheet, gán nhãn theo guideline, hai người gán độc lập một tập mẫu và đo Cohen's kappa, chạy lại pipeline, điền bảng 3.3, phân tích lỗi và thảo luận khác biệt giữa các phương pháp.

## Tài liệu tham khảo

1. Manning, Raghavan, Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008.
2. Jurafsky, Martin. *Speech and Language Processing* (3rd ed. draft), chương Naive Bayes và Text Classification.
3. Cortes, Vapnik. Support-vector networks. *Machine Learning*, 20(3), 1995.
4. Pedregosa et al. Scikit-learn: Machine Learning in Python. *JMLR* 12, 2011.
5. Nguyen, Nguyen. PhoBERT: Pre-trained language models for Vietnamese. *Findings of EMNLP*, 2020.
6. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*, 2019.
7. Trần Văn Đặng. PyVi – Vietnamese tokenizer. https://github.com/trungtv/pyvi
