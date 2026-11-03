# BÀI TẬP LỚN 1 – PHÂN LOẠI SPAM/HAM BÌNH LUẬN CỦA SINH VIÊN PTIT TRÊN FACEBOOK GROUP

Nhóm 2 – Môn Xử lý ngôn ngữ tự nhiên, PTIT

> **Lưu ý:** dữ liệu `raw_fb.csv` chưa có nhãn; nhãn dùng ở đây là **nhãn yếu** (luật + duyệt tay) chưa được hai người gán độc lập kiểm chứng. Test chỉ có 11 mẫu SPAM nên các kết luận mang tính tham khảo.

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

**3.1 Dữ liệu.** `raw_fb.csv` gồm 2948 dòng (500 bài đăng, 2448 bình luận) từ 7 group/page: confessions.ptit (744), 2k5ptit (569), 584397217391365 (525), 1130813927934700 (416), sinhvienptithn (359), 1542129366737375 (274), ptit.onthi.aptis (61). `<USER>`, `<PHONE>` đã được ẩn danh; `author_hash` là hash. Cột `label` trống nên nhóm gán nhãn bằng luật regex (quảng cáo/chào mời khóa học Aptis, bán key/tài liệu, tuyển dụng của công ty, bài đăng quảng bá dạng page, bình luận lặp ≥3 lần có từ khóa chào mời) cộng danh sách ghi đè sau khi đọc tay các ứng viên. Kết quả: **SPAM 68, HAM 2853, REVIEW 27** (CLB tuyển thành viên, thông báo của trường; loại khỏi huấn luyện). Sau khi bỏ trùng/rỗng còn 2807 mẫu, tỉ lệ SPAM ≈ 2,4%, rất lệch lớp. SPAM tập trung ở các thread về Aptis (nhóm 1130813927934700: 31, ptit.onthi.aptis: 15).

**3.2 Chia dữ liệu và cấu hình.** Chia ~80/20 bằng `StratifiedGroupKFold` theo `post_id` để bình luận cùng bài không rơi vào cả train và test: train 2295 (45 SPAM), test 512 (11 SPAM). Vectorizer nằm trong `Pipeline` (chỉ fit trên train). Tinh chỉnh `GridSearchCV` 3-fold theo nhóm, tối ưu F1 lớp SPAM: `ngram_range` ∈ {(1,1),(1,2)}, `min_df` ∈ {1,2}, NB `alpha` ∈ {0.1,0.5,1}, SVM `C` ∈ {0.1,1,10}, SVM dùng `class_weight=balanced`. Tham số chọn được: NB+BoW (α=0,5; min_df=2; unigram), NB+TF-IDF (α=0,1; min_df=2; 1–2-gram), SVM+BoW (C=0,1; min_df=2; 1–2-gram), SVM+TF-IDF (C=1; min_df=1; 1–2-gram). PhoBERT-base: 4 epoch, batch 16, lr 3e-5, max_len 96, loss có trọng số lớp, chọn epoch theo F1 validation, chạy CPU.

**3.3 Kết quả trên tập test (512 mẫu, SPAM là lớp dương).**

| Mô hình | Đặc trưng | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP | CV-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Naive Bayes | BoW | 0.9824 | 0.5714 | 0.7273 | 0.6400 | 495 | 6 | 3 | 8 | 0.718 |
| Naive Bayes | TF-IDF | 0.9863 | 0.7500 | 0.5455 | 0.6316 | 499 | 2 | 5 | 6 | 0.702 |
| SVM | BoW | 0.9902 | 1.0000 | 0.5455 | 0.7059 | 501 | 0 | 5 | 6 | 0.722 |
| SVM | TF-IDF | **0.9922** | 1.0000 | 0.6364 | **0.7778** | 501 | 0 | 4 | 7 | 0.770 |
| PhoBERT | Tokenizer/Embedding | 0.9824 | 1.0000 | 0.1818 | 0.3077 | 501 | 0 | 9 | 2 | 0.400 (val) |

Hình: `figures/confusion_matrices.png`, `figures/compare.png`.

**3.4 Nhận xét.**
- Accuracy ≥ 98% ở mọi mô hình chỉ vì HAM chiếm ~97,6%; nên đọc F1/Recall của lớp SPAM.
- **SVM + TF-IDF tốt nhất** (F1 0,778; cũng cao nhất ở CV-F1 0,770), Precision 1,0 (không chặn nhầm). SVM nhìn chung có Precision cao hơn NB; NB (đặc biệt với BoW) có Recall cao hơn nhưng chặn nhầm 2–6 bình luận. TF-IDF nhỉnh hơn BoW với SVM, ngược lại với NB (chênh lệch 1–2 mẫu, không đủ kết luận).
- **PhoBERT kém nhất (F1 0,31)**: chỉ có ~45 mẫu SPAM để fine-tune nên mô hình nghiêng về HAM, F1 validation chỉ 0,40 sau 4 epoch và vẫn đang tăng. Kết quả này phản ánh dữ liệu ít và cấu hình chạy CPU, không phải PhoBERT kém hơn nói chung; cần thêm dữ liệu SPAM và nhiều epoch hơn (GPU).
- **Phân tích lỗi** (`errors.csv` trong notebook): lọt spam (FN) thường là spam ngầm (seeding kiểu “Cô … nhiệt tình 1000%…”), bài viết quảng bá nội dung dài như “APTIS WRITING PART 4”, hoặc bình luận rất ngắn “ib mình đang gom aptis nhé”. Một số “chặn nhầm” (FP) của NB thực ra là nội dung quảng bá mà luật gán nhãn đã bỏ sót (ví dụ “Em nhắn chị gửi tài liệu free nha”, “TOP 1 CHỨNG CHỈ SINH VIÊN PTIT…”), cho thấy **nhãn còn nhiễu** và số liệu có thể bị đánh giá thấp.
- **Hạn chế:** nhãn yếu (mô hình có thể chỉ học lại luật), test chỉ 11 SPAM (1 mẫu đổi F1 ~ vài điểm %), SPAM tập trung ở chủ đề Aptis nên chưa kiểm chứng tổng quát hóa. Cần: hai người gán độc lập một tập mẫu và tính Cohen's kappa, xử lý 27 dòng REVIEW, thu thêm SPAM, dùng cross-validation nhiều fold khi báo cáo.

## Tài liệu tham khảo

1. Manning, Raghavan, Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008.
2. Jurafsky, Martin. *Speech and Language Processing* (3rd ed. draft), chương Naive Bayes và Text Classification.
3. Cortes, Vapnik. Support-vector networks. *Machine Learning*, 20(3), 1995.
4. Pedregosa et al. Scikit-learn: Machine Learning in Python. *JMLR* 12, 2011.
5. Nguyen, Nguyen. PhoBERT: Pre-trained language models for Vietnamese. *Findings of EMNLP*, 2020.
6. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*, 2019.
7. Trần Văn Đặng. PyVi – Vietnamese tokenizer. https://github.com/trungtv/pyvi
