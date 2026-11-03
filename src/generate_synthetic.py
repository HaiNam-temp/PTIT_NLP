"""Sinh dataset BÌNH LUẬN GIẢ LẬP (synthetic) SPAM/HAM mô phỏng group sinh viên PTIT.

CẢNH BÁO: đây KHÔNG phải dữ liệu thật. Dùng để chạy thử pipeline. Thay file
data/raw/comments.csv bằng dữ liệu thật (cùng schema) rồi chạy lại toàn bộ.
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

random.seed(42)
GROUPS = ["Cộng đồng sinh viên PTIT", "Góc thông tin PTIT", "D26 PTIT", "PTIT Confession",
          "D26 Học Viện Công Nghệ Bưu Chính Viễn Thông"]

MON = ["giải tích 2", "vật lý 1", "cấu trúc dữ liệu", "xử lý ngôn ngữ tự nhiên", "mạng máy tính",
       "toán rời rạc", "tiếng anh", "lập trình hướng đối tượng", "cơ sở dữ liệu", "xác suất thống kê"]
NOI = ["hội trường", "phòng 201 nhà A2", "giảng đường 2", "thư viện", "căng tin", "ký túc xá Hà Đông",
       "nhà xe", "phòng máy", "sân bóng"]
GV = ["thầy Nam", "cô Lan", "thầy Hùng", "cô Hoa", "thầy Minh"]
EMO = ["", "", "", " 😂", " 😭", " 🥲", " ❤️", " 🙏", " :))", " =))"]
TEEN = [("không", "ko"), ("được", "đc"), ("bạn", "bn"), ("với", "vs"), ("mình", "mk"), ("biết", "bt")]

HAM = [
    "có ai học {m} khóa trước cho mình xin tài liệu với",
    "lịch thi {m} năm nay thi ở {n} đúng không mọi người",
    "{g} dạy {m} có dễ qua môn không mọi người ơi",
    "mai có ai đi học {m} ở {n} không, mình đi chung với",
    "bạn nào nhặt được thẻ sinh viên ở {n} thì cho mình xin lại với ạ",
    "cho mình hỏi đăng ký học phần đợt 2 mở lúc mấy giờ vậy",
    "hôm nay {n} đóng cửa sớm hả mọi người",
    "cảm ơn bạn nhé, mình nộp bài đúng hạn rồi",
    "deadline bài tập lớn {m} là thứ mấy vậy các bạn",
    "mình thấy {g} chấm điểm {m} khá công bằng, bài giữa kỳ cũng ổn",
    "ai biết quy định miễn học tiếng anh đầu ra không chỉ mình với",
    "tuần sau có buổi giao lưu CLB lập trình ở {n}, ai đi không",
    "ký túc xá mình ở dạo này mất điện liên tục, có ai bị giống mình ko",
    "xin kinh nghiệm thực tập mùa hè cho sinh viên năm 3 với ạ",
    "cho mình xin đề cương ôn tập {m} với, thi sát rồi mà chưa học gì",
    "bạn ơi cho mình hỏi học bổng kỳ này công bố khi nào",
    "hôm qua thi {m} khó quá, câu cuối mình không làm được",
    "thanh lý giáo trình {m} cũ cho bạn nào cần, giá sinh viên, ib mình nhé",
    "mình đang tìm bạn cùng nhóm làm bài tập lớn {m}, ai rảnh nhắn mình",
    "link đăng ký thi chứng chỉ tiếng anh của trường đây mọi người: https://qldt.ptit.edu.vn",
    "tối nay có trận bóng giao hữu ở {n}, mọi người qua cổ vũ nha",
    "xe buýt tuyến 02 chạy tới mấy giờ vậy mọi người",
    "tham gia group học nhóm {m} đi các bạn, mình lập rồi nè",
    "mình mới nhập học, có ai chỉ mình cách vào wifi trường với",
    "chúc các bạn thi tốt nhé, cố lên mọi người",
    "ai có slide {m} tuần 5 cho mình xin với, hôm đó mình nghỉ ốm",
    "đồng ý với bạn trên, bài thực hành {m} khá nhiều nhưng học được nhiều thứ",
    "confession: crush mình học cùng lớp {m} mà chưa dám nói",
    "bao giờ trường có lịch nghỉ tết chính thức vậy ạ",
]
SPAM = [
    "💥💥 KIẾM TIỀN ONLINE tại nhà 500k-1tr/ngày, không cần vốn, ib ngay zalo {p} để nhận việc 💥💥",
    "Bạn cần vay tiền nhanh? Duyệt trong 5 phút, không thế chấp, lãi thấp. Liên hệ {p}",
    "Nhận làm bài tập lớn, đồ án, luận văn thuê giá rẻ uy tín, bao qua môn. Zalo {p}",
    "SALE SỐC iPhone 15 giảm 50% chỉ hôm nay, click ngay http://bit.ly/{c} để mua 🔥🔥",
    "Tuyển CTV bán hàng online hoa hồng cao 30%, ai quan tâm inbox mình nhé {p}",
    "Hack điểm thi, sửa điểm học bạ, uy tín 100%, liên hệ {p} bảo mật tuyệt đối",
    "Follow page mình để nhận quà miễn phí nha mọi người, like share comment nhận ngay voucher 100k",
    "Cho thuê trọ giá rẻ gần trường chỉ 1tr5, view đẹp, ib em {p} xem phòng ngay ạ",
    "Chúc mừng bạn đã trúng thưởng 50 triệu! Nhấn vào link http://win-{c}.xyz để nhận thưởng",
    "Đầu tư tiền ảo lãi 20%/tuần cam kết lợi nhuận, vào nhóm {c} ngay để được hướng dẫn",
    "Bán tài khoản netflix spotify giá rẻ chỉ 30k/tháng, bảo hành đầy đủ, nhắn {p}",
    "tặng 100 suất học bổng du học giả, đăng ký tại http://hocbong-{c}.top nhanh kẻo hết",
    "Thuốc giảm cân cấp tốc 7 ngày giảm 5kg, không tác dụng phụ, đặt hàng {p}",
    "Kết bạn làm quen gái xinh khu vực Hà Nội, xem ảnh nóng tại http://{c}.vip",
    "🔥🔥 Shop thời trang sinh viên SALE 70% toàn bộ, freeship, vào xem ngay https://shop-{c}.com 🔥🔥",
    "Cần tuyển gấp nhân viên part time lương 15tr/tháng, chỉ cần có điện thoại, liên hệ {p}",
    "Dịch vụ tăng like tăng follow facebook tiktok giá rẻ, uy tín, nhắn {p}",
    "Học tiếng anh giao tiếp cam kết đầu ra 7.0 sau 1 tháng, đăng ký ngay {p} giảm 60%",
    "ADMIN ƠI DUYỆT BÀI GIÚP MÌNH, bán sim số đẹp giá rẻ, ib {p} ạ ạ ạ",
    "mn ơi vào link này xem video hot nhé {c} http://xem-{c}.net",
]
# Spam "khó": ngắn, teencode, không có link; Ham "khó": có link/giá/ib
SPAM_HARD = [
    "ib mk nhận việc nhẹ lương cao nhé", "ai cần vay ib em hỗ trợ lãi thấp ạ", "nhận đồ án giá rẻ ib em nhé",
    "bán acc giá rẻ ai mua ib", "cần tuyển ctv ib ngay", "kiếm tiền tại nhà ib mình hướng dẫn",
]
HAM_HARD = [
    "mình bán lại cái quạt ký túc xá 50k, bạn nào ở cùng khu thì ib mình",
    "ai cần tìm trọ gần trường thì ib mình chỉ chỗ, mình ở đó 2 năm rồi",
    "đây là link nộp bài thầy gửi trên nhóm: https://forms.gle/{c}",
    "mua giáo trình {m} ở đâu rẻ vậy mọi người, ib mình với",
    "vừa mua được vé xe về quê giá 150k, ai về cùng ib mình",
]


def phone():
    return "09" + "".join(random.choice("0123456789") for _ in range(8))


def code():
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(6))


def fill(t):
    return t.format(m=random.choice(MON), n=random.choice(NOI), g=random.choice(GV), p=phone(), c=code())


def vary(t, label):
    if random.random() < 0.35:
        for a, b in TEEN:
            if random.random() < 0.6:
                t = t.replace(a, b)
    if random.random() < 0.15:
        t = t.upper() if label == "HAM" else t
    return t + random.choice(EMO)


def make(n_ham=420, n_spam=280):
    rows, base = [], datetime(2025, 9, 1)
    def add(label, text, hard=False):
        rows.append(dict(text=text, group=random.choice(GROUPS), label=label,
                         timestamp=(base + timedelta(minutes=random.randint(0, 60 * 24 * 90))).isoformat(),
                         source_url="", annotator="synthetic", review_status="SYNTHETIC"))
    for _ in range(n_ham):
        add("HAM", vary(fill(random.choice(HAM)), "HAM"))
    for _ in range(int(n_ham * 0.07)):
        add("HAM", vary(fill(random.choice(HAM_HARD)), "HAM"))
    for _ in range(n_spam):
        add("SPAM", vary(fill(random.choice(SPAM)), "SPAM"))
    for _ in range(int(n_spam * 0.12)):
        add("SPAM", vary(fill(random.choice(SPAM_HARD)), "SPAM"))
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "comments.csv"
    df = make()
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.label.value_counts(), "->", out)
