import pandas as pd, re
d=pd.read_csv(r'C:/Users/HAINAM/Downloads/raw_fb.csv')
t=d.text.str.lower()
R=lambda p: t.str.contains(p,regex=True)
promo_course = R(r'aptis one|cam kết đầu ra|khóa ôn|khoá ôn|khóa học aptis|web luyện|lộ trình (rõ|tinh|từ|ôn|chi)|giáo viên band|gv band|ưu đãi hphi|ưu đãi nữa|nhận ôn aptis|inbox bên chị|bên chị có|bên c có')
sell_key   = R(r'full key|key update|có tài liệu ôn|có tài liệu,|đang gom aptis|mua key')
company    = R(r'tuyển dụng|tuyển cộng tác|tuyển thực tập|thực tập sinh kinh doanh|bna vietnam|smile media') 
page_post  = (d.kind=='post') & d.text.str.match(r'^\s*[📢🚨🏍️📚💌📸✨🌷🎓🔥💥📌⭐]') & R(r'zalo|đăng ký|đăng kí|liên hệ|aptis|livestream|cảnh báo')
dup3 = d.text.map(d.text.value_counts())>=3
dup_promo = dup3 & (promo_course|R(r'cam kết|khóa|khoá|ib|lh|liên hệ|zalo')) & (d.text.str.len()>25)
spam = promo_course|sell_key|company|page_post|dup_promo
# giữ ham: câu hỏi tìm nơi học
ask = R(r'(ở đâu|chỗ nào|có ai|ai biết|mn cho em hỏi|cho em hỏi|có nơi nào|giới thiệu cho)') & (d.kind=='post') & (d.text.str.len()<200)
spam &= ~ask
review = (R(r'clb|câu lạc bộ|rcm') & R(r'tuyển|ứng tuyển|đơn|rcm')) & ~spam
print(spam.sum(), review.sum(), len(d))
print(d[spam].groupby('group').size())
for i in d[spam].index[:400]: print(i,d.kind[i][0],d.text[i][:90].replace('\n',' '))
