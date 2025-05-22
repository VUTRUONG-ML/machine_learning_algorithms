import numpy as np 
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
def predict(w, X):
    return np.sign(X @ w) # X shape (n_samples, n_features), w shape (n_features,)

def perceptronAl(w_init, X, y):
    w = [w_init]
    i = 0 
    while True:
        i += 1 
        y_pred = predict(w[-1], X)
        lstMiss = np.where(np.equal(y_pred, y) == False)[0]
        if len(lstMiss) == 0 or i == 5000:
            break 
        random_id = np.random.choice(lstMiss, 1)[0]
        w_new = w[-1] + X[random_id]*y[random_id]
        w.append(w_new)
    return w[-1]


# Tải danh sách stopwords (nếu chưa có, chạy `nltk.download('stopwords')`)
vietnamese_stopwords = {"bạn", "đã", "và", "nhận", "ngay", "có", "là", "của", "tôi"}

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in vietnamese_stopwords]
    return " ".join(words)

data = [
    ("Bạn đã trúng thưởng 10 triệu đồng! Nhấp vào link để nhận.", "Spam"),
    ("Anh ơi, tối nay mình đi ăn không?", "Ham"),
    ("Khách hàng thân mến, quý vị được giảm giá 50% cho đơn hàng tiếp theo!", "Spam"),
    ("Nhớ làm bài tập nhé, mai có kiểm tra đấy!", "Ham"),
    ("Nhận ngay 500.000 VND khi đăng ký tài khoản tại đây: www.lừađảo.com", "Spam"),
    ("Em đã gửi email, anh check giúp em nhé.", "Ham"),
    ("Bố mẹ khỏe không con? Bao giờ con về thăm nhà?", "Ham"),
    ("💰 Vay ngay 100 triệu không cần thế chấp, lãi suất 0% tháng đầu tiên! 💰", "Spam"),
    ("Bạn có đơn hàng chưa thanh toán, hãy thanh toán sớm để tránh bị hủy.", "Ham"),
    ("Tài khoản của bạn có dấu hiệu đăng nhập lạ, vui lòng xác minh ngay!", "Spam"),
    ("Gọi ngay để nhận ưu đãi khủng từ nhà mạng của bạn!", "Spam"),
    ("Nhớ uống đủ nước mỗi ngày để cơ thể khỏe mạnh nhé!", "Ham"),
    ("Công ty chúng tôi đang tuyển dụng, lương cao, không yêu cầu kinh nghiệm!", "Spam"),
    ("Cảm ơn bạn đã mua hàng tại cửa hàng của chúng tôi!", "Ham"),
    ("Lịch thi đấu bóng đá hôm nay: 20h00 - Việt Nam vs Thái Lan", "Ham"),
    ("Nạp thẻ điện thoại ngay để nhận thêm 50% giá trị khuyến mãi!", "Spam"),
    ("Học lập trình Python không? Mình có khóa học miễn phí đây!", "Spam"),
    ("Bạn có hẹn gặp bác sĩ vào ngày mai, đừng quên nhé!", "Ham"),
    ("Truy cập ngay website của chúng tôi để nhận phần thưởng bất ngờ!", "Spam"),
    ("Mẹ đang nấu món con thích nhất, về nhà ngay nhé!", "Ham"),
    ("Tải ngay ứng dụng mới để nhận 100.000 VND miễn phí!", "Spam"),
    ("Hẹn gặp lại bạn vào tuần sau nhé!", "Ham"),
    ("Nhập mã khuyến mãi XYZ để giảm ngay 20%!", "Spam"),
    ("Anh có thể gửi lại file báo cáo không?", "Ham"),
    ("Đăng ký ngay hôm nay để nhận quà tặng hấp dẫn!", "Spam"),
    ("Sáng mai em có lịch họp, anh nhớ đến nhé!", "Ham"),
    ("Chúc mừng bạn đã nhận được mã giảm giá 30%", "Spam"),
    ("Tối nay rảnh không? Đi xem phim nhé!", "Ham"),
    ("Cảnh báo! Tài khoản của bạn có hoạt động đáng ngờ!", "Spam"),
    ("Bạn ơi, có thể giúp mình bài tập này không?", "Ham"),
    ("Mở tài khoản ngay để nhận lãi suất hấp dẫn!", "Spam"),
    ("Hôm nay trời đẹp quá, đi dạo không?", "Ham"),
    ("Mua 1 tặng 1 trong hôm nay! Nhanh tay lên!", "Spam"),
    ("Sếp yêu cầu gửi báo cáo gấp!", "Ham"),
    ("Nhận ngay ưu đãi đặc biệt khi đặt hàng online!", "Spam"),
    ("Chúc mừng sinh nhật! Chúc bạn một ngày vui vẻ!", "Ham"),
    ("Cảnh báo: Máy tính của bạn có nguy cơ nhiễm virus!", "Spam"),
    ("Anh đã đặt bàn chưa? Nhớ báo em nhé!", "Ham"),
    ("Nhấn vào đây để nhận quà may mắn!", "Spam"),
    ("Mai có buổi họp quan trọng, đừng quên nhé!", "Ham"),
    ("Tải ứng dụng ngay để nhận 10GB data miễn phí!", "Spam"),
    ("Mình đến nơi rồi, bạn đang ở đâu?", "Ham"),
    ("Số dư tài khoản ngân hàng của bạn đã thay đổi!", "Spam"),
    ("Hôm nay có lịch hẹn với bác sĩ lúc 14h", "Ham"),
    ("Nhận ngay mã giảm giá khi mua sắm tại cửa hàng!", "Spam"),
    ("Lịch học tuần này có thay đổi, nhớ kiểm tra nhé!", "Ham"),
    ("Bạn đã trúng giải đặc biệt, liên hệ ngay để nhận thưởng!", "Spam"),
    ("Mẹ nhớ con lắm, bao giờ con về?", "Ham")
]

# Tạo DataFrame
df = pd.DataFrame(data, columns=["Message", "Label"])

df["Message"] = df["Message"].apply(clean_text)

# Chuyển văn bản thành vector TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Message"])
# Chuyển nhãn thành số (Spam = 1, Ham = 0)
y = df["Label"].map({"Spam": 1, "Ham": 0})

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test = X_train.toarray(), X_test.toarray()
y_train = y_train.to_numpy()  # Chuyển Series -> numpy array
y_test = y_test.to_numpy()



w_init = np.zeros(X_train.shape[1])
# Train mô hình
w_final = perceptronAl(w_init, X_train, y_train)

# Kiểm tra trên tập test
y_pred = predict(w_final, X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")