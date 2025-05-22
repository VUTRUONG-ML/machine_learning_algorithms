from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score
# Tạo dữ liệu email mẫu
emails = [
    ("Win a free iPhone now!", "spam"),
    ("Congratulations! You've been selected for a prize.", "spam"),
    ("Earn money from home easily!", "spam"),
    ("Limited offer: Buy 1 get 1 free!", "spam"),
    ("Meeting tomorrow at 10 AM", "ham"),
    ("Your order has been shipped.", "ham"),
    ("Let's have lunch next week.", "ham"),
    ("Can you send me the report by EOD?", "ham"),
    ("Reminder: Dentist appointment at 3 PM.", "ham"),
    # 20 email bổ sung
    ("Get your free vacation trip today!", "spam"),
    ("You are a winner! Claim your reward now.", "spam"),
    ("Earn $5000 a week working from home.", "spam"),
    ("Special promotion just for you!", "spam"),
    ("Click here to win big prizes!", "spam"),
    ("Final notice: Update your payment information.", "spam"),
    ("Don't miss this limited-time offer!", "spam"),
    ("Free trial: No credit card required.", "spam"),
    ("Important: Your account will be suspended soon.", "spam"),
    ("Your loan application has been approved.", "spam"),
    ("Hi, can we reschedule our meeting to next Friday?", "ham"),
    ("The document you requested is attached.", "ham"),
    ("Looking forward to our discussion next week.", "ham"),
    ("Please find the project updates attached.", "ham"),
    ("Lunch meeting confirmed for Tuesday.", "ham"),
    ("Let me know your thoughts on the presentation.", "ham"),
    ("Thanks for your help with the project.", "ham"),
    ("We will need additional budget approval.", "ham"),
    ("See you at the conference next month.", "ham"),
    ("Could you review the proposal by tomorrow?", "ham"),
]
2
# Tách thành danh sách email và nhãn
email_texts = [email[0] for email in emails]
email_labels = [email[1] for email in emails]

# Tạo đặc trưng (Bag of word)
vectorizer1 = CountVectorizer() # tạo bộ dữ liệu cho mulinomialNB
vectorizer = CountVectorizer(binary=True) # Them binary = True nếu muốn chuyển thành 0/1

X = vectorizer1.fit_transform(email_texts) # Chuyển X thành ma trận đặc trưng các cặp số và value ví dụ 1 hàng của X [(0, 42)       1], trên mảng email có nhiều từ xếp thành dãy nên từ thứ 42 trong dãy xuất hiện trong email đầu tiên có 1 từ 
X_bernourli = vectorizer.fit_transform(email_texts)

# Bat dau huan luyen 
X_train, X_test, y_train, y_test = train_test_split(X, email_labels, test_size=0.3, random_state=42)
X_train_bernoulli, X_test_bernoulli, y_train_bernoulli, y_test_bernoulli = train_test_split(X_bernourli, email_labels, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

model_bernoulli = BernoulliNB() 
model_bernoulli.fit(X_train_bernoulli, y_train_bernoulli)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(y_test)
print(y_pred)

print("Bernoulli")
y_pred = model_bernoulli.predict(X_test_bernoulli)
print("Accuracy:", accuracy_score(y_test_bernoulli, y_pred))
print(y_test_bernoulli)
print(y_pred)
