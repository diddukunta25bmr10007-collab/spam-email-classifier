import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("data.csv")
X = data['text']
y = data['label']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)
email = input("Enter email text: ")
email_vec = vectorizer.transform([email])
prediction = model.predict(email_vec)
print("hi im Prediction:", prediction[0])
print("Model Accuracy:", model.score(X_test, y_test))
prediction = model.predict(email_vec)
print("\n======================")
print("EMAIL ANALYSIS RESULT")
print("======================")
if prediction[0] == "spam":
    print(" This is SPAM email")
else:
    print(" This is NOT SPAM email")
print("\nModel Accuracy:", round(model.score(X_test, y_test), 2))