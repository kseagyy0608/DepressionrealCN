import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import string
import joblib

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

# Read dataset
df = pd.read_csv('nlp.csv')

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        tokens = nltk.word_tokenize(text)  # Tokenize
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        return ' '.join(tokens)
    else:
        return ""

# Apply preprocessing
df['preprocessed_text'] = df['text'].apply(preprocess_text)

# Ensure that the emotion column contains strings
df['emotion'] = df['emotion'].astype(str)

# Split the data
X = df['preprocessed_text']
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use TF-IDF for feature extraction
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train_tfidf, y_train)
y_pred_logistic = logistic_classifier.predict(X_test_tfidf)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Logistic Regression Accuracy: ", accuracy_logistic)

# SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy: ", accuracy_svm)

# Use the model if accuracy is high enough
best_classifier = None
best_accuracy = max(accuracy_logistic, accuracy_svm)

if best_accuracy >= 0.8:  # Choose a threshold for minimum acceptable accuracy
    if best_accuracy == accuracy_logistic:
        best_classifier = logistic_classifier
    else:
        best_classifier = svm_classifier
    
    # Save the best model and vectorizer
    joblib.dump(best_classifier, 'best_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    textInput = input("How are you today?")
    preprocessed_text = preprocess_text(textInput)
    text_tfidf = vectorizer.transform([preprocessed_text])
    predicted_emotion = best_classifier.predict(text_tfidf)
    print("Your mood is", predicted_emotion[0])
else:
    print("Model is not available.")
