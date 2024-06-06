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

def load_and_preprocess_data(csv_file):
    # Read dataset
    df = pd.read_csv(csv_file)

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Apply preprocessing
    df['preprocessed_text'] = df['text'].apply(preprocess_text)

    # Ensure that the emotion column contains strings
    df['emotion'] = df['emotion'].astype(str)

    return df

def prepare_features(df):
    X = df['preprocessed_text']
    y = df['emotion']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save_model(featuresets):
    X_train, X_test, y_train, y_test = featuresets
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

    # Select the best classifier
    best_classifier = svm_classifier if accuracy_svm > accuracy_logistic else logistic_classifier
    best_accuracy = max(accuracy_logistic, accuracy_svm)

    # Save the best model and vectorizer
    joblib.dump(best_classifier, 'best_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return best_classifier, vectorizer, best_accuracy

def load_model_and_vectorizer():
    classifier = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return classifier, vectorizer

def predict_emotion(classifier, text, vectorizer):
    preprocessed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([preprocessed_text])
    predicted_emotion = classifier.predict(text_tfidf)
    return predicted_emotion[0]

if __name__ == "__main__":
    # Load and preprocess data
    data_filepath = 'nlp.csv'  # Adjust this path to your CSV file
    df = load_and_preprocess_data(data_filepath)
    featuresets = prepare_features(df)
    classifier, vectorizer, accuracy = train_and_save_model(featuresets)
    
    if accuracy >= 0.8:  # Choose a threshold for minimum acceptable accuracy
        textInput = input("How are you today?")
        preprocessed_text = preprocess_text(textInput)
        text_tfidf = vectorizer.transform([preprocessed_text])
        predicted_emotion = classifier.predict(text_tfidf)
        print("Your mood is", predicted_emotion[0])
    else:
        print("Model is not available.")
