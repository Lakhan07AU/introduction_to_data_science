from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model + TF-IDF
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 2]
    return " ".join(tokens)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["jobtext"].strip()
    if not text:
        return render_template("index.html", prediction="⚠️ Please enter a job description before checking.", color="orange")

    # Rule-based suspicious keyword check
    suspicious_keywords = [
        "earn", "weekly", "aadhaar", "limited seats", "urgent",
        "paytm", "google pay", "work from home", "no experience",
        "daily payment", "send details", "training provided"
    ]

    if any(word in text.lower() for word in suspicious_keywords):
        result = "FAKE JOB POSTING"
        color = "red"
    else:
        # Fall back to ML model
        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        result = "FAKE JOB POSTING" if prediction == 1 else "REAL JOB POSTING"
        color = "red" if prediction == 1 else "green"

    return render_template("index.html", prediction=result, color=color)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)