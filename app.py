from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
model_data = joblib.load("rf_post_popularity.joblib")
model = model_data["model"]
scaler = model_data["scaler"]
feature_cols = model_data["feature_columns"]

def preprocess_input(likes, comments, followers, hashtags, text, timestamp):
    hashtag_list = [t.strip('#').lower() for t in hashtags.split() if t.startswith('#')]
    hashtag_count = len(hashtag_list)
    unique_hashtags = len(set(hashtag_list))
    text_length = len(text)
    avg_word_length = np.mean([len(w) for w in text.split()]) if len(text.split()) > 0 else 0
    total_engagement = likes + comments
    engagement_ratio = total_engagement / followers if followers > 0 else 0
    hashtag_density = (hashtag_count / (text_length if text_length > 0 else 1)) * 100

    try:
        # Handle datetime-local format (YYYY-MM-DDTHH:MM)
        posting_hour = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M').hour
    except Exception:
        try:
            # Fallback to ISO format
            posting_hour = datetime.fromisoformat(timestamp).hour
        except Exception:
            posting_hour = 12  # Default value

    data = pd.DataFrame([[
        likes, comments, followers, hashtag_count, unique_hashtags,
        text_length, avg_word_length, engagement_ratio,
        hashtag_density, posting_hour
    ]], columns=feature_cols)

    data_scaled = scaler.transform(data)
    return data_scaled


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        likes = int(request.form['likes'])
        comments = int(request.form['comments'])
        followers = int(request.form['followers'])
        hashtags = request.form['hashtags']
        text = request.form['text']
        timestamp = request.form['timestamp']

        X_input = preprocess_input(likes, comments, followers, hashtags, text, timestamp)
        prediction = model.predict(X_input)[0]
        result = "🌟 High Chance of Popularity" if prediction == 1 else "⚪ Low Chance of Popularity"

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"⚠️ Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)