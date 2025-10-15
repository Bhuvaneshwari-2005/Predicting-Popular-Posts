import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# ----- Step 1: Create synthetic dataset -----
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'likes': np.random.randint(10, 5000, n),
    'comments': np.random.randint(1, 1000, n),
    'followers': np.random.randint(100, 50000, n),
    'hashtags_count': np.random.randint(0, 15, n),
    'unique_hashtags': np.random.randint(0, 10, n),
    'text_length': np.random.randint(20, 500, n),
    'avg_word_length': np.random.uniform(3, 8, n),
    'engagement_ratio': np.random.uniform(0.001, 0.5, n),
    'hashtag_density': np.random.uniform(0, 1, n),
    'posting_hour': np.random.randint(0, 24, n),
})

# Label (popular = 1 if engagement > 0.1)
data['popular'] = ((data['likes'] + data['comments']) / data['followers']) > 0.1
data['popular'] = data['popular'].astype(int)

# ----- Step 2: Split data -----
X = data.drop('popular', axis=1)
y = data['popular']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----- Step 3: Train model -----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----- Step 4: Evaluate -----
acc = model.score(X_test, y_test)
print(f"✅ Model trained successfully with accuracy: {acc:.2f}")

# ----- Step 5: Save model -----
joblib.dump({
    'model': model,
    'scaler': scaler,
    'feature_columns': list(X.columns)
}, "rf_post_popularity.joblib")

print("💾 Model saved successfully as rf_post_popularity.joblib")
