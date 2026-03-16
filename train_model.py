import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv(
    r"C:\Users\sruthi\Pictures\Downloads\social_media_ad_engagement_20000_rows.csv"
)

# Target
y = df["engagement_rate"]

# Features (ONLY things known before posting)
X = df[
    [
        "platform",
        "ad_format",
        "placement",
        "caption_length",
        "emoji_count",
        "sentiment_score",
        "hour",
        "is_weekend",
        "age_group",
        "interest_category",
        "spend",
    ]
]

# Categorical & numeric columns
cat_cols = [
    "platform",
    "ad_format",
    "placement",
    "age_group",
    "interest_category",
]
num_cols = [
    "caption_length",
    "emoji_count",
    "sentiment_score",
    "hour",
    "is_weekend",
    "spend",
]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "engagement_model.pkl")

print("✅ Model trained and saved as engagement_model.pkl")
