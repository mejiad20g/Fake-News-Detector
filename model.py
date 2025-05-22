import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import re
from textblob import TextBlob
import warnings
import ssl
from sklearn.ensemble import VotingClassifier, VotingRegressor
import nltk
warnings.filterwarnings('ignore')

# Handle SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")
    print("Using basic stopwords list instead...")
    # Basic English stopwords as fallback
    STOP_WORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
                  'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
                  'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what',
                  'when', 'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too',
                  'very', 'can', 'will', 'just', 'should', 'now'}

def get_sentiment_features(text):
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

def get_text_features(text):
    words = text.split()
    return {
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'max_word_length': max([len(w) for w in words]) if words else 0,
        'min_word_length': min([len(w) for w in words]) if words else 0,
        'word_count': len(words),
        'unique_word_ratio': len(set(words)) / len(words) if words else 0,
        'uppercase_ratio': sum(1 for w in words if w.isupper()) / len(words) if words else 0
    }

# Enhanced text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove stopwords
    words = text.split()
    text = ' '.join([word for word in words if word not in STOP_WORDS])
    return text

# Load all three parts of the dataset
folder_path = "liar_dataset"
columns = [
    "label", "statement", "subject", "speaker", "job_title",
    "state_info", "party_affiliation", "barely_true_counts",
    "false_counts", "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts", "context"
]

train_df = pd.read_csv(os.path.join(folder_path, "train.tsv"), sep='\t', names=columns)
valid_df = pd.read_csv(os.path.join(folder_path, "valid.tsv"), sep='\t', names=columns)
test_df  = pd.read_csv(os.path.join(folder_path, "test.tsv"),  sep='\t', names=columns)

# Combine all into one DataFrame
df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

# Map labels to scores
label_to_score = {
    "pants-fire": 0.0,
    "false": 0.25,
    "barely-true": 0.5,  # Combined with half-true
    "half-true": 0.5,    # Combined with barely-true
    "mostly-true": 0.75,
    "true": 1.0,
}
df["truth_score"] = df["label"].map(label_to_score)

# Print unique labels and their counts
print("\nUnique labels in dataset:")
print(df["label"].value_counts())
print("\nNaN values in truth_score:", df["truth_score"].isna().sum())

# Filter out rows with invalid labels if any
df = df[df["truth_score"].notnull()].copy()

# Print unique truth scores and their counts
print("\nUnique truth scores after filtering:")
print(df["truth_score"].value_counts().sort_index())

# Fill missing values in text columns
for col in ["statement", "context", "speaker", "party_affiliation", "subject", "job_title"]:
    df[col] = df[col].fillna("").astype(str)

# Apply preprocessing
for col in ["statement", "context"]:
    df[col] = df[col].apply(preprocess_text)

# Create additional features
df['text_length'] = df['statement'].apply(len)
df['word_count'] = df['statement'].apply(lambda x: len(x.split()))
df['context_length'] = df['context'].apply(len)
df['context_word_count'] = df['context'].apply(lambda x: len(x.split()))

# Add sentiment features
sentiment_features = df['statement'].apply(get_sentiment_features)
df['statement_polarity'] = sentiment_features.apply(lambda x: x['polarity'])
df['statement_subjectivity'] = sentiment_features.apply(lambda x: x['subjectivity'])

# Add text features
text_features = df['statement'].apply(get_text_features)
for feature in text_features.iloc[0].keys():
    df[f'statement_{feature}'] = text_features.apply(lambda x: x[feature])

# Add TF-IDF features
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_features = tfidf.fit_transform(df['statement']).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
df = pd.concat([df, tfidf_df], axis=1)

# Create interaction features
df['statement_context_similarity'] = df.apply(
    lambda x: len(set(x['statement'].split()) & set(x['context'].split())) / 
    max(len(set(x['statement'].split()) | set(x['context'].split())), 1), 
    axis=1
)

# Add historical features
df['total_claims'] = df[['barely_true_counts', 'false_counts', 'half_true_counts', 
                        'mostly_true_counts', 'pants_on_fire_counts']].sum(axis=1)
df['truth_ratio'] = (df['mostly_true_counts'] + df['half_true_counts']) / df['total_claims']
df['false_ratio'] = (df['false_counts'] + df['pants_on_fire_counts']) / df['total_claims']

FEATURE_COLS = [
    "statement",
    "context",
    "speaker",
    "party_affiliation",
    "subject",
    "job_title",
    "text_length",
    "word_count",
    "context_length",
    "context_word_count",
    "statement_polarity",
    "statement_subjectivity",
    "statement_context_similarity",
    "total_claims",
    "truth_ratio",
    "false_ratio"
] + [f'statement_{feature}' for feature in text_features.iloc[0].keys()] + [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]

TARGET_COL = "truth_score"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# Verify no NaN values in target
print("\nNaN values in target variable:", y.isna().sum())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y  # Use the truth scores directly for stratification
)

# Load sentence transformer
model_name = 'all-MiniLM-L6-v2'
sbert_model = SentenceTransformer(model_name)

# Compute embeddings with better parameters
print("Computing statement embeddings for training data...")
stmt_train_emb = sbert_model.encode(
    X_train["statement"].tolist(),
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True
)
print("Computing context embeddings for training data...")
ctx_train_emb = sbert_model.encode(
    X_train["context"].tolist(),
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True
)

print("Computing statement embeddings for test data...")
stmt_test_emb = sbert_model.encode(
    X_test["statement"].tolist(),
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True
)
print("Computing context embeddings for test data...")
ctx_test_emb = sbert_model.encode(
    X_test["context"].tolist(),
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True
)

# One-hot encode categorical features with better handling
categorical_cols = ["speaker", "party_affiliation", "subject", "job_title"]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
cat_train = ohe.fit_transform(X_train[categorical_cols])
cat_test = ohe.transform(X_test[categorical_cols])

# Scale numerical features
numerical_cols = [col for col in FEATURE_COLS if col not in categorical_cols + ["statement", "context"]]
scaler = StandardScaler()
num_train = scaler.fit_transform(X_train[numerical_cols])
num_test = scaler.transform(X_test[numerical_cols])

# Combine all features
X_train_features = np.hstack([stmt_train_emb, ctx_train_emb, cat_train, num_train])
X_test_features = np.hstack([stmt_test_emb, ctx_test_emb, cat_test, num_test])

# Handle NaN values in features
X_train_features = np.nan_to_num(X_train_features, nan=0.0)
X_test_features = np.nan_to_num(X_test_features, nan=0.0)

# Convert scores to categories for classification
y_train_cat = (y_train * 4).astype(int)
y_test_cat = (y_test * 4).astype(int)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_features, y_train_cat)

# Stage 1: Coarse classifier (True vs False)
coarse_labels = (y_train_balanced >= 2).astype(int)
coarse_classifier = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    class_weight='balanced',
    random_state=42
)
coarse_classifier.fit(X_train_balanced, coarse_labels)

# Stage 2: Fine classifier for each coarse category
fine_classifiers = {}
for coarse_label in [0, 1]:
    mask = coarse_labels == coarse_label
    if coarse_label == 0:
        fine_labels = y_train_balanced[mask].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))
    else:
        fine_labels = y_train_balanced[mask].apply(lambda x: 3 if x == 2 else (4 if x == 3 else 5))
    
    fine_classifiers[coarse_label] = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42
    )
    fine_classifiers[coarse_label].fit(X_train_balanced[mask], fine_labels)

# Stage 3: Regressor for fine-grained scores
regressor = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
regressor.fit(X_train_balanced, y_train_balanced / 4)  # Convert back to 0-1 scale

# Make predictions
coarse_pred = coarse_classifier.predict(X_test_features)
fine_pred = np.zeros_like(coarse_pred)
for coarse_label in [0, 1]:
    mask = coarse_pred == coarse_label
    if mask.any():
        fine_pred[mask] = fine_classifiers[coarse_label].predict(X_test_features[mask])

# Convert predictions to scores
score_pred = regressor.predict(X_test_features)

# Evaluate results
print("\nClassification Report:")
print(classification_report(y_test_cat, fine_pred))

print("\nRegression Metrics:")
print(f"Mean Squared Error: {mean_squared_error(y_test, score_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, score_pred):.4f}")
print(f"R² Score: {r2_score(y_test, score_pred):.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_cat, fine_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save models
joblib.dump(coarse_classifier, 'coarse_classifier.joblib')
for label, model in fine_classifiers.items():
    joblib.dump(model, f'fine_classifier_{label}.joblib')
joblib.dump(regressor, 'regressor.joblib')
joblib.dump(ohe, 'onehot_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

# Show sample predictions
X_test_statements = X_test["statement"].tolist()
y_test_actual = y_test.tolist()
y_pred_pct = score_pred * 100

print("\nSample predictions (first 10):")
for stmt, actual, pct in zip(X_test_statements[:10], y_test_actual[:10], y_pred_pct[:10]):
    print("Statement:", stmt)
    print(f"Actual truthfulness: {actual * 100:.1f}%")
    print(f"Estimated truthfulness: {pct:.1f}%")
    print("-" * 60)
