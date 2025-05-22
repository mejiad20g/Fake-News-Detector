import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Load NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")
    STOP_WORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
                  'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
                  'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what',
                  'when', 'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too',
                  'very', 'can', 'will', 'just', 'should', 'now'}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    text = ' '.join([word for word in words if word not in STOP_WORDS])
    return text

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

# Load models
@st.cache_resource
def load_models():
    model_name = 'all-MiniLM-L6-v2'
    sbert_model = SentenceTransformer(model_name)
    coarse_classifier = joblib.load('coarse_classifier.joblib')
    fine_classifier_0 = joblib.load('fine_classifier_0.joblib')
    fine_classifier_1 = joblib.load('fine_classifier_1.joblib')
    regressor = joblib.load('regressor.joblib')
    ohe = joblib.load('onehot_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    return sbert_model, coarse_classifier, fine_classifier_0, fine_classifier_1, regressor, ohe, scaler, tfidf

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Load models
sbert_model, coarse_classifier, fine_classifier_0, fine_classifier_1, regressor, ohe, scaler, tfidf = load_models()

# App title and description
st.title("Fake News Detector")
st.write("Enter a statement and its context to analyze its truthfulness and party bias.")

# Input fields
statement = st.text_area("Statement", height=100)

# Speaker as text input
speaker = st.text_input("Speaker")

# Context options from LIAR dataset
context_options = [
    "a speech", "a debate", "a press conference", "a TV interview", "a radio interview",
    "a campaign ad", "a news release", "a statement", "a tweet", "a Facebook post",
    "a television ad", "a news conference", "a town hall", "a rally", "a press release",
    "a campaign speech", "a debate", "a news article", "a blog post", "a social media post"
]
context_type = st.selectbox("Context Type", context_options)

# Party affiliation options from LIAR dataset
party_options = ["democrat", "republican", "independent", "libertarian", "none", "organization"]
party_affiliation = st.selectbox("Party Affiliation", party_options)

# Subject options from LIAR dataset
subject_options = [
    "economy", "health-care", "immigration", "education", "taxes", "jobs",
    "federal-budget", "state-budget", "foreign-policy", "elections", "guns",
    "abortion", "climate-change", "energy", "environment", "crime", "criminal-justice",
    "campaign-finance", "ethics", "government-regulation", "social-security",
    "transportation", "welfare", "poverty", "religion", "science", "technology"
]
subject = st.selectbox("Subject", subject_options)

# Job title options from LIAR dataset
job_title_options = [
    "President", "Governor", "U.S. Senator", "U.S. Representative", "State Senator",
    "State Representative", "Attorney General", "Mayor", "Lieutenant Governor",
    "Speaker of the House", "Secretary of State", "none"
]
job_title = st.selectbox("Job Title", job_title_options)

# Analyze button
if st.button("Analyze"):
    if not statement:
        st.error("Please enter a statement to analyze.")
    else:
        # Preprocess text
        processed_statement = preprocess_text(statement)
        
        # Get embeddings
        stmt_emb = sbert_model.encode([processed_statement], normalize_embeddings=True)
        # Add placeholder context embeddings (384 dimensions)
        ctx_emb = np.zeros((1, 384))
        
        # Prepare categorical features
        cat_features = pd.DataFrame({
            "speaker": [speaker],
            "party_affiliation": [party_affiliation],
            "subject": [subject],
            "job_title": [job_title]
        })
        cat_features = ohe.transform(cat_features)
        
        # Prepare numerical features
        sentiment_features = get_sentiment_features(statement)
        text_features = get_text_features(statement)
        
        numerical_features = pd.DataFrame({
            "text_length": [len(statement)],
            "word_count": [len(statement.split())],
            "context_length": [0],
            "context_word_count": [0],
            "statement_polarity": [sentiment_features['polarity']],
            "statement_subjectivity": [sentiment_features['subjectivity']],
            "statement_context_similarity": [0],
            "total_claims": [0], 
            "truth_ratio": [0],
            "false_ratio": [0]
        })
        
        # Add text features
        for feature, value in text_features.items():
            numerical_features[f'statement_{feature}'] = [value]
        
        # Add TF-IDF features
        tfidf_features = tfidf.transform([processed_statement]).toarray()
        for i in range(tfidf_features.shape[1]):
            numerical_features[f'tfidf_{i}'] = [tfidf_features[0, i]]
        
        # Scale numerical features
        numerical_features = scaler.transform(numerical_features)
        
        # Combine all features
        X = np.hstack([stmt_emb, ctx_emb, cat_features, numerical_features])
        
        # Make predictions
        coarse_pred = coarse_classifier.predict(X)
        fine_pred = fine_classifier_0.predict(X) if coarse_pred[0] == 0 else fine_classifier_1.predict(X)
        score_pred = regressor.predict(X)[0]
        
        # Display results
        st.session_state.analyzed = True
        
        # Truthfulness score
        st.subheader("Truthfulness Analysis")
        truth_score = score_pred * 100
        st.metric("Truthfulness Score", f"{truth_score:.1f}%")
        
        # Party bias analysis
        st.subheader("Party Bias Analysis")
        # Simple party bias calculation based on sentiment and party affiliation
        sentiment = sentiment_features['polarity']
        if party_affiliation == "democrat":
            bias_score = (sentiment + 1) / 2 * 100
        elif party_affiliation == "republican":
            bias_score = (1 - sentiment) / 2 * 100
        else:
            bias_score = 50  # Neutral for other parties
        
        st.metric("Party Bias Score", f"{bias_score:.1f}%")
        
        # Visualize party bias
        st.progress(bias_score / 100)
        if bias_score > 60:
            st.write("This statement shows a strong bias towards the stated party.")
        elif bias_score < 40:
            st.write("This statement shows a strong bias against the stated party.")
        else:
            st.write("This statement appears relatively neutral in terms of party bias.")
        
        # Additional insights
        st.subheader("Additional Insights")
        if sentiment > 0.3:
            st.write("The statement has a positive sentiment.")
        elif sentiment < -0.3:
            st.write("The statement has a negative sentiment.")
        else:
            st.write("The statement has a neutral sentiment.")
            
        if sentiment_features['subjectivity'] > 0.5:
            st.write("The statement appears to be more subjective than objective.")
        else:
            st.write("The statement appears to be more objective than subjective.") 