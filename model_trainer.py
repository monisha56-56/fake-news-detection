import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from nlp_pipeline import clean_text

def train_model():
    # Load dataset
    df = pd.read_csv("data/dataset.csv")
    
    # Preprocess text based on language
    df['clean_text'] = df.apply(lambda x: clean_text(x['text'], x['language']), axis=1)
    
    # Split data
    X = df['clean_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Save model
    joblib.dump(pipeline, "models/fake_news_model.pkl")
    print("Model trained and saved to models/fake_news_model.pkl")
    
    # Evaluate
    score = pipeline.score(X_test, y_test)
    print(f"Model accuracy on test set: {score:.2f}")

if __name__ == "__main__":
    train_model()
