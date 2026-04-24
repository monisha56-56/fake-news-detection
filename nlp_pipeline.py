import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_english(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return " ".join(filtered_tokens)

def preprocess_tamil(text):
    # Simplified Tamil preprocessing
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\u0B80-\u0BFF\s]', '', text) # Keep only Tamil characters and whitespace
    tokens = text.split()
    # Tamil stopwords list (minimal)
    tamil_stopwords = ["அது", "இந்த", "அந்த", "எந்த", "ஒரு", "ஆகும்", "உள்ளது"]
    filtered_tokens = [w for w in tokens if not w in tamil_stopwords]
    return " ".join(filtered_tokens)

def clean_text(text, language="English"):
    if language == "Tamil":
        return preprocess_tamil(text)
    return preprocess_english(text)

if __name__ == "__main__":
    test_en = "This is a fake news about COVID-19! Check https://fake.com"
    test_ta = "இது ஒரு போலியான செய்தி! https://fake.com"
    print(f"English: {clean_text(test_en)}")
    print(f"Tamil: {clean_text(test_ta, 'Tamil')}")
