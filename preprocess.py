import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(filtered_tokens)
