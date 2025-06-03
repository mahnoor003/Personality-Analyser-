from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import joblib
import nltk
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer
import warnings
import os
from model.predictor import predict_personality_batch  # Import from updated file
from tqdm import tqdm

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

warnings.filterwarnings("ignore")

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("data/cleaned_texts.csv")  # Ensure this file exists

if 'text' not in df.columns:
    raise ValueError("The CSV must contain a 'text' column.")

texts = df["text"].astype(str).tolist()
if len(texts) == 0:
    raise ValueError("No data found in 'text' column.")

# Generate sentence embeddings
print("🔄 Generating sentence embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device.type)
try:
    embeddings = embedder.encode(texts, show_progress_bar=True)
except Exception as e:
    raise RuntimeError(f"Error during embedding generation: {e}")

embeddings = np.array(embeddings)
if embeddings.ndim != 2:
    raise ValueError("Embeddings must be a 2D array.")

# Generate pseudo labels using BERT model
print("🔮 Generating pseudo labels...")
trait_keys = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
pseudo_labels = []
valid_embeddings = []

# Run batch predictions
try:
    all_trait_scores = predict_personality_batch(texts)
except Exception as e:
    raise RuntimeError(f"❌ Error during batched prediction: {e}")

# Match predictions to embeddings with progress bar
for idx, (traits, emb) in tqdm(enumerate(zip(all_trait_scores, embeddings)), total=len(all_trait_scores), desc="🔄 Matching pseudo labels"):
    if not isinstance(emb, (np.ndarray, list)) or len(emb) == 0:
        continue
    try:
        label_vector = [traits[k] for k in trait_keys]
        pseudo_labels.append(label_vector)
        valid_embeddings.append(emb)
    except Exception as e:
        print(f"⚠️ Error at index {idx}: {e}")

# Prepare data for training
X = np.array(valid_embeddings)
y = np.array(pseudo_labels)

print("✅ Data ready for training.")
print("🔢 Feature shape (X):", X.shape)
print("🔢 Target shape (y):", y.shape)

if len(X) == 0 or len(y) == 0:
    raise ValueError("❌ No valid training data available.")

# Split dataset
print("📊 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🎯 Training model...")
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
regression_model = MultiOutputRegressor(base_model)

# Simulate training progress
for i in tqdm(range(1), desc="🏋️ Training model"):  # Just to visualize progress (since `.fit` is not iterable)
    regression_model.fit(X_train, y_train)

# Evaluate model
print("📈 Evaluating model...")
y_pred = regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Mean Squared Error: {mse:.4f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(regression_model, "model/personality_model.pkl")
print("💾 Model saved to model/personality_model.pkl")