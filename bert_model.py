from sentence_transformers import SentenceTransformer
import numpy as np

# Load SentenceTransformer model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_bert_embedding(text: str) -> np.ndarray:
    """
    Generate embedding vector for a single text input.
    """
    embedding = model.encode([text], convert_to_numpy=True)[0]
    return embedding

def get_bert_embeddings_batch(text_list: list[str], batch_size: int = 16) -> list[np.ndarray]:
    """
    Generate embeddings for a batch of texts.
    """
    embeddings = model.encode(text_list, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()
