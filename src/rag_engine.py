import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the encoder once for the whole project
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Professional Knowledge Base (Can be expanded to 100s of rules)
EXPERT_RULES = [
    "High RSI (>70) suggests overbought conditions; expect price reversal.",
    "Low RSI (<30) indicates oversold conditions; potential for bullish bounce.",
    "MACD bullish crossover suggests rising upward momentum.",
    "Prices touching Upper Bollinger Band often lead to mean-reversion pullbacks.",
    "Breakouts on high volume confirm the strength of the new trend."
]

def initialize_vector_db():
    """Builds and returns a FAISS index of financial expert rules."""
    vectors = embedder.encode(EXPERT_RULES)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    return index

def retrieve_context(query, index, k=3):
    """Retrieves top-k rules relevant to the user's input headline."""
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), k)
    return [EXPERT_RULES[i] for i in indices[0]]