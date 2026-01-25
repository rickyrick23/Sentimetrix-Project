import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge Base of Expert Logic
EXPERT_RULES = [
    "Prices touching Lower Bollinger Band often indicate oversold conditions.",
    "High RSI (>70) suggests overbought conditions; expect price reversal.",
    "MACD bullish crossover suggests rising upward momentum.",
    "Breakouts on high volume confirm the strength of the new trend."
]

def build_research_index():
    """Builds the FAISS index for real-time retrieval."""
    vectors = embedder.encode(EXPERT_RULES)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors).astype('float32'))
    return index

def retrieve_alpha_context(headline, index, k=3):
    """Retrieves top-k rules relevant to news context."""
    query_vec = embedder.encode([headline])
    _, indices = index.search(np.array(query_vec).astype('float32'), k)
    return [EXPERT_RULES[i] for i in indices[0]]