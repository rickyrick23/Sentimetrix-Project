from transformers import pipeline
import torch

class SentimentEngine:
    def __init__(self):
        print("Loading Sentiment Engine...")
        device = 0 if torch.cuda.is_available() else -1
        # Use a lightweight robust model for financial sentiment if possible, or general.
        # 'distilbert-base-uncased-finetuned-sst-2-english' is good for general.
        # For finance, 'ProsusAI/finbert' is better but larger. Let's stick to a fast one.
        self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
        print("Sentiment Engine Loaded.")

    def analyze(self, text):
        """Returns label (POSITIVE/NEGATIVE) and score."""
        try:
            result = self.analyzer(text)[0]
            # Map to simpler terms if needed
            return result
        except Exception as e:
            print(f"Sentiment Error: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
