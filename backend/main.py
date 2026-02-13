from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import torch
import joblib
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import get_alpha_live_data, search_ticker
from src.rag_retriever import build_research_index, retrieve_alpha_context, embedder
from src.model_engine import SentimetrixTCN, predict_signal
from src.intelligence_engine import IntelligenceEngine
from src.sentiment_engine import SentimentEngine
from src.news_engine import NewsEngine

app = FastAPI(title="Sentimetrix-TCN API")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev (or specific ["http://localhost:5173"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Assets
model = None
index = None
scaler = None
llm = None
sentiment_engine = None
news_engine = None

@app.on_event("startup")
def load_assets():
    global model, index, scaler, llm, sentiment_engine, news_engine
    
    # 1. TCN Model
    model = SentimetrixTCN(input_size=19)
    if os.path.exists("models/alpha_weights.pth"):
        model.load_state_dict(torch.load("models/alpha_weights.pth"))
        model.eval()
    else:
        print("Warning: No weights found.")
        
    # 2. Scaler
    if os.path.exists("models/scaler.pkl"):
        scaler = joblib.load("models/scaler.pkl")
        
    # 3. RAG Index
    index = build_research_index()
    
    # 4. LLM
    llm = IntelligenceEngine()
    
    # 5. Sentiment
    sentiment_engine = SentimentEngine()
    
    # 6. News
    news_engine = NewsEngine()
    
    print("All Systems Online.")

# --- API Models ---
class AnalysisRequest(BaseModel):
    ticker: str
    news_context: str = None # Optional now

class ChatRequest(BaseModel):
    ticker: str
    query: str
    context: str
    signal: int

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "online", "system": "Sentimetrix-TCN"}

@app.get("/market-data/{ticker}")
def get_market_data(ticker: str):
    """Fetch live data, RSI, MACD"""
    try:
        df, kpis, resolved_ticker = get_alpha_live_data(ticker)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found (and search failed).")
        
        # Convert last 100 rows to JSON friendly format
        # Ensure date column is named 'date' for Recharts
        df_history = df.tail(100).reset_index()
        df_history.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True) # Handle varied yfinance outputs
        history = df_history.to_dict(orient="records")
        # Handle datetime conversion if needed (pandas validation)
        
        return {
            "ticker": resolved_ticker, # Return the ACTUAL ticker found
            "original_ticker": ticker,
            "kpis": kpis,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/{ticker}")
def get_news(ticker: str):
    """Fetch live news context and articles"""
    try:
        # Resolve ticker for better matching (e.g. "Bitcoin" -> "BTC-USD")
        resolved = search_ticker(ticker)
        query_ticker = resolved if resolved else ticker
        
        context, articles = news_engine.fetch_company_news(query_ticker)
        return {"context": context, "articles": articles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest):
    """Run TCN + RAG + Sentiment Analysis"""
    try:
        # Fetch Data
        df, _, resolved_ticker = get_alpha_live_data(req.ticker)
        
        if df.empty:
             raise HTTPException(status_code=404, detail=f"Ticker '{req.ticker}' data not found.")

        # Determine Context: Use provided or fetch live
        news_context = req.news_context
        if not news_context:
            # Fetch news for the RESOLVED ticker if possible, or original?
            # If resolved is "BTC-USD", fetching news for "BTC-USD" on Google News might be weird?
            # "Bitcoin USD"?
            # News engine logic is separate. Let's try resolved ticker first.
            news_context, _ = news_engine.fetch_company_news(resolved_ticker)
            
        if not news_context or news_context == "No recent news found.":
             news_context = "Market volatility observed in the sector."
            
        # RAG
        rules = retrieve_alpha_context(news_context, index)
        
        # TCN Inference
        if scaler:
            pred_class, conf, weights = predict_signal(model, df, rules, embedder, scaler)
        else:
            raise HTTPException(status_code=500, detail="Scaler not loaded")
            
        # Sentiment
        sentiment = sentiment_engine.analyze(news_context)
        
        return {
            "ticker": resolved_ticker, # Inform UI of the actual ticker used
            "signal_class": pred_class,
            "confidence": conf,
            "rules_triggered": rules,
            "sentiment": sentiment,
            "news_context_used": news_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_analyst(req: ChatRequest):
    """Generate LLM Response"""
    try:
        # We accept context list as string? Or pass list?
        # Let's assume context is a string joined by newlines or we parse it.
        # For simplicity, let's treat context as a single string block
        response = llm.generate_analysis(
            context_rules=[req.context], # list expected
            technical_signal=req.signal,
            ticker=req.ticker,
            user_query=req.query
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Return model accuracy/precision (static/simulated or read from file)"""
    # Ideally read from metrics.json
    metrics = {"accuracy": 0.47, "precision": 0.50} # Default fallback
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
    return metrics
