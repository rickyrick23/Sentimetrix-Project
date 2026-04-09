from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import torch
import joblib
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data_fetcher import get_alpha_live_data, search_ticker
from src.rag_retriever import build_research_index, retrieve_alpha_context, embedder
from src.model_engine import SentimetrixTCN, predict_signal
from src.intelligence_engine import IntelligenceEngine
from src.sentiment_engine import SentimentEngine
from src.news_engine import NewsEngine

app = FastAPI(title="Sentimetrix-TCN API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Assets ---
model = None
index = None
scaler = None
llm = None
sentiment_engine = None
news_engine = None


# ✅ SAFE LAZY LOADER
def safe_load_assets(load_model=True, load_llm=True):
    global model, index, scaler, llm, sentiment_engine, news_engine

    try:
        # MODEL + SCALER
        if load_model:
            if model is None:
                print("Loading model...")
                model_path = os.path.join(PROJECT_ROOT, "models", "alpha_weights.pth")
                model = SentimetrixTCN(input_size=19)
                if os.path.exists(model_path):
                    device = torch.device("cpu")  # force CPU
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()

            if scaler is None:
                print("Loading scaler...")
                scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)

            if index is None:
                print("Loading FAISS index...")
                index = build_research_index()

        # LLM
        if load_llm and llm is None:
            print("Loading LLM...")
            llm = IntelligenceEngine()

        # Sentiment + News (lightweight)
        if sentiment_engine is None:
            sentiment_engine = SentimentEngine()

        if news_engine is None:
            news_engine = NewsEngine()

    except Exception as e:
        print("Lazy load error:", e)


# --- API Models ---
class AnalysisRequest(BaseModel):
    ticker: str
    news_context: str = None


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
    try:
        df, kpis, resolved_ticker = get_alpha_live_data(ticker)

        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker not found")

        df_history = df.tail(100).reset_index()
        df_history.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)

        return {
            "ticker": resolved_ticker,
            "original_ticker": ticker,
            "kpis": kpis,
            "history": df_history.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news/{ticker}")
def get_news(ticker: str):
    try:
        safe_load_assets(load_model=False, load_llm=False)

        resolved = search_ticker(ticker)
        query_ticker = resolved if resolved else ticker

        context, articles = news_engine.fetch_company_news(query_ticker)
        return {"context": context, "articles": articles}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze_stock(req: AnalysisRequest):
    try:
        safe_load_assets(load_model=True, load_llm=False)

        df, _, resolved_ticker = get_alpha_live_data(req.ticker)

        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker data not found")

        news_context = req.news_context
        if not news_context:
            news_context, _ = news_engine.fetch_company_news(resolved_ticker)

        if not news_context:
            news_context = "Market volatility observed."

        rules = retrieve_alpha_context(news_context, index)

        if scaler:
            pred_class, conf, weights = predict_signal(model, df, rules, embedder, scaler)
        else:
            raise HTTPException(status_code=500, detail="Scaler not loaded")

        sentiment = sentiment_engine.analyze(news_context)

        return {
            "ticker": resolved_ticker,
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
    try:
        safe_load_assets(load_model=False, load_llm=True)

        response = llm.generate_analysis(
            context_rules=[req.context],
            technical_signal=req.signal,
            ticker=req.ticker,
            user_query=req.query
        )

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    metrics = {"accuracy": 0.47, "precision": 0.50}
    metrics_path = os.path.join(PROJECT_ROOT, "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return metrics