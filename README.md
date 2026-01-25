# Sentimetrix-TCN: Multimodal Financial Decision Support System

Sentimetrix-TCN is a state-of-the-art financial analysis platform that combines Deep Learning (TCN), Generative AI (LLM), and Real-Time Data to provide "Strong Buy/Sell" signals with explainable insights.

![Status](https://img.shields.io/badge/Status-Active-success)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-89.6%25-brightgreen)
![Tech](https://img.shields.io/badge/Stack-FastAPI%20%7C%20React%20%7C%20PyTorch-blue)

## 🚀 Key Features

*   ⚡ Real-Time "Alpha" Detection:
    *   Analyzing 17+ technical indicators (RSI, MACD, Bollinger Bands, ATR) in real-time.
    *   Powered by a Temporal Convolutional Network (TCN) trained on deep historical data.
    *   Accuracy: >89% (Validation).

*   🧠 The "Intelligence Engine":
    *   Local LLM (DistilGPT-2): Generates natural language analysis of market trends.
    *   RAG (Retrieval-Augmented Generation): Uses expert financial rules (FAISS) to explain *why* a signal was triggered.
    *   Sentiment Analysis: Scans live news headlines to gauge market mood (Positive/Negative).

*   📰 Live News Stream:
    *   Aggregates real-time news from Yahoo Finance and Google News RSS.
    *   Auto-feeds news context into the AI Analyst.

*   💻 Modern Web Dashboard:
    *   Frontend: React + Vite + Tailwind CSS (Glassmorphism Dark Mode).
    *   Charts: Interactive Price & RSI/MACD charts using Recharts.
    *   Sentiment Panel: Dedicated sidebar showing overall sentiment score and live news headlines.

---

## 🛠️ Technology Stack

*   Backend: Python 3.10+, FastAPI, Uvicorn, PyTorch, Transformers, Pandas, TA-Lib/Pandas-TA, Feedparser, FAISS.
*   Frontend: Node.js, React, Vite, Tailwind CSS, Recharts, Lucide-React, Axios.
*   Data Sources: Yahoo Finance (yfinance), Google News (RSS).

---

## 📦 Dependencies

### Backend (`requirements.txt`)
```text
fastapi
uvicorn
torch
transformers
pandas
numpy
scikit-learn
joblib
yfinance
ta
feedparser
faiss-cpu
python-multipart
requests
```

### Frontend (`package.json`)
*   `react`, `react-dom`
*   `tailwindcss`, `@tailwindcss/postcss`
*   `recharts`, `framer-motion`, `lucide-react`, `axios`

---

## 🔧 Installation & Run Instructions

### Prerequisites
1.  Python 3.10+: Ensure `python` and `pip` are installed.
2.  Node.js 18+: Ensure `npm` is installed.

### Step 1: Backend Setup
1.  Navigate to the project root:
    ```bash
    cd Sentimetrix-Project
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Backend Server:
    ```bash
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    *The API will start at `http://localhost:8000`.*

### Step 2: Frontend Setup
1.  Open a new terminal and navigate to the frontend folder:
    ```bash
    cd frontend
    ```
2.  Install Node dependencies:
    ```bash
    npm install
    ```
3.  Run the Web App:
    ```bash
    npm run dev
    ```
    *The App will start at `http://localhost:5173`.*

---

## 🖥️ How to Use

1.  Open the App: Go to `http://localhost:5173` in your browser.
2.  Enter Ticker: Type a valid US Stock Ticker (e.g., `NVDA`, `AAPL`, `TSLA`, `AMD`).
3.  View Real-Time Data:
    *   The Live News Stream will populate immediately.
    *   Prices and Indicators will load in the charts.
4.  Run Analysis:
    *   Click "RUN DEEP ANALYSIS".
    *   The AI will compute the Signal (Buy/Sell/Hold), Confidence Score, and Sentiment.
    *   Read the "Key Drivers" section to see which rules triggered the decision.
5.  Check Sentiment:
    *   Use the sidebar to view the aggregated Sentiment Score (Positive/Negative) and read the latest news headlines.

---

## ⚠️ Troubleshooting

1. "Port already in use" (WinError 10013)
*   This means the server is already running.
*   Solution: Close the terminal using the port or run:
    ```bash
    taskkill /IM uvicorn.exe /F
    ```

2. "Invalid Crumb" / "401 Unauthorized" (Yahoo Finance)
*   The system uses a patched session to handle this. If it persists:
    ```bash
    pip install --upgrade yfinance
    ```
*   Restart the backend after upgrading.

3. "PostCSS / Tailwind Error"
*   Ensure you have installed the postcss adapter:
    ```bash
    cd frontend
    npm install @tailwindcss/postcss
    ```



