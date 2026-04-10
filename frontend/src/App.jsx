import React, { useState, useEffect } from 'react';
import DeepChart from './components/DeepChart';
import SentimentPanel from './components/SentimentPanel';
import axios from 'axios';
import { Activity, TrendingUp, AlertCircle } from 'lucide-react';

// ✅ LIVE BACKEND URL
const API_BASE = "https://sentimetrix-project.onrender.com";

const App = () => {
  const [ticker, setTicker] = useState('NVDA');
  const [activeTicker, setActiveTicker] = useState('NVDA');
  const [analysis, setAnalysis] = useState(null);
  const [news, setNews] = useState("");
  const [articles, setArticles] = useState([]);
  const [loadingNews, setLoadingNews] = useState(false);

  useEffect(() => {
    runAnalysis();
  }, []);

  const fetchNews = async (symbol) => {
    setLoadingNews(true);
    try {
      const res = await axios.get(`${API_BASE}/news/${symbol}`);
      setArticles(res.data.articles);
      setNews(res.data.context);
    } catch (err) {
      console.error("News fetch error:", err);
    } finally {
      setLoadingNews(false);
    }
  };

  const handleTickerChange = (e) => {
    setTicker(e.target.value.toUpperCase());
  };

  const runAnalysis = async () => {
    setActiveTicker(ticker);
    setLoadingNews(true);

    try {
      // ✅ Fetch news
      const newsRes = await axios.get(`${API_BASE}/news/${ticker}`);
      const latestArticles = newsRes.data.articles;
      const latestContext = newsRes.data.context;

      setArticles(latestArticles);
      setNews(latestContext);

      // ✅ Run analysis
      const res = await axios.post(`${API_BASE}/analyze`, {
        ticker: ticker,
        news_context: latestContext
      });

      setAnalysis(res.data);
      if (res.data.news_context_used) setNews(res.data.news_context_used);

    } catch (err) {
      console.error(err);
      alert("Analysis failed. Backend online?");
    } finally {
      setLoadingNews(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col font-sans">
      {/* Navbar */}
      <div className="bg-slate-900 border-b border-slate-800 p-4 flex justify-between items-center sticky top-0 z-10 shadow-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold">SM</div>
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
            Sentimetrix-TCN
          </h1>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Main Content */}
        <div className="flex-1 p-6 overflow-y-auto">

          {/* Controls */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <input
              className="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500 outline-none"
              value={ticker}
              onChange={handleTickerChange}
              placeholder="Ticker (e.g. AAPL)"
            />

            <button
              onClick={runAnalysis}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold rounded-lg p-3 shadow-lg transition-all transform hover:scale-105"
            >
              RUN DEEP ANALYSIS
            </button>
          </div>

          {/* Context */}
          <details className="mb-6">
            <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-300">
              View/Edit Analysis Context
            </summary>
            <textarea
              className="w-full mt-2 bg-slate-800 border border-slate-700 rounded-lg p-3 text-xs text-slate-300"
              rows={2}
              value={news}
              onChange={e => setNews(e.target.value)}
            />
          </details>

          {/* Dashboard */}
          {analysis && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">

              {/* Signal */}
              <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl">
                <h3 className="text-slate-400 text-sm mb-2">INTELLIGENCE SIGNAL</h3>
                <div className="text-4xl font-extrabold">
                  {analysis.signal_class === 2
                    ? 'STRONG BUY'
                    : analysis.signal_class === 0
                      ? 'STRONG SELL'
                      : 'HOLD'}
                </div>
                <p className="mt-2 text-sm text-slate-400">
                  Confidence: {analysis.confidence?.toFixed(1)}%
                </p>
              </div>

              {/* Rules */}
              <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl">
                <h3 className="text-slate-400 text-sm mb-2">KEY DRIVERS</h3>
                <ul className="text-sm space-y-2">
                  {analysis.rules_triggered?.map((rule, idx) => (
                    <li key={idx}>• {rule}</li>
                  ))}
                </ul>
              </div>

            </div>
          )}

          {/* Chart */}
          <DeepChart ticker={activeTicker} />
        </div>

        {/* Sidebar */}
        <div className="w-96 border-l border-slate-800 bg-slate-900">
          <SentimentPanel
            ticker={activeTicker}
            sentiment={analysis?.sentiment}
            articles={articles}
            loading={loadingNews}
          />
        </div>
      </div>
    </div>
  );
};

export default App;