import React, { useState, useEffect } from 'react';
import DeepChart from './components/DeepChart';
import SentimentPanel from './components/SentimentPanel';
import axios from 'axios';
import { Activity, TrendingUp, AlertCircle } from 'lucide-react';

const App = () => {
  const [ticker, setTicker] = useState('NVDA');
  const [analysis, setAnalysis] = useState(null);
  const [metrics, setMetrics] = useState({ accuracy: 0, precision: 0 });
  const [news, setNews] = useState("");
  const [articles, setArticles] = useState([]);
  const [loadingNews, setLoadingNews] = useState(false);

  useEffect(() => {
    // Load Metrics
    axios.get('http://localhost:8000/metrics')
      .then(res => setMetrics(res.data))
      .catch(err => console.error(err));

    // Initial News Fetch
    fetchNews(ticker);
  }, []);

  const fetchNews = async (symbol) => {
    setLoadingNews(true);
    try {
      const res = await axios.get(`http://localhost:8000/news/${symbol}`);
      setArticles(res.data.articles);
      setNews(res.data.context); // Auto-populate context for analysis
    } catch (err) {
      console.error("News fetch error:", err);
    } finally {
      setLoadingNews(false);
    }
  };

  const handleTickerChange = (e) => {
    setTicker(e.target.value.toUpperCase());
  };

  const handleTickerBlur = () => {
    if (ticker) fetchNews(ticker);
  };

  const runAnalysis = async () => {
    try {
      const res = await axios.post('http://localhost:8000/analyze', {
        ticker: ticker,
        news_context: news
      });
      setAnalysis(res.data);
      if (res.data.news_context_used) setNews(res.data.news_context_used);
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Backend online?");
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col font-sans">
      {/* Navbar / Metric Banner */}
      <div className="bg-slate-900 border-b border-slate-800 p-4 flex justify-between items-center sticky top-0 z-10 shadow-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold">SM</div>
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
            Sentimetrix-TCN
          </h1>
        </div>
        <div className="flex gap-6 text-sm">
          <div className="flex items-center gap-2 text-emerald-400">
            <Activity size={16} />
            <span>Model Accuracy: {(metrics.accuracy * 100).toFixed(1)}%</span>
          </div>
          <div className="flex items-center gap-2 text-blue-400">
            <TrendingUp size={16} />
            <span>Precision: {(metrics.precision * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Main Content */}
        <div className="flex-1 p-6 overflow-y-auto">
          {/* Controls */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="relative">
              <input
                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500 outline-none"
                value={ticker}
                onChange={handleTickerChange}
                onBlur={handleTickerBlur}
                placeholder="Ticker (e.g. AAPL)"
              />
            </div>

            <button
              onClick={runAnalysis}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold rounded-lg p-3 shadow-lg transition-all transform hover:scale-105"
            >
              RUN DEEP ANALYSIS
            </button>
          </div>

          {/* Analysis Context Area (Auto-filled but editable) - Kept Compact */}
          <details className="mb-6">
            <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-300 select-none">View/Edit Analysis Context</summary>
            <textarea
              className="w-full mt-2 bg-slate-800 border border-slate-700 rounded-lg p-3 text-xs text-slate-300 focus:ring-2 focus:ring-blue-500 outline-none"
              rows={2}
              value={news}
              onChange={e => setNews(e.target.value)}
              placeholder="AI Context..."
            />
          </details>

          {/* Dashboard Grid */}
          {analysis && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {/* Signal Card - Expanded Width since there are only 2 items now */}
                <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl relative overflow-hidden">
                  <div className={`absolute top-0 right-0 p-2 ${analysis.signal_class === 2 ? 'bg-emerald-500' : analysis.signal_class === 0 ? 'bg-red-500' : 'bg-gray-500'} bg-opacity-20 text-xs font-bold rounded-bl-xl`}>
                    CONFIDENCE: {analysis.confidence.toFixed(1)}%
                  </div>
                  <h3 className="text-slate-400 text-sm font-semibold mb-2">INTELLIGENCE SIGNAL</h3>
                  <div className={`text-4xl font-extrabold ${analysis.signal_class === 2 ? 'text-emerald-400' : analysis.signal_class === 0 ? 'text-red-400' : 'text-gray-200'}`}>
                    {analysis.signal_class === 2 ? 'STRONG BUY' : analysis.signal_class === 0 ? 'STRONG SELL' : 'HOLD / NEUTRAL'}
                  </div>
                </div>

                {/* Key Drivers (RAG) - Replaces the Sentiment Card which moved to sidebar */}
                <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl overflow-y-auto max-h-48 scrollbar-thin scrollbar-thumb-slate-600">
                  <h3 className="text-slate-400 text-sm font-semibold mb-2">KEY DRIVERS (RAG)</h3>
                  <ul className="text-sm text-slate-300 space-y-2">
                    {analysis.rules_triggered && analysis.rules_triggered.length > 0 ? (
                      analysis.rules_triggered.map((rule, idx) => (
                        <li key={idx} className="flex gap-2">
                          <span className="text-blue-400">•</span>
                          {rule}
                        </li>
                      ))
                    ) : (
                      <li className="italic text-slate-500">No specific expert rules triggered.</li>
                    )}
                  </ul>
                </div>
              </div>
            </>
          )}

          {/* Charts */}
          <DeepChart ticker={ticker} />
        </div>

        {/* Sentiment Analysis Sidebar (Replaces AI Analyst) */}
        <div className="w-96 border-l border-slate-800 bg-slate-900 flex-shrink-0">
          <SentimentPanel
            ticker={ticker}
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
