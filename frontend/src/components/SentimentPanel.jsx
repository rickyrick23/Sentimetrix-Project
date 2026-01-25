import React from 'react';
import { Newspaper, TrendingUp, TrendingDown, Minus } from 'lucide-react';

const SentimentPanel = ({ ticker, sentiment, articles, loading }) => {
    // Helper to determine sentiment color
    const getSentimentColor = (label) => {
        if (label === 'POSITIVE') return 'text-emerald-400';
        if (label === 'NEGATIVE') return 'text-red-400';
        return 'text-slate-400';
    };

    const getSentimentIcon = (label) => {
        if (label === 'POSITIVE') return <TrendingUp size={24} />;
        if (label === 'NEGATIVE') return <TrendingDown size={24} />;
        return <Minus size={24} />;
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 border-l border-slate-700">
            {/* Header */}
            <div className="p-4 border-b border-slate-700 bg-slate-800">
                <h2 className="flex items-center gap-2 text-xl font-bold text-blue-400">
                    <Newspaper size={24} /> News & Sentiment
                </h2>
            </div>

            {/* Content Container */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6">

                {/* 1. Overall Sentiment Score */}
                <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg text-center">
                    <h3 className="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">Overall Sentiment</h3>
                    {sentiment ? (
                        <div className="flex flex-col items-center gap-2">
                            <div className={`p-3 rounded-full bg-opacity-20 ${sentiment.label === 'POSITIVE' ? 'bg-emerald-500' : sentiment.label === 'NEGATIVE' ? 'bg-red-500' : 'bg-slate-500'} ${getSentimentColor(sentiment.label)}`}>
                                {getSentimentIcon(sentiment.label)}
                            </div>
                            <div className={`text-2xl font-extrabold ${getSentimentColor(sentiment.label)}`}>
                                {sentiment.label}
                            </div>
                            <div className="text-sm text-slate-500">
                                Score: {(sentiment.score * 100).toFixed(1)}%
                            </div>
                        </div>
                    ) : (
                        <div className="text-slate-500 italic text-sm py-4">Run analysis to see sentiment.</div>
                    )}
                </div>

                {/* 2. Live News Stream */}
                <div>
                    <div className="flex justify-between items-center mb-3">
                        <h3 className="text-slate-400 text-xs font-bold uppercase tracking-wider">Live Headlines</h3>
                        <span className="text-[10px] bg-blue-900/50 text-blue-300 px-2 py-0.5 rounded-full">{articles.length} Items</span>
                    </div>

                    <div className="space-y-3">
                        {loading ? (
                            <div className="text-center py-8 text-slate-500 animate-pulse text-sm">Fetching latest news...</div>
                        ) : articles.length > 0 ? (
                            articles.map((art, i) => (
                                <div key={i} className="bg-slate-800/50 p-3 rounded-lg border border-slate-700 hover:border-blue-500/50 transition-colors group">
                                    <div className="flex justify-between items-start gap-2 mb-1">
                                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${art.source === 'Google' ? 'bg-orange-500/20 text-orange-400' : 'bg-purple-500/20 text-purple-400'}`}>
                                            {art.source || 'Yahoo'}
                                        </span>
                                        <span className="text-[10px] text-slate-500 whitespace-nowrap">{new Date(art.published).toLocaleDateString()}</span>
                                    </div>
                                    <a href={art.link} target="_blank" rel="noopener noreferrer" className="block text-sm text-slate-200 font-medium leading-tight group-hover:text-blue-400 transition-colors">
                                        {art.title}
                                    </a>
                                </div>
                            ))
                        ) : (
                            <div className="text-slate-500 text-xs italic text-center py-4">No recent news found for {ticker}.</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SentimentPanel;
