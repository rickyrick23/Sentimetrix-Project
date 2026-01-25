import React, { useState } from 'react';
import axios from 'axios';
import { Send, Bot, User } from 'lucide-react';

const AIAnalyst = ({ ticker, context, signal }) => {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: `Hello! I am your AI Analyst for ${ticker}. Ask me anything about the technicals.` }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);

    // Reset chat when ticker changes
    React.useEffect(() => {
        setMessages([
            { role: 'assistant', content: `Hello! I am your AI Analyst for ${ticker}. Ask me anything about the technicals.` }
        ]);
    }, [ticker]);

    const handleSend = async () => {
        if (!input.trim()) return;
        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const res = await axios.post('http://localhost:8000/chat', {
                ticker: ticker,
                query: userMsg.content,
                context: context || "No news context provided.",
                signal: signal || 1
            });

            const aiMsg = { role: 'assistant', content: res.data.response };
            setMessages(prev => [...prev, aiMsg]);
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not reach the Intelligence Engine." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 border-l border-slate-700">
            <div className="p-4 border-b border-slate-700 bg-slate-800">
                <h2 className="flex items-center gap-2 text-xl font-bold text-emerald-400">
                    <Bot size={24} /> AI Analyst
                </h2>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((m, i) => (
                    <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[80%] p-3 rounded-xl ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-200'}`}>
                            <div className="flex items-center gap-2 mb-1 opacity-50 text-xs">
                                {m.role === 'user' ? <User size={12} /> : <Bot size={12} />}
                                {m.role === 'user' ? 'You' : 'AI'}
                            </div>
                            <div>{m.content}</div>
                        </div>
                    </div>
                ))}
                {loading && <div className="text-slate-500 text-sm animate-pulse">Thinking...</div>}
            </div>

            <div className="p-4 border-t border-slate-700 bg-slate-800">
                <div className="flex gap-2">
                    <input
                        className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about trend, MACD, etc..."
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                    />
                    <button
                        onClick={handleSend}
                        className="bg-emerald-500 hover:bg-emerald-600 text-white p-2 rounded-lg transition-colors"
                    >
                        <Send size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default AIAnalyst;
