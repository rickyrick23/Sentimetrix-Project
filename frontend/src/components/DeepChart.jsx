import React, { useEffect, useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import axios from 'axios';
import { Loader2 } from 'lucide-react';

const DeepChart = ({ ticker }) => {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const res = await axios.get(`http://localhost:8000/market-data/${ticker}`);
                // Transform data if necessary. Backend returns 'history' list.
                // Assuming history has 'date', 'close', 'rsi_14', etc.
                setData(res.data.history);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [ticker]);

    if (loading) return <div className="h-96 flex items-center justify-center"><Loader2 className="animate-spin text-blue-500" size={48} /></div>;

    return (
        <div className="space-y-6">
            {/* Price Chart */}
            <div className="bg-slate-800 p-4 rounded-xl shadow-lg border border-slate-700">
                <h3 className="text-xl font-bold mb-4 text-blue-400">Price Action ({ticker})</h3>
                <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data}>
                            <defs>
                                <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="date" hide />
                            <YAxis domain={['auto', 'auto']} stroke="#94a3b8" />
                            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                            <Area type="monotone" dataKey="close" stroke="#3b82f6" fillOpacity={1} fill="url(#colorClose)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* RSI Chart */}
            <div className="bg-slate-800 p-4 rounded-xl shadow-lg border border-slate-700">
                <h3 className="text-lg font-bold mb-2 text-purple-400">RSI (14)</h3>
                <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="date" hide />
                            <YAxis domain={[0, 100]} stroke="#94a3b8" />
                            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                            <Line type="monotone" dataKey="RSI_14" stroke="#a855f7" dot={false} strokeWidth={2} />
                            {/* Reference Lines at 30 and 70 (approx via grid or custom RefLine if imported) */}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default DeepChart;
