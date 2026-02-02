import React, { useState } from 'react';
import axios from 'axios';

const API_URL = "http://localhost:8000/predict";

function App() {
    const [message, setMessage] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handlePredict = async () => {
        if (!message.trim()) return;

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await axios.post(API_URL, { message });
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError("Failed to connect to the server. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-4">
            <div className="bg-slate-800 p-8 rounded-2xl shadow-2xl max-w-2xl w-full border border-slate-700">
                <h1 className="text-4xl font-bold text-center mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
                    Spam Detective
                </h1>
                <p className="text-slate-400 text-center mb-8">
                    AI-Powered Email Classification
                </p>

                <div className="space-y-4">
                    <textarea
                        className="w-full h-40 bg-slate-900 border border-slate-700 rounded-xl p-4 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all resize-none placeholder-slate-500"
                        placeholder="Paste email content here..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                    ></textarea>

                    <button
                        onClick={handlePredict}
                        disabled={loading || !message.trim()}
                        className={`w-full py-4 rounded-xl font-bold text-lg transition-all duration-300 ${loading || !message.trim()
                                ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                                : 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white shadow-lg hover:shadow-blue-500/25'
                            }`}
                    >
                        {loading ? (
                            <span className="flex items-center justify-center gap-2">
                                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Analyzing...
                            </span>
                        ) : (
                            'Analyze Message'
                        )}
                    </button>
                </div>

                {error && (
                    <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-center">
                        {error}
                    </div>
                )}

                {result && (
                    <div className="mt-8 animate-fade-in text-center">
                        <div className={`inline-block px-6 py-2 rounded-full text-sm font-bold tracking-wide mb-4 ${result.prediction === 'SPAM'
                                ? 'bg-red-500/20 text-red-400 border border-red-500/20'
                                : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/20'
                            }`}>
                            {result.prediction}
                        </div>

                        <div className="text-6xl font-black mb-2 text-white">
                            {(result.confidence * 100).toFixed(1)}%
                        </div>
                        <p className="text-slate-400 font-medium">Confidence Score</p>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
