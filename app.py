import streamlit as st
import plotly.graph_objects as go
import torch
import sys
import os
import joblib

# --- PATH INJECTION (Fail-safe for imports) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- MODULAR IMPORTS ---
try:
    from src.data_fetcher import get_alpha_live_data
    from src.rag_retriever import build_research_index, retrieve_alpha_context, embedder
    from src.model_engine import SentimetrixTCN, predict_signal
    from src.intelligence_engine import IntelligenceEngine
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error(f"❌ Import Error: {e}. Please ensure your 'src' folder contains empty __init__.py and all engine files.")
    st.stop()

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Sentimetrix AlphaFeed", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- CSS FIX HERE (unsafe_allow_html instead of unsafe_allow_index) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 8px; border: 1px solid #2d303e; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 4px; color: #fff; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1a1c24; border-bottom: 2px solid #4CAF50; }
    </style>
    """, unsafe_allow_html=True) 

# --- 2. ASSET LOADING (Cached) ---
@st.cache_resource
def load_deep_assets():
    # Initialize Model
    model = SentimetrixTCN(input_size=17)
    # If you have trained weights, load them here:
    if os.path.exists("models/alpha_weights.pth"):
        model.load_state_dict(torch.load("models/alpha_weights.pth"))
        # print("Loaded trained weights.")
    else:
        print("Warning: No trained weights found. Using random init.")
    
    model.eval()
    
    # Initialize RAG Vector DB
    index = build_research_index()
    
    # Load Scaler
    scaler = None
    if os.path.exists("models/scaler.pkl"):
        scaler = joblib.load("models/scaler.pkl")
        
    # Initialize Intelligence Engine (LLM)
    llm = IntelligenceEngine()
        
    return model, index, scaler, llm

model, faiss_index, scaler, llm = load_deep_assets()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("⚡ AlphaFeed Control")
st.sidebar.caption("Deep Multimodal Analysis Engine")

ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL", help="Enter any valid US stock ticker").upper()
news_input = st.sidebar.text_area("Live News Context:", 
                                  value="Tech sector rallies as AI demand surges, boosting semiconductor stocks.",
                                  height=100)

if st.sidebar.button("Run Deep Analysis", type="primary"):
    
    # --- STEP A: LIVE DATA INGESTION ---
    with st.spinner(f"📡 Fetching live data for {ticker}..."):
        try:
            df, kpis = get_alpha_live_data(ticker)
            if df.empty:
                st.error("No data found. Please check the ticker symbol.")
                st.stop()
        except Exception as e:
            st.error(f"API Error: {e}")
            st.stop()

    # --- STEP B: HEADER METRICS ---
    st.markdown(f"## 📊 {ticker} Market Intelligence")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Price", f"${df['close'].iloc[-1]:.2f}", delta=f"{df['close'].pct_change().iloc[-1]*100:.2f}%")
    col2.metric("Market Cap", kpis.get("Market Cap", "N/A"))
    col3.metric("52W High", f"${kpis.get('52W High', 0):.2f}")
    col4.metric("RSI (14D)", f"{df['RSI_14'].iloc[-1]:.2f}")

    # --- STEP C: DEEP DIVE TABS ---
    tab_chart, tab_brain, tab_chat, tab_data = st.tabs(["📉 Price Action", "🧠 AI Reasoning", "💬 AI Analyst", "💾 Raw Data"])

    # === TAB 1: INTERACTIVE CHART ===
    with tab_chart:
        st.subheader(f"Technical Analysis: {ticker}")
        
        # Create Subplots: Row 1=Price, Row 2=RSI, Row 3=MACD
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=(f"{ticker} Price", "RSI (14)", "MACD"))

        # 1. Candlestick Price
        fig.add_trace(go.Candlestick(
            x=df.index[-100:],
            open=df['open'][-100:], high=df['high'][-100:],
            low=df['low'][-100:], close=df['close'][-100:],
            name="OHLC"
        ), row=1, col=1)
        
        # 2. RSI
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['RSI_14'][-100:], name="RSI", line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 3. MACD
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['MACD_12_26_9'][-100:], name="MACD", line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['MACDs_12_26_9'][-100:], name="Signal", line=dict(color='orange')), row=3, col=1)
        fig.add_bar(x=df.index[-100:], y=df['MACDh_12_26_9'][-100:], name="Hist", marker_color='gray', row=3, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=700, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig)

    # === TAB 2: AI REASONING (The "Brain") ===
    with tab_brain:
        st.subheader("🤖 Neural Engine Output")
        
        # 1. Retrieve Context
        rules = retrieve_alpha_context(news_input, faiss_index)
        
        # 2. Run Inference
        with st.spinner("Calculating Alpha Signal..."):
            pred_class, conf, weights = predict_signal(model, df, rules, embedder, scaler)
        
        # 3. Dynamic Signal Display
        signals = {
            0: ("🔴 STRONG SELL", "Bearish divergence detected."), 
            1: ("⚪ NEUTRAL / HOLD", "Market consolidation phase."), 
            2: ("🟢 STRONG BUY", "Bullish momentum confirmed.")
        }
        label, description = signals[pred_class]
        
        # Create a visually distinct signal box
        st.markdown(f"""
            <div style='background-color: #1a1c24; padding: 20px; border-radius: 10px; border-left: 5px solid {"#ff4b4b" if pred_class==0 else "#4CAF50"};'>
                <h2 style='margin:0; color: {"#ff4b4b" if pred_class==0 else "#4CAF50"};'>{label}</h2>
                <p style='margin:0; font-size: 1.1em;'>Confidence: <b>{conf:.1f}%</b></p>
                <p style='margin-top:5px; color: #888;'><i>{description}</i></p>
            </div>
        """, unsafe_allow_html=True)
        
        # 4. Explainable AI (Attention Weights)
        st.write("---")
        st.markdown("### 🧠 Logic Attribution")
        st.caption("Which expert rules did the TCN model focus on?")
        
        for i, rule in enumerate(rules):
            col_txt, col_bar = st.columns([3, 1])
            with col_txt:
                st.info(f"💡 {rule}")
            with col_bar:
                st.progress(float(weights[i]), text="Weight")

    # === TAB 3: AI ANALYST CHAT ===
    with tab_chat:
        st.subheader("💬 Ask the Intelligence Engine")
        st.caption("Powered by RAG + Generative Transformers")
        
        # Chat History Session State
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about the technicals, strategy, or market logic..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = llm.generate_analysis(
                            context_rules=rules, 
                            technical_signal=pred_class, 
                            ticker=ticker, 
                            user_query=prompt
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Analysis Generation Error: {e}")

    # === TAB 4: DATA INSPECTION ===
    with tab_data:
        st.dataframe(df.tail(50).style.highlight_max(axis=0), width='stretch')

else:
    # Empty State
    st.info("👈 Enter a stock ticker and news context in the sidebar to start the AlphaFeed Engine.")