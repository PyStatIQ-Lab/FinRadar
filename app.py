import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize session state
def init_session_state():
    if 'assets' not in st.session_state:
        st.session_state.assets = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'name': ['Apple', 'Microsoft', 'Alphabet', 'Amazon', 'Tesla'],
            'price': [182.32, 420.54, 172.65, 178.22, 265.12],
            'prev_close': [180.45, 418.32, 170.55, 177.89, 260.54],
            'change': [1.87, 2.22, 2.10, 0.33, 4.58],
            'volume': [58000000, 35000000, 28000000, 42000000, 72000000],
            'volatility': [0.28, 0.24, 0.31, 0.35, 0.65],
            'beta': [1.02, 0.98, 1.05, 1.12, 2.10],
            'sector': ['Technology', 'Technology', 'Technology', 'Consumer', 'Automotive'],
            'risk_score': [0.45, 0.38, 0.52, 0.58, 0.82],
            'status': ['Monitored', 'Monitored', 'Monitored', 'Watch', 'High Risk'],
            'hedge_id': ['', '', '', '', ''],
            'last_updated': [datetime.now()] * 5
        })
        
    if 'strategies' not in st.session_state:
        st.session_state.strategies = pd.DataFrame({
            'id': [f'HEDGE{i+1}' for i in range(20)],
            'type': ['Options Collar', 'Futures', 'Pair Trade', 'Put Options'] * 5,
            'cost_bps': np.random.uniform(5, 25, 20),
            'effectiveness': np.random.uniform(0.65, 0.95, 20),
            'status': ['Available'] * 20,
            'asset_symbol': [''] * 20,
            'engaged_until': [0.0] * 20
        })
        
    if 'command_log' not in st.session_state:
        st.session_state.command_log = []
        
    if 'last_news_fetch' not in st.session_state:
        st.session_state.last_news_fetch = time.time() - 300  # 5 min ago
        
    if 'market_pulse' not in st.session_state:
        st.session_state.market_pulse = 0.65
        
    if 'selected_asset' not in st.session_state:
        st.session_state.selected_asset = 'AAPL'
        
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
        
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()

# Fetch real-time market data
def fetch_market_data():
    assets = st.session_state.assets
    for idx, row in assets.iterrows():
        try:
            ticker = yf.Ticker(row['symbol'])
            data = ticker.history(period='1d')
            if not data.empty:
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else row['prev_close']
                current_price = data['Close'].iloc[-1]
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                # Update volatility (simplified)
                hist = ticker.history(period='30d')['Close']
                volatility = hist.pct_change().std() * np.sqrt(252)
                
                assets.at[idx, 'price'] = current_price
                assets.at[idx, 'prev_close'] = prev_close
                assets.at[idx, 'change'] = change_pct
                assets.at[idx, 'volatility'] = volatility
                assets.at[idx, 'last_updated'] = datetime.now()
        except:
            st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: Error fetching data for {row['symbol']}")
    
    st.session_state.assets = assets

# Calculate risk scores
def calculate_risk_scores():
    assets = st.session_state.assets
    for idx, row in assets.iterrows():
        # Risk factors: volatility, beta, price change, volume
        risk_score = (0.4 * min(row['volatility'] / 0.5, 1.0) + \
                     (0.3 * min(abs(row['beta']), 2.0) / 2.0 + \
                     (0.2 * min(abs(row['change']) / 10, 1.0)) + \
                     (0.1 * min(row['volume'] / 100000000, 1.0))
        
        assets.at[idx, 'risk_score'] = min(risk_score, 1.0)
        
        # Update status based on risk
        if risk_score >= 0.8:
            assets.at[idx, 'status'] = 'High Risk'
        elif risk_score >= 0.6:
            assets.at[idx, 'status'] = 'Watch'
        else:
            assets.at[idx, 'status'] = 'Monitored'
    
    st.session_state.assets = assets

# Fetch news and analyze sentiment
def fetch_news():
    current_time = time.time()
    if current_time - st.session_state.last_news_fetch < 300:  # 5 min cache
        return
    
    try:
        # Fetch news data from API
        url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=10"
        response = requests.get(url)
        news_data = response.json().get('data', [])
        
        # Process news
        sentiment_scores = []
        for news in news_data[:5]:  # Process top 5 news
            title = news.get('title', '')
            content = news.get('content', '')[:500]  # First 500 chars
            
            # Sentiment analysis
            blob = TextBlob(title + " " + content)
            sentiment = blob.sentiment.polarity  # -1 to 1
            
            # Store news in session state
            news_entry = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'title': title,
                'sentiment': sentiment,
                'impact': 'High' if abs(sentiment) > 0.5 else 'Medium' if abs(sentiment) > 0.2 else 'Low'
            }
            st.session_state.command_log.append(f"{news_entry['time']}: NEWS: {title} (Sentiment: {sentiment:.2f})")
            
            sentiment_scores.append(sentiment)
        
        # Update market pulse (average sentiment)
        if sentiment_scores:
            st.session_state.market_pulse = (sum(sentiment_scores) / len(sentiment_scores) * 0.5 + 0.5
            
        st.session_state.last_news_fetch = current_time
    except Exception as e:
        st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: News fetch error: {str(e)}")

# Predict future events (simplified)
def predict_future_events():
    assets = st.session_state.assets
    predictions = []
    
    for idx, row in assets.iterrows():
        # Simple prediction based on recent performance
        momentum = 1.0 if row['change'] > 0 else -1.0
        volatility_factor = row['volatility'] * 0.5
        prediction_score = momentum * (0.7 + volatility_factor)
        
        prediction = {
            'symbol': row['symbol'],
            'direction': 'Up' if prediction_score > 0.75 else 'Down' if prediction_score < 0.25 else 'Neutral',
            'confidence': min(abs(prediction_score), 0.95)
        }
        predictions.append(prediction)
    
    return predictions

# Assign hedging strategy
def assign_hedge(asset_symbol, strategy_type):
    assets = st.session_state.assets
    strategies = st.session_state.strategies
    current_time = time.time()
    
    # Find asset
    asset_idx = assets.index[assets['symbol'] == asset_symbol].tolist()
    if not asset_idx:
        return f"Asset {asset_symbol} not found"
    
    asset_idx = asset_idx[0]
    
    # Find available strategy
    available = strategies[(strategies['status'] == 'Available') & 
                          (strategies['type'] == strategy_type)]
    
    if available.empty:
        return "No available strategies of this type"
    
    strategy = available.iloc[0]
    strategy_id = strategy['id']
    
    # Update strategy
    strategies.loc[strategies['id'] == strategy_id, 
                   ['status', 'asset_symbol', 'engaged_until']] = \
        ['Active', asset_symbol, current_time + 3600]  # Active for 1 hour
    
    # Update asset
    assets.at[asset_idx, 'hedge_id'] = strategy_id
    assets.at[asset_idx, 'status'] = f"Hedged ({strategy_type})"
    
    st.session_state.assets = assets
    st.session_state.strategies = strategies
    
    return f"Hedge {strategy_id} ({strategy_type}) assigned to {asset_symbol}"

# Update hedging strategies
def update_hedges():
    strategies = st.session_state.strategies
    assets = st.session_state.assets
    current_time = time.time()
    
    active_strategies = strategies[strategies['status'] == 'Active']
    
    for idx, strategy in active_strategies.iterrows():
        if current_time >= strategy['engaged_until']:
            # Strategy expired
            asset_symbol = strategy['asset_symbol']
            strategies.at[idx, 'status'] = 'Available'
            strategies.at[idx, 'asset_symbol'] = ''
            strategies.at[idx, 'engaged_until'] = 0.0
            
            # Update asset
            asset_idx = assets.index[assets['symbol'] == asset_symbol].tolist()
            if asset_idx:
                asset_idx = asset_idx[0]
                assets.at[asset_idx, 'hedge_id'] = ''
                calculate_risk_scores()  # Recalculate risk
    
    st.session_state.strategies = strategies
    st.session_state.assets = assets

# Create financial dashboard
def create_dashboard():
    assets = st.session_state.assets
    strategies = st.session_state.strategies
    
    # Create Plotly figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatterpolar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "heatmap"}]],
        subplot_titles=("Risk Radar", "Price Performance", "Sector Exposure", "Correlation Matrix"),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Risk Radar
    categories = ['Volatility', 'Beta', 'Price Change', 'Volume', 'Liquidity']
    max_vals = [0.7, 2.5, 10, 100, 1]
    
    for idx, row in assets.iterrows():
        values = [
            row['volatility'] / max_vals[0],
            min(row['beta'], max_vals[1]) / max_vals[1],
            min(abs(row['change']), max_vals[2]) / max_vals[2],
            min(row['volume']/1000000, max_vals[3]) / max_vals[3],
            0.8  # Placeholder for liquidity
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['symbol'],
            line=dict(color='red' if row['risk_score'] > 0.7 else 'orange' if row['risk_score'] > 0.5 else 'green'),
            opacity=0.7
        ), row=1, col=1)
    
    fig.update_polars(row=1, col=1, bgcolor='#1a1a3a', radialaxis=dict(visible=True, range=[0,1]))
    
    # Price Performance
    for idx, row in assets.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['symbol']],
            y=[row['change']],
            mode='markers',
            marker=dict(
                size=abs(row['change'])*5 + 10,
                color='green' if row['change'] > 0 else 'red',
                opacity=0.7
            ),
            name=row['symbol'],
            text=f"{row['change']:.2f}%",
            hoverinfo='text'
        ), row=1, col=2)
    
    fig.update_xaxes(title_text="Assets", row=1, col=2)
    fig.update_yaxes(title_text="Price Change (%)", row=1, col=2)
    
    # Sector Exposure
    sector_exposure = assets.groupby('sector')['risk_score'].mean().reset_index()
    fig.add_trace(go.Bar(
        x=sector_exposure['sector'],
        y=sector_exposure['risk_score'],
        marker_color=['#ff3333', '#ff9900', '#ffff00', '#00ccff'],
        opacity=0.8
    ), row=2, col=1)
    
    fig.update_xaxes(title_text="Sector", row=2, col=1)
    fig.update_yaxes(title_text="Average Risk Score", row=2, col=1)
    
    # Correlation Matrix (simulated)
    symbols = assets['symbol'].tolist()
    corr_matrix = pd.DataFrame(
        np.random.uniform(-1, 1, size=(len(symbols), len(symbols))),
        columns=symbols,
        index=symbols
    )
    
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=symbols,
        y=symbols,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Financial Risk Dashboard",
        template="plotly_dark",
        paper_bgcolor="#0a0a23",
        plot_bgcolor="#1a1a3a",
        font=dict(color="#00ffcc"),
        showlegend=False
    )
    
    return fig

# Create asset monitor
def create_asset_monitor():
    assets = st.session_state.assets
    fig = go.Figure()
    
    for idx, row in assets.iterrows():
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=row['risk_score'] * 100,
            delta={'reference': 60, 'increasing': {'color': "#ff3333"}},
            domain={'x': [0.1 * (idx % 5), 0.1 * (idx % 5) + 0.08], 'y': [0.5, 1]},
            title={'text': f"{row['symbol']}<br>{row['price']:.2f}"},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#00ffcc"},
                'bar': {'color': "#ff3333" if row['risk_score'] > 0.7 else "#ff9900" if row['risk_score'] > 0.5 else "#00ccff"},
                'bgcolor': "#1a1a3a",
                'steps': [
                    {'range': [0, 50], 'color': '#00ccff'},
                    {'range': [50, 70], 'color': '#ff9900'},
                    {'range': [70, 100], 'color': '#ff3333'}],
            }
        ))
    
    fig.update_layout(
        height=300,
        margin=dict(t=20, b=0),
        paper_bgcolor="#0a0a23",
        font=dict(color="#00ffcc")
    )
    
    return fig

# Main app
def main():
    st.set_page_config(
        page_title="Quantum Financial Risk Mitigation",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    :root {
        --primary: #00ffcc;
        --secondary: #cc00ff;
        --bg-dark: #0a0a23;
        --panel-bg: #1a1a3a;
        --warning: #ff9900;
        --danger: #ff3333;
        --success: #00ccff;
    }
    
    body {
        background-color: var(--bg-dark);
        color: var(--primary);
        font-family: 'Courier New', monospace;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a23 0%, #1a1a3a 100%);
    }
    
    .stButton>button {
        background-color: var(--panel-bg);
        color: var(--primary);
        border: 2px solid var(--primary);
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 0 10px var(--primary);
    }
    
    .stButton>button:hover {
        background-color: var(--primary);
        color: var(--bg-dark);
        box-shadow: 0 0 15px var(--primary);
    }
    
    .stSelectbox>div>div {
        background-color: var(--panel-bg);
        color: var(--primary);
        border: 2px solid var(--primary);
    }
    
    .stTextInput>div>div>input {
        background-color: var(--panel-bg);
        color: var(--primary);
        border: 2px solid var(--primary);
    }
    
    .stDataFrame {
        background-color: var(--panel-bg);
        border: 2px solid var(--primary);
        border-radius: 5px;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--primary);
        text-shadow: 0 0 10px var(--primary);
    }
    
    .panel {
        background-color: var(--panel-bg);
        border: 2px solid var(--primary);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 0 15px var(--primary);
    }
    
    .log-panel {
        max-height: 200px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        background-color: #0a0a23;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid var(--primary);
    }
    
    .data-feed {
        background-color: #0a0a23;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid var(--success);
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Header
    st.title("📊 Quantum Financial Risk Mitigation Dashboard")
    st.markdown("---")
    
    # Control Panel
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            st.session_state.selected_asset = st.selectbox(
                "Select Asset", 
                st.session_state.assets['symbol'].tolist(),
                key='asset_selector'
            )
            
        with col2:
            strategy_type = st.selectbox(
                "Hedging Strategy", 
                ["Options Collar", "Futures", "Pair Trade", "Put Options"],
                key='strategy_selector'
            )
            
        with col3:
            st.session_state.risk_model = st.selectbox(
                "Risk Model", 
                ["Volatility-Based", "Correlation Matrix", "Monte Carlo"],
                key='model_selector'
            )
            
        with col4:
            st.write("")
            st.write("")
            execute_btn = st.button("Apply Hedge", key='execute_btn')
    
    # Market Pulse
    pulse_colors = {
        'high': '#ff3333',
        'medium': '#ff9900',
        'low': '#00ccff'
    }
    pulse_status = 'high' if st.session_state.market_pulse > 0.7 else 'medium' if st.session_state.market_pulse > 0.5 else 'low'
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Market Pulse", f"{st.session_state.market_pulse*100:.1f}%", 
                  delta_color="inverse", 
                  help="Market sentiment based on news analysis")
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {pulse_colors[pulse_status]} {st.session_state.market_pulse*100}%, #1a1a3a {st.session_state.market_pulse*100}%);
                    height: 30px; 
                    border-radius: 15px; 
                    border: 1px solid {pulse_colors[pulse_status]};
                    margin-top: 10px;"></div>
        """, unsafe_allow_html=True)
    
    # Simulation Control
    with col3:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.session_state.simulation_running:
                if st.button("⏹️ Stop Simulation"):
                    st.session_state.simulation_running = False
            else:
                if st.button("▶️ Start Simulation"):
                    st.session_state.simulation_running = True
                    st.session_state.last_update_time = time.time()
        
        with col_b:
            st.session_state.sim_speed = st.slider("Simulation Speed", 1, 10, 3, 1,
                                                   key='speed_slider')
    
    # Run simulation step
    if st.session_state.simulation_running:
        current_time = time.time()
        if current_time - st.session_state.last_update_time >= (10 / st.session_state.sim_speed):
            fetch_market_data()
            calculate_risk_scores()
            fetch_news()
            update_hedges()
            st.session_state.last_update_time = current_time
            st.experimental_rerun()
    
    # Handle button actions
    if execute_btn:
        result = assign_hedge(st.session_state.selected_asset, strategy_type)
        st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: {result}")
        st.experimental_rerun()
    
    # Asset Monitor
    st.subheader("Asset Risk Monitor")
    asset_fig = create_asset_monitor()
    st.plotly_chart(asset_fig, use_container_width=True)
    
    # Main Dashboard
    st.subheader("Financial Risk Dashboard")
    dashboard_fig = create_dashboard()
    st.plotly_chart(dashboard_fig, use_container_width=True)
    
    # Data Feed
    st.subheader("Market Event Feed")
    predictions = predict_future_events()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.container():
            if st.session_state.command_log:
                log_html = "<div class='log-panel'>"
                for entry in st.session_state.command_log[-10:]:
                    if "NEWS" in entry:
                        log_html += f"<div style='color:#00ccff; margin: 5px 0;'>{entry}</div>"
                    elif "Error" in entry:
                        log_html += f"<div style='color:#ff3333; margin: 5px 0;'>{entry}</div>"
                    elif "Hedge" in entry:
                        log_html += f"<div style='color:#00ffcc; margin: 5px 0;'>{entry}</div>"
                    else:
                        log_html += f"<div style='color:#ffffff; margin: 5px 0;'>{entry}</div>"
                log_html += "</div>"
                st.markdown(log_html, unsafe_allow_html=True)
            else:
                st.markdown("<div class='log-panel'>No market events detected</div>", unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.subheader("Predictions")
            for pred in predictions[:3]:
                color = "#00ccff" if pred['direction'] == 'Up' else "#ff3333" if pred['direction'] == 'Down' else "#ffff00"
                st.markdown(f"""
                <div class='data-feed'>
                    <b>{pred['symbol']}</b>: {pred['direction']}<br>
                    Confidence: {pred['confidence']:.0%}
                </div>
                """, unsafe_allow_html=True)
    
    # Data Tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Portfolio")
        display_assets = st.session_state.assets[['symbol', 'name', 'price', 'change', 'volatility', 'risk_score', 'status']]
        st.dataframe(display_assets.style.background_gradient(
            subset=['risk_score', 'change'], 
            cmap='RdYlGn_r' if st.session_state.risk_model == "Volatility-Based" else 'viridis'
        ), height=300)
    
    with col2:
        st.subheader("Hedging Strategies")
        display_strategies = st.session_state.strategies[['id', 'type', 'cost_bps', 'effectiveness', 'status', 'asset_symbol']]
        st.dataframe(display_strategies, height=300)

if __name__ == "__main__":
    main()
