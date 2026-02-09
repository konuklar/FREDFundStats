import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas_datareader.data as web
import warnings
from scipy import stats
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FRED Mutual Fund Flows - Institutional Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for superior institutional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        text-align: center;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #6B7280;
        margin-bottom: 2.5rem;
        text-align: center;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    .institution-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 0 0 20px 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        color: white;
    }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid #F3F4F6;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #3B82F6, #10B981);
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1F2937;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-subtext {
        font-size: 0.85rem;
        color: #9CA3AF;
        margin-top: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        font-weight: 700;
        margin: 3rem 0 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #3B82F6, #10B981) 1;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .section-header::before {
        content: '▸';
        font-size: 2rem;
        color: #3B82F6;
    }
    .analysis-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        padding: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        padding: 0 28px;
        border-radius: 10px;
        background: white;
        border: 2px solid #E5E7EB;
        font-weight: 600;
        color: #4B5563;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.4);
        transform: scale(1.05);
    }
    .data-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-success { background: #D1FAE5; color: #065F46; }
    .badge-warning { background: #FEF3C7; color: #92400E; }
    .badge-error { background: #FEE2E2; color: #991B1B; }
    .badge-info { background: #DBEAFE; color: #1E40AF; }
    .stat-box {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    .trend-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-left: 10px;
    }
    .trend-up { background: #10B981; color: white; }
    .trend-down { background: #EF4444; color: white; }
    .trend-neutral { background: #6B7280; color: white; }
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #F3F4F6;
    }
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: #6B7280;
        font-size: 0.9rem;
        margin-top: 4rem;
        border-top: 1px solid #E5E7EB;
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 0 0 20px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title with superior institutional styling
st.markdown("""
<div class="institution-banner">
    <h1 class="main-header">Institutional Fund Flow Analysis</h1>
    <p class="sub-header">Comprehensive U.S. Mutual Fund & ETF Flow Analytics | Federal Reserve Economic Data (FRED)</p>
    <div style="display: flex; gap: 2rem; justify-content: center; margin-top: 1.5rem;">
        <span class="data-badge badge-success">Real-time FRED Data</span>
        <span class="data-badge badge-warning">Advanced Statistical Analysis</span>
        <span class="data-badge badge-info">Professional Risk Metrics</span>
        <span class="data-badge badge-error">Institutional Grade Reporting</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced FRED Series IDs with fallbacks
FRED_SERIES = {
    'Total Mutual Fund Assets': {
        'monthly': 'TOTMFS',
        'weekly': 'H8B3092NCBA',
        'description': 'Total Mutual Fund Assets',
        'category': 'total'
    },
    'Money Market Funds': {
        'monthly': 'MMMFFAQ027S',
        'weekly': 'WMFAA',
        'description': 'Money Market Fund Assets',
        'category': 'cash'
    },
    'Equity Funds': {
        'monthly': 'TOTCI',
        'weekly': 'H8B3053NCBA',
        'description': 'Equity Mutual Fund Assets',
        'category': 'equity'
    },
    'Bond Funds': {
        'monthly': 'BOGZ1FL413065005Q',  # Fixed: Bond fund assets quarterly
        'weekly': 'H8B3094NCBA',  # Commercial paper
        'description': 'Bond/Income Fund Assets',
        'category': 'fixed_income'
    },
    'Municipal Bond Funds': {
        'monthly': 'MBCI',
        'weekly': 'H8B3095NCBA',
        'description': 'Municipal Bond Fund Assets',
        'category': 'fixed_income'
    },
    'Government Bond Funds': {
        'monthly': 'BOGZ1FL413065015Q',  # Treasury funds
        'weekly': 'H8B3093NCBA',
        'description': 'Government Bond Fund Assets',
        'category': 'fixed_income'
    },
    'Corporate Bond Funds': {
        'monthly': 'BOGZ1FL413065025Q',
        'weekly': None,
        'description': 'Corporate Bond Fund Assets',
        'category': 'fixed_income'
    }
}

# Additional market data for correlation analysis
MARKET_INDICES = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DJIA': '^DJI',
    'Russell 2000': '^RUT',
    'Bloomberg Barclays US Agg Bond': 'AGG',
    '10-Year Treasury Yield': '^TNX'
}

@st.cache_data(ttl=3600, show_spinner="📊 Fetching institutional data from FRED...")
def fetch_fred_series(series_id, start_date, end_date, retries=3):
    """Robust FRED data fetching with error handling"""
    for attempt in range(retries):
        try:
            df = web.DataReader(series_id, 'fred', start=start_date, end=end_date)
            if not df.empty:
                return df
        except Exception as e:
            if attempt == retries - 1:
                st.warning(f"Could not fetch {series_id}: {str(e)[:100]}")
                return pd.DataFrame()
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_all_fred_data(selected_series, start_date, frequency):
    """Fetch data for all selected FRED series with comprehensive error handling"""
    data_dict = {}
    end_date = datetime.today()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (category, info) in enumerate(selected_series.items()):
        status_text.text(f"Loading {category} data...")
        
        series_id = info.get(frequency)
        if series_id:
            df = fetch_fred_series(series_id, start_date, end_date)
            
            if not df.empty:
                # Calculate flows
                df_flows = df.diff()
                df_flows.columns = [f'{category}_Flow']
                
                # Calculate percentage changes
                df_pct = df.pct_change() * 100
                df_pct.columns = [f'{category}_Pct_Change']
                
                # Store comprehensive data
                data_dict[category] = {
                    'assets': df,
                    'flows': df_flows,
                    'pct_change': df_pct,
                    'description': info['description'],
                    'category': info['category']
                }
        
        progress_bar.progress((idx + 1) / len(selected_series))
    
    progress_bar.empty()
    status_text.empty()
    
    return data_dict

@st.cache_data(ttl=3600)
def fetch_market_data(start_date):
    """Fetch market indices for correlation analysis"""
    market_data = {}
    end_date = datetime.today()
    
    for name, symbol in MARKET_INDICES.items():
        try:
            if symbol.startswith('^'):
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                # Resample to monthly for consistency
                monthly_data = data['Adj Close'].resample('M').last()
                monthly_pct = monthly_data.pct_change() * 100
                
                market_data[name] = {
                    'price': monthly_data,
                    'returns': monthly_pct
                }
        except Exception as e:
            continue
    
    return market_data

def create_superior_metrics(data_dict, frequency):
    """Create institutional-grade metrics dashboard"""
    st.markdown("## 🎯 Executive Dashboard")
    
    cols = st.columns(5)
    metrics_summary = []
    
    for idx, (category, data) in enumerate(list(data_dict.items())[:5]):
        with cols[idx]:
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0]
                avg_flow = data['flows'].mean().iloc[0]
                flow_std = data['flows'].std().iloc[0]
                
                # Calculate trend
                recent_avg = data['flows'].iloc[-3:].mean().iloc[0] if len(data['flows']) >= 3 else latest_flow
                trend = "up" if latest_flow > recent_avg else "down" if latest_flow < recent_avg else "neutral"
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{category}</div>
                    <div class='metric-value'>${abs(latest_flow):,.0f}M</div>
                    <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                        <span class="trend-indicator trend-{trend}">
                            {'↗' if trend == 'up' else '↘' if trend == 'down' else '→'} 
                            {'+' if latest_flow > 0 else ''}{latest_flow:,.0f}M
                        </span>
                    </div>
                    <div class='metric-subtext'>
                        σ: ${flow_std:,.0f}M | Avg: ${avg_flow:,.0f}M
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                metrics_summary.append({
                    'Category': category,
                    'Latest Flow': f"${latest_flow:,.0f}M",
                    '3M Trend': f"{'↑' if trend == 'up' else '↓' if trend == 'down' else '→'}",
                    'Volatility': f"${flow_std:,.0f}M",
                    'Sharpe Ratio': f"{latest_flow/flow_std:.2f}" if flow_std != 0 else "N/A"
                })
    
    return pd.DataFrame(metrics_summary)

def create_enhanced_growth_charts(data_dict):
    """Create superior normalized growth charts without overlapping"""
    st.markdown('<div class="section-header">📈 Advanced Growth Dynamics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        norm_method = st.selectbox(
            "Normalization Method",
            ["Cumulative Index (Base=100)", "Z-Score Standardization", "Log Returns", 
             "Rolling Sharpe Ratio", "Risk-Adjusted Returns"],
            key="norm_method_select"
        )
    
    with col2:
        window_size = st.slider("Analysis Window", 1, 24, 12, key="analysis_window")
    
    with col3:
        chart_type = st.radio("Chart Style", ["Line", "Area", "Bar"], horizontal=True)
    
    # Prepare normalized data
    normalized_series = []
    dates = None
    
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets'].copy()
            assets.columns = [category]
            
            if dates is None:
                dates = assets.index
            
            # Apply selected normalization
            if norm_method == "Cumulative Index (Base=100)":
                norm_data = 100 * assets / assets.iloc[0]
            elif norm_method == "Z-Score Standardization":
                norm_data = (assets - assets.mean()) / assets.std()
            elif norm_method == "Log Returns":
                norm_data = np.log(assets / assets.shift(1)).fillna(0) * 100
            elif norm_method == "Rolling Sharpe Ratio":
                returns = assets.pct_change()
                rolling_sharpe = returns.rolling(window=window_size).mean() / returns.rolling(window=window_size).std()
                norm_data = rolling_sharpe.fillna(0)
            elif norm_method == "Risk-Adjusted Returns":
                returns = assets.pct_change()
                norm_data = returns.rolling(window=window_size).mean() / returns.rolling(window=window_size).std()
                norm_data = norm_data.fillna(0)
            
            normalized_series.append(norm_data)
    
    if not normalized_series:
        st.warning("No data available for growth analysis")
        return
    
    # Combine data
    combined_df = pd.concat(normalized_series, axis=1).dropna()
    
    # Create superior visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Normalized Growth Trajectories', 
                       'Rolling Correlation Matrix',
                       'Risk-Return Profile',
                       'Volatility Surface'],
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
        specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
               [{'type': 'scatter'}, {'type': 'surface'}]]
    )
    
    # 1. Normalized growth trajectories
    colors = px.colors.qualitative.Vivid
    for i, column in enumerate(combined_df.columns):
        y_data = combined_df[column].rolling(window=3).mean() if window_size > 1 else combined_df[column]
        
        fig.add_trace(
            go.Scatter(
                x=combined_df.index,
                y=y_data,
                name=column,
                mode='lines',
                line=dict(width=3, color=colors[i % len(colors)]),
                hovertemplate=f'%{{x|%b %Y}}<br>{column}: %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Rolling correlation matrix
    if len(combined_df.columns) > 1:
        rolling_corr = combined_df.rolling(window=window_size).corr().dropna()
        if not rolling_corr.empty:
            latest_corr = rolling_corr.iloc[-len(combined_df.columns):]
            
            fig.add_trace(
                go.Heatmap(
                    z=latest_corr.values,
                    x=combined_df.columns,
                    y=combined_df.columns,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Correlation"),
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
    
    # 3. Risk-return profile
    risk_return_data = []
    for column in combined_df.columns:
        returns = combined_df[column].pct_change().dropna()
        if len(returns) > 0:
            annual_return = returns.mean() * 12
            annual_vol = returns.std() * np.sqrt(12)
            sharpe = annual_return / annual_vol if annual_vol != 0 else 0
            
            risk_return_data.append({
                'Category': column,
                'Return': annual_return,
                'Risk': annual_vol,
                'Sharpe': sharpe
            })
    
    if risk_return_data:
        rr_df = pd.DataFrame(risk_return_data)
        
        fig.add_trace(
            go.Scatter(
                x=rr_df['Risk'],
                y=rr_df['Return'],
                mode='markers+text',
                text=rr_df['Category'],
                textposition='top center',
                marker=dict(
                    size=rr_df['Sharpe'].abs() * 50 + 10,
                    color=rr_df['Sharpe'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                hovertemplate='%{text}<br>Risk: %{x:.3f}<br>Return: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text=f"Institutional Growth Analysis: {norm_method}",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("#### 📊 Statistical Summary")
    
    stats_data = []
    for column in combined_df.columns:
        series = combined_df[column].dropna()
        if len(series) > 1:
            stats_data.append({
                'Category': column,
                'Mean': f"{series.mean():.4f}",
                'Std Dev': f"{series.std():.4f}",
                'Skewness': f"{series.skew():.4f}",
                'Kurtosis': f"{series.kurtosis():.4f}",
                'Max Drawdown': f"{((series.expanding().max() - series) / series.expanding().max()).max():.2%}",
                'Sharpe Ratio': f"{(series.mean() / series.std() * np.sqrt(12)):.3f}" if series.std() != 0 else "N/A"
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

def create_comprehensive_flow_analysis(data_dict, frequency):
    """Create superior flow analysis with advanced metrics"""
    st.markdown('<div class="section-header">💸 Advanced Flow Dynamics</div>', unsafe_allow_html=True)
    
    # Inflow vs Outflow analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Detailed Inflow Analysis")
        
        inflow_data = {}
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].copy()
                inflows = flows[flows.iloc[:, 0] > 0]
                if not inflows.empty:
                    inflow_data[category] = inflows.iloc[:, 0]
        
        if inflow_data:
            inflows_df = pd.DataFrame(inflow_data).fillna(0)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Monthly Inflow Composition', 'Cumulative Inflow Trends'],
                vertical_spacing=0.15
            )
            
            # Stacked bar chart for inflows
            colors = px.colors.qualitative.Pastel
            for i, column in enumerate(inflows_df.columns):
                fig.add_trace(
                    go.Bar(
                        x=inflows_df.index,
                        y=inflows_df[column],
                        name=column,
                        marker_color=colors[i % len(colors)],
                        hovertemplate=f'%{{x|%b %Y}}<br>{column}: $%{{y:,.0f}}M<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Cumulative inflows
            cumulative_inflows = inflows_df.cumsum()
            for i, column in enumerate(cumulative_inflows.columns):
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_inflows.index,
                        y=cumulative_inflows[column],
                        name=f"{column} Cumulative",
                        mode='lines',
                        line=dict(width=2, color=colors[i % len(colors)]),
                        hovertemplate=f'%{{x|%b %Y}}<br>{column}: $%{{y:,.0f}}M<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, barmode='stack', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### 📉 Detailed Outflow Analysis")
        
        outflow_data = {}
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].copy()
                outflows = flows[flows.iloc[:, 0] < 0].abs()
                if not outflows.empty:
                    outflow_data[category] = outflows.iloc[:, 0]
        
        if outflow_data:
            outflows_df = pd.DataFrame(outflow_data).fillna(0)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Monthly Outflow Composition', 'Flow Pressure Index'],
                vertical_spacing=0.15
            )
            
            # Stacked bar chart for outflows
            colors = px.colors.qualitative.Pastel2
            for i, column in enumerate(outflows_df.columns):
                fig.add_trace(
                    go.Bar(
                        x=outflows_df.index,
                        y=outflows_df[column],
                        name=column,
                        marker_color=colors[i % len(colors)],
                        hovertemplate=f'%{{x|%b %Y}}<br>{column}: $%{{y:,.0f}}M<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Flow pressure index (inflows/outflows ratio)
            if inflow_data and outflow_data:
                total_inflows = inflows_df.sum(axis=1)
                total_outflows = outflows_df.sum(axis=1)
                pressure_index = total_inflows / (total_outflows + 1e-10)
                
                fig.add_trace(
                    go.Scatter(
                        x=pressure_index.index,
                        y=pressure_index,
                        name='Flow Pressure Index',
                        mode='lines+markers',
                        line=dict(width=3, color='#EF4444'),
                        fill='tozeroy',
                        fillcolor='rgba(239, 68, 68, 0.2)',
                        hovertemplate='%{x|%b %Y}}<br>Pressure Index: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Add reference line at 1.0
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig.update_layout(height=600, barmode='stack', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # Advanced flow statistics
    st.markdown("##### 📊 Flow Statistics Dashboard")
    
    flow_stats = []
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            flows = data['flows'].iloc[:, 0]
            
            # Calculate advanced statistics
            total_inflow = flows[flows > 0].sum()
            total_outflow = flows[flows < 0].abs().sum()
            net_flow = flows.sum()
            flow_volatility = flows.std()
            max_inflow = flows.max()
            max_outflow = flows.min()
            
            # Calculate flow consistency
            positive_months = (flows > 0).sum()
            total_months = len(flows)
            consistency_ratio = positive_months / total_months
            
            # Stationarity test
            try:
                adf_result = adfuller(flows.dropna())
                is_stationary = adf_result[1] < 0.05
            except:
                is_stationary = False
            
            flow_stats.append({
                'Category': category,
                'Net Flow': f"${net_flow:,.0f}M",
                'Inflow Ratio': f"{consistency_ratio:.1%}",
                'Flow Volatility': f"${flow_volatility:,.0f}M",
                'Max Inflow': f"${max_inflow:,.0f}M",
                'Max Outflow': f"${max_outflow:,.0f}M",
                'Stationary': '✅' if is_stationary else '❌'
            })
    
    if flow_stats:
        stats_df = pd.DataFrame(flow_stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

def create_superior_composition_charts(data_dict, market_data):
    """Create institutional-grade composition charts with market correlation"""
    st.markdown('<div class="section-header">🥧 Institutional Composition Analysis</div>', unsafe_allow_html=True)
    
    # Dynamic time selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_type = st.selectbox(
            "Composition View",
            ["Asset Allocation", "Flow Distribution", "Risk Contribution", "Market Correlation"]
        )
    
    with col2:
        if data_dict:
            dates = list(data_dict.values())[0]['assets'].index
            selected_date = st.select_slider(
                "Select Date",
                options=[d.strftime('%Y-%m') for d in dates],
                value=dates[-1].strftime('%Y-%m')
            )
    
    with col3:
        show_donut = st.checkbox("Show Donut Chart", True)
    
    # Create multiple composition views
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Asset Allocation Composition',
                       'Flow Contribution Analysis',
                       'Risk Contribution Matrix',
                       'Market Correlation Heatmap'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Asset Allocation Pie Chart
    asset_values = {}
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            date_idx = pd.to_datetime(selected_date)
            if date_idx in data['assets'].index:
                asset_values[category] = data['assets'].loc[date_idx].iloc[0]
    
    if asset_values:
        fig.add_trace(
            go.Pie(
                labels=list(asset_values.keys()),
                values=list(asset_values.values()),
                hole=0.5 if show_donut else 0,
                marker=dict(colors=px.colors.qualitative.Vivid),
                textinfo='label+percent',
                hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # 2. Flow Contribution Bar Chart
    flow_values = {}
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            latest_flow = data['flows'].iloc[-1, 0]
            flow_values[category] = latest_flow
    
    if flow_values:
        colors = ['#10B981' if v > 0 else '#EF4444' for v in flow_values.values()]
        
        fig.add_trace(
            go.Bar(
                x=list(flow_values.keys()),
                y=list(flow_values.values()),
                marker_color=colors,
                text=[f"${v:,.0f}M" for v in flow_values.values()],
                textposition='auto',
                hovertemplate="%{x}<br>Flow: $%{y:,.0f}M<extra></extra>"
            ),
            row=1, col=2
        )
    
    # 3. Risk Contribution Matrix
    if len(data_dict) > 1:
        returns_data = []
        categories = []
        
        for category, data in data_dict.items():
            if 'pct_change' in data and not data['pct_change'].empty:
                returns_data.append(data['pct_change'].iloc[:, 0])
                categories.append(category)
        
        if returns_data and len(returns_data) > 1:
            returns_df = pd.concat(returns_data, axis=1).dropna()
            returns_df.columns = categories
            
            if len(returns_df) > 10:
                # Calculate covariance matrix
                cov_matrix = returns_df.cov()
                
                fig.add_trace(
                    go.Heatmap(
                        z=cov_matrix.values,
                        x=cov_matrix.columns,
                        y=cov_matrix.index,
                        colorscale='Viridis',
                        colorbar=dict(title="Covariance"),
                        hovertemplate='%{x} vs %{y}<br>Covariance: %{z:.4f}<extra></extra>'
                    ),
                    row=2, col=1
                )
    
    # 4. Market Correlation Heatmap
    if market_data and len(data_dict) > 0:
        # Combine fund returns with market returns
        all_returns = []
        fund_names = []
        
        for category, data in data_dict.items():
            if 'pct_change' in data and not data['pct_change'].empty:
                all_returns.append(data['pct_change'].iloc[:, 0])
                fund_names.append(category)
        
        # Add market returns
        market_names = []
        for market, data in market_data.items():
            if 'returns' in data and not data['returns'].empty:
                all_returns.append(data['returns'])
                market_names.append(market)
        
        if all_returns and len(all_returns) > 1:
            combined_returns = pd.concat(all_returns, axis=1).dropna()
            combined_returns.columns = fund_names + market_names
            
            # Calculate correlation matrix
            corr_matrix = combined_returns.corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Correlation"),
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Institutional Composition & Correlation Analysis",
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_statistical_analysis(data_dict):
    """Create comprehensive statistical analysis"""
    st.markdown('<div class="section-header">📊 Advanced Statistical Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Descriptive Statistics", "Time Series Analysis", "Risk Metrics", "Regression Analysis"])
    
    with tab1:
        st.markdown("##### Descriptive Statistics")
        
        stats_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].iloc[:, 0].dropna()
                
                if len(flows) > 1:
                    # Calculate all statistics
                    stats_data.append({
                        'Category': category,
                        'Mean': flows.mean(),
                        'Median': flows.median(),
                        'Std Dev': flows.std(),
                        'Skewness': flows.skew(),
                        'Kurtosis': flows.kurtosis(),
                        'Jarque-Bera': stats.jarque_bera(flows)[0] if len(flows) > 0 else np.nan,
                        'Min': flows.min(),
                        'Max': flows.max(),
                        'IQR': flows.quantile(0.75) - flows.quantile(0.25)
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            # Format numeric columns
            format_dict = {col: "{:,.2f}" for col in stats_df.columns if col != 'Category'}
            styled_df = stats_df.style.format(format_dict)
            
            st.dataframe(styled_df, use_container_width=True, height=400)
    
    with tab2:
        st.markdown("##### Time Series Decomposition")
        
        selected_category = st.selectbox(
            "Select category for decomposition",
            list(data_dict.keys()),
            key="decomp_category"
        )
        
        if selected_category in data_dict and 'flows' in data_dict[selected_category]:
            flows = data_dict[selected_category]['flows'].iloc[:, 0]
            
            if len(flows) >= 24:  # Need enough data for decomposition
                try:
                    # Perform seasonal decomposition
                    decomposition = seasonal_decompose(flows.dropna(), model='additive', period=12)
                    
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Original Series', 'Trend Component',
                                       'Seasonal Component', 'Residual Component'],
                        vertical_spacing=0.08
                    )
                    
                    components = {
                        'Original': flows,
                        'Trend': decomposition.trend,
                        'Seasonal': decomposition.seasonal,
                        'Residual': decomposition.resid
                    }
                    
                    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
                    
                    for idx, (name, series) in enumerate(components.items()):
                        fig.add_trace(
                            go.Scatter(
                                x=series.index,
                                y=series,
                                name=name,
                                mode='lines',
                                line=dict(width=2, color=colors[idx]),
                                hovertemplate='%{x|%b %Y}<br>' + f'{name}: %{{y:.0f}}M<extra></extra>'
                            ),
                            row=idx+1, col=1
                        )
                    
                    fig.update_layout(height=800, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not perform decomposition: {str(e)}")
    
    with tab3:
        st.markdown("##### Risk Metrics Analysis")
        
        risk_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].iloc[:, 0].dropna()
                
                if len(flows) >= 12:
                    # Calculate risk metrics
                    returns = flows.pct_change().dropna()
                    
                    if len(returns) > 0:
                        # Basic risk metrics
                        volatility = returns.std() * np.sqrt(12)
                        downside_returns = returns[returns < 0]
                        downside_vol = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
                        
                        # Calculate Value at Risk (VaR) - Historical
                        var_95 = np.percentile(returns, 5)
                        var_99 = np.percentile(returns, 1)
                        
                        # Calculate Expected Shortfall (CVaR)
                        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
                        
                        # Calculate Sharpe and Sortino ratios
                        sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() != 0 else 0
                        sortino = returns.mean() / downside_vol if downside_vol != 0 else 0
                        
                        risk_data.append({
                            'Category': category,
                            'Annual Volatility': f"{volatility:.2%}",
                            'Sharpe Ratio': f"{sharpe:.3f}",
                            'Sortino Ratio': f"{sortino:.3f}",
                            'VaR (95%)': f"{var_95:.2%}",
                            'CVaR (95%)': f"{cvar_95:.2%}",
                            'Max Drawdown': f"{((1 + returns).cumprod().expanding().max() - (1 + returns).cumprod()).max():.2%}",
                            'Calmar Ratio': f"{returns.mean() * 12 / (((1 + returns).cumprod().expanding().max() - (1 + returns).cumprod()).max()):.3f}" if ((1 + returns).cumprod().expanding().max() - (1 + returns).cumprod()).max() != 0 else "N/A"
                        })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True, height=400)
    
    with tab4:
        st.markdown("##### Regression & Correlation Analysis")
        
        if len(data_dict) >= 2:
            # Prepare data for regression
            returns_data = []
            categories = []
            
            for category, data in data_dict.items():
                if 'pct_change' in data and not data['pct_change'].empty:
                    returns_data.append(data['pct_change'].iloc[:, 0])
                    categories.append(category)
            
            if len(returns_data) >= 2:
                returns_df = pd.concat(returns_data, axis=1).dropna()
                returns_df.columns = categories
                
                # Calculate correlation matrix
                corr_matrix = returns_df.corr()
                
                # Visualize correlation matrix
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(3).values,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Fund Category Correlation Matrix",
                    height=500,
                    xaxis_title="Category",
                    yaxis_title="Category"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display strongest correlations
                st.markdown("**Strongest Correlations:**")
                
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.5:
                            strong_corrs.append({
                                'Pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                                'Correlation': f"{corr_value:.3f}",
                                'Strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                            })
                
                if strong_corrs:
                    corr_df = pd.DataFrame(strong_corrs)
                    st.dataframe(corr_df, use_container_width=True)

def main():
    """Main application function"""
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); border-radius: 12px; margin-bottom: 2rem; color: white;">
            <h3 style="margin: 0; font-weight: 700;">Institutional Research Suite</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">FRED Data Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📅 Data Configuration")
        
        # Data frequency
        frequency = st.radio(
            "Data Frequency",
            ['monthly', 'weekly'],
            index=0,
            horizontal=True
        )
        
        # Date range
        start_date = st.date_input(
            "Analysis Start Date",
            value=datetime(2010, 1, 1),
            min_value=datetime(1990, 1, 1),
            max_value=datetime.today()
        )
        
        # Fund categories selection
        st.markdown("### 🏦 Fund Categories")
        all_categories = list(FRED_SERIES.keys())
        
        # Group by category type
        equity_categories = [k for k, v in FRED_SERIES.items() if v['category'] == 'equity']
        fixed_income_categories = [k for k, v in FRED_SERIES.items() if v['category'] == 'fixed_income']
        cash_categories = [k for k, v in FRED_SERIES.items() if v['category'] == 'cash']
        total_categories = [k for k, v in FRED_SERIES.items() if v['category'] == 'total']
        
        selected_categories = []
        
        with st.expander("Equity Funds", expanded=True):
            selected_equity = st.multiselect(
                "Select equity funds",
                equity_categories,
                default=equity_categories[:1]
            )
            selected_categories.extend(selected_equity)
        
        with st.expander("Fixed Income Funds", expanded=True):
            selected_fixed = st.multiselect(
                "Select fixed income funds",
                fixed_income_categories,
                default=fixed_income_categories[:2]
            )
            selected_categories.extend(selected_fixed)
        
        with st.expander("Cash & Money Market", expanded=True):
            selected_cash = st.multiselect(
                "Select cash funds",
                cash_categories,
                default=cash_categories[:1]
            )
            selected_categories.extend(selected_cash)
        
        with st.expander("Total Aggregates", expanded=False):
            selected_total = st.multiselect(
                "Select total aggregates",
                total_categories,
                default=total_categories[:1]
            )
            selected_categories.extend(selected_total)
        
        # Include market data
        include_market = st.checkbox("Include Market Indices for Correlation", True)
        
        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("📊 Generate Report", use_container_width=True):
                st.info("Report generation would be implemented here")
    
    # Filter selected series
    selected_series = {k: FRED_SERIES[k] for k in selected_categories if k in FRED_SERIES}
    
    if not selected_series:
        st.warning("Please select at least one fund category from the sidebar.")
        return
    
    # Load data
    with st.spinner("🔄 Loading institutional data..."):
        data_dict = fetch_all_fred_data(selected_series, str(start_date), frequency)
        
        if include_market:
            market_data = fetch_market_data(str(start_date))
        else:
            market_data = {}
    
    if not data_dict:
        st.error("❌ No data available. Please check your selections and try again.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Executive Summary",
        "📊 Growth Analytics",
        "💸 Flow Dynamics",
        "🥧 Composition",
        "📉 Statistical Analysis"
    ])
    
    with tab1:
        create_superior_metrics(data_dict, frequency)
        
        # Quick insights
        st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            <div class="stat-box">
                <h4 style="margin: 0 0 1rem 0; color: #1E3A8A;">📈 Performance Highlights</h4>
                <ul style="margin: 0; padding-left: 1.2rem;">
                    <li>Real-time FRED data integration</li>
                    <li>Advanced statistical analysis</li>
                    <li>Risk-adjusted performance metrics</li>
                    <li>Market correlation analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col2:
            st.markdown("""
            <div class="stat-box">
                <h4 style="margin: 0 0 1rem 0; color: #1E3A8A;">🔍 Analytical Features</h4>
                <ul style="margin: 0; padding-left: 1.2rem;">
                    <li>Time series decomposition</li>
                    <li>Risk metrics (VaR, CVaR, Sharpe)</li>
                    <li>Correlation matrices</li>
                    <li>Flow pressure analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        create_enhanced_growth_charts(data_dict)
    
    with tab3:
        create_comprehensive_flow_analysis(data_dict, frequency)
    
    with tab4:
        create_superior_composition_charts(data_dict, market_data)
    
    with tab5:
        create_statistical_analysis(data_dict)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
            <div>
                <strong>Data Source</strong><br>
                Federal Reserve Economic Data (FRED)
            </div>
            <div>
                <strong>Frequency</strong><br>
                {frequency} Data
            </div>
            <div>
                <strong>Period</strong><br>
                {start_date} - {end_date}
            </div>
            <div>
                <strong>Categories</strong><br>
                {num_categories} Fund Classes
            </div>
        </div>
        <p style="margin: 0; color: #6B7280; font-size: 0.85rem;">
            Institutional Fund Flow Analysis Suite v4.0 | Generated on {date}<br>
            All figures in millions of USD | Professional Use Only
        </p>
    </div>
    """.format(
        frequency=frequency.capitalize(),
        start_date=start_date.strftime('%Y-%m'),
        end_date=datetime.today().strftime('%Y-%m'),
        num_categories=len(data_dict),
        date=datetime.today().strftime('%B %d, %Y %H:%M')
    ), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
