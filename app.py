import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
from scipy import stats
import calendar
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Institutional Fund Flow Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional, clean CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        font-weight: 600;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .data-table-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e0e0e0;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 24px;
        border-radius: 6px;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        font-weight: 500;
        color: #495057;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #2c3e50 !important;
        color: white !important;
        border: 1px solid #2c3e50 !important;
    }
    .analysis-section {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .stat-box {
        background: #f8f9fa;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2c3e50;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666666;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown('<h1 class="main-header">Institutional Fund Flow Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Analysis of Mutual Fund & ETF Flows using Federal Reserve Economic Data (FRED)</p>', unsafe_allow_html=True)

# FRED Series IDs - Using working series
FRED_SERIES = {
    'Total Mutual Fund Assets': {
        'monthly': 'TOTMFS',
        'weekly': 'TOTMFS',  # Fallback to monthly for weekly if needed
        'description': 'Total Mutual Fund Assets',
        'color': '#2c3e50'
    },
    'Money Market Funds': {
        'monthly': 'MMMFFAQ027S',
        'weekly': 'MMMFFAQ027S',  # Fallback to monthly
        'description': 'Money Market Fund Assets',
        'color': '#3498db'
    },
    'Equity Funds': {
        'monthly': 'TOTCI',
        'weekly': 'TOTCI',  # Fallback to monthly
        'description': 'Equity Mutual Fund Assets',
        'color': '#27ae60'
    },
    'Bond Funds': {
        'monthly': 'TBCI',
        'weekly': 'TBCI',  # Fallback to monthly
        'description': 'Bond/Income Fund Assets',
        'color': '#e74c3c'
    },
    'Municipal Bond Funds': {
        'monthly': 'MBCI',
        'weekly': 'MBCI',  # Fallback to monthly
        'description': 'Municipal Bond Fund Assets',
        'color': '#9b59b6'
    }
}

# Professional color palette
PROFESSIONAL_COLORS = ['#2c3e50', '#3498db', '#27ae60', '#e74c3c', '#9b59b6', '#f39c12']

@st.cache_data(ttl=3600)
def fetch_fred_data(series_id, start_date, end_date):
    """Fetch data from FRED with robust error handling"""
    try:
        # FRED CSV API
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}"
        
        # Add end date if specified
        if end_date:
            url += f"&coed={end_date}"
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200 and 'DATE' in response.text:
            # Parse CSV
            df = pd.read_csv(StringIO(response.text))
            df['DATE'] = pd.to_datetime(df['DATE'])
            df.set_index('DATE', inplace=True)
            
            # Check if we have data
            if not df.empty and df.iloc[:, 0].notna().any():
                return df
            else:
                st.warning(f"No data available for series {series_id}")
                return pd.DataFrame()
        else:
            st.warning(f"Failed to fetch {series_id}: HTTP {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        st.warning(f"Error fetching {series_id}: {str(e)[:100]}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def generate_sample_data(start_date, end_date, frequency):
    """Generate realistic sample data for demonstration"""
    if frequency == 'monthly':
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    
    n = len(dates)
    np.random.seed(42)
    
    # Generate realistic patterns
    data = {}
    
    # Equity funds - volatile with trend
    equity_base = 15000 + np.linspace(0, 10000, n)
    equity_noise = np.random.normal(0, 4000, n)
    equity_seasonal = 2000 * np.sin(2 * np.pi * np.arange(n) / 12)
    data['Equity Funds'] = equity_base + equity_noise + equity_seasonal
    
    # Bond funds - more stable
    bond_base = 8000 + np.linspace(0, 6000, n)
    bond_noise = np.random.normal(0, 2000, n)
    data['Bond Funds'] = bond_base + bond_noise
    
    # Money Market - flight to quality
    mm_base = 12000 + np.linspace(0, 8000, n)
    mm_noise = np.random.normal(0, 3000, n)
    data['Money Market Funds'] = mm_base + mm_noise
    
    # Total - sum
    data['Total Mutual Fund Assets'] = data['Equity Funds'] + data['Bond Funds'] + data['Money Market Funds']
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    return df

def load_fund_data(selected_categories, start_date, frequency, use_sample=False):
    """Load fund data - use sample if FRED fails"""
    data_dict = {}
    end_date = datetime.today()
    
    if use_sample:
        # Generate sample data
        sample_data = generate_sample_data(start_date, end_date, frequency)
        
        for category in selected_categories:
            if category in sample_data.columns:
                df = pd.DataFrame(sample_data[category])
                df.columns = ['Value']
                
                # Calculate flows
                df_flows = df.diff()
                df_flows.columns = ['Flow']
                
                data_dict[category] = {
                    'assets': df,
                    'flows': df_flows,
                    'description': FRED_SERIES[category]['description'],
                    'color': FRED_SERIES[category]['color']
                }
        
        st.info("Using sample data for demonstration")
        return data_dict
    
    # Try to fetch from FRED
    for category in selected_categories:
        if category in FRED_SERIES:
            series_id = FRED_SERIES[category][frequency]
            df = fetch_fred_data(series_id, start_date, end_date)
            
            if not df.empty:
                # Rename column
                df.columns = ['Value']
                
                # Calculate flows
                df_flows = df.diff()
                df_flows.columns = ['Flow']
                
                data_dict[category] = {
                    'assets': df,
                    'flows': df_flows,
                    'description': FRED_SERIES[category]['description'],
                    'color': FRED_SERIES[category]['color']
                }
            else:
                # If FRED fails, use sample for this category
                st.warning(f"Using sample data for {category}")
                
                # Generate sample for this category
                sample_data = generate_sample_data(start_date, end_date, frequency)
                if category in sample_data.columns:
                    df = pd.DataFrame(sample_data[category])
                    df.columns = ['Value']
                    
                    df_flows = df.diff()
                    df_flows.columns = ['Flow']
                    
                    data_dict[category] = {
                        'assets': df,
                        'flows': df_flows,
                        'description': FRED_SERIES[category]['description'],
                        'color': FRED_SERIES[category]['color']
                    }
    
    return data_dict

def create_executive_summary(data_dict, frequency):
    """Create clean executive summary"""
    st.markdown("## Executive Summary")
    
    if not data_dict:
        st.warning("No data available")
        return
    
    cols = st.columns(min(4, len(data_dict)))
    
    for idx, (category, data) in enumerate(list(data_dict.items())[:4]):
        with cols[idx % len(cols)]:
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0] if len(data['flows']) > 0 else 0
                avg_flow = data['flows'].mean().iloc[0]
                
                flow_class = "positive" if latest_flow > 0 else "negative"
                flow_color = "#27ae60" if latest_flow > 0 else "#e74c3c"
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{category}</div>
                    <div class='metric-value'>${abs(latest_flow):,.0f}M</div>
                    <div style='color: {flow_color}; font-size: 0.9rem; font-weight: 500;'>
                        {'+' if latest_flow > 0 else ''}{latest_flow:,.0f}M {frequency}
                    </div>
                    <div style='font-size: 0.8rem; color: #666666; margin-top: 8px;'>
                        Avg: ${avg_flow:,.0f}M
                    </div>
                </div>
                """, unsafe_allow_html=True)

def create_professional_growth_charts(data_dict):
    """Create professional growth analysis"""
    st.markdown("## Growth Dynamics Analysis")
    
    if not data_dict:
        st.warning("No data available for growth analysis")
        return
    
    # Control panel
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Cumulative Growth", "Rolling Returns", "Annual Performance"],
            key="growth_type"
        )
    
    with col2:
        window_size = st.slider("Rolling Window", 1, 24, 12, key="growth_window")
    
    # Prepare growth data
    fig = go.Figure()
    
    for idx, (category, data) in enumerate(data_dict.items()):
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets']
            
            if analysis_type == "Cumulative Growth":
                # Normalize to starting value = 100
                normalized = 100 * assets['Value'] / assets['Value'].iloc[0]
                y_label = "Growth Index (Base=100)"
                
            elif analysis_type == "Rolling Returns":
                # Calculate rolling returns
                returns = assets['Value'].pct_change()
                rolling_returns = returns.rolling(window=window_size).mean() * 100
                normalized = rolling_returns
                y_label = f"{window_size}-Period Rolling Return (%)"
                
            elif analysis_type == "Annual Performance":
                # Calculate annual performance
                returns = assets['Value'].pct_change()
                annualized = returns.rolling(window=12).mean() * 12 * 100
                normalized = annualized
                y_label = "Annualized Return (%)"
            
            color = data.get('color', PROFESSIONAL_COLORS[idx % len(PROFESSIONAL_COLORS)])
            
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized,
                name=category,
                mode='lines',
                line=dict(width=2, color=color),
                hovertemplate='%{x|%b %Y}<br>' + f'{category}: %{{y:.2f}}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=f"{analysis_type} Analysis",
        xaxis_title="Date",
        yaxis_title=y_label,
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("### Statistical Summary")
    
    stats_data = []
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets']['Value']
            
            if len(assets) > 1:
                returns = assets.pct_change().dropna()
                
                stats_data.append({
                    'Category': category,
                    'Total Growth (%)': f"{(assets.iloc[-1] / assets.iloc[0] - 1) * 100:.2f}",
                    'Annual Return (%)': f"{returns.mean() * 12 * 100:.2f}",
                    'Annual Volatility (%)': f"{returns.std() * np.sqrt(12) * 100:.2f}",
                    'Sharpe Ratio': f"{returns.mean() / returns.std() * np.sqrt(12):.2f}" if returns.std() > 0 else "N/A"
                })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

def create_flow_analysis(data_dict, frequency):
    """Create professional flow analysis"""
    st.markdown("## Flow Dynamics Analysis")
    
    if not data_dict:
        st.warning("No data available for flow analysis")
        return
    
    # Create inflow/outflow analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Inflow Analysis")
        
        # Get latest inflows
        inflow_data = []
        categories = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0] if len(data['flows']) > 0 else 0
                if latest_flow > 0:
                    inflow_data.append(latest_flow)
                    categories.append(category)
        
        if inflow_data:
            fig_in = go.Figure()
            fig_in.add_trace(go.Bar(
                x=categories,
                y=inflow_data,
                marker_color='#27ae60',
                opacity=0.8,
                hovertemplate='%{x}<br>Inflow: $%{y:,.0f}M<extra></extra>'
            ))
            
            fig_in.update_layout(
                title=f"Latest {frequency.capitalize()} Inflows",
                xaxis_title="Category",
                yaxis_title="Inflows (Millions USD)",
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig_in, use_container_width=True)
        else:
            st.info("No inflows in latest period")
    
    with col2:
        st.markdown("### Outflow Analysis")
        
        # Get latest outflows
        outflow_data = []
        categories = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0] if len(data['flows']) > 0 else 0
                if latest_flow < 0:
                    outflow_data.append(abs(latest_flow))
                    categories.append(category)
        
        if outflow_data:
            fig_out = go.Figure()
            fig_out.add_trace(go.Bar(
                x=categories,
                y=outflow_data,
                marker_color='#e74c3c',
                opacity=0.8,
                hovertemplate='%{x}<br>Outflow: $%{y:,.0f}M<extra></extra>'
            ))
            
            fig_out.update_layout(
                title=f"Latest {frequency.capitalize()} Outflows",
                xaxis_title="Category",
                yaxis_title="Outflows (Millions USD)",
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig_out, use_container_width=True)
        else:
            st.info("No outflows in latest period")
    
    # Net flow trend
    st.markdown("### Net Flow Trend")
    
    # Find common dates
    all_dates = None
    for data in data_dict.values():
        if 'flows' in data and not data['flows'].empty:
            if all_dates is None:
                all_dates = data['flows'].index
            else:
                all_dates = all_dates.intersection(data['flows'].index)
    
    if all_dates is not None and len(all_dates) > 0:
        # Create net flow dataframe
        net_flows = pd.DataFrame(index=all_dates)
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                # Align with common dates
                aligned_flows = data['flows'].reindex(all_dates)
                net_flows[category] = aligned_flows['Flow']
        
        if not net_flows.empty:
            fig_net = go.Figure()
            
            for idx, column in enumerate(net_flows.columns):
                color = data_dict[column].get('color', PROFESSIONAL_COLORS[idx % len(PROFESSIONAL_COLORS)])
                
                fig_net.add_trace(go.Scatter(
                    x=net_flows.index,
                    y=net_flows[column],
                    name=column,
                    mode='lines',
                    line=dict(width=1.5, color=color),
                    hovertemplate='%{x|%b %Y}<br>' + f'{column}: $%{{y:,.0f}}M<extra></extra>'
                ))
            
            fig_net.update_layout(
                title="Net Flows Over Time",
                xaxis_title="Date",
                yaxis_title="Net Flow (Millions USD)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_net, use_container_width=True)

def create_composition_analysis(data_dict):
    """Create professional composition analysis - FIXED"""
    st.markdown("## Composition Analysis")
    
    if not data_dict:
        st.warning("No data available for composition analysis")
        return
    
    # Get the latest date from any dataset
    latest_date = None
    for data in data_dict.values():
        if 'assets' in data and not data['assets'].empty:
            if latest_date is None or data['assets'].index[-1] > latest_date:
                latest_date = data['assets'].index[-1]
    
    if latest_date is None:
        st.warning("No valid dates found in data")
        return
    
    # Convert to string for display
    latest_date_str = latest_date.strftime('%Y-%m')
    
    # Get available dates for selection
    available_dates = []
    for data in data_dict.values():
        if 'assets' in data and not data['assets'].empty:
            available_dates.extend(data['assets'].index.tolist())
    
    if not available_dates:
        st.warning("No dates available for selection")
        return
    
    available_dates = sorted(set(available_dates))
    date_options = [d.strftime('%Y-%m') for d in available_dates]
    
    # Date selection
    selected_date_str = st.selectbox(
        "Select date for composition analysis",
        options=date_options,
        index=len(date_options) - 1
    )
    
    # Convert back to datetime for comparison
    try:
        selected_date = pd.to_datetime(selected_date_str + '-01')
    except:
        selected_date = pd.to_datetime(selected_date_str)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Asset Composition ({selected_date_str})")
        
        # Get asset values for selected date
        asset_values = {}
        asset_colors = []
        
        for category, data in data_dict.items():
            if 'assets' in data and not data['assets'].empty:
                # Find the closest date to selected date
                dates = data['assets'].index
                closest_date = min(dates, key=lambda x: abs(x - selected_date))
                
                if abs((closest_date - selected_date).days) <= 30:  # Within 30 days
                    asset_value = data['assets'].loc[closest_date, 'Value']
                    if not pd.isna(asset_value):
                        asset_values[category] = asset_value
                        asset_colors.append(data.get('color', PROFESSIONAL_COLORS[len(asset_colors) % len(PROFESSIONAL_COLORS)]))
        
        if asset_values:
            fig_assets = go.Figure(data=[go.Pie(
                labels=list(asset_values.keys()),
                values=list(asset_values.values()),
                hole=0.3,
                marker_colors=asset_colors,
                textinfo='label+percent',
                hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
            )])
            
            fig_assets.update_layout(
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
            
            st.plotly_chart(fig_assets, use_container_width=True)
        else:
            st.info("No asset data available for selected date")
    
    with col2:
        st.markdown("### Flow Composition (Latest)")
        
        # Get latest flows
        flow_values = {}
        flow_colors = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0] if len(data['flows']) > 0 else 0
                if not pd.isna(latest_flow):
                    flow_values[category] = abs(latest_flow)
                    flow_colors.append(data.get('color', PROFESSIONAL_COLORS[len(flow_colors) % len(PROFESSIONAL_COLORS)]))
        
        if flow_values:
            # Separate positive and negative
            positive_flows = {}
            negative_flows = {}
            
            for category, data in data_dict.items():
                if 'flows' in data and not data['flows'].empty:
                    latest_flow = data['flows'].iloc[-1, 0] if len(data['flows']) > 0 else 0
                    if not pd.isna(latest_flow):
                        if latest_flow > 0:
                            positive_flows[category] = latest_flow
                        elif latest_flow < 0:
                            negative_flows[category] = abs(latest_flow)
            
            if positive_flows:
                st.markdown("**Inflows:**")
                fig_inflows = go.Figure(data=[go.Pie(
                    labels=list(positive_flows.keys()),
                    values=list(positive_flows.values()),
                    hole=0.4,
                    marker_colors=[data_dict[cat].get('color', '#27ae60') for cat in positive_flows.keys()],
                    hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
                )])
                
                fig_inflows.update_layout(
                    height=250,
                    showlegend=False
                )
                
                st.plotly_chart(fig_inflows, use_container_width=True)
            
            if negative_flows:
                st.markdown("**Outflows:**")
                fig_outflows = go.Figure(data=[go.Pie(
                    labels=list(negative_flows.keys()),
                    values=list(negative_flows.values()),
                    hole=0.4,
                    marker_colors=[data_dict[cat].get('color', '#e74c3c') for cat in negative_flows.keys()],
                    hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
                )])
                
                fig_outflows.update_layout(
                    height=250,
                    showlegend=False
                )
                
                st.plotly_chart(fig_outflows, use_container_width=True)
            
            if not positive_flows and not negative_flows:
                st.info("No flow data available")
        else:
            st.info("No flow data available")

def create_statistical_analysis(data_dict):
    """Create professional statistical analysis"""
    st.markdown("## Statistical Analysis")
    
    if not data_dict:
        st.warning("No data available for statistical analysis")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Descriptive Statistics", "Risk Analysis", "Correlation Analysis"])
    
    with tab1:
        st.markdown("### Descriptive Statistics")
        
        stats_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows']['Flow'].dropna()
                
                if len(flows) > 1:
                    stats_data.append({
                        'Category': category,
                        'Observations': len(flows),
                        'Mean (M$)': f"{flows.mean():,.1f}",
                        'Std Dev (M$)': f"{flows.std():,.1f}",
                        'Minimum (M$)': f"{flows.min():,.1f}",
                        'Maximum (M$)': f"{flows.max():,.1f}",
                        'Skewness': f"{flows.skew():.3f}",
                        'Kurtosis': f"{flows.kurtosis():.3f}"
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, height=400)
        else:
            st.info("Insufficient data for descriptive statistics")
    
    with tab2:
        st.markdown("### Risk Analysis")
        
        risk_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows']['Flow'].dropna()
                
                if len(flows) >= 12:
                    returns = flows.pct_change().dropna()
                    
                    if len(returns) > 0:
                        # Calculate risk metrics
                        volatility = returns.std() * np.sqrt(12)
                        sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() != 0 else 0
                        
                        # Value at Risk
                        var_95 = np.percentile(returns, 5)
                        
                        # Maximum drawdown
                        cum_returns = (1 + returns).cumprod()
                        running_max = cum_returns.expanding().max()
                        drawdown = (cum_returns - running_max) / running_max
                        max_drawdown = drawdown.min()
                        
                        risk_data.append({
                            'Category': category,
                            'Annual Volatility': f"{volatility:.2%}",
                            'Sharpe Ratio': f"{sharpe:.3f}",
                            'VaR (95%)': f"{var_95:.2%}",
                            'Max Drawdown': f"{max_drawdown:.2%}"
                        })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True, height=400)
        else:
            st.info("Insufficient data for risk analysis (need at least 12 periods)")
    
    with tab3:
        st.markdown("### Correlation Analysis")
        
        # Prepare returns data for correlation
        returns_data = {}
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows']['Flow'].dropna()
                if len(flows) > 1:
                    returns = flows.pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[category] = returns
        
        if len(returns_data) >= 2:
            # Combine returns
            combined_returns = pd.DataFrame(returns_data)
            correlation_matrix = combined_returns.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
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
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.3:
                        strong_corrs.append({
                            'Pair': f"{correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}",
                            'Correlation': f"{corr_value:.3f}",
                            'Strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate' if abs(corr_value) > 0.5 else 'Weak'
                        })
            
            if strong_corrs:
                corr_df = pd.DataFrame(strong_corrs)
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.info("No significant correlations found (all < 0.3)")
        else:
            st.info("Need at least 2 categories for correlation analysis")

def create_data_explorer(data_dict, frequency):
    """Create advanced data explorer"""
    st.markdown("## Data Explorer")
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Create combined dataframe
    combined_data = []
    
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            df = data['flows'].copy()
            df.columns = [f'{category}_Flow']
            combined_data.append(df)
    
    if not combined_data:
        st.warning("No flow data available")
        return
    
    # Combine all data
    all_data = pd.concat(combined_data, axis=1)
    
    # Display options
    st.markdown("### Data Preview")
    
    # Row limit selector
    row_limit = st.slider("Rows to display", 10, 200, 50)
    
    # Format display dataframe
    display_df = all_data.copy()
    display_df.index.name = 'Date'
    display_df = display_df.reset_index()
    
    # Add formatted columns
    formatted_df = display_df.copy()
    for col in formatted_df.columns:
        if col != 'Date' and formatted_df[col].dtype in ['float64', 'int64']:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"${x:,.0f}M" if pd.notnull(x) else ""
            )
    
    st.dataframe(
        formatted_df.head(row_limit),
        use_container_width=True,
        height=400
    )
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    
    summary_data = []
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            flows = data['flows']['Flow']
            
            summary_data.append({
                'Category': category,
                'Mean (M$)': f"{flows.mean():,.1f}",
                'Std Dev (M$)': f"{flows.std():,.1f}",
                'Min (M$)': f"{flows.min():,.1f}",
                'Max (M$)': f"{flows.max():,.1f}",
                'Latest (M$)': f"{flows.iloc[-1] if len(flows) > 0 else 0:,.1f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, height=300)
    
    # Export options
    st.markdown("### Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = all_data.to_csv()
        st.download_button(
            label="Download CSV (Raw Data)",
            data=csv,
            file_name=f"fund_flows_{frequency}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if summary_data:
            summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
            st.download_button(
                label="Download CSV (Summary)",
                data=summary_csv,
                file_name=f"summary_stats_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    """Main application function"""
    
    # Sidebar with clean design
    with st.sidebar:
        st.markdown("### Data Configuration")
        
        # Frequency selection
        frequency = st.radio(
            "Data Frequency",
            ['monthly', 'weekly'],
            index=0,
            help="Select weekly or monthly frequency"
        )
        
        # Date range
        start_date = st.date_input(
            "Start Date",
            value=datetime(2015, 1, 1),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.today()
        )
        
        st.markdown("---")
        
        # Fund categories selection
        st.markdown("### Fund Categories")
        
        all_categories = list(FRED_SERIES.keys())
        selected_categories = st.multiselect(
            "Select categories to analyze",
            all_categories,
            default=['Total Mutual Fund Assets', 'Equity Funds', 'Bond Funds', 'Money Market Funds']
        )
        
        # Data source option
        st.markdown("---")
        st.markdown("### Data Source")
        
        use_sample = st.checkbox(
            "Use sample data (if FRED fails)",
            value=False,
            help="Use realistic sample data if FRED API fails"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    if not selected_categories:
        st.warning("Please select at least one fund category from the sidebar.")
        return
    
    with st.spinner(f"Loading {frequency} data..."):
        data_dict = load_fund_data(selected_categories, str(start_date), frequency, use_sample)
    
    if not data_dict:
        st.error("Failed to load data. Please try using sample data or check your selections.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary",
        "Growth Dynamics",
        "Flow Dynamics",
        "Composition",
        "Statistical Analysis",
        "Data Explorer"
    ])
    
    with tab1:
        create_executive_summary(data_dict, frequency)
    
    with tab2:
        create_professional_growth_charts(data_dict)
    
    with tab3:
        create_flow_analysis(data_dict, frequency)
    
    with tab4:
        create_composition_analysis(data_dict)
    
    with tab5:
        create_statistical_analysis(data_dict)
    
    with tab6:
        create_data_explorer(data_dict, frequency)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Institutional Fund Flow Analytics v2.0</strong></p>
        <p>Data Source: Federal Reserve Economic Data (FRED) | Frequency: {frequency} | Period: {start_date} to {end_date}</p>
        <p>All figures in millions of USD | Professional institutional platform</p>
    </div>
    """.format(
        frequency=frequency,
        start_date=start_date.strftime('%Y-%m'),
        end_date=datetime.today().strftime('%Y-%m')
    ), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
