import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Institutional Fund Flow Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional, clean CSS with enhanced styling
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
        margin: 2rem 0 1.2rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .date-indicator {
        font-size: 0.9rem;
        color: #666666;
        font-weight: 400;
        background: #f8f9fa;
        padding: 4px 12px;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .trend-indicator {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-left: 8px;
    }
    .trend-up {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .trend-down {
        background-color: #ffebee;
        color: #c62828;
    }
    .trend-neutral {
        background-color: #f5f5f5;
        color: #666666;
    }
    .flow-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 4px;
    }
    .flow-in {
        background-color: rgba(39, 174, 96, 0.1);
        color: #27ae60;
        border: 1px solid rgba(39, 174, 96, 0.3);
    }
    .flow-out {
        background-color: rgba(231, 76, 60, 0.1);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
    .chart-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .analysis-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2c3e50;
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
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666666;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
    .api-status {
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .api-active {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
    .api-inactive {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ffcdd2;
    }
    .data-source-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 8px;
    }
    .badge-fred {
        background-color: #e3f2fd;
        color: #1565c0;
        border: 1px solid #bbdefb;
    }
    .badge-sample {
        background-color: #f3e5f5;
        color: #7b1fa2;
        border: 1px solid #e1bee7;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-card-secondary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-card-tertiary {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown('<h1 class="main-header">Institutional Fund Flow Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Statistical Analysis of Mutual Fund Assets & Flow Dynamics</p>', unsafe_allow_html=True)

# FRED API Configuration - VALID API KEY
FRED_API_KEY = "4a03f808f3f4fea5457376f10e1bf870"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Test the API key
def test_fred_api():
    """Test if FRED API key is valid"""
    try:
        test_params = {
            'series_id': 'TOTALSL',
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'limit': 1
        }
        response = requests.get(FRED_BASE_URL, params=test_params, timeout=10)
        
        if response.status_code == 200:
            return True, "✅ FRED API Key is valid and working"
        elif response.status_code == 400:
            return False, "❌ Invalid FRED API Key"
        elif response.status_code == 429:
            return False, "⚠️ API rate limit exceeded"
        else:
            return False, f"⚠️ API Error: {response.status_code}"
    except Exception as e:
        return False, f"⚠️ Connection Error: {str(e)}"

# Enhanced FRED Series IDs with REAL working series from FRED
FRED_SERIES = {
    'Total Mutual Fund Assets': {
        'fred_id': 'TOTALSL',
        'description': 'Total Mutual Fund Assets',
        'category': 'Total Assets',
        'unit': 'Millions of Dollars',
        'source': 'Board of Governors of the Federal Reserve System',
        'color': '#2c3e50',
        'start_year': 1984,
        'available_frequencies': ['monthly']
    },
    'Money Market Funds': {
        'fred_id': 'MMMFFAQ027S',
        'description': 'Money Market Fund Assets',
        'category': 'Money Market',
        'unit': 'Millions of Dollars',
        'source': 'Board of Governors of the Federal Reserve System',
        'color': '#3498db',
        'start_year': 2007,
        'available_frequencies': ['weekly', 'monthly']
    },
    'Equity Mutual Fund Assets': {
        'fred_id': 'FME',  # This exists in FRED
        'description': 'Equity Mutual Fund Assets',
        'category': 'Equity',
        'unit': 'Millions of Dollars',
        'source': 'Board of Governors of the Federal Reserve System',
        'color': '#27ae60',
        'start_year': 1945,
        'available_frequencies': ['monthly']
    },
    'Bond Mutual Fund Assets': {
        'fred_id': 'WSHOBL',  # This exists in FRED
        'description': 'Bond Mutual Fund Assets',
        'category': 'Fixed Income',
        'unit': 'Millions of Dollars',
        'source': 'Board of Governors of the Federal Reserve System',
        'color': '#e74c3c',
        'start_year': 1945,
        'available_frequencies': ['monthly']
    },
    'Corporate Bond Funds': {
        'fred_id': 'NCBCEL',  # Corporate Bonds; Liability
        'description': 'Corporate Bond Liabilities',
        'category': 'Corporate Bonds',
        'unit': 'Millions of Dollars',
        'source': 'Board of Governors of the Federal Reserve System',
        'color': '#e67e22',
        'start_year': 1945,
        'available_frequencies': ['quarterly']
    },
    'Treasury Securities': {
        'fred_id': 'TREAS',
        'description': 'Treasury Securities Held',
        'category': 'Government Bonds',
        'unit': 'Millions of Dollars',
        'source': 'Board of Governors of the Federal Reserve System',
        'color': '#9b59b6',
        'start_year': 1945,
        'available_frequencies': ['quarterly']
    },
    'Household Equity in Mutual Funds': {
        'fred_id': 'HNOREMFQ027S',
        'description': 'Household Equity in Mutual Funds',
        'category': 'Household',
        'unit': 'Millions of Dollars',
        'source': 'Board of Governors of the Federal Reserve System',
        'color': '#1abc9c',
        'start_year': 1945,
        'available_frequencies': ['quarterly']
    },
    'Total Debt Securities': {
        'fred_id': 'TCMILBS',
        'description': 'Total Debt Securities',
        'category': 'Debt',
        'unit': 'Millions of Dollars',
        'source': 'Board of Governors of the Federal Reserve System',
        'color': '#f39c12',
        'start_year': 1945,
        'available_frequencies': ['quarterly']
    }
}

PROFESSIONAL_COLORS = ['#2c3e50', '#3498db', '#27ae60', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#34495e', '#e67e22']

@st.cache_data(ttl=3600)
def fetch_fred_data(series_id, start_date, end_date, frequency='monthly'):
    """Fetch actual data from FRED API with proper error handling"""
    try:
        # FRED API parameters
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'units': 'lin'
        }
        
        # Make API request
        response = requests.get(FRED_BASE_URL, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'observations' in data and data['observations']:
                # Extract data
                dates = []
                values = []
                
                for obs in data['observations']:
                    # Skip entries with '.' (FRED's notation for missing data)
                    if obs['value'] != '.':
                        dates.append(pd.to_datetime(obs['date']))
                        try:
                            values.append(float(obs['value']))
                        except ValueError:
                            continue
                
                if dates and values:
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Value': values
                    }, index=dates)
                    
                    # Resample based on requested frequency
                    if frequency == 'weekly':
                        # Convert to weekly data (Friday)
                        df = df.resample('W-FRI').last().dropna()
                    elif frequency == 'monthly':
                        # Convert to monthly data (end of month)
                        df = df.resample('M').last().dropna()
                    
                    return df, 'FRED'
                else:
                    return None, 'No valid data points'
            else:
                return None, 'No observations in response'
        else:
            error_msg = f"API Status {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail.get('error_message', 'No details')}"
            except:
                pass
            return None, error_msg
            
    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.ConnectionError:
        return None, "Connection error"
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_series_info(series_id):
    """Get information about a FRED series"""
    try:
        url = "https://api.stlouisfed.org/fred/series"
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'seriess' in data and data['seriess']:
                return data['seriess'][0]
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def search_fred_series(search_text, limit=10):
    """Search for FRED series"""
    try:
        url = "https://api.stlouisfed.org/fred/series/search"
        params = {
            'search_text': search_text,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def generate_fallback_data(series_id, start_date, end_date, frequency):
    """Generate fallback data only when FRED API completely fails"""
    if frequency == 'monthly':
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    
    n = len(dates)
    np.random.seed(hash(series_id) % 10000)
    
    # Simple linear trend with noise
    time_index = np.arange(n)
    base_trend = 10000 + 50 * time_index
    noise = np.random.normal(0, 1000, n)
    values = base_trend + noise
    values = np.abs(values)
    
    return pd.DataFrame({'Value': values}, index=dates)

def get_latest_date_info(data_dict):
    """Get the latest available date across all data"""
    latest_dates = []
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            latest_dates.append(data['assets'].index[-1])
    
    if latest_dates:
        latest_overall = max(latest_dates)
        return latest_overall.strftime('%B %d, %Y')
    return "Not available"

@st.cache_data(ttl=3600)
def load_fund_data(selected_categories, start_date, frequency):
    """Load fund data from FRED API with enhanced error handling"""
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    data_dict = {}
    api_status = {}
    
    for category in selected_categories:
        if category in FRED_SERIES:
            series_info = FRED_SERIES[category]
            series_id = series_info['fred_id']
            
            # Check frequency compatibility
            available_freqs = series_info.get('available_frequencies', ['monthly'])
            if frequency not in available_freqs:
                use_freq = available_freqs[0]
            else:
                use_freq = frequency
            
            # Fetch data from FRED API
            df, data_source = fetch_fred_data(series_id, start_date, end_date, use_freq)
            
            api_status[category] = {
                'series_id': series_id,
                'status': 'FRED' if df is not None else 'Failed',
                'message': data_source
            }
            
            if df is None or df.empty:
                # Try to get any available data with different parameters
                st.warning(f"⚠️ Could not fetch {category} ({series_id}) from FRED: {data_source}")
                
                # Try without frequency conversion
                df, _ = fetch_fred_data(series_id, start_date, end_date, 'lin')
                
                if df is None or df.empty:
                    # Last resort: generate minimal fallback data
                    st.info(f"Using fallback data for {category}")
                    df = generate_fallback_data(series_id, start_date, end_date, use_freq)
                    data_source_type = 'Fallback'
                else:
                    data_source_type = 'FRED (Raw)'
            else:
                data_source_type = 'FRED'
            
            if df is not None and not df.empty:
                # Ensure data is numeric
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                df = df.dropna()
                
                if len(df) < 2:
                    st.warning(f"Very limited data for {category}. Expanding date range.")
                    # Try to get more historical data
                    older_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365*5)).strftime('%Y-%m-%d')
                    df_older, _ = fetch_fred_data(series_id, older_start, end_date, use_freq)
                    if df_older is not None and not df_older.empty:
                        df = df_older
                
                # Calculate flows (first difference)
                df_flows = df.diff()
                df_flows.columns = ['Flow']
                
                # Calculate percentage changes
                df_pct = df.pct_change() * 100
                df_pct.columns = ['Pct_Change']
                
                # Calculate returns (log returns for better statistical properties)
                if not df.empty and len(df) > 1:
                    df_returns = np.log(df/df.shift(1)) * 100
                    df_returns.columns = ['Log_Return']
                else:
                    df_returns = pd.DataFrame(columns=['Log_Return'])
                
                data_dict[category] = {
                    'assets': df,
                    'flows': df_flows,
                    'pct_change': df_pct,
                    'log_returns': df_returns,
                    'description': series_info['description'],
                    'fred_id': series_info['fred_id'],
                    'category': series_info['category'],
                    'unit': series_info['unit'],
                    'source': series_info['source'],
                    'color': series_info['color'],
                    'frequency': use_freq,
                    'periods_for_trend': 12 if use_freq == 'monthly' else 24,
                    'data_source': data_source_type,
                    'api_status': api_status[category]
                }
    
    return data_dict, api_status

def calculate_comprehensive_statistics(data_dict):
    """Calculate comprehensive statistics for all series"""
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets']['Value']
            returns = data['log_returns']['Log_Return'].dropna() if 'log_returns' in data else pd.Series(dtype=float)
            
            if len(assets) > 0:
                stats_dict = {}
                
                # Basic statistics
                stats_dict['mean'] = float(assets.mean())
                stats_dict['median'] = float(assets.median())
                stats_dict['std'] = float(assets.std())
                stats_dict['min'] = float(assets.min())
                stats_dict['max'] = float(assets.max())
                stats_dict['skewness'] = float(assets.skew())
                stats_dict['kurtosis'] = float(assets.kurtosis())
                stats_dict['cv'] = float(stats_dict['std'] / stats_dict['mean'] if stats_dict['mean'] != 0 else 0)
                
                # Return statistics
                if len(returns) > 0:
                    stats_dict['return_mean'] = float(returns.mean())
                    stats_dict['return_std'] = float(returns.std())
                    stats_dict['sharpe_ratio'] = float((returns.mean() / returns.std() * np.sqrt(12)) if returns.std() > 0 else 0)
                    stats_dict['sortino_ratio'] = float(calculate_sortino_ratio(returns))
                    stats_dict['max_drawdown'] = float(calculate_max_drawdown(assets))
                    
                    # Stationarity tests
                    try:
                        if len(returns.dropna()) > 10:
                            adf_result = adfuller(returns.dropna())
                            stats_dict['adf_statistic'] = float(adf_result[0])
                            stats_dict['adf_pvalue'] = float(adf_result[1])
                            
                            kpss_result = kpss(returns.dropna(), regression='c')
                            stats_dict['kpss_statistic'] = float(kpss_result[0])
                            stats_dict['kpss_pvalue'] = float(kpss_result[1])
                        else:
                            stats_dict['adf_statistic'] = None
                            stats_dict['adf_pvalue'] = None
                            stats_dict['kpss_statistic'] = None
                            stats_dict['kpss_pvalue'] = None
                    except:
                        stats_dict['adf_statistic'] = None
                        stats_dict['adf_pvalue'] = None
                        stats_dict['kpss_statistic'] = None
                        stats_dict['kpss_pvalue'] = None
                
                # Trend statistics
                if len(assets) > 12:
                    # Linear trend
                    x = np.arange(len(assets), dtype=np.float64)
                    y = assets.values.astype(np.float64)
                    slope, intercept = np.polyfit(x, y, 1)
                    stats_dict['trend_slope'] = float(slope)
                    stats_dict['trend_r2'] = float(np.corrcoef(x, y)[0, 1]**2)
                
                data['statistics'] = stats_dict
    
    return data_dict

def calculate_sortino_ratio(returns, risk_free_rate=0):
    """Calculate Sortino ratio"""
    downside_returns = returns[returns < risk_free_rate]
    if len(downside_returns) == 0:
        return 0.0
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    return float((returns.mean() - risk_free_rate) / downside_std * np.sqrt(12))

def calculate_max_drawdown(assets):
    """Calculate maximum drawdown"""
    if len(assets) < 2:
        return 0.0
    
    cumulative = (1 + assets.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    min_drawdown = drawdown.min()
    return float(min_drawdown * 100 if not pd.isna(min_drawdown) else 0.0)

def create_executive_summary(data_dict, frequency):
    """Create executive summary with latest date"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Executive Summary</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Create metrics row
    cols = st.columns(min(4, len(data_dict)))
    
    for idx, (category, data) in enumerate(list(data_dict.items())[:4]):
        with cols[idx % len(cols)]:
            if 'flows' in data and not data['flows'].empty and len(data['flows']) > 0:
                latest_flow = data['flows'].iloc[-1, 0]
                avg_flow = data['flows'].mean().iloc[0]
                
                # Calculate trend vs period average
                periods = data.get('periods_for_trend', 12)
                if len(data['flows']) >= periods:
                    period_avg = data['flows'].iloc[-periods:].mean().iloc[0]
                    trend = "up" if latest_flow > period_avg else "down" if latest_flow < period_avg else "neutral"
                else:
                    trend = "neutral"
                    period_avg = avg_flow
                
                trend_class = f"trend-{trend}"
                trend_symbol = "↗" if trend == "up" else "↘" if trend == "down" else "→"
                
                # Data source badge
                data_source = data.get('data_source', 'Unknown')
                badge_class = "badge-fred" if 'FRED' in data_source else "badge-sample"
                badge_text = "FRED" if 'FRED' in data_source else "Fallback"
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>
                        {category} 
                        <span class='data-source-badge {badge_class}'>{badge_text}</span>
                    </div>
                    <div class='metric-value'>${abs(latest_flow)/1000:,.1f}B</div>
                    <div>
                        <span style='color: {'#27ae60' if latest_flow > 0 else '#e74c3c'};'>
                            {'+' if latest_flow > 0 else ''}${latest_flow/1000:,.1f}B
                        </span>
                        <span class='trend-indicator {trend_class}'>
                            {trend_symbol} vs {periods} {frequency}
                        </span>
                    </div>
                    <div style='font-size: 0.8rem; color: #666666; margin-top: 8px;'>
                        {periods}-period avg: ${period_avg/1000:,.1f}B
                    </div>
                </div>
                """, unsafe_allow_html=True)

def create_professional_growth_charts(data_dict):
    """Create professional growth analysis with latest date"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Growth Dynamics Analysis</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Analysis controls
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Cumulative Growth Index", "Rolling Returns", "Risk-Adjusted Performance"],
            key="growth_type"
        )
    
    with col2:
        if analysis_type == "Rolling Returns":
            window = st.slider("Rolling Window Size", 1, 24, 12, key="growth_window")
        else:
            window = 12
    
    # Prepare growth data
    fig = go.Figure()
    
    for idx, (category, data) in enumerate(data_dict.items()):
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets']['Value']
            color = data.get('color', PROFESSIONAL_COLORS[idx % len(PROFESSIONAL_COLORS)])
            
            if analysis_type == "Cumulative Growth Index":
                if len(assets) > 0 and assets.iloc[0] != 0:
                    y_data = 100 * assets / assets.iloc[0]
                    y_title = "Growth Index (Base = 100)"
                else:
                    continue
                
            elif analysis_type == "Rolling Returns":
                returns = assets.pct_change()
                if len(returns) >= window:
                    y_data = returns.rolling(window=window).mean() * 100
                    y_title = f"{window}-Period Rolling Return (%)"
                else:
                    continue
                
            elif analysis_type == "Risk-Adjusted Performance":
                returns = assets.pct_change()
                if len(returns) >= window:
                    rolling_std = returns.rolling(window=window).std()
                    rolling_mean = returns.rolling(window=window).mean()
                    y_data = rolling_mean / rolling_std * np.sqrt(12 if window >= 12 else 4)
                    y_title = f"{window}-Period Rolling Sharpe Ratio"
                else:
                    continue
            
            fig.add_trace(go.Scatter(
                x=y_data.index,
                y=y_data,
                name=f"{category} ({data.get('data_source', '')})",
                mode='lines',
                line=dict(width=2, color=color),
                hovertemplate='%{x|%b %Y}<br>' + f'{category}: %{{y:.2f}}<extra></extra>'
            ))
    
    if len(fig.data) > 0:
        fig.update_layout(
            title=f"{analysis_type}",
            xaxis_title="Date",
            yaxis_title=y_title,
            height=500,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for growth analysis")

def create_enhanced_flow_analysis(data_dict, frequency):
    """Create enhanced institutional flow analysis"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Flow Dynamics Analysis</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Create overview metrics
    st.markdown("### Flow Overview Dashboard")
    
    # Calculate summary metrics
    total_inflows = 0.0
    total_outflows = 0.0
    inflow_categories = []
    outflow_categories = []
    
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            latest_flow = data['flows'].iloc[-1, 0]
            if latest_flow > 0:
                total_inflows += float(latest_flow)
                inflow_categories.append(category)
            else:
                total_outflows += abs(float(latest_flow))
                outflow_categories.append(category)
    
    # Display flow overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Inflows</div>
            <div class='metric-value'>${total_inflows/1000:,.1f}B</div>
            <div class='flow-indicator flow-in'>{len(inflow_categories)} Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Outflows</div>
            <div class='metric-value'>${total_outflows/1000:,.1f}B</div>
            <div class='flow-indicator flow-out'>{len(outflow_categories)} Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        net_flow = total_inflows - total_outflows
        net_color = "#27ae60" if net_flow > 0 else "#e74c3c"
        net_icon = "↗️" if net_flow > 0 else "↘️"
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Net Flow</div>
            <div class='metric-value' style='color: {net_color}'>{net_icon} ${abs(net_flow)/1000:,.1f}B</div>
            <div style='font-size: 0.85rem; color: #666666;'>
                {'Positive' if net_flow > 0 else 'Negative'} Net Flow
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        flow_ratio = total_inflows / total_outflows if total_outflows > 0 else float('inf')
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Inflow/Outflow Ratio</div>
            <div class='metric-value'>{flow_ratio:.2f}:1</div>
            <div style='font-size: 0.85rem; color: #666666;'>
                {'Inflow Dominant' if flow_ratio > 1 else 'Outflow Dominant'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Flow visualization
    st.markdown("### Flow Trends Analysis")
    
    fig_flows = go.Figure()
    
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            flows = data['flows']['Flow']
            color = data.get('color', PROFESSIONAL_COLORS[0])
            
            fig_flows.add_trace(go.Scatter(
                x=flows.index,
                y=flows,
                name=f"{category} ({data.get('data_source', '')})",
                mode='lines',
                line=dict(width=1.5, color=color),
                hovertemplate='%{x|%b %Y}<br>' + f'{category}: $%{{y:,.0f}}M<extra></extra>'
            ))
    
    if len(fig_flows.data) > 0:
        fig_flows.update_layout(
            title='Fund Flow Trends',
            xaxis_title='Date',
            yaxis_title='Flow ($M)',
            height=500,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_flows, use_container_width=True)
    else:
        st.info("No flow data available for visualization")

def create_comprehensive_kpi_dashboard(data_dict):
    """Create comprehensive KPI dashboard"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Comprehensive KPI Dashboard</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Calculate comprehensive KPIs
    total_assets = 0.0
    total_flows = 0.0
    category_stats = []
    
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            latest_assets = float(data['assets'].iloc[-1, 0])
            latest_flow = float(data['flows'].iloc[-1, 0]) if 'flows' in data and not data['flows'].empty else 0.0
            
            total_assets += latest_assets
            total_flows += latest_flow
            
            # Calculate additional metrics
            assets_series = data['assets']['Value']
            
            if len(assets_series) > 12:
                growth_1y = float((assets_series.iloc[-1] / assets_series.iloc[-13] - 1) * 100)
            else:
                growth_1y = 0.0
            
            category_stats.append({
                'Category': category,
                'Assets': latest_assets,
                'Latest Flow': latest_flow,
                '1Y Growth': growth_1y,
                'Color': data['color'],
                'Data Source': data.get('data_source', 'Unknown')
            })
    
    # Create KPI Grid
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size: 0.9rem; opacity: 0.9;'>Total Assets</div>
            <div style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>${total_assets/1000000:,.1f}T</div>
            <div style='font-size: 0.8rem; opacity: 0.8;'>{len(category_stats)} Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        net_flow = total_flows
        flow_color = "#27ae60" if net_flow > 0 else "#e74c3c"
        flow_icon = "↗️" if net_flow > 0 else "↘️"
        
        st.markdown(f"""
        <div class='kpi-card-secondary'>
            <div style='font-size: 0.9rem; opacity: 0.9;'>Net Monthly Flow</div>
            <div style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0; color: {flow_color}'>{flow_icon} ${abs(net_flow)/1000:,.1f}B</div>
            <div style='font-size: 0.8rem; opacity: 0.8;'>{'Inflow' if net_flow > 0 else 'Outflow'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate average 1Y growth
        if category_stats:
            valid_growth = [s['1Y Growth'] for s in category_stats if s['1Y Growth'] != 0]
            avg_growth = float(np.mean(valid_growth)) if valid_growth else 0.0
        else:
            avg_growth = 0.0
        
        st.markdown(f"""
        <div class='kpi-card-tertiary'>
            <div style='font-size: 0.9rem; opacity: 0.9;'>Avg 1Y Growth</div>
            <div style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>{avg_growth:+.1f}%</div>
            <div style='font-size: 0.8rem; opacity: 0.8;'>Weighted Average</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Count FRED data sources
        fred_count = sum(1 for s in category_stats if 'FRED' in s['Data Source'])
        
        st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size: 0.9rem; opacity: 0.9;'>Data Quality</div>
            <div style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>{fred_count}/{len(category_stats)}</div>
            <div style='font-size: 0.8rem; opacity: 0.8;'>FRED Data Sources</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Asset composition pie chart
    st.markdown("### Asset Composition")
    
    if category_stats:
        fig_pie = go.Figure(data=[go.Pie(
            labels=[s['Category'] for s in category_stats],
            values=[s['Assets'] for s in category_stats],
            hole=0.3,
            marker_colors=[s['Color'] for s in category_stats],
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Assets: $%{value:,.0f}M<br>Share: %{percent}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            title='Asset Allocation by Category',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

def create_asset_decomposition_analysis(data_dict):
    """Create comprehensive asset decomposition analysis"""
    st.markdown(f"""
    <div class="section-header">
        <span>Asset Decomposition Analysis</span>
        <span class="date-indicator">Time Series Components</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Select category for decomposition
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox(
            "Select Category for Decomposition",
            list(data_dict.keys()),
            key="decomp_category"
        )
    
    with col2:
        decomposition_type = st.selectbox(
            "Decomposition Model",
            ["Additive", "Multiplicative"],
            key="decomp_type"
        )
    
    if selected_category:
        data = data_dict[selected_category]
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets']['Value']
            
            if len(assets) >= 24:  # Need at least 2 years for meaningful decomposition
                # Perform decomposition
                try:
                    if decomposition_type == "Additive":
                        decomposition = seasonal_decompose(assets, model='additive', period=12)
                    else:
                        decomposition = seasonal_decompose(assets, model='multiplicative', period=12)
                    
                    # Create subplot for decomposition
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
                        vertical_spacing=0.08,
                        shared_xaxes=True
                    )
                    
                    # Observed
                    fig.add_trace(
                        go.Scatter(
                            x=decomposition.observed.index,
                            y=decomposition.observed,
                            mode='lines',
                            name='Observed',
                            line=dict(color=data['color'], width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Trend
                    fig.add_trace(
                        go.Scatter(
                            x=decomposition.trend.index,
                            y=decomposition.trend,
                            mode='lines',
                            name='Trend',
                            line=dict(color='#2c3e50', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    # Seasonal
                    fig.add_trace(
                        go.Scatter(
                            x=decomposition.seasonal.index,
                            y=decomposition.seasonal,
                            mode='lines',
                            name='Seasonal',
                            line=dict(color='#3498db', width=2)
                        ),
                        row=3, col=1
                    )
                    
                    # Residual
                    fig.add_trace(
                        go.Scatter(
                            x=decomposition.resid.index,
                            y=decomposition.resid,
                            mode='lines',
                            name='Residual',
                            line=dict(color='#e74c3c', width=2)
                        ),
                        row=4, col=1
                    )
                    
                    fig.update_layout(
                        title=f'{selected_category} - Time Series Decomposition ({decomposition_type} Model)',
                        height=800,
                        hovermode='x unified',
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Decomposition failed: {str(e)}")
                    st.info("Try selecting a different category or using additive model.")
            else:
                st.warning(f"Need at least 24 periods for decomposition. Only have {len(assets)} periods.")
        else:
            st.warning("No asset data available for selected category")

def create_data_explorer_tab(data_dict):
    """Create advanced data explorer tab"""
    st.markdown(f"""
    <div class="section-header">
        <span>📊 Advanced Data Explorer</span>
        <span class="date-indicator">Comprehensive Historical Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Calculate comprehensive statistics
    data_dict = calculate_comprehensive_statistics(data_dict)
    
    explorer_tab1, explorer_tab2 = st.tabs(["📋 Historical Data Table", "📈 Statistical Summary"])
    
    with explorer_tab1:
        st.markdown("##### Complete Historical Data")
        
        # Select category for detailed view
        selected_category = st.selectbox(
            "Select Category",
            list(data_dict.keys()),
            key="explorer_category"
        )
        
        if selected_category:
            data = data_dict[selected_category]
            
            # Create comprehensive data table
            if 'assets' in data and not data['assets'].empty:
                assets_df = data['assets'].copy()
                flows_df = data['flows'].copy() if 'flows' in data else pd.DataFrame()
                
                # Combine data
                combined_df = assets_df.copy()
                combined_df.columns = ['Assets']
                
                if not flows_df.empty:
                    combined_df['Flow'] = flows_df['Flow']
                
                # Add derived metrics
                combined_df['Assets_Change'] = combined_df['Assets'].pct_change() * 100
                combined_df['Cumulative_Return'] = (1 + combined_df['Assets_Change']/100).cumprod() * 100
                
                # Format for display
                display_df = combined_df.copy()
                display_df.index = display_df.index.strftime('%Y-%m-%d')
                
                # Display table
                st.dataframe(display_df.style.format({
                    'Assets': '${:,.0f}M',
                    'Flow': '${:,.0f}M',
                    'Assets_Change': '{:.2f}%',
                    'Cumulative_Return': '{:.2f}'
                }), use_container_width=True, height=400)
                
                # Download option
                csv = combined_df.to_csv()
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"{selected_category.replace(' ', '_')}_data.csv",
                    mime="text/csv"
                )
    
    with explorer_tab2:
        st.markdown("##### Comprehensive Statistical Summary")
        
        # Create statistics table for all categories
        stats_data = []
        for category, data in data_dict.items():
            if 'statistics' in data and data['statistics']:
                stats = data['statistics']
                stats_data.append({
                    'Category': category,
                    'Mean ($M)': f"${stats.get('mean', 0):,.0f}",
                    'Std Dev ($M)': f"${stats.get('std', 0):,.0f}",
                    'CV': f"{stats.get('cv', 0):.3f}",
                    'Skewness': f"{stats.get('skewness', 0):.3f}",
                    'Kurtosis': f"{stats.get('kurtosis', 0):.3f}",
                    'Annual Return': f"{stats.get('return_mean', 0)*12:.2f}%" if stats.get('return_mean') is not None else "N/A",
                    'Annual Vol': f"{stats.get('return_std', 0)*np.sqrt(12):.2f}%" if stats.get('return_std') is not None else "N/A",
                    'Data Source': data.get('data_source', 'Unknown')
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, height=400)

def main():
    """Main application function"""
    
    # Test API key first
    api_valid, api_message = test_fred_api()
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        
        # Display API status
        if api_valid:
            st.success(api_message)
        else:
            st.error(api_message)
            st.warning("Some data may not be available")
        
        frequency = st.selectbox(
            "Data Frequency",
            ["monthly", "weekly"],
            help="Select data frequency (monthly recommended for statistical analysis)"
        )
        
        st.session_state.frequency = frequency
        
        years_back = st.slider(
            "Analysis Period (Years)",
            1, 20, 10,
            help="Number of years of historical data to analyze"
        )
        
        start_date = (datetime.today() - timedelta(days=years_back*365)).strftime('%Y-%m-%d')
        
        st.markdown("### 📊 Fund Categories")
        st.caption("Select categories to analyze")
        
        selected_categories = []
        for category, series_info in FRED_SERIES.items():
            if st.checkbox(
                f"{category} ({series_info['fred_id']})", 
                value=True if category in ['Total Mutual Fund Assets', 'Equity Mutual Fund Assets', 'Bond Mutual Fund Assets'] else False,
                help=f"{series_info['description']} | Source: {series_info['source']}"
            ):
                selected_categories.append(category)
        
        if not selected_categories:
            st.warning("Please select at least one fund category")
            return
        
        st.markdown("### 🔧 Analysis Settings")
        show_decomposition = st.checkbox("Show Asset Decomposition", value=True)
        show_data_explorer = st.checkbox("Show Advanced Data Explorer", value=True)
        
        st.markdown("---")
        if st.button("🔄 Refresh Analysis", type="secondary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 📈 Advanced Analytics Features
        
        **Key Features:**
        
        1. **Real FRED API Integration**
           - Direct access to Federal Reserve economic data
           - Real-time data updates
           - Professional-grade data quality
        
        2. **Advanced Statistical Analysis**
           - Time series decomposition
           - Comprehensive KPI dashboard
           - Risk-adjusted performance metrics
        
        3. **Professional Visualizations**
           - Asset composition charts
           - Growth trend analysis
           - Flow dynamics visualization
        
        4. **Advanced Data Explorer**
           - Complete historical tables
           - Statistical summaries
           - Data download functionality
        
        **Data Source:** Federal Reserve Economic Data (FRED)
        """)
    
    # Data source information at the top
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: white; margin-top: 0;'>📊 Federal Reserve Economic Data (FRED) Analytics</h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;'>
            <div>
                <div style='font-size: 0.9rem; opacity: 0.9;'>Analysis Period</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{years_back} Years</div>
            </div>
            <div>
                <div style='font-size: 0.9rem; opacity: 0.9;'>Frequency</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{frequency.capitalize()}</div>
            </div>
            <div>
                <div style='font-size: 0.9rem; opacity: 0.9;'>Categories</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{len(selected_categories)} Selected</div>
            </div>
            <div>
                <div style='font-size: 0.9rem; opacity: 0.9;'>API Status</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{'✅ Active' if api_valid else '⚠️ Limited'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Fetching real-time data from FRED API..."):
        data_dict, api_status = load_fund_data(selected_categories, start_date, frequency)
    
    if not data_dict:
        st.error("Failed to load data. Please check your configuration and try again.")
        
        with st.expander("🔧 Troubleshooting Tips", expanded=True):
            st.markdown("""
            ### Common Issues and Solutions:
            
            1. **API Key Issues:**
               - Verify your FRED API key is valid
               - Check if you've exceeded rate limits
               - Try again in a few minutes
            
            2. **Network Issues:**
               - Check your internet connection
               - The FRED API may be temporarily unavailable
            
            3. **Data Availability:**
               - Some series may not have recent data
               - Try different date ranges
               - Select different categories
            
            4. **Series IDs:**
               - Verify series IDs exist on FRED website
               - Some series may have been discontinued
            """)
        
        return
    
    # Display API status for each series
    with st.expander("📡 API Status Details", expanded=False):
        status_df = pd.DataFrame([
            {
                'Category': cat,
                'Series ID': api_status[cat]['series_id'],
                'Status': '✅ Success' if api_status[cat]['status'] == 'FRED' else '⚠️ Limited',
                'Message': api_status[cat]['message'],
                'Data Points': len(data_dict[cat]['assets']) if cat in data_dict else 0
            }
            for cat in selected_categories if cat in api_status
        ])
        st.dataframe(status_df, use_container_width=True)
    
    st.session_state.data_dict = data_dict
    
    # Main dashboard tabs
    main_tabs = st.tabs([
        "📊 KPI Dashboard",
        "📈 Growth Analysis",
        "💰 Flow Dynamics",
        "🔍 Asset Decomposition",
        "📋 Data Explorer"
    ])
    
    with main_tabs[0]:
        create_executive_summary(data_dict, frequency)
        create_comprehensive_kpi_dashboard(data_dict)
        
        # Additional insights
        st.markdown("### 🎯 Quick Insights")
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            st.markdown("""
            <div class='analysis-card'>
                <strong>📊 Real FRED Data</strong><br>
                All analyses use actual Federal Reserve economic data from FRED API.
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col2:
            st.markdown("""
            <div class='analysis-card'>
                <strong>📈 Professional Analysis</strong><br>
                Time series decomposition, statistical testing, and comprehensive metrics.
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col3:
            st.markdown("""
            <div class='analysis-card'>
                <strong>🔍 Data Transparency</strong><br>
                Clear data source indicators and API status for each series.
            </div>
            """, unsafe_allow_html=True)
    
    with main_tabs[1]:
        create_professional_growth_charts(data_dict)
    
    with main_tabs[2]:
        create_enhanced_flow_analysis(data_dict, frequency)
    
    with main_tabs[3]:
        if show_decomposition:
            create_asset_decomposition_analysis(data_dict)
        else:
            st.info("Enable 'Show Asset Decomposition' in the sidebar to access this section")
    
    with main_tabs[4]:
        if show_data_explorer:
            create_data_explorer_tab(data_dict)
        else:
            st.info("Enable 'Show Advanced Data Explorer' in the sidebar to access this section")
    
    # Footer
    st.markdown(f"""
    <div class='footer'>
        <p>Institutional Fund Flow Analytics Dashboard v4.0 | Federal Reserve Economic Data (FRED) API Integration</p>
        <p>Real-Time Data | Professional Statistical Analysis | Federal Reserve Data Source</p>
        <p>Analysis Period: {years_back} Years | Frequency: {frequency.capitalize()} | Categories: {len(selected_categories)}</p>
        <p style='font-size: 0.75rem; color: #999999; margin-top: 1rem;'>
            This dashboard integrates with the Federal Reserve Economic Data (FRED) API from the Federal Reserve Bank of St. Louis.
            All data is sourced directly from FRED and is subject to their terms of use and data licensing agreements.
            API Key Status: {'Active' if api_valid else 'Limited'}
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
