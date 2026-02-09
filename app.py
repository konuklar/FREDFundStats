import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="ICI Mutual Fund Flows Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📈 ICI Monthly Mutual Fund Flows Dashboard")
st.markdown("""
This dashboard displays monthly net new cash flows by US investors into various mutual fund investment classes.
Data source: **Investment Company Institute (ICI)** - Estimated Long-Term Mutual Fund Flows
""")

# ICI data URLs
ICI_DATA_URLS = {
    "monthly": "https://www.icidata.org/research/trends_ffs.xls",
    "weekly": "https://www.icidata.org/research/ffs.xls"
}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_ici_data():
    """Load ICI mutual fund flows data from their Excel files"""
    try:
        # Try to load monthly data
        response = requests.get(ICI_DATA_URLS["monthly"], timeout=10)
        response.raise_for_status()
        
        # Read Excel file - there are typically multiple sheets
        excel_data = pd.ExcelFile(BytesIO(response.content))
        
        # Different Excel files have different sheet names
        # Try to find the sheet with monthly flows data
        sheets = excel_data.sheet_names
        
        # Common sheet names for monthly data
        possible_sheets = ["Monthly", "Sheet1", "Data", "LT Flows", "Long-Term"]
        
        target_sheet = None
        for sheet in possible_sheets:
            if sheet in sheets:
                target_sheet = sheet
                break
        
        if not target_sheet:
            target_sheet = sheets[0]
        
        # Read the data
        df = pd.read_excel(BytesIO(response.content), sheet_name=target_sheet, header=None)
        
        # Process the data - ICI files have specific formats
        # Look for the header row (usually contains "Month" or date references)
        for i in range(min(20, len(df))):
            if df.iloc[i].astype(str).str.contains('Month|Date|Net').any():
                header_row = i
                break
        else:
            header_row = 0
        
        # Read again with proper header
        df = pd.read_excel(
            BytesIO(response.content), 
            sheet_name=target_sheet, 
            header=header_row
        )
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        return df
        
    except Exception as e:
        st.error(f"Error loading ICI data: {str(e)}")
        
        # Fallback: Load sample data structure
        st.info("Using sample data structure. Actual ICI data loading failed.")
        
        # Create sample data for demonstration
        dates = pd.date_range(start='2007-01-01', end=datetime.now(), freq='MS')
        categories = {
            'Equity': np.random.randint(-20000, 40000, len(dates)).cumsum(),
            'Bond': np.random.randint(-10000, 25000, len(dates)).cumsum(),
            'Hybrid': np.random.randint(-5000, 15000, len(dates)).cumsum(),
            'Money Market': np.random.randint(-15000, 30000, len(dates)).cumsum(),
            'Total Net Assets': np.random.randint(500000, 1000000, len(dates))
        }
        
        df = pd.DataFrame(categories, index=dates)
        df['Month'] = dates.strftime('%Y-%m')
        df = df.reset_index(drop=True)
        
        return df

def process_ici_data(df):
    """Process and clean the ICI data"""
    # Make a copy
    processed_df = df.copy()
    
    # Look for date column
    date_columns = [col for col in processed_df.columns if any(
        term in str(col).lower() for term in ['date', 'month', 'year', 'period']
    )]
    
    if date_columns:
        date_col = date_columns[0]
        # Convert to datetime
        processed_df['Date'] = pd.to_datetime(processed_df[date_col], errors='coerce')
    
    # Look for flow data columns
    flow_columns = []
    for col in processed_df.columns:
        col_str = str(col).lower()
        if any(term in col_str for term in ['equity', 'stock', 'domestic', 'world']):
            flow_columns.append((col, 'Equity'))
        elif any(term in col_str for term in ['bond', 'fixed income', 'taxable', 'municipal']):
            flow_columns.append((col, 'Bond'))
        elif any(term in col_str for term in ['hybrid', 'balanced', 'mixed']):
            flow_columns.append((col, 'Hybrid'))
        elif any(term in col_str for term in ['money market', 'mmf', 'liquid']):
            flow_columns.append((col, 'Money Market'))
        elif any(term in col_str for term in ['total', 'net', 'flow', 'assets']):
            if 'date' not in col_str:
                flow_columns.append((col, 'Total'))
    
    # If we can't identify columns, create default structure
    if not flow_columns:
        st.info("Could not auto-identify fund categories. Using default structure.")
        # Create sample columns
        for i in range(1, min(6, len(processed_df.columns))):
            col_name = f"Fund_Category_{i}"
            processed_df[col_name] = pd.to_numeric(processed_df.iloc[:, i], errors='coerce')
        flow_columns = [(f"Fund_Category_{i}", f"Category_{i}") for i in range(1, 6)]
    
    # Filter numeric columns
    numeric_data = {}
    for col, category in flow_columns:
        if col in processed_df.columns:
            numeric_data[category] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Create processed dataframe
    result_df = pd.DataFrame(numeric_data)
    
    # Add date if available
    if 'Date' in processed_df.columns:
        result_df['Date'] = processed_df['Date']
    elif 'Month' in processed_df.columns:
        result_df['Date'] = pd.to_datetime(processed_df['Month'], errors='coerce')
    
    # Drop rows with no date
    result_df = result_df.dropna(subset=['Date'])
    
    # Sort by date
    result_df = result_df.sort_values('Date')
    
    # Calculate cumulative flows
    for col in result_df.columns:
        if col != 'Date' and col != 'Total':
            cum_col = f"{col}_Cumulative"
            result_df[cum_col] = result_df[col].cumsum()
    
    return result_df

def create_visualizations(df):
    """Create interactive visualizations"""
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(max_date - timedelta(days=365*3), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
            df = df[mask]
    
    # Category selection
    flow_cols = [col for col in df.columns if col != 'Date' and not col.endswith('_Cumulative')]
    selected_categories = st.sidebar.multiselect(
        "Select Fund Categories",
        options=flow_cols,
        default=flow_cols[:min(4, len(flow_cols))]
    )
    
    # Main dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Net Cash Flows")
        
        # Bar chart for monthly flows
        if selected_categories and 'Date' in df.columns:
            fig = go.Figure()
            
            for category in selected_categories:
                fig.add_trace(go.Bar(
                    name=category,
                    x=df['Date'],
                    y=df[category],
                    opacity=0.8
                ))
            
            fig.update_layout(
                barmode='group',
                title="Monthly Net New Cash Flows",
                xaxis_title="Date",
                yaxis_title="Millions USD",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cumulative Cash Flows")
        
        # Line chart for cumulative flows
        cum_cols = [f"{cat}_Cumulative" for cat in selected_categories if f"{cat}_Cumulative" in df.columns]
        
        if cum_cols and 'Date' in df.columns:
            fig2 = go.Figure()
            
            for cum_col in cum_cols:
                category = cum_col.replace('_Cumulative', '')
                fig2.add_trace(go.Scatter(
                    name=category,
                    x=df['Date'],
                    y=df[cum_col],
                    mode='lines',
                    line=dict(width=3)
                ))
            
            fig2.update_layout(
                title="Cumulative Net Cash Flows",
                xaxis_title="Date",
                yaxis_title="Millions USD",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics section
    st.subheader("📊 Summary Statistics")
    
    if selected_categories:
        stats_cols = st.columns(len(selected_categories))
        
        for idx, category in enumerate(selected_categories):
            with stats_cols[idx]:
                if category in df.columns:
                    latest_value = df[category].iloc[-1] if len(df) > 0 else 0
                    total_flow = df[category].sum()
                    avg_flow = df[category].mean()
                    
                    st.metric(
                        label=category,
                        value=f"${latest_value:,.0f}M",
                        delta=f"Avg: ${avg_flow:,.0f}M"
                    )
                    st.caption(f"Total: ${total_flow:,.0f}M")
    
    # Data table
    st.subheader("📋 Raw Data")
    
    display_cols = ['Date'] + selected_categories
    if all(col in df.columns for col in display_cols):
        display_df = df[display_cols].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m')
        
        # Format numeric columns
        for col in selected_categories:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}M" if pd.notnull(x) else "")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="ici_fund_flows.csv",
            mime="text/csv"
        )

def main():
    """Main application function"""
    
    # Load data
    with st.spinner("Loading ICI data..."):
        raw_data = load_ici_data()
    
    # Process data
    processed_data = process_ici_data(raw_data)
    
    # Create visualizations
    create_visualizations(processed_data)
    
    # Information section
    with st.expander("ℹ️ About this Data"):
        st.markdown("""
        ### Data Source
        - **Provider**: Investment Company Institute (ICI)
        - **Dataset**: Estimated Long-Term Mutual Fund Flows
        - **Frequency**: Monthly (actual numbers from "Trends in Mutual Fund Investing")
        - **Coverage**: 98% of industry assets
        - **Currency**: Millions of US Dollars (nominal)
        - **Time Period**: 2007 to present
        
        ### Notes
        1. Weekly cash flows are estimates based on reporting covering 98% of industry assets
        2. Monthly flows are actual numbers as reported in ICI's "Trends in Mutual Fund Investing"
        3. Negative values represent net outflows
        4. Positive values represent net inflows
        
        ### Fund Categories
        - **Equity**: Stock-based mutual funds
        - **Bond**: Fixed-income mutual funds
        - **Hybrid**: Balanced/mixed allocation funds
        - **Money Market**: Short-term liquid funds
        """)

if __name__ == "__main__":
    main()
