import streamlit as st
import sqlite3
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Corporate Guidance Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    
    /* Global font override */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container background - dark navy */
    .stApp {
        background: #1a1f3c;
    }
    
    /* Main header styling - coral/salmon gradient */
    .main-header {
        background: linear-gradient(135deg, #e8b4a6 0%, #d4a99a 50%, #c99b8e 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: #2d3250;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    .main-header h1 {
        margin: 0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #2d3250;
    }
    .main-header p {
        margin: 0.75rem 0 0 0;
        opacity: 0.8;
        font-size: 1.1rem;
        font-weight: 400;
        color: #2d3250;
    }
    
    /* Metric cards - dark navy with subtle border */
    div[data-testid="stMetric"] {
        background: #252b48;
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8b9dc3 !important;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2rem;
    }
    
    /* Remove individual card colors - use uniform dark style */
    div[data-testid="column"]:nth-child(1) div[data-testid="stMetric"],
    div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"],
    div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"],
    div[data-testid="column"]:nth-child(4) div[data-testid="stMetric"],
    div[data-testid="column"]:nth-child(5) div[data-testid="stMetric"] {
        background: #252b48;
    }
    
    /* Sidebar styling - darker navy */
    section[data-testid="stSidebar"] {
        background: #151a30;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: #d4a99a !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(212, 169, 154, 0.2);
    }
    section[data-testid="stSidebar"] .stCheckbox label span {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #8b9dc3 !important;
    }
    
    /* Multiselect and selectbox dark theme */
    .stMultiSelect > div, .stSelectbox > div {
        background: #252b48 !important;
        border: 1px solid rgba(212, 169, 154, 0.3) !important;
        border-radius: 10px !important;
    }
    .stMultiSelect span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #d4a99a, #c99b8e) !important;
        color: #2d3250 !important;
    }
    
    /* Table section header - coral accent */
    .section-header {
        background: #252b48;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        border-left: 4px solid #d4a99a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #d4a99a;
    }
    .section-header h3 {
        margin: 0;
        color: #e2e8f0;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Info and warning boxes */
    .stAlert {
        background: #252b48 !important;
        border: 1px solid rgba(212, 169, 154, 0.3) !important;
        border-radius: 12px !important;
    }
    
    /* Download button - coral theme */
    .stDownloadButton button {
        background: linear-gradient(135deg, #d4a99a 0%, #c99b8e 100%) !important;
        color: #2d3250 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(212, 169, 154, 0.4) !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #8b9dc3 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: #8b9dc3;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 2rem;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1f3c;
    }
    ::-webkit-scrollbar-thumb {
        background: #d4a99a;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #e8b4a6;
    }
    
    /* Search box styling */
    .stTextInput > div > div {
        background: #252b48 !important;
        border: 1px solid rgba(212, 169, 154, 0.3) !important;
        border-radius: 10px !important;
    }
    .stTextInput input {
        color: #e2e8f0 !important;
    }
    .stTextInput input::placeholder {
        color: #8b9dc3 !important;
    }
    
    /* Date input styling */
    .stDateInput > div > div {
        background: #252b48 !important;
        border: 1px solid rgba(212, 169, 154, 0.3) !important;
        border-radius: 10px !important;
    }
    .stDateInput input {
        color: #e2e8f0 !important;
    }
    
    /* Charts container */
    .chart-container {
        background: #252b48;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #252b48;
        border-radius: 8px;
        color: #8b9dc3;
        border: 1px solid rgba(212, 169, 154, 0.2);
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #d4a99a 0%, #c99b8e 100%) !important;
        color: #2d3250 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #252b48 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .streamlit-expanderContent {
        background: #1a1f3c !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Text area for SQL */
    .stTextArea textarea {
        background: #252b48 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(212, 169, 154, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #d4a99a 0%, #c99b8e 100%) !important;
        color: #2d3250 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Database connection
DB_PATH = Path("finance_data.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def load_data():
    conn = get_connection()
    query = """
    SELECT 
        g.ticker AS "Ticker",
        g.guidance_type AS "Type",
        g.metric_name AS "Metric",
        g.reporting_period AS "Period",
        g.guided_range_low AS "Range Low",
        g.guided_range_high AS "Range High",
        g.unit AS "Unit",
        g.is_revision AS "Is Revision",
        g.revision_direction AS "Revision Dir",
        g.qualitative_direction AS "Direction",
        c.published_at AS "Published",
        c.link AS "Source"
    FROM guidance g
    LEFT JOIN contents c ON g.content_uid = c.uid
    ORDER BY c.published_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def format_range(row):
    """Format guided range with unit"""
    low = row['Range Low']
    high = row['Range High']
    unit = row['Unit'] if pd.notna(row['Unit']) else ''
    
    if pd.isna(low) and pd.isna(high):
        return "—"
    elif pd.isna(low):
        return f"≤ {high:,.2f} {unit}".strip()
    elif pd.isna(high):
        return f"≥ {low:,.2f} {unit}".strip()
    elif low == high:
        return f"{low:,.2f} {unit}".strip()
    else:
        return f"{low:,.2f} – {high:,.2f} {unit}".strip()

def format_date(date_str):
    """Format date for display"""
    if pd.isna(date_str):
        return "—"
    try:
        dt = pd.to_datetime(date_str, utc=True)
        return dt.strftime('%b %d, %Y')
    except:
        return str(date_str)[:10]

def create_excel_download(df):
    """Create formatted Excel file with multiple sheets"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main data sheet
        export_df = df.copy()
        export_df.to_excel(writer, sheet_name='Guidance Data', index=False)
        
        # Summary by ticker
        ticker_summary = df.groupby('Ticker').agg({
            'Metric': 'count',
            'Is Revision': 'sum'
        }).rename(columns={'Metric': 'Total Guidance', 'Is Revision': 'Revisions'})
        ticker_summary.to_excel(writer, sheet_name='By Ticker')
        
        # Summary by type
        type_summary = df.groupby('Type').agg({
            'Metric': 'count'
        }).rename(columns={'Metric': 'Count'})
        type_summary.to_excel(writer, sheet_name='By Type')
        
        # Summary by period
        period_summary = df.groupby('Period').agg({
            'Metric': 'count'
        }).rename(columns={'Metric': 'Count'})
        period_summary.to_excel(writer, sheet_name='By Period')
    
    return output.getvalue()

def get_guidance_type_colors():
    """Color palette for guidance types - rainbow for distinction"""
    return {
        'revenue': '#f97316',      # orange
        'earnings': '#eab308',     # yellow
        'EPS': '#22c55e',          # green
        'margin': '#06b6d4',       # cyan
        'capex': '#3b82f6',        # blue
        'opex': '#8b5cf6',         # violet
        'cash_flow': '#ec4899',    # pink
        'ebitda': '#ef4444',       # red
        'other': '#94a3b8'         # slate
    }

# Header
st.markdown("""
<div class="main-header">
    <h1>Corporate Guidance Tracker</h1>
    <p>Monitor forward-looking statements and financial guidance from corporate filings</p>
</div>
""", unsafe_allow_html=True)

# Load data
try:
    df = load_data()
    
    if df.empty:
        st.warning("⚠️ No guidance data found in the database.")
        st.info("Run the extraction pipeline to populate the database with guidance events.")
    else:
        # Sidebar Filters
        with st.sidebar:
            st.markdown("### Filters")
            st.markdown("---")
            
            # Search box
            search_query = st.text_input(
                "Search",
                placeholder="Search companies, metrics...",
                help="Search across all columns"
            )
            
            st.markdown("---")
            
            # Date Range Filter
            st.markdown("##### Date Range")
            
            # Get date range from data
            df['Published_dt'] = pd.to_datetime(df['Published'], errors='coerce', utc=True)
            min_date = df['Published_dt'].min()
            max_date = df['Published_dt'].max()
            
            if pd.notna(min_date) and pd.notna(max_date):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    start_date = st.date_input(
                        "From",
                        value=min_date.date(),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        label_visibility="collapsed"
                    )
                with col_d2:
                    end_date = st.date_input(
                        "To", 
                        value=max_date.date(),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        label_visibility="collapsed"
                    )
            else:
                start_date = None
                end_date = None
            
            st.markdown("---")
            
            # Ticker Filter
            tickers = sorted(df['Ticker'].dropna().unique().tolist())
            selected_tickers = st.multiselect(
                "Tickers",
                options=tickers,
                default=[],
                placeholder="All tickers"
            )
            
            # Guidance Type Filter
            guidance_types = sorted(df['Type'].dropna().unique().tolist())
            selected_types = st.multiselect(
                "Guidance Types",
                options=guidance_types,
                default=[],
                placeholder="All types"
            )
            
            # Period Filter
            periods = sorted(df['Period'].dropna().unique().tolist())
            selected_periods = st.multiselect(
                "Reporting Periods",
                options=periods,
                default=[],
                placeholder="All periods"
            )
            
            # Revision filter
            st.markdown("---")
            show_revisions_only = st.checkbox("Show revisions only", value=False)
        
        # Apply filters
        filtered_df = df.copy()
        
        # Ensure Published_dt is datetime in filtered_df
        filtered_df['Published_dt'] = pd.to_datetime(filtered_df['Published'], errors='coerce', utc=True)
        
        # Apply search filter
        if search_query:
            search_lower = search_query.lower()
            mask = filtered_df.apply(
                lambda row: any(search_lower in str(val).lower() for val in row.values), 
                axis=1
            )
            filtered_df = filtered_df[mask]
        
        # Apply date filter
        if start_date and end_date:
            # Filter to rows with valid dates first, then apply date range
            valid_dates = filtered_df['Published_dt'].notna()
            filtered_df = filtered_df[valid_dates]
            if not filtered_df.empty:
                # Convert to date for comparison
                pub_dates = pd.to_datetime(filtered_df['Published_dt'], utc=True).dt.date
                filtered_df = filtered_df[
                    (pub_dates >= start_date) &
                    (pub_dates <= end_date)
                ]
        
        if selected_tickers:
            filtered_df = filtered_df[filtered_df['Ticker'].isin(selected_tickers)]
        if selected_types:
            filtered_df = filtered_df[filtered_df['Type'].isin(selected_types)]
        if selected_periods:
            filtered_df = filtered_df[filtered_df['Period'].isin(selected_periods)]
        if show_revisions_only:
            filtered_df = filtered_df[filtered_df['Is Revision'] == 1]

        # Key Metrics Row
        st.markdown("")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Guidance Events",
                value=f"{len(filtered_df):,}"
            )
        with col2:
            st.metric(
                label="Companies",
                value=filtered_df['Ticker'].nunique()
            )
        with col3:
            st.metric(
                label="Unique Metrics",
                value=filtered_df['Metric'].nunique()
            )
        with col4:
            revision_count = filtered_df['Is Revision'].sum() if 'Is Revision' in filtered_df.columns else 0
            st.metric(
                label="Revisions",
                value=int(revision_count)
            )
        with col5:
            latest_date = pd.to_datetime(filtered_df['Published'], errors='coerce', utc=True).max()
            st.metric(
                label="Latest Update",
                value=format_date(latest_date) if pd.notnull(latest_date) else "N/A"
            )

        # Charts Section - Guidance Type Breakdown & Activity Timeline
        st.markdown("")
        chart_col1, chart_col2 = st.columns([1, 2])
        
        with chart_col1:
            st.markdown("""
            <div class="section-header">
                <h3>Guidance by Type</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
            
            # Donut chart data
            type_counts = filtered_df['Type'].value_counts()
            if not type_counts.empty:
                colors = get_guidance_type_colors()
                
                chart_data = pd.DataFrame({
                    'Type': type_counts.index,
                    'Count': type_counts.values
                })
                chart_data['Color'] = chart_data['Type'].map(colors).fillna('#64748b')
                chart_data['Percentage'] = (chart_data['Count'] / chart_data['Count'].sum() * 100).round(1)
                
                donut = alt.Chart(chart_data).mark_arc(innerRadius=60, outerRadius=100).encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(
                        field="Type",
                        type="nominal",
                        scale=alt.Scale(
                            domain=list(colors.keys()),
                            range=list(colors.values())
                        ),
                        legend=alt.Legend(
                            orient='bottom',
                            columns=3,
                            labelColor='#a5b4fc',
                            titleColor='#a5b4fc'
                        )
                    ),
                    tooltip=['Type', 'Count', 'Percentage']
                ).properties(
                    height=320,
                    padding={'top': 30, 'bottom': 10, 'left': 10, 'right': 10}
                ).configure_view(
                    strokeWidth=0
                ).configure(
                    background='transparent'
                )
                
                st.altair_chart(donut, use_container_width=True)
        
        with chart_col2:
            st.markdown("""
            <div class="section-header">
                <h3>Activity Timeline</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Timeline/sparkline chart showing guidance events over time
            if 'Published_dt' in filtered_df.columns:
                timeline_df = filtered_df.copy()
                timeline_df['Published_dt'] = pd.to_datetime(timeline_df['Published_dt'], errors='coerce', utc=True)
                timeline_df = timeline_df[timeline_df['Published_dt'].notna()]
                
                if not timeline_df.empty:
                    # Group by week - total events only
                    timeline_df['Week'] = timeline_df['Published_dt'].dt.to_period('W').dt.start_time
                    weekly_counts = timeline_df.groupby('Week').size().reset_index(name='Count')
                    
                    area_chart = alt.Chart(weekly_counts).mark_area(
                        opacity=0.7,
                        interpolate='monotone',
                        color='#d4a99a',
                        line={'color': '#e8b4a6', 'strokeWidth': 2}
                    ).encode(
                        x=alt.X('Week:T', title='', axis=alt.Axis(labelColor='#8b9dc3', format='%b %Y')),
                        y=alt.Y('Count:Q', title='Events', axis=alt.Axis(labelColor='#8b9dc3')),
                        tooltip=['Week:T', 'Count']
                    ).properties(
                        height=220
                    ).configure_view(
                        strokeWidth=0
                    ).configure(
                        background='transparent'
                    ).configure_axis(
                        gridColor='rgba(139, 157, 195, 0.1)'
                    )
                    
                    st.altair_chart(area_chart, use_container_width=True)
                else:
                    st.info("No timeline data available")

        # Data Table Section
        st.markdown("""
        <div class="section-header">
            <h3>Guidance Events</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if filtered_df.empty:
            st.info("No records match the selected filters. Try adjusting your filter criteria.")
        else:
            # Prepare display dataframe
            display_df = filtered_df.copy()
            
            # Format the guided range into a single column
            display_df['Guided Range'] = display_df.apply(format_range, axis=1)
            
            # Format the published date
            display_df['Published'] = display_df['Published'].apply(format_date)
            
            # Format Is Revision as Yes/No
            display_df['Is Revision'] = display_df['Is Revision'].apply(
                lambda x: "Yes" if x == 1 else "No" if x == 0 else "—"
            )
            
            # Handle missing source links - replace None/NaN with placeholder
            display_df['Source'] = display_df['Source'].apply(
                lambda x: x if pd.notna(x) and x else None
            )
            
            # Select and reorder columns for display
            display_columns = [
                'Published',
                'Ticker',
                'Type',
                'Metric',
                'Period',
                'Guided Range',
                'Direction',
                'Is Revision',
                'Source'
            ]
            
            # Only include columns that exist
            display_columns = [col for col in display_columns if col in display_df.columns]
            display_df = display_df[display_columns]
            
            # Configure column display
            column_config = {
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Metric": st.column_config.TextColumn("Metric", width="large"),
                "Period": st.column_config.TextColumn("Period", width="small"),
                "Guided Range": st.column_config.TextColumn("Guided Range", width="medium"),
                "Direction": st.column_config.TextColumn("Direction", width="small"),
                "Is Revision": st.column_config.TextColumn("Revision", width="small"),
                "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "Published": st.column_config.TextColumn("Published", width="small"),
                "Source": st.column_config.LinkColumn(
                    "Source",
                    width="small",
                    display_text="View"
                ),
            }
            
            st.dataframe(
                display_df,
                column_config=column_config,
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Download buttons
            st.markdown("")
            col_dl1, col_dl2, col_dl3, col_dl4 = st.columns([1, 1, 1, 2])
            with col_dl1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"guidance_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col_dl2:
                try:
                    excel_data = create_excel_download(filtered_df)
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name=f"guidance_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.button("Download Excel", disabled=True, help="Install openpyxl for Excel export")
            with col_dl3:
                st.caption(f"Showing {len(display_df):,} of {len(df):,} records")
        
        # Ticker Sparklines Section
        if len(filtered_df) > 0 and filtered_df['Ticker'].nunique() <= 20:
            st.markdown("""
            <div class="section-header">
                <h3>Trends by Ticker</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Get top tickers by guidance count
            top_tickers = filtered_df['Ticker'].value_counts().head(8).index.tolist()
            
            if top_tickers and 'Published_dt' in filtered_df.columns:
                sparkline_df = filtered_df[filtered_df['Ticker'].isin(top_tickers)].copy()
                sparkline_df['Published_dt'] = pd.to_datetime(sparkline_df['Published_dt'], errors='coerce', utc=True)
                sparkline_df = sparkline_df[sparkline_df['Published_dt'].notna()]
                
                if not sparkline_df.empty:
                    sparkline_df['Month'] = sparkline_df['Published_dt'].dt.to_period('M').dt.start_time
                    monthly_by_ticker = sparkline_df.groupby(['Month', 'Ticker']).size().reset_index(name='Count')
                    
                    sparklines = alt.Chart(monthly_by_ticker).mark_line(
                        strokeWidth=2,
                        point=alt.OverlayMarkDef(size=30)
                    ).encode(
                        x=alt.X('Month:T', title='', axis=alt.Axis(labelColor='#8b9dc3', format='%b')),
                        y=alt.Y('Count:Q', title='Events', axis=alt.Axis(labelColor='#8b9dc3')),
                        color=alt.Color(
                            'Ticker:N',
                            scale=alt.Scale(scheme='brownbluegreen'),
                            legend=alt.Legend(
                                orient='bottom',
                                columns=4,
                                labelColor='#8b9dc3'
                            )
                        ),
                        tooltip=['Month:T', 'Ticker', 'Count']
                    ).properties(
                        height=250
                    ).configure_view(
                        strokeWidth=0
                    ).configure(
                        background='transparent'
                    ).configure_axis(
                        gridColor='rgba(139, 157, 195, 0.1)'
                    )
                    
                    st.altair_chart(sparklines, use_container_width=True)

        # SQL Query Section
        st.markdown("""
        <div class="section-header">
            <h3>Custom SQL Query</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Run custom SQL queries", expanded=False):
            st.caption("Available tables: `guidance`, `contents`. Use SELECT queries only.")
            
            default_query = """SELECT 
    g.ticker, 
    g.guidance_type, 
    g.metric_name,
    g.reporting_period,
    g.guided_range_low,
    g.guided_range_high,
    g.unit
FROM guidance g
LIMIT 100"""
            
            sql_query = st.text_area(
                "SQL Query",
                value=default_query,
                height=150,
                label_visibility="collapsed"
            )
            
            if st.button("Run Query"):
                if sql_query.strip().lower().startswith("select"):
                    try:
                        conn = get_connection()
                        result_df = pd.read_sql_query(sql_query, conn)
                        conn.close()
                        st.success(f"Returned {len(result_df)} rows")
                        st.dataframe(result_df, use_container_width=True, height=400)
                        
                        # Download option
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            "Download Result",
                            data=csv,
                            file_name="query_result.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Query error: {e}")
                else:
                    st.warning("Only SELECT queries are allowed for safety.")

        # Footer
        st.markdown("""
        <div class="footer">
            <p>Data sourced from SEC filings and corporate announcements</p>
        </div>
        """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ Database file not found")
    st.info("Please ensure 'finance_data.db' exists in the project directory.")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("Please ensure 'finance_data.db' exists and has the correct schema.")

