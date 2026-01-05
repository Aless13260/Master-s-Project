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
    
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%);
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .main-header p {
        margin: 0.75rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Metric cards - each with unique gradient */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e1e3f 0%, #2a2a5a 100%);
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2rem;
    }
    
    /* Color accents for metric cards */
    div[data-testid="column"]:nth-child(1) div[data-testid="stMetric"] {
        border-top: 3px solid #6366f1;
    }
    div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] {
        border-top: 3px solid #8b5cf6;
    }
    div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] {
        border-top: 3px solid #d946ef;
    }
    div[data-testid="column"]:nth-child(4) div[data-testid="stMetric"] {
        border-top: 3px solid #f97316;
    }
    div[data-testid="column"]:nth-child(5) div[data-testid="stMetric"] {
        border-top: 3px solid #22d3ee;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122a 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: #a5b4fc !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(165, 180, 252, 0.2);
    }
    section[data-testid="stSidebar"] .stCheckbox label span {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
    }
    
    /* Multiselect and selectbox dark theme */
    .stMultiSelect > div, .stSelectbox > div {
        background: rgba(30, 30, 63, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
    }
    .stMultiSelect span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    }
    
    /* Table section header */
    .section-header {
        background: linear-gradient(145deg, #1e1e3f 0%, #2a2a5a 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        border-left: 4px solid #8b5cf6;
        border: 1px solid rgba(255, 255, 255, 0.08);
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
        background: rgba(30, 30, 63, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #94a3b8 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: #64748b;
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
        background: #1a1a3e;
    }
    ::-webkit-scrollbar-thumb {
        background: #6366f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #8b5cf6;
    }
    
    /* Search box styling */
    .stTextInput > div > div {
        background: rgba(30, 30, 63, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
    }
    .stTextInput input {
        color: #e2e8f0 !important;
    }
    .stTextInput input::placeholder {
        color: #64748b !important;
    }
    
    /* Date input styling */
    .stDateInput > div > div {
        background: rgba(30, 30, 63, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
    }
    .stDateInput input {
        color: #e2e8f0 !important;
    }
    
    /* Charts container */
    .chart-container {
        background: linear-gradient(145deg, #1e1e3f 0%, #2a2a5a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Revision row highlighting */
    .revision-row {
        background: rgba(251, 191, 36, 0.1) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 30, 63, 0.6);
        border-radius: 8px;
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
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
        g.company AS "Company",
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
        
        # Summary by company
        company_summary = df.groupby('Company').agg({
            'Metric': 'count',
            'Is Revision': 'sum'
        }).rename(columns={'Metric': 'Total Guidance', 'Is Revision': 'Revisions'})
        company_summary.to_excel(writer, sheet_name='By Company')
        
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
    """Color palette for guidance types"""
    return {
        'revenue': '#6366f1',
        'earnings': '#8b5cf6', 
        'EPS': '#d946ef',
        'margin': '#f97316',
        'capex': '#22d3ee',
        'opex': '#10b981',
        'cash_flow': '#f43f5e',
        'ebitda': '#fbbf24',
        'other': '#64748b'
    }

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Corporate Guidance Tracker</h1>
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
            st.markdown("### 🔍 Filters")
            st.markdown("---")
            
            # Search box
            search_query = st.text_input(
                "🔎 Search",
                placeholder="Search companies, metrics...",
                help="Search across all columns"
            )
            
            st.markdown("---")
            
            # Date Range Filter
            st.markdown("##### 📆 Date Range")
            
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
            
            # Company Filter
            companies = sorted(df['Company'].dropna().unique().tolist())
            selected_companies = st.multiselect(
                "🏢 Companies",
                options=companies,
                default=[],
                placeholder="All companies"
            )
            
            # Guidance Type Filter
            guidance_types = sorted(df['Type'].dropna().unique().tolist())
            selected_types = st.multiselect(
                "📋 Guidance Types",
                options=guidance_types,
                default=[],
                placeholder="All types"
            )
            
            # Period Filter
            periods = sorted(df['Period'].dropna().unique().tolist())
            selected_periods = st.multiselect(
                "📅 Reporting Periods",
                options=periods,
                default=[],
                placeholder="All periods"
            )
            
            # Revision filter
            st.markdown("---")
            show_revisions_only = st.checkbox("📝 Show revisions only", value=False)
            
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            st.caption(f"**Database:** {len(df)} total records")
            st.caption(f"**Companies:** {df['Company'].nunique()}")
            st.caption(f"**Date Range:** {format_date(df['Published'].min())} to {format_date(df['Published'].max())}")
        
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
        
        if selected_companies:
            filtered_df = filtered_df[filtered_df['Company'].isin(selected_companies)]
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
                label="📄 Guidance Events",
                value=f"{len(filtered_df):,}"
            )
        with col2:
            st.metric(
                label="🏢 Companies",
                value=filtered_df['Company'].nunique()
            )
        with col3:
            st.metric(
                label="📊 Unique Metrics",
                value=filtered_df['Metric'].nunique()
            )
        with col4:
            revision_count = filtered_df['Is Revision'].sum() if 'Is Revision' in filtered_df.columns else 0
            st.metric(
                label="📝 Revisions",
                value=int(revision_count)
            )
        with col5:
            latest_date = pd.to_datetime(filtered_df['Published'], errors='coerce', utc=True).max()
            st.metric(
                label="🕐 Latest Update",
                value=format_date(latest_date) if pd.notnull(latest_date) else "N/A"
            )

        # Charts Section - Guidance Type Breakdown & Activity Timeline
        st.markdown("")
        chart_col1, chart_col2 = st.columns([1, 2])
        
        with chart_col1:
            st.markdown("""
            <div class="section-header">
                <h3>📊 Guidance by Type</h3>
            </div>
            """, unsafe_allow_html=True)
            
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
                
                donut = alt.Chart(chart_data).mark_arc(innerRadius=50, outerRadius=80).encode(
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
                    height=250
                ).configure_view(
                    strokeWidth=0
                ).configure(
                    background='transparent'
                )
                
                st.altair_chart(donut, use_container_width=True)
        
        with chart_col2:
            st.markdown("""
            <div class="section-header">
                <h3>📈 Guidance Activity Timeline</h3>
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
                        color='#8b5cf6',
                        line={'color': '#a78bfa', 'strokeWidth': 2}
                    ).encode(
                        x=alt.X('Week:T', title='', axis=alt.Axis(labelColor='#a5b4fc', format='%b %Y')),
                        y=alt.Y('Count:Q', title='Events', axis=alt.Axis(labelColor='#a5b4fc')),
                        tooltip=['Week:T', 'Count']
                    ).properties(
                        height=220
                    ).configure_view(
                        strokeWidth=0
                    ).configure(
                        background='transparent'
                    ).configure_axis(
                        gridColor='rgba(165, 180, 252, 0.1)'
                    )
                    
                    st.altair_chart(area_chart, use_container_width=True)
                else:
                    st.info("No timeline data available")

        # Data Table Section
        st.markdown("""
        <div class="section-header">
            <h3>📋 Guidance Events Table</h3>
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
                'Company',
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
                "Company": st.column_config.TextColumn("Company", width="medium"),
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
                    display_text="🔗 View"
                ),
            }
            
            # Display the dataframe with revision highlighting
            def highlight_revisions(row):
                if row.get('Is Revision') == 'Yes':
                    return ['background-color: rgba(251, 191, 36, 0.15)'] * len(row)
                return [''] * len(row)
            
            styled_df = display_df.style.apply(highlight_revisions, axis=1)
            
            st.dataframe(
                styled_df,
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
                    label="📥 CSV",
                    data=csv,
                    file_name=f"guidance_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col_dl2:
                try:
                    excel_data = create_excel_download(filtered_df)
                    st.download_button(
                        label="📊 Excel",
                        data=excel_data,
                        file_name=f"guidance_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.button("📊 Excel", disabled=True, help="Install openpyxl for Excel export")
            with col_dl3:
                st.caption(f"Showing {len(display_df):,} of {len(df):,} records")
        
        # Company Sparklines Section
        if len(filtered_df) > 0 and filtered_df['Company'].nunique() <= 20:
            st.markdown("""
            <div class="section-header">
                <h3>📉 Guidance Trends by Company</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Get top companies by guidance count
            top_companies = filtered_df['Company'].value_counts().head(8).index.tolist()
            
            if top_companies and 'Published_dt' in filtered_df.columns:
                sparkline_df = filtered_df[filtered_df['Company'].isin(top_companies)].copy()
                sparkline_df['Published_dt'] = pd.to_datetime(sparkline_df['Published_dt'], errors='coerce', utc=True)
                sparkline_df = sparkline_df[sparkline_df['Published_dt'].notna()]
                
                if not sparkline_df.empty:
                    sparkline_df['Month'] = sparkline_df['Published_dt'].dt.to_period('M').dt.start_time
                    monthly_by_company = sparkline_df.groupby(['Month', 'Company']).size().reset_index(name='Count')
                    
                    sparklines = alt.Chart(monthly_by_company).mark_line(
                        strokeWidth=2,
                        point=alt.OverlayMarkDef(size=30)
                    ).encode(
                        x=alt.X('Month:T', title='', axis=alt.Axis(labelColor='#a5b4fc', format='%b')),
                        y=alt.Y('Count:Q', title='Events', axis=alt.Axis(labelColor='#a5b4fc')),
                        color=alt.Color(
                            'Company:N',
                            scale=alt.Scale(scheme='plasma'),
                            legend=alt.Legend(
                                orient='bottom',
                                columns=4,
                                labelColor='#a5b4fc'
                            )
                        ),
                        tooltip=['Month:T', 'Company', 'Count']
                    ).properties(
                        height=250
                    ).configure_view(
                        strokeWidth=0
                    ).configure(
                        background='transparent'
                    ).configure_axis(
                        gridColor='rgba(165, 180, 252, 0.1)'
                    )
                    
                    st.altair_chart(sparklines, use_container_width=True)

        # Footer
        st.markdown("""
        <div class="footer">
            <p>Data sourced from SEC filings and corporate announcements • Updated automatically</p>
        </div>
        """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ Database file not found")
    st.info("Please ensure 'finance_data.db' exists in the project directory.")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("Please ensure 'finance_data.db' exists and has the correct schema.")

