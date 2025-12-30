import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

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
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #555;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1a1a2e;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #eef1f6 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        font-weight: 600;
        color: #333;
    }
    
    /* Table section header */
    .section-header {
        background: #f0f2f6;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #667eea;
    }
    .section-header h3 {
        margin: 0;
        color: #333;
    }
    
    /* Revision badge styling */
    .revision-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .revision-yes { background: #ffeeba; color: #856404; }
    .revision-no { background: #d4edda; color: #155724; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.85rem;
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
        g.sentiment_label AS "Sentiment",
        g.sentiment_score AS "Sentiment Score",
        g.published_at AS "Published",
        g.source_url AS "Source"
    FROM guidance g
    ORDER BY g.published_at DESC
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
        dt = pd.to_datetime(date_str)
        return dt.strftime('%b %d, %Y')
    except:
        return str(date_str)[:10]

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
            latest_date = pd.to_datetime(filtered_df['Published']).max()
            st.metric(
                label="🕐 Latest Update",
                value=format_date(latest_date) if pd.notnull(latest_date) else "N/A"
            )

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
                'Company',
                'Type',
                'Metric',
                'Period',
                'Guided Range',
                'Direction',
                'Is Revision',
                'Sentiment',
                'Published',
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
            
            # Display the dataframe
            st.dataframe(
                display_df,
                column_config=column_config,
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Download button
            st.markdown("")
            col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 3])
            with col_dl1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"guidance_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col_dl2:
                st.caption(f"Showing {len(display_df):,} of {len(df):,} records")

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

