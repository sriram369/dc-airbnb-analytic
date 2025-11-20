import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="The Capital Investor", layout="wide")

# --- CUSTOM CSS FOR DARK MODE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { color: #b2b5be !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('clean_airbnb_dc.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ File 'clean_airbnb_dc.csv' not found.")
    st.stop()

# --- 3. TRAIN MODEL (Behind the Scenes) ---
@st.cache_resource
def train_model(df):
    features = ['bedrooms', 'accommodates', 'dist_to_mall', 'number_of_reviews']
    X = df[features]
    y = df['price']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# --- 4. APP LAYOUT ---
st.title("🏠 The Capital Investor: AI & Analytics Suite")
st.markdown("### Washington D.C. Short-Term Rental Intelligence")

# Create Two Tabs: One for Viz (Power BI style), One for AI (Prediction)
tab1, tab2 = st.tabs(["📊 Market Insights (Dashboard)", "🤖 ROI Calculator (AI)"])

# ==========================================
# TAB 1: MARKET INSIGHTS (Recreating Power BI)
# ==========================================
with tab1:
    st.header("Market Overview")
    
    # Top Row KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Active Listings", f"{len(df):,}")
    col2.metric("Avg. Nightly Price", f"${df['price'].mean():.2f}")
    col3.metric("Avg. Distance to Mall", f"{df['dist_to_mall'].mean():.1f} miles")
    
    st.divider()

    # Row 2: Map and Bar Chart
    c1, c2 = st.columns([3, 2]) # 60% Map, 40% Charts
    
    with c1:
        st.subheader("📍 Geospatial Price Heatmap")
        # Interactive Map (Uses Lat/Lon like Power BI)
        fig_map = px.scatter_mapbox(
            df, 
            lat="latitude", 
            lon="longitude", 
            color="price", 
            size="price",
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15, 
            zoom=10,
            mapbox_style="carto-positron",
            hover_name="neighbourhood_cleansed",
            title="Listings by Price (Size & Color)"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with c2:
        st.subheader("💰 Price by Neighborhood")
        # Aggregating data for the Bar Chart
        nbhd_stats = df.groupby('neighbourhood_cleansed')['price'].mean().reset_index()
        nbhd_stats = nbhd_stats.sort_values(by='price', ascending=True).tail(15) # Top 15
        
        fig_bar = px.bar(
            nbhd_stats, 
            x='price', 
            y='neighbourhood_cleansed', 
            orientation='h',
            title="Top 15 Most Expensive Areas",
            color='price',
            color_continuous_scale='Bluered'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 2: ROI CALCULATOR (The AI Tool)
# ==========================================
with tab2:
    st.header("Predictive Valuation Engine")
    
    # Layout: Inputs on Left, Results on Right
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        st.subheader("🏗️ Property Specs")
        beds = st.slider("Bedrooms", 1, 6, 2)
        guests = st.slider("Guest Capacity", 1, 12, 4)
        dist = st.slider("Distance to Mall (Miles)", 0.1, 10.0, 1.5)
        reviews = st.number_input("Est. Review Count", value=50)
        
        # Real-time prediction logic
        input_data = pd.DataFrame({
            'bedrooms': [beds],
            'accommodates': [guests],
            'dist_to_mall': [dist],
            'number_of_reviews': [reviews]
        })
        pred_price = model.predict(input_data)[0]
        occupancy = 0.65
        revenue = pred_price * 30 * occupancy
        
    with col_result:
        st.subheader("💵 Financial Projections")
        
        # The Big Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Nightly Rate", f"${pred_price:.2f}")
        m2.metric("Est. Monthly Revenue", f"${revenue:,.2f}")
        m3.metric("Implied Asset Value", f"${revenue * 12 * 15:,.0f}")
        
        # The AI Advisor Logic
        st.info("💡 **AI Strategy Note:**")
        if dist < 1.0:
            st.write("This property is in a **High-Traffic Tourist Zone**. Prioritize luxury amenities (e.g., high-end coffee machine, premium linens) to justify the premium rate.")
        elif dist > 4.0:
            st.write("This property is in a **Residential/Commuter Zone**. Compete on price and offer 'Work-from-Home' setups (fast WiFi, monitors) to attract long-term stays.")
        else:
            st.write("This is a **Balanced Location**. Ideal for families. Stock the property with family-friendly gear (cribs, games) to maximize occupancy.")

# --- 5. FOOTER ---
st.divider()
st.caption("Analytics powered by Python, Plotly, and Scikit-Learn. Data: InsideAirbnb.")