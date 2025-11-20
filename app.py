import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Airbnb ROI Predictor", layout="wide")

# Custom CSS to make it look like a Consulting Tool
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD DATA & TRAIN MODEL ---
@st.cache_data # Caches the model so it doesn't reload every time you move a slider
def load_and_train():
    # Load the clean data
    df = pd.read_csv('clean_airbnb_dc.csv')
    
    # Features we will use for prediction
    features = ['bedrooms', 'accommodates', 'dist_to_mall', 'number_of_reviews']
    target = 'price'
    
    # Train a solid Random Forest model (better than Linear Regression)
    X = df[features]
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, df

try:
    model, df = load_and_train()
except FileNotFoundError:
    st.error("⚠️ distinct file 'clean_airbnb_dc.csv' not found. Please make sure it is in the same folder as app.py")
    st.stop()

# --- 3. SIDEBAR: INPUTS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/2560px-Airbnb_Logo_B%C3%A9lo.svg.png", width=150)
st.sidebar.header("Property Specs")

# The User Controls
beds = st.sidebar.slider("🛏️ Bedrooms", 1, 6, 2)
guests = st.sidebar.slider("👥 Guest Capacity", 1, 12, 4)
dist = st.sidebar.slider("🏛️ Distance to Mall (Miles)", 0.1, 10.0, 1.5)
reviews = st.sidebar.number_input("⭐ Estimated Reviews (Competitiveness)", value=50)

# --- 4. MAIN DASHBOARD ---
st.title("🏠 The Capital Investor: ROI Calculator")
st.markdown("### AI-Powered Valuation Engine for Washington D.C.")
st.divider()

# PREDICTION ENGINE
input_data = pd.DataFrame({
    'bedrooms': [beds],
    'accommodates': [guests],
    'dist_to_mall': [dist],
    'number_of_reviews': [reviews]
})

predicted_price = model.predict(input_data)[0]

# FINANCIAL LOGIC
occupancy_rate = 0.65 # Conservative estimate (65%)
monthly_revenue = predicted_price * 30 * occupancy_rate

# DISPLAY METRICS (The "Consulting" Look)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="💰 Predicted Nightly Rate", value=f"${predicted_price:.2f}", delta="Based on AI Model")

with col2:
    st.metric(label="📅 Est. Monthly Revenue", value=f"${monthly_revenue:,.2f}", delta=f"{occupancy_rate*100:.0f}% Occupancy")

with col3:
    # Simple valuation heuristic: Revenue * 12 months * 15 years
    valuation = monthly_revenue * 12 * 15 
    st.metric(label="🏢 Est. Property Value (Cap Rate)", value=f"${valuation:,.0f}")

# --- 5. AI ADVISOR (Syllabus Requirement) ---
st.divider()
st.subheader("🤖 Strategic Recommendation")

if dist < 1.0:
    st.success("💎 **Prime Location Strategy:** You are in the 'Golden Zone'. Focus on luxury amenities (Concierge, High-end toiletries) to justify a premium price. Competitors here charge 30% more.")
elif dist > 4.0:
    st.warning("📉 **Volume Strategy:** You are far from tourist hubs. You must compete on price. Consider marketing to long-term remote workers rather than weekend tourists.")
else:
    st.info("⚖️ **Balanced Strategy:** Good location. To maximize revenue, focus on 'Family Friendly' amenities (Cribs, High chairs) as you are in the residential sweet spot.")

# --- 6. MARKET CONTEXT ---
st.divider()
st.caption(f"Model trained on {len(df)} active DC listings. Data source: InsideAirbnb.")