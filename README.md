# 🏠 The Capital Investor: Airbnb Analytics Suite

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0-FF4B4B)
![Power BI](https://img.shields.io/badge/Power_BI-Desktop-F2C811)
![Status](https://img.shields.io/badge/Status-Deployment_Ready-success)

**A Business Intelligence & AI-Powered investment tool for Washington D.C. Real Estate Investors.**

---

## 📊 Executive Summary
Real estate investors in Washington D.C. currently lack predictive tools to assess the profitability of short-term rentals. This project solves the "Blind Investment" problem by providing a **Two-Tiered Intelligence System**:
1.  **Strategic Dashboard (Power BI):** For high-level market overview, supply/demand heatmaps, and root-cause price analysis.
2.  **Valuation Engine (Streamlit + AI):** A Machine Learning application that predicts nightly revenue and property value based on amenities and geospatial proximity to tourist hubs.

## 🛠️ Technical Architecture
This project integrates **Data Engineering**, **Business Intelligence**, and **Machine Learning**:

| Component | Technology | Key Function |
| :--- | :--- | :--- |
| **Data Processing** | Python (Pandas, NumPy) | Cleaning 6,000+ listings, handling nulls, currency conversion. |
| **Feature Engineering** | Python (Geopy) | Calculating geodesic distance from every property to the National Mall. |
| **Visualization** | Microsoft Power BI | Geospatial mapping, Decomposition Trees (Root Cause Analysis). |
| **Predictive AI** | Scikit-Learn (Random Forest) | Predicting nightly rates based on bedrooms, location, and reviews. |
| **App Deployment** | Streamlit | Interactive web-based ROI calculator for end-users. |

---

## 📂 Repository Structure
```text
├── app.py                   # The main Streamlit application (AI Valuation Engine)
├── clean_airbnb_dc.csv      # The processed dataset (Ready for ML & BI)
├── requirements.txt         # Dependencies for cloud deployment
├── GroupProject_Dashboard.pbix # The Power BI Executive Dashboard file
└── README.md                # Project Documentation
