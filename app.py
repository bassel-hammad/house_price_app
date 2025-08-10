import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
        padding: 1rem;
        background-color: #e8f5e8;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-high { color: #27ae60; }
    .confidence-medium { color: #f39c12; }
    .confidence-low { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_prediction(house_features):
    """Get prediction from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=house_features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_sample_data():
    """Get sample data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/sample-data", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Check API once
    api_status = check_api_health()
    if not api_status:
        st.error("❌ **API is not running!**")
        st.stop()
    
    st.success("✅ **API is running successfully!**")
    
    # Use form to prevent constant re-runs
    with st.form("prediction_form"):
        st.subheader("🏡 House Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            size_sqft = st.slider("House Size (sqft)", 500, 5000, 2000, 50)
            bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5,6,7,8,9,10], index=2)
            bathrooms = st.selectbox("Bathrooms", [1,2,3,4,5,6,7,8,9,10], index=1)
        
        with col2:
            age_years = st.slider("House Age (years)", 0, 100, 10)
            location_factor = st.slider("Location Quality", 0.5, 3.0, 1.2, 0.1)
        
        # This button only triggers when clicked
        submitted = st.form_submit_button("🔮 Predict Price", type="primary")
        
        if submitted:
            # Prepare data for API
            house_features = {
                "size_sqft": float(size_sqft),
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "age_years": float(age_years),
                "location_factor": float(location_factor)
            }
            
            # Get prediction
            with st.spinner("🤖 Getting prediction..."):
                prediction, error = get_prediction(house_features)
            
            if error:
                st.error(f"❌ **Prediction failed:** {error}")
            else:
                # Display prediction results
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Main prediction
                    st.markdown(
                        f'<div class="prediction-result">💰 {prediction["formatted_price"]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Confidence level
                    confidence = prediction["confidence_level"]
                    confidence_class = f"confidence-{confidence.lower()}"
                    
                    st.markdown(
                        f'<p style="text-align: center; font-size: 1.2rem;">Confidence: <span class="{confidence_class}"><strong>{confidence}</strong></span></p>',
                        unsafe_allow_html=True
                    )
                
                # Detailed results
                st.subheader("📊 Prediction Details")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Predicted Price",
                        f"${prediction['predicted_price']:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "Price per sqft",
                        f"${prediction['predicted_price']/size_sqft:.0f}"
                    )
                
                with col3:
                    st.metric(
                        "Confidence Level",
                        confidence
                    )
                
                with col4:
                    features_count = len(prediction['features_used'])
                    st.metric(
                        "Features Used",
                        f"{features_count}/5"
                    )
                
                # Feature breakdown visualization
                st.subheader("🏗️ Price Breakdown Analysis")
                
                # Calculate feature contributions (approximate)
                base_contributions = {
                    "Size (sqft)": size_sqft * 150,
                    "Bedrooms": bedrooms * 10000,
                    "Bathrooms": bathrooms * 8000,
                    "Age Factor": (50 - age_years) * 1000,
                    "Location Premium": prediction['predicted_price'] * (location_factor - 1)
                }
                
                # Create breakdown chart
                fig = px.bar(
                    x=list(base_contributions.keys()),
                    y=list(base_contributions.values()),
                    title="Estimated Price Contribution by Feature",
                    labels={'x': 'Features', 'y': 'Price Contribution ($)'},
                    color=list(base_contributions.values()),
                    color_continuous_scale='viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Model Information Section
    st.sidebar.markdown("---")
    if st.sidebar.button("📈 Show Model Info"):
        model_info = get_model_info()
        if model_info:
            st.subheader("🤖 Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Details:**")
                st.write(f"• **Type:** {model_info.get('model_type', 'N/A')}")
                st.write(f"• **Features:** {model_info.get('n_features', 'N/A')}")
                st.write(f"• **Training Date:** {model_info.get('training_date', 'N/A')}")
            
            with col2:
                st.markdown("**Performance Metrics:**")
                st.write(f"• **RMSE:** ${model_info.get('test_rmse', 0):,.0f}")
                st.write(f"• **R² Score:** {model_info.get('test_r2', 0):.3f}")
                st.write(f"• **Accuracy:** {model_info.get('test_r2', 0)*100:.1f}%")
    
    # Sample Data Section
    if st.sidebar.button("📋 Show Sample Data"):
        sample_data = get_sample_data()
        if sample_data:
            st.subheader("📋 Sample House Data")
            df = pd.DataFrame(sample_data['samples'])
            
            # Format the dataframe for better display
            df['price'] = df['price'].apply(lambda x: f"${x:,.0f}")
            df['size_sqft'] = df['size_sqft'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(df, use_container_width=True)
            
            # Quick stats
            st.markdown("**Quick Statistics:**")
            col1, col2, col3 = st.columns(3)
            
            raw_df = pd.DataFrame(sample_data['samples'])
            with col1:
                st.metric("Avg Price", f"${raw_df['price'].mean():,.0f}")
            with col2:
                st.metric("Avg Size", f"{raw_df['size_sqft'].mean():,.0f} sqft")
            with col3:
                st.metric("Avg Age", f"{raw_df['age_years'].mean():.1f} years")
    
    # About section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **🏠 House Price Predictor** uses machine learning to estimate house prices based on key features.
        
        **🎯 How it works:**
        1. **Linear Regression Model** trained on synthetic house data
        2. **5 Key Features:** Size, Bedrooms, Bathrooms, Age, Location
        3. **Real-time Predictions** via FastAPI backend
        4. **Confidence Scoring** based on training data similarity
        
        **📊 Model Performance:**
        - R² Score: ~94% (explains 94% of price variation)
        - RMSE: ~$37k (typical prediction error)
        - Features: Size, bedrooms, bathrooms, age, location quality
        
        **🔧 Tech Stack:**
        - **Frontend:** Streamlit
        - **Backend:** FastAPI
        - **ML:** Scikit-learn (Linear Regression)
        - **Data:** Synthetic house price dataset
        """)

if __name__ == "__main__":
    main()