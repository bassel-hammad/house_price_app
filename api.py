from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
import os

'''
FastAPI
A modern Python web framework for building APIs
Purpose: Create the backend that serves your ML model
Why: Fast, automatic API documentation, type hints support
'''

'''
Pydantic
Data validation library that works seamlessly with FastAPI
Purpose: Define and validate input/output data structures
Example: Ensure house features (bedrooms, sqft, etc.) are correct types
'''

# Load trained model artifacts at startup
try:
    model = joblib.load('model/house_price_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    features = joblib.load('model/features.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    features = None

# Create FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using machine learning",
    version="1.0.0"
)

# Pydantic models for input validation
class HouseFeatures(BaseModel):
    size_sqft: float = Field(..., gt=0, description="House size in square feet")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, le=10, description="Number of bathrooms")
    age_years: float = Field(..., ge=0, le=100, description="Age of house in years")
    location_factor: float = Field(..., gt=0, le=3, description="Location quality factor")

    class Config:
        json_schema_extra = {  # Changed from 'schema_extra' to 'json_schema_extra'
            "example": {
                "size_sqft": 2000,
                "bedrooms": 3,
                "bathrooms": 2,
                "age_years": 10,
                "location_factor": 1.2
            }
        }

class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str
    confidence_level: str
    features_used: List[str]

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Check if the API is running and model is loaded
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "message": "House Price Prediction API is running",
        "model_loaded": True
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_price(house: HouseFeatures):
    """
    Predict house price based on features
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([{
            'size_sqft': house.size_sqft,
            'bedrooms': house.bedrooms,
            'bathrooms': house.bathrooms,
            'age_years': house.age_years,
            'location_factor': house.location_factor
        }])
        
        # Scale features (same as training)
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Ensure prediction is positive
        prediction = max(prediction, 50000)  # Minimum $50k
        
        # Determine confidence level based on input ranges
        confidence = determine_confidence(house)
        
        return PredictionResponse(
            predicted_price=round(prediction, 2),
            formatted_price=f"${prediction:,.0f}",
            confidence_level=confidence,
            features_used=features
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

def determine_confidence(house: HouseFeatures) -> str:
    """
    Confidence = How trustworthy is our prediction?
    The confidence level tells users how reliable the model's prediction is,
    based on whether the input features are similar to the training data.
    """
    # Check if features are within training ranges
    confidence_factors = []
    
    # Size check (training range: ~500-5000)
    if 800 <= house.size_sqft <= 4000:
        confidence_factors.append(True)
    else:
        confidence_factors.append(False)
    
    # Age check (training range: 0-50)
    if 0 <= house.age_years <= 50:
        confidence_factors.append(True)
    else:
        confidence_factors.append(False)
    
    # Location factor check (training range: 0.5-2.0)
    if 0.5 <= house.location_factor <= 2.0:
        confidence_factors.append(True)
    else:
        confidence_factors.append(False)
    
    # Determine overall confidence
    confidence_score = sum(confidence_factors) / len(confidence_factors)
    
    if confidence_score >= 0.8:
        return "High"
    elif confidence_score >= 0.6:
        return "Medium"
    else:
        return "Low"

# Get model info endpoint
@app.get("/model/info")
async def get_model_info():
    """
    Get information about the trained model
    """
    try:
        with open('model/metadata.json', 'r') as f:
            import json
            metadata = json.load(f)
        
        return {
            "model_type": metadata.get('model_type'),
            "features": metadata.get('features'),
            "n_features": metadata.get('n_features'),
            "test_rmse": metadata.get('test_rmse'),
            "test_r2": metadata.get('test_r2'),
            "training_date": metadata.get('training_date')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load model info: {str(e)}")

# Get sample data endpoint
@app.get("/sample-data")
async def get_sample_data():
    """
    Get sample house data for testing
    """
    try:
        sample_data = pd.read_csv('data/house_data.csv')
        return {
            "sample_count": len(sample_data),
            "samples": sample_data.head(5).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load sample data: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """
    API welcome message
    """
    return {
        "message": "Welcome to House Price Prediction API!",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "model_info": "/model/info",
            "sample_data": "/sample-data"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)