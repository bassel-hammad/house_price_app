# 🏠 House Price Prediction App

A complete machine learning application for predicting house prices using Linear Regression, built to apply and extend concepts from Andrew Ng's Machine Learning Course.

## 🚀 Features

- **Real-time price predictions** with confidence scoring
- **Interactive web interface** built with Streamlit
- **RESTful API** with automatic documentation
- **Price breakdown analysis** showing feature contributions
- **Model performance metrics** and sample data exploration
- **Intelligent confidence system** based on training data similarity

## 📱 Live Application Demo

### **🎨 Streamlit Frontend**
<img width="850" height="400" alt="image" src="https://github.com/user-attachments/assets/22a886e1-cd88-49e8-b1db-9fc1be3bb858" />

*Clean, responsive interface for house price predictions*

<img width="550" height="250" alt="image" src="https://github.com/user-attachments/assets/9a03d769-d6ff-4cb1-bd8a-5bea1f56da59" />

*Real-time predictions with confidence scoring and interactive visualizations*



### **🤖 Model Training**
<img width="450" height="145" alt="image" src="https://github.com/user-attachments/assets/2b4f7078-27fc-47cf-8a07-dfe9cfa011a2" />
<img width="680" height="260" alt="image" src="https://github.com/user-attachments/assets/3056bc4e-242f-4d89-be72-11325ed29e30" />


*Model training results showing excellent performance metrics (R² = 94.3%)*

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI  
- **ML Model**: Scikit-learn (Linear Regression)
- **Data**: Synthetic house price dataset
- **Visualization**: Plotly
- **API Documentation**: Swagger/OpenAPI

## 🎓 Key Learning Outcomes

This project demonstrates practical application of machine learning concepts and modern development practices:

### **Machine Learning Fundamentals Applied:**
- **Train/Test Split (80/20)** - Implemented proper data separation to prevent overfitting
- **Feature Scaling (StandardScaler)** - Applied Z-score normalization for better model convergence
- **Model Evaluation Metrics** - Used RMSE, MAE, and R² to assess performance comprehensively
- **Overfitting Detection** - Implemented comparison between training and test performance
- **Synthetic Data Generation** - Created realistic datasets with controlled relationships

### **Advanced ML Engineering Concepts Learned:**
- **Model Persistence** - Saved trained models using joblib for production deployment
- **Feature Engineering** - Designed meaningful features for house price prediction
- **Confidence Scoring** - Built intelligent system to assess prediction reliability
- **Data Validation** - Implemented input validation and error handling

### **Full-Stack Development Skills Acquired:**
- **API Development with FastAPI** - Built RESTful endpoints with automatic documentation
- **Pydantic Data Validation** - Learned type safety and input validation
- **Frontend Development with Streamlit** - Created interactive web interfaces
- **Performance Optimization** - Solved UI responsiveness issues using forms and caching
- **Production Considerations** - Implemented error handling, health checks, and user feedback

### **Software Engineering Best Practices:**
- **Project Structure** - Organized code into logical modules and folders
- **Environment Management** - Used virtual environments and requirements.txt
- **Version Control** - Proper .gitignore configuration for ML projects
- **Documentation** - Comprehensive README and inline code comments
- **Testing Strategy** - API testing through Swagger UI and manual validation

## 📊 Technical Deep Dive

### **Model Architecture:**
```python
# Linear Regression with 5 features:
price = w₁×size_sqft + w₂×bedrooms + w₃×bathrooms + w₄×age_years + w₅×location_factor + bias

# Feature scaling applied: scaled_value = (original_value - mean) / std_deviation
# Confidence scoring based on feature range similarity to training data
```

### **Performance Metrics:**
- **R² Score**: 94.3% (explains 94% of price variation)
- **RMSE**: ~$37k (typical prediction error)
- **Training vs Test Performance**: Minimal gap (no overfitting detected)
- **Prediction Range**: $50k - $1M (realistic house prices)

## 📦 Installation

1. **Clone the repository**
2. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

1. **Train the model:**
   ```bash
   python train_model.py
   ```
   - Generates synthetic house data (1000 samples)
   - Trains linear regression model
   - Saves model artifacts for production use

2. **Start the API backend:**
   ```bash
   python api.py
   ```
   - Loads trained model
   - Starts FastAPI server on port 8000
   - Provides automatic API documentation

3. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```
   - Starts Streamlit frontend on port 8501
   - Connects to FastAPI backend for predictions

4. **Access the application:**
   - **Web App**: http://localhost:8501
   - **API Documentation**: http://localhost:8000/docs
   - **API Health Check**: http://localhost:8000/health

## 🎮 API Endpoints

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/health` | GET | API health check | Status and model info |
| `/predict` | POST | Get price prediction | Price, confidence, breakdown |
| `/model/info` | GET | Model performance metrics | R², RMSE, training date |
| `/sample-data` | GET | Sample house data | 5 example houses |
| `/docs` | GET | Interactive API documentation | Swagger UI |

## 🔧 Key Implementation Insights

### **Confidence Scoring Logic:**
```python
# Innovative approach to prediction reliability
confidence_factors = [
    800 <= size_sqft <= 4000,      # Size within training range
    0 <= age_years <= 50,          # Age within training range  
    0.5 <= location_factor <= 2.0  # Location within training range
]
confidence_score = sum(confidence_factors) / len(confidence_factors)
# Result: "High" (>80%), "Medium" (>60%), or "Low" confidence
```

### **Streamlit Performance Optimization:**
- **Problem Solved**: Eliminated constant re-runs when adjusting sliders
- **Solution**: Used `st.form()` to batch user inputs
- **Result**: Smooth, responsive user interface

### **Production-Ready Features:**
- **Error Handling**: Graceful API failures and user-friendly messages
- **Input Validation**: Pydantic models prevent invalid data
- **Caching**: Optimized API calls for better performance
- **Health Monitoring**: API status checks and model loading verification

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │───▶│    FastAPI      │───▶│   ML Model      │
│   Frontend      │    │    Backend      │    │   (Trained)     │
│                 │    │                 │    │                 │
│ • Input forms   │    │ • /predict      │    │ • Linear Reg    │
│ • Visualizations│    │ • /health       │    │ • Feature proc  │
│ • Results       │    │ • Swagger docs  │    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        ▲                       ▲
        │                       │
        └───────────────────────┼──────── Postman Testing
                                │
                        ┌───────▼─────────┐
                        │   Swagger UI    │
                        │ (Auto-generated)│
                        └─────────────────┘
```

## 🎯 Project Highlights

### **Problem-Solving Examples:**
1. **UI Responsiveness**: Identified and solved Streamlit re-run issues using forms
2. **Data Validation**: Implemented two-layer validation (Pydantic + Confidence)
3. **Model Deployment**: Successfully bridged ML training and web deployment
4. **User Experience**: Created intuitive interface for non-technical users

### **Skills Demonstrated:**
- **Applied ML Theory**: Successfully translated Andrew Ng's concepts to working code
- **Full-Stack Thinking**: Connected data science to user-facing applications
- **Production Mindset**: Built with error handling, documentation, and scalability
- **Continuous Learning**: Adapted and solved new challenges throughout development

## 👨‍💻 Author

**Built as a learning project to bridge the gap between academic ML knowledge and production applications.**

*This project demonstrates the successful application of Andrew Ng's Machine Learning Course concepts in a real-world, deployable application, showcasing both theoretical understanding and practical implementation skills.*

---

### 📈 **Learning Journey Summary:**
From understanding basic linear regression concepts to building a complete, production-ready ML application with modern web technologies - this project represents a comprehensive learning experience in machine learning engineering.

