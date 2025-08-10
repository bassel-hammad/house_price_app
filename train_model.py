import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic house price data for training
    Based on realistic relationships between features and price
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate features
    size_sqft = np.random.normal(2000, 800, n_samples)  # Mean: 2000 sqft, std: 800
    size_sqft = np.clip(size_sqft, 500, 5000)  # Realistic range
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    bathrooms = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.5, 0.25, 0.05])
    age_years = np.random.uniform(0, 50, n_samples)  # 0-50 years old
    
    # Location factor (simulates neighborhood quality)
    location_factor = np.random.normal(1.0, 0.3, n_samples)
    location_factor = np.clip(location_factor, 0.5, 2.0)
    
    # Create realistic price based on features (like Andrew Ng's examples)
    base_price = (
        size_sqft * 150 +           # $150 per sqft
        bedrooms * 10000 +          # $10k per bedroom
        bathrooms * 8000 +          # $8k per bathroom
        (50 - age_years) * 1000     # Newer houses worth more
    )
    
    # Apply location factor and add noise
    price = base_price * location_factor
    noise = np.random.normal(0, 20000, n_samples)  # Add realistic noise
    price = price + noise
    price = np.clip(price, 50000, 1000000)  # Realistic price range
    
    # Create DataFrame
    data = pd.DataFrame({
        'size_sqft': size_sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age_years,
        'location_factor': location_factor,
        'price': price
    })
    
    return data

def preprocess_data(data):
    """
    Preprocess the data for training
    """
    # Features and target
    features = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'location_factor']
    X = data[features]
    y = data['price']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {features}")
    print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    return X, y, features

def train_model(X, y):
    """
    Train linear regression model (like in Andrew Ng's course)
    """
    # Split data (80/20 split) - Separate data for training vs testing to evaluate real performance
    # Prevent overfitting by testing on data model has never seen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling (normalization - important for linear regression)
    # scaled_value = (original_value - mean) / standard_deviation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    return model, scaler, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def evaluate_model(y_train, y_test, y_train_pred, y_test_pred):
    """
    Evaluate model performance (like Andrew Ng teaches)
    """
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred) # Raw error (squared)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_rmse = np.sqrt(train_mse) # Error in same units as price
    test_rmse = np.sqrt(test_mse)

    train_mae = mean_absolute_error(y_train, y_train_pred) # Average absolute error - Formula: Average of |actual - predicted|
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred) # How much variance explained
    test_r2 = r2_score(y_test, y_test_pred)
    # Formula: 1 - (Sum of squared residuals / Total sum of squares)
    # R² = 1 - (SSR / TSS)
    # SSR = Σ(actual - predicted)²
    # TSS = Σ(actual - mean)²
    # Range: 0 to 1 (higher is better)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Training RMSE: ${train_rmse:,.0f}")
    print(f"Testing RMSE:  ${test_rmse:,.0f}")
    print(f"Training MAE:  ${train_mae:,.0f}")
    print(f"Testing MAE:   ${test_mae:,.0f}")
    print(f"Training R²:   {train_r2:.3f}")
    print(f"Testing R²:    {test_r2:.3f}")
    
    # Check for overfitting
    if abs(train_rmse - test_rmse) > 0.1 * train_rmse:
        print("\n⚠️  Warning: Possible overfitting detected!")
    else:
        print("\n✅ Good! No significant overfitting detected.")
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

def save_model_and_data(model, scaler, features, data, metrics):
    """
    Save trained model and related artifacts
    """
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, 'model/house_price_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    
    # Save feature names
    joblib.dump(features, 'model/features.pkl')
    
    # Save sample data for API testing
    sample_data = data.head(10)
    sample_data.to_csv('data/house_data.csv', index=False)
    
    # Save model metadata
    metadata = {
        'model_type': 'Linear Regression',
        'features': features,
        'n_features': len(features),
        'test_rmse': metrics['test_rmse'],
        'test_r2': metrics['test_r2'],
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    import json
    with open('model/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved successfully!")
    print(f"📁 Model files saved in: model/")
    print(f"📊 Sample data saved in: data/house_data.csv")

def main():
    """
    Main training pipeline
    """
    print("🏠 HOUSE PRICE PREDICTION MODEL TRAINING")
    print("=" * 50)
    
    # 1. Generate synthetic data
    print("📊 Generating synthetic house price data...")
    data = generate_synthetic_data(n_samples=1000)
    
    # 2. Preprocess data
    print("\n🔧 Preprocessing data...")
    X, y, features = preprocess_data(data)
    
    # 3. Train model
    print("\n🤖 Training linear regression model...")
    model, scaler, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_model(X, y)
    
    # 4. Evaluate model
    print("\n📈 Evaluating model performance...")
    metrics = evaluate_model(y_train, y_test, y_train_pred, y_test_pred)
    
    # 5. Save everything
    print("\n💾 Saving model and artifacts...")
    save_model_and_data(model, scaler, features, data, metrics)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"🚀 Ready to use in FastAPI and Streamlit!")

if __name__ == "__main__":
    main()