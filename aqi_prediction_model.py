# train_aqi_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_aqi_model(data_file='cleaned_final_dataset.csv'):
    """
    Train Random Forest Regressor for AQI prediction
    """
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found!")
        print("Please run clean_dataset.py first")
        return None, None
    
    # Load cleaned dataset
    try:
        df = pd.read_csv(data_file)
        print(f"Dataset loaded with {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
    
    # Check if we have the required columns
    required_columns = ['temperature', 'pressure', 'gas', 'pm25', 'aqi']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return None, None
    
    # Remove any rows with missing values
    df = df.dropna(subset=required_columns)
    print(f"After removing NaNs: {len(df)} samples")
    
    # Features and target
    feature_columns = ['temperature', 'pressure', 'gas', 'pm25']
    X = df[feature_columns]
    y = df['aqi']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100,  # Reduced for faster training
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    print("\n" + "="*50)
    print("AQI PREDICTION MODEL PERFORMANCE")
    print("="*50)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nR² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE FOR AQI PREDICTION")
    print("="*50)
    print(feature_importance)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/aqi_model.pkl')
    print("\nModel saved as 'models/aqi_model.pkl'")
    
    return model, feature_importance

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    model, importance = train_aqi_model()