# train_health_risk_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def categorize_health_risk(aqi):
    """
    Categorize AQI into health risk levels
    """
    if aqi <= 50:
        return 'Safe'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy_Sensitive'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very_Unhealthy'
    else:
        return 'Hazardous'

def train_health_risk_model():
    """
    Train model to predict health risk from sensor readings
    """
    print("Training Health Risk Prediction Model...")
    
    # Load cleaned dataset
    if not os.path.exists('cleaned_final_dataset.csv'):
        print("Error: cleaned_final_dataset.csv not found!")
        return None
    
    df = pd.read_csv('cleaned_final_dataset.csv')
    print(f"Dataset loaded with {len(df)} samples")
    
    # Create health risk labels
    df['health_risk'] = df['aqi'].apply(categorize_health_risk)
    
    print("\nHealth Risk Distribution:")
    print(df['health_risk'].value_counts())
    
    # Features and target
    feature_columns = ['temperature', 'pressure', 'gas', 'pm25', 'aqi']
    X = df[feature_columns]
    y = df['health_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    print("\n" + "="*50)
    print("HEALTH RISK PREDICTION MODEL PERFORMANCE")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE FOR HEALTH RISK PREDICTION")
    print("="*50)
    print(feature_importance)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/health_risk_model.pkl')
    print("\nModel saved as 'models/health_risk_model.pkl'")
    
    return model, feature_importance

if __name__ == "__main__":
    model, importance = train_health_risk_model()