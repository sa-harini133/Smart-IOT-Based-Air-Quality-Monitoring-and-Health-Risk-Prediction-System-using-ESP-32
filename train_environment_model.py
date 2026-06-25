# train_environment_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_environment_model():
    """
    Train Random Forest Classifier for environment detection
    """
    print("Training Environment Classification Model...")
    
    # Load cleaned dataset
    if not os.path.exists('cleaned_final_dataset.csv'):
        print("Error: cleaned_final_dataset.csv not found!")
        return None
    
    df = pd.read_csv('cleaned_final_dataset.csv')
    print(f"Dataset loaded with {len(df)} samples")
    
    # Check environment distribution
    print("\nEnvironment distribution:")
    print(df['environment'].value_counts())
    
    # Features and target
    feature_columns = ['temperature', 'pressure', 'gas', 'pm25', 'aqi']
    X = df[feature_columns]
    y = df['environment']
    
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
    print("ENVIRONMENT CLASSIFICATION MODEL PERFORMANCE")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\nCross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE FOR ENVIRONMENT CLASSIFICATION")
    print("="*50)
    print(feature_importance)
    
    # Visualize feature importance
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance for Environment Classification')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('environment_feature_importance.png')
        print("\nFeature importance plot saved as 'environment_feature_importance.png'")
    except:
        print("Could not save plot (matplotlib may not be configured correctly)")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/environment_model.pkl')
    print("\nModel saved as 'models/environment_model.pkl'")
    
    return model, feature_importance

if __name__ == "__main__":
    model, importance = train_environment_model()