"""
Train product return prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def train_model():
    print("🛍️  Training Product Return Prediction Model")
    print("=" * 70)
    
    # Load training data
    print("\n📥 Loading training data...")
    df = pd.read_csv('../data/train.csv')
    
    print(f"   Total orders: {len(df)}")
    print(f"   Returns: {df['returned'].sum()} ({df['returned'].mean():.1%})")
    print(f"   Kept: {(~df['returned'].astype(bool)).sum()} ({(~df['returned'].astype(bool)).mean():.1%})")
    
    # Encode categorical variables
    print("\n🔧 Encoding categories...")
    le_category = LabelEncoder()
    le_size = LabelEncoder()
    
    df['category_encoded'] = le_category.fit_transform(df['product_category'])
    df['size_encoded'] = le_size.fit_transform(df['size_ordered'])
    
    # Save encoders
    joblib.dump(le_category, 'category_encoder.pkl')
    joblib.dump(le_size, 'size_encoder.pkl')
    
    # Select features
    feature_columns = [
        'product_price',
        'customer_purchase_history', 
        'review_rating',
        'delivery_time_days',
        'discount_percentage',
        'previous_returns',
        'days_since_purchase',
        'category_encoded',
        'size_encoded'
    ]
    
    X = df[feature_columns]
    y = df['returned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} orders")
    print(f"   Test: {len(X_test)} orders")
    
    # Train model
    print("\n🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📊 Model Performance:")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Kept', 'Returned']))
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\nConfusion Matrix:")
    print(f"   Correctly predicted KEPT: {tn}")
    print(f"   Predicted RETURN but actually KEPT: {fp}")
    print(f"   Predicted KEEP but actually RETURNED: {fn}")
    print(f"   Correctly predicted RETURNED: {tp}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Feature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']:.<30} {row['importance']:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    joblib.dump(model, 'return_prediction_model.pkl')
    
    # Save baseline stats
    baseline_stats = {
        'feature_means': X_train.mean().to_dict(),
        'feature_stds': X_train.std().to_dict(),
        'feature_mins': X_train.min().to_dict(),
        'feature_maxs': X_train.max().to_dict(),
        'return_rate': float(y_train.mean()),
        'n_samples': len(X_train),
        'feature_names': feature_columns
    }
    
    with open('baseline_stats.json', 'w') as f:
        json.dump(baseline_stats, f, indent=2)
    
    print("✅ Model and stats saved!")
    print("=" * 70)
    
    print(f"\n💡 Key Insight: {feature_importance.iloc[0]['feature']} is most important!")
    
    return model

if __name__ == "__main__":
    train_model()