"""
Generate synthetic e-commerce product return dataset with drift scenarios
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Product categories
CATEGORIES = ['Clothing', 'Electronics', 'Home_Decor', 'Sports', 'Beauty']
SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'NA']

def generate_baseline_data(n_samples=5000):
    """Generate baseline training data"""
    X, y = make_classification(
        n_samples=n_samples, n_features=7, n_informative=6,
        n_redundant=1, n_classes=2, weights=[0.85, 0.15],
        flip_y=0.05, random_state=42
    )
    
    df = pd.DataFrame({
        'product_price': np.clip(X[:, 0] * 50 + 100, 10, 500).astype(int),
        'customer_purchase_history': np.clip(X[:, 1] * 20 + 10, 0, 100).astype(int),
        'review_rating': np.clip(X[:, 2] * 1.5 + 3.5, 1, 5).round(1),
        'delivery_time_days': np.clip(X[:, 3] * 2 + 4, 1, 10).astype(int),
        'discount_percentage': np.clip(X[:, 4] * 20 + 20, 0, 80).astype(int),
        'previous_returns': np.clip(X[:, 5] * 3, 0, 10).astype(int),
        'days_since_purchase': np.clip(X[:, 6] * 10 + 5, 1, 30).astype(int),
        'returned': y
    })
    
    df['product_category'] = np.random.choice(CATEGORIES, n_samples)
    df['size_ordered'] = np.random.choice(SIZES, n_samples, 
                                         p=[0.05, 0.15, 0.35, 0.25, 0.15, 0.03, 0.02])
    return df

def generate_flash_sale_drift(n_samples=1000):
    """Flash sale - deep discounts, impulse buyers, high returns"""
    X, y = make_classification(
        n_samples=n_samples, n_features=7, n_informative=6,
        n_redundant=1, n_classes=2, weights=[0.65, 0.35],
        flip_y=0.05, random_state=123
    )
    
    df = pd.DataFrame({
        'product_price': np.clip(X[:, 0] * 40 + 80, 10, 400).astype(int),
        'customer_purchase_history': np.clip(X[:, 1] * 10 + 3, 0, 50).astype(int),
        'review_rating': np.clip(X[:, 2] * 1.5 + 3.5, 1, 5).round(1),
        'delivery_time_days': np.clip(X[:, 3] * 2.5 + 5, 1, 12).astype(int),
        'discount_percentage': np.clip(X[:, 4] * 30 + 50, 40, 80).astype(int),
        'previous_returns': np.clip(X[:, 5] * 4, 0, 15).astype(int),
        'days_since_purchase': np.clip(X[:, 6] * 10 + 5, 1, 30).astype(int),
        'returned': y
    })
    
    df['product_category'] = np.random.choice(CATEGORIES, n_samples, 
                                              p=[0.6, 0.15, 0.1, 0.1, 0.05])
    df['size_ordered'] = np.random.choice(SIZES, n_samples,
                                         p=[0.08, 0.18, 0.30, 0.25, 0.15, 0.03, 0.01])
    return df

def generate_supply_chain_crisis(n_samples=1000):
    """Supply chain issues - wrong items, delays, very high returns"""
    X, y = make_classification(
        n_samples=n_samples, n_features=7, n_informative=6,
        n_redundant=1, n_classes=2, weights=[0.55, 0.45],
        flip_y=0.08, random_state=456
    )
    
    df = pd.DataFrame({
        'product_price': np.clip(X[:, 0] * 50 + 100, 10, 500).astype(int),
        'customer_purchase_history': np.clip(X[:, 1] * 20 + 10, 0, 100).astype(int),
        'review_rating': np.clip(X[:, 2] * 1.2 + 2.5, 1, 4.5).round(1),
        'delivery_time_days': np.clip(X[:, 3] * 3 + 7, 3, 15).astype(int),
        'discount_percentage': np.clip(X[:, 4] * 20 + 20, 0, 80).astype(int),
        'previous_returns': np.clip(X[:, 5] * 3.5, 0, 12).astype(int),
        'days_since_purchase': np.clip(X[:, 6] * 10 + 5, 1, 30).astype(int),
        'returned': y
    })
    
    df['product_category'] = np.random.choice(CATEGORIES, n_samples)
    df['size_ordered'] = np.random.choice(SIZES, n_samples,
                                         p=[0.10, 0.20, 0.25, 0.20, 0.20, 0.03, 0.02])
    return df

# Generate and save all datasets
print("🛍️  Generating Product Return Dataset...")
print("=" * 70)

train_df = generate_baseline_data(5000)
batch_1 = generate_baseline_data(1000)
batch_2 = generate_flash_sale_drift(1000)
batch_3 = generate_supply_chain_crisis(1000)

train_df.to_csv('train.csv', index=False)
batch_1.to_csv('batch_1_normal.csv', index=False)
batch_2.to_csv('batch_2_flash_sale.csv', index=False)
batch_3.to_csv('batch_3_supply_crisis.csv', index=False)

print("\n✅ All datasets created!")
print(f"   Training: {len(train_df)} orders, {train_df['returned'].mean():.1%} returns")
print(f"   Batch 1: {len(batch_1)} orders, {batch_1['returned'].mean():.1%} returns")
print(f"   Batch 2 (Flash Sale): {len(batch_2)} orders, {batch_2['returned'].mean():.1%} returns")
print(f"   Batch 3 (Supply Crisis): {len(batch_3)} orders, {batch_3['returned'].mean():.1%} returns")
print("=" * 70)