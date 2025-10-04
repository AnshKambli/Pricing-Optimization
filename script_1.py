# Import required libraries for advanced modeling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Dynamic Pricing Optimization - Advanced Analytics")
print("=" * 55)

# Load the dataset
df = pd.read_csv('dynamic_pricing_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

# 1. PRICE ELASTICITY ANALYSIS
print("\n1. PRICE ELASTICITY ANALYSIS")
print("-" * 30)

def calculate_price_elasticity(data, product_id):
    """Calculate price elasticity using the midpoint method"""
    product_data = data[data['product_id'] == product_id].copy()
    product_data = product_data.sort_values('price')
    
    # Calculate percentage changes
    product_data['price_change'] = product_data['price'].pct_change()
    product_data['quantity_change'] = product_data['actual_sales'].pct_change()
    
    # Remove infinite and NaN values
    product_data = product_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(product_data) < 2:
        return np.nan
    
    # Calculate elasticity
    elasticity = product_data['quantity_change'].mean() / product_data['price_change'].mean()
    return abs(elasticity)  # Return absolute value

# Calculate elasticity for each product
elasticity_results = []
for product in df['product_id'].unique()[:20]:  # Sample first 20 products
    elasticity = calculate_price_elasticity(df, product)
    category = df[df['product_id'] == product]['category'].iloc[0]
    store = df[df['product_id'] == product]['store'].iloc[0]
    
    elasticity_results.append({
        'product_id': product,
        'category': category,
        'store': store,
        'calculated_elasticity': elasticity,
        'theoretical_elasticity': df[df['product_id'] == product]['price_elasticity'].iloc[0]
    })

elasticity_df = pd.DataFrame(elasticity_results)
print("Price Elasticity Analysis Results:")
print(elasticity_df.head(10))

# Elasticity by category
category_elasticity = elasticity_df.groupby('category')['calculated_elasticity'].agg(['mean', 'std']).round(3)
print("\nAverage Price Elasticity by Category:")
print(category_elasticity)

# 2. DEMAND FORECASTING MODELS
print("\n\n2. DEMAND FORECASTING MODELS")
print("-" * 32)

# Prepare features for modeling
def prepare_features(data):
    """Prepare features for machine learning models"""
    data = data.copy()
    
    # Date features
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day_of_year'] = data['date'].dt.dayofyear
    
    # Lag features
    data = data.sort_values(['product_id', 'date'])
    data['sales_lag_1'] = data.groupby('product_id')['actual_sales'].shift(1)
    data['price_lag_1'] = data.groupby('product_id')['price'].shift(1)
    data['revenue_lag_1'] = data.groupby('product_id')['revenue'].shift(1)
    
    # Moving averages
    data['sales_ma_4'] = data.groupby('product_id')['actual_sales'].rolling(4, min_periods=1).mean().values
    data['price_ma_4'] = data.groupby('product_id')['price'].rolling(4, min_periods=1).mean().values
    
    # Price difference from competitor
    data['price_diff'] = data['price'] - data['competitor_price']
    data['price_ratio'] = data['price'] / data['competitor_price']
    
    # Encode categorical variables
    le_category = LabelEncoder()
    le_store = LabelEncoder()
    le_product = LabelEncoder()
    
    data['category_encoded'] = le_category.fit_transform(data['category'])
    data['store_encoded'] = le_store.fit_transform(data['store'])
    data['product_encoded'] = le_product.fit_transform(data['product_id'])
    
    return data

# Prepare the data
df_features = prepare_features(df)

# Select features for modeling
feature_columns = [
    'price', 'competitor_price', 'inventory_level', 'seasonal_factor',
    'marketing_spend', 'weather_impact', 'price_elasticity',
    'year', 'month', 'day_of_week', 'day_of_year',
    'sales_lag_1', 'price_lag_1', 'revenue_lag_1',
    'sales_ma_4', 'price_ma_4', 'price_diff', 'price_ratio',
    'category_encoded', 'store_encoded', 'product_encoded',
    'is_weekend', 'is_holiday'
]

# Remove rows with NaN values (due to lag features)
df_modeling = df_features.dropna(subset=feature_columns + ['actual_sales'])

X = df_modeling[feature_columns]
y = df_modeling['actual_sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

model_results = {}

for name, model in models.items():
    if name in ['Linear Regression', 'Ridge Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_results[name] = {
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'R²': round(r2, 4)
    }

results_df = pd.DataFrame(model_results).T
print("Demand Forecasting Model Performance:")
print(results_df)

# Feature importance from Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features for Demand Prediction:")
print(feature_importance.head(10))

# Save model results
model_results_df = pd.DataFrame(model_results).T
model_results_df.to_csv('model_performance_results.csv')
feature_importance.to_csv('feature_importance.csv', index=False)

print(f"\nModel results saved to 'model_performance_results.csv'")
print(f"Feature importance saved to 'feature_importance.csv'")