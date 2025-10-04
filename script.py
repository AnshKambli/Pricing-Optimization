import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Creating comprehensive Dynamic Pricing Optimization dataset...")
print("=" * 60)

# Generate synthetic dataset for dynamic pricing analysis
def generate_dynamic_pricing_dataset():
    # Date range for 2 years
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Product categories and their characteristics
    product_categories = {
        'Electronics': {'base_price': 500, 'elasticity': -1.2, 'seasonal_factor': 0.3},
        'Clothing': {'base_price': 80, 'elasticity': -1.8, 'seasonal_factor': 0.5},
        'Home & Garden': {'base_price': 150, 'elasticity': -0.8, 'seasonal_factor': 0.4},
        'Sports': {'base_price': 120, 'elasticity': -1.5, 'seasonal_factor': 0.6},
        'Books': {'base_price': 25, 'elasticity': -0.6, 'seasonal_factor': 0.2}
    }
    
    # Stores with different characteristics
    stores = {
        'Premium_Store': {'price_multiplier': 1.3, 'demand_multiplier': 0.7},
        'Budget_Store': {'price_multiplier': 0.8, 'demand_multiplier': 1.4},
        'Online_Store': {'price_multiplier': 0.9, 'demand_multiplier': 1.2},
        'Flagship_Store': {'price_multiplier': 1.1, 'demand_multiplier': 1.0}
    }
    
    data = []
    product_id = 1
    
    for category, cat_info in product_categories.items():
        for store, store_info in stores.items():
            for i in range(5):  # 5 products per category per store
                for date in dates[::7]:  # Weekly data
                    # Calculate seasonal effect
                    day_of_year = date.timetuple().tm_yday
                    seasonal_effect = 1 + cat_info['seasonal_factor'] * np.sin(2 * np.pi * day_of_year / 365)
                    
                    # Base demand with random variation
                    base_demand = np.random.normal(100, 20) * seasonal_effect * store_info['demand_multiplier']
                    
                    # Competitor pricing (random around base price)
                    competitor_price = cat_info['base_price'] * store_info['price_multiplier'] * np.random.normal(1.0, 0.1)
                    
                    # Our price (strategic pricing around competitor)
                    price_strategy = np.random.choice(['competitive', 'premium', 'discount'], p=[0.5, 0.3, 0.2])
                    if price_strategy == 'competitive':
                        price = competitor_price * np.random.normal(1.0, 0.05)
                    elif price_strategy == 'premium':
                        price = competitor_price * np.random.normal(1.15, 0.05)
                    else:  # discount
                        price = competitor_price * np.random.normal(0.85, 0.05)
                    
                    # Calculate demand based on price elasticity
                    price_ratio = price / (cat_info['base_price'] * store_info['price_multiplier'])
                    demand_effect = (price_ratio) ** cat_info['elasticity']
                    demand = base_demand * demand_effect
                    
                    # Add inventory constraints
                    inventory_level = np.random.randint(50, 500)
                    actual_sales = min(demand, inventory_level)
                    
                    # External factors
                    is_weekend = date.weekday() >= 5
                    is_holiday = date.month == 12 and date.day >= 20  # Holiday season
                    weather_impact = np.random.normal(1.0, 0.1)  # Weather effect on demand
                    
                    # Marketing campaigns (random)
                    marketing_spend = np.random.exponential(100) if np.random.random() < 0.3 else 0
                    
                    # Calculate final metrics
                    revenue = actual_sales * price
                    cost = actual_sales * (price * 0.6)  # 60% cost ratio
                    profit = revenue - cost
                    
                    data.append({
                        'date': date,
                        'product_id': f'PROD_{product_id:04d}',
                        'category': category,
                        'store': store,
                        'price': round(price, 2),
                        'competitor_price': round(competitor_price, 2),
                        'demand': round(demand, 0),
                        'actual_sales': round(actual_sales, 0),
                        'inventory_level': inventory_level,
                        'revenue': round(revenue, 2),
                        'profit': round(profit, 2),
                        'is_weekend': is_weekend,
                        'is_holiday': is_holiday,
                        'seasonal_factor': round(seasonal_effect, 3),
                        'marketing_spend': round(marketing_spend, 2),
                        'weather_impact': round(weather_impact, 3),
                        'price_elasticity': cat_info['elasticity']
                    })
                
                product_id += 1
    
    return pd.DataFrame(data)

# Generate the dataset
df = generate_dynamic_pricing_dataset()

print(f"Generated dataset with {len(df)} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Categories: {df['category'].unique()}")
print(f"Stores: {df['store'].unique()}")

# Display basic statistics
print("\nDataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Save to CSV
df.to_csv('dynamic_pricing_dataset.csv', index=False)
print(f"\nDataset saved as 'dynamic_pricing_dataset.csv'")