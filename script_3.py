# 4. REVENUE OPTIMIZATION FUNCTIONS
print("\n4. REVENUE OPTIMIZATION FUNCTIONS")
print("-" * 35)

from scipy.optimize import minimize_scalar, minimize
import warnings
warnings.filterwarnings('ignore')

class RevenueOptimizer:
    """Revenue optimization using mathematical optimization"""
    
    def __init__(self, demand_model, cost_ratio=0.6):
        self.demand_model = demand_model  # Trained ML model
        self.cost_ratio = cost_ratio
        self.scaler = scaler  # From earlier
        
    def predict_demand(self, price, features):
        """Predict demand given price and other features"""
        # Create feature vector with the given price
        feature_vector = features.copy()
        feature_vector[0] = price  # Assuming price is first feature
        
        # Reshape and scale
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict demand
        demand = self.demand_model.predict(feature_vector_scaled)[0]
        return max(0, demand)  # Ensure non-negative
    
    def revenue_function(self, price, features):
        """Calculate revenue for given price"""
        demand = self.predict_demand(price, features)
        revenue = price * demand
        return -revenue  # Negative for minimization
    
    def profit_function(self, price, features):
        """Calculate profit for given price"""
        demand = self.predict_demand(price, features)
        revenue = price * demand
        cost = demand * price * self.cost_ratio
        profit = revenue - cost
        return -profit  # Negative for minimization
    
    def optimize_price(self, features, min_price=10, max_price=1000, objective='profit'):
        """Find optimal price using scipy optimization"""
        if objective == 'profit':
            objective_func = self.profit_function
        else:
            objective_func = self.revenue_function
        
        result = minimize_scalar(
            objective_func,
            bounds=(min_price, max_price),
            method='bounded',
            args=(features,)
        )
        
        optimal_price = result.x
        optimal_value = -result.fun  # Convert back from negative
        predicted_demand = self.predict_demand(optimal_price, features)
        
        return {
            'optimal_price': optimal_price,
            'predicted_demand': predicted_demand,
            f'optimal_{objective}': optimal_value,
            'revenue': optimal_price * predicted_demand,
            'cost': predicted_demand * optimal_price * self.cost_ratio,
            'profit': optimal_price * predicted_demand * (1 - self.cost_ratio)
        }

# Initialize optimizer with best model (Gradient Boosting)
best_model = models['Gradient Boosting']
optimizer = RevenueOptimizer(best_model)

# Test optimization on sample data
sample_data = df_modeling.iloc[:10].copy()
optimization_results = []

print("Price Optimization Results (Sample Products):")
print("-" * 50)

for idx, row in sample_data.iterrows():
    # Get feature vector (excluding target variable)
    features = row[feature_columns].values
    
    result = optimizer.optimize_price(features, min_price=50, max_price=1000, objective='profit')
    
    optimization_results.append({
        'product_id': row['product_id'],
        'category': row['category'],
        'current_price': row['price'],
        'optimal_price': result['optimal_price'],
        'price_change': ((result['optimal_price'] - row['price']) / row['price']) * 100,
        'current_sales': row['actual_sales'],
        'predicted_sales': result['predicted_demand'],
        'current_revenue': row['revenue'],
        'optimal_revenue': result['revenue'],
        'revenue_uplift': ((result['revenue'] - row['revenue']) / row['revenue']) * 100,
        'current_profit': row['profit'],
        'optimal_profit': result['profit'],
        'profit_uplift': ((result['profit'] - row['profit']) / row['profit']) * 100
    })

optimization_df = pd.DataFrame(optimization_results)

# Display results
print("Top 5 Optimization Results:")
display_cols = ['product_id', 'category', 'current_price', 'optimal_price', 'price_change', 'revenue_uplift', 'profit_uplift']
print(optimization_df[display_cols].head().round(2))

# Summary statistics
print(f"\nOptimization Summary:")
print(f"Average price change: {optimization_df['price_change'].mean():.1f}%")
print(f"Average revenue uplift: {optimization_df['revenue_uplift'].mean():.1f}%")
print(f"Average profit uplift: {optimization_df['profit_uplift'].mean():.1f}%")
print(f"Products with positive uplift: {(optimization_df['profit_uplift'] > 0).sum()}/{len(optimization_df)}")

# 5. SENSITIVITY ANALYSIS
print("\n\n5. SENSITIVITY ANALYSIS")
print("-" * 25)

def sensitivity_analysis(base_features, price_range, feature_idx, feature_name):
    """Perform sensitivity analysis for a specific feature"""
    results = []
    
    for price in price_range:
        for feature_multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
            features_modified = base_features.copy()
            features_modified[0] = price  # Price
            features_modified[feature_idx] = base_features[feature_idx] * feature_multiplier
            
            demand = optimizer.predict_demand(price, features_modified)
            revenue = price * demand
            profit = revenue * (1 - optimizer.cost_ratio)
            
            results.append({
                'price': price,
                'feature_multiplier': feature_multiplier,
                'feature_name': feature_name,
                'demand': demand,
                'revenue': revenue,
                'profit': profit
            })
    
    return pd.DataFrame(results)

# Sensitivity analysis for key features
base_row = sample_data.iloc[0]
base_features = base_row[feature_columns].values
price_range = np.linspace(50, 200, 10)

# Analyze sensitivity to competitor price
competitor_sensitivity = sensitivity_analysis(
    base_features, price_range, 1, 'competitor_price'
)

# Analyze sensitivity to inventory level
inventory_sensitivity = sensitivity_analysis(
    base_features, price_range, 2, 'inventory_level'
)

# Analyze sensitivity to seasonal factor
seasonal_sensitivity = sensitivity_analysis(
    base_features, price_range, 3, 'seasonal_factor'
)

print("Sensitivity Analysis completed for:")
print("- Competitor Price")
print("- Inventory Level") 
print("- Seasonal Factor")

# Save all optimization results
optimization_df.to_csv('price_optimization_results.csv', index=False)
competitor_sensitivity.to_csv('competitor_sensitivity_analysis.csv', index=False)
inventory_sensitivity.to_csv('inventory_sensitivity_analysis.csv', index=False)
seasonal_sensitivity.to_csv('seasonal_sensitivity_analysis.csv', index=False)

print(f"\nOptimization results saved:")
print(f"- Price optimization: 'price_optimization_results.csv'")
print(f"- Sensitivity analyses: '*_sensitivity_analysis.csv'")

# 6. BUSINESS IMPACT SUMMARY
print("\n\n6. BUSINESS IMPACT SUMMARY")
print("-" * 28)

# Calculate total business impact
total_current_revenue = optimization_df['current_revenue'].sum()
total_optimal_revenue = optimization_df['optimal_revenue'].sum()
total_revenue_uplift = total_optimal_revenue - total_current_revenue
revenue_uplift_pct = (total_revenue_uplift / total_current_revenue) * 100

total_current_profit = optimization_df['current_profit'].sum()
total_optimal_profit = optimization_df['optimal_profit'].sum()
total_profit_uplift = total_optimal_profit - total_current_profit
profit_uplift_pct = (total_profit_uplift / total_current_profit) * 100

print(f"Business Impact Analysis (Sample of {len(optimization_df)} products):")
print(f"{'='*60}")
print(f"Current Total Revenue:     ${total_current_revenue:,.2f}")
print(f"Optimized Total Revenue:   ${total_optimal_revenue:,.2f}")
print(f"Revenue Uplift:            ${total_revenue_uplift:,.2f} ({revenue_uplift_pct:.1f}%)")
print()
print(f"Current Total Profit:      ${total_current_profit:,.2f}")
print(f"Optimized Total Profit:    ${total_optimal_profit:,.2f}")
print(f"Profit Uplift:             ${total_profit_uplift:,.2f} ({profit_uplift_pct:.1f}%)")
print()

# Price change distribution
price_increase = (optimization_df['price_change'] > 0).sum()
price_decrease = (optimization_df['price_change'] < 0).sum()
price_maintain = (abs(optimization_df['price_change']) < 1).sum()

print(f"Price Change Recommendations:")
print(f"- Increase Price: {price_increase} products ({price_increase/len(optimization_df)*100:.1f}%)")
print(f"- Decrease Price: {price_decrease} products ({price_decrease/len(optimization_df)*100:.1f}%)")
print(f"- Maintain Price: {price_maintain} products ({price_maintain/len(optimization_df)*100:.1f}%)")

# Create comprehensive summary
summary_stats = {
    'Total_Products_Analyzed': len(optimization_df),
    'Current_Revenue': total_current_revenue,
    'Optimized_Revenue': total_optimal_revenue,
    'Revenue_Uplift_Dollar': total_revenue_uplift,
    'Revenue_Uplift_Percent': revenue_uplift_pct,
    'Current_Profit': total_current_profit,
    'Optimized_Profit': total_optimal_profit,
    'Profit_Uplift_Dollar': total_profit_uplift,
    'Profit_Uplift_Percent': profit_uplift_pct,
    'Avg_Price_Change_Percent': optimization_df['price_change'].mean(),
    'Products_Price_Increase': price_increase,
    'Products_Price_Decrease': price_decrease,
    'Products_Price_Maintain': price_maintain
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('business_impact_summary.csv', index=False)

print(f"\nBusiness impact summary saved to 'business_impact_summary.csv'")
print("\n" + "="*60)
print("DYNAMIC PRICING OPTIMIZATION PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)