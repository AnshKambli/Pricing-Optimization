# 3. REINFORCEMENT LEARNING PRICING AGENT
print("\n3. REINFORCEMENT LEARNING PRICING AGENT")
print("-" * 40)

import random
from collections import defaultdict, deque

class DynamicPricingEnvironment:
    """Environment for dynamic pricing using reinforcement learning"""
    
    def __init__(self, product_data, base_price=100, max_price_change=0.3):
        self.product_data = product_data.copy()
        self.base_price = base_price
        self.max_price_change = max_price_change
        self.current_price = base_price
        self.time_step = 0
        self.total_revenue = 0
        self.total_profit = 0
        self.demand_history = deque(maxlen=5)
        
        # Price elasticity from data
        self.price_elasticity = product_data['price_elasticity'].iloc[0]
        
    def get_state(self):
        """Get current state representation"""
        # Normalize features to create state
        current_data = self.product_data.iloc[self.time_step % len(self.product_data)]
        
        state = (
            round(self.current_price / self.base_price, 2),  # Price ratio
            round(current_data['competitor_price'] / self.base_price, 2),  # Competitor price ratio
            round(current_data['seasonal_factor'], 2),  # Seasonal factor
            int(current_data['is_weekend']),  # Weekend flag
            int(current_data['is_holiday']),  # Holiday flag
            round(current_data['inventory_level'] / 500, 2),  # Inventory ratio
            round(np.mean(self.demand_history) / 100 if self.demand_history else 0.5, 2)  # Demand history
        )
        return state
    
    def get_demand(self, price):
        """Calculate demand based on price and external factors"""
        current_data = self.product_data.iloc[self.time_step % len(self.product_data)]
        
        # Base demand from data
        base_demand = current_data['demand']
        
        # Price elasticity effect
        price_ratio = price / self.base_price
        price_effect = (price_ratio) ** self.price_elasticity
        
        # External factors
        seasonal_effect = current_data['seasonal_factor']
        weekend_effect = 1.2 if current_data['is_weekend'] else 1.0
        holiday_effect = 1.5 if current_data['is_holiday'] else 1.0
        marketing_effect = 1 + (current_data['marketing_spend'] / 1000)
        
        # Calculate final demand
        demand = base_demand * price_effect * seasonal_effect * weekend_effect * holiday_effect * marketing_effect
        demand = max(0, demand)  # Ensure non-negative demand
        
        return demand
    
    def step(self, action):
        """Take action and return reward"""
        # Action mapping: 0=decrease, 1=maintain, 2=increase
        price_changes = [-self.max_price_change, 0, self.max_price_change]
        price_change = price_changes[action]
        
        # Update price
        new_price = self.current_price * (1 + price_change)
        new_price = max(self.base_price * 0.5, min(new_price, self.base_price * 2.0))  # Price bounds
        self.current_price = new_price
        
        # Calculate demand and sales
        demand = self.get_demand(new_price)
        current_data = self.product_data.iloc[self.time_step % len(self.product_data)]
        sales = min(demand, current_data['inventory_level'])
        
        # Update history
        self.demand_history.append(demand)
        
        # Calculate reward (profit)
        revenue = sales * new_price
        cost = sales * (new_price * 0.6)  # 60% cost ratio
        profit = revenue - cost
        
        # Reward function: profit with penalty for extreme prices
        reward = profit
        if new_price < self.base_price * 0.7 or new_price > self.base_price * 1.4:
            reward *= 0.8  # Penalty for extreme pricing
        
        # Update totals
        self.total_revenue += revenue
        self.total_profit += profit
        
        # Move to next time step
        self.time_step += 1
        
        # Check if episode is done
        done = self.time_step >= len(self.product_data)
        
        return self.get_state(), reward, done, {'sales': sales, 'revenue': revenue, 'profit': profit}

class QLearningAgent:
    """Q-Learning agent for dynamic pricing"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])  # 3 actions: decrease, maintain, increase
        
    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best action
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula"""
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * max(self.q_table[next_state])
        
        # Q-learning update
        self.q_table[state][action] += self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_pricing_agent(product_data, episodes=500):
    """Train the Q-learning pricing agent"""
    env = DynamicPricingEnvironment(product_data)
    agent = QLearningAgent()
    
    episode_rewards = []
    episode_revenues = []
    episode_profits = []
    
    print(f"Training RL agent for {episodes} episodes...")
    
    for episode in range(episodes):
        # Reset environment
        env = DynamicPricingEnvironment(product_data)
        state = env.get_state()
        total_reward = 0
        
        while True:
            # Choose and take action
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_revenues.append(env.total_revenue)
        episode_profits.append(env.total_profit)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_revenue = np.mean(episode_revenues[-100:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Revenue = ${avg_revenue:,.2f}, Epsilon = {agent.epsilon:.3f}")
    
    return agent, episode_rewards, episode_revenues, episode_profits

# Train the agent on Electronics products from Premium Store
electronics_data = df[
    (df['category'] == 'Electronics') & 
    (df['store'] == 'Premium_Store') & 
    (df['product_id'] == 'PROD_0001')
].copy().reset_index(drop=True)

# Train the RL agent
trained_agent, rewards, revenues, profits = train_pricing_agent(electronics_data, episodes=300)

print(f"\nTraining completed!")
print(f"Final average reward (last 50 episodes): ${np.mean(rewards[-50:]):,.2f}")
print(f"Final average revenue (last 50 episodes): ${np.mean(revenues[-50:]):,.2f}")
print(f"Final average profit (last 50 episodes): ${np.mean(profits[-50:]):,.2f}")

# Test the trained agent
def test_pricing_strategy(agent, test_data, strategy_name="RL Agent"):
    """Test pricing strategy and return results"""
    env = DynamicPricingEnvironment(test_data)
    state = env.get_state()
    
    results = []
    
    while True:
        if strategy_name == "RL Agent":
            action = np.argmax(agent.q_table[state])  # Use best action (no exploration)
        elif strategy_name == "Fixed Price":
            action = 1  # Always maintain price
        elif strategy_name == "Competitive":
            # Simple competitive strategy
            current_data = test_data.iloc[env.time_step % len(test_data)]
            if env.current_price > current_data['competitor_price']:
                action = 0  # Decrease
            elif env.current_price < current_data['competitor_price'] * 0.95:
                action = 2  # Increase
            else:
                action = 1  # Maintain
        
        next_state, reward, done, info = env.step(action)
        
        results.append({
            'time_step': env.time_step - 1,
            'price': env.current_price,
            'sales': info['sales'],
            'revenue': info['revenue'],
            'profit': info['profit']
        })
        
        state = next_state
        if done:
            break
    
    return pd.DataFrame(results), env.total_revenue, env.total_profit

# Compare different strategies
test_data = electronics_data[:52]  # One year of data for testing

strategies = {
    "RL Agent": trained_agent,
    "Fixed Price": None,
    "Competitive": None
}

strategy_results = {}

print("\nTesting Different Pricing Strategies:")
print("-" * 40)

for strategy_name, agent in strategies.items():
    results_df, total_revenue, total_profit = test_pricing_strategy(agent, test_data, strategy_name)
    
    strategy_results[strategy_name] = {
        'Total Revenue': total_revenue,
        'Total Profit': total_profit,
        'Avg Price': results_df['price'].mean(),
        'Avg Sales': results_df['sales'].mean(),
        'Results_DF': results_df
    }
    
    print(f"{strategy_name}:")
    print(f"  Total Revenue: ${total_revenue:,.2f}")
    print(f"  Total Profit: ${total_profit:,.2f}")
    print(f"  Average Price: ${results_df['price'].mean():.2f}")
    print(f"  Average Sales: {results_df['sales'].mean():.1f} units")
    print()

# Create comparison summary
comparison_df = pd.DataFrame({
    strategy: {
        'Total Revenue': results['Total Revenue'],
        'Total Profit': results['Total Profit'],
        'Avg Price': results['Avg Price'],
        'Avg Sales': results['Avg Sales']
    }
    for strategy, results in strategy_results.items()
}).T

print("Strategy Comparison Summary:")
print(comparison_df.round(2))

# Save results
comparison_df.to_csv('strategy_comparison.csv')
pd.DataFrame({'episode': range(len(rewards)), 'reward': rewards, 'revenue': revenues, 'profit': profits}).to_csv('rl_training_history.csv', index=False)

print(f"\nResults saved:")
print(f"- Strategy comparison: 'strategy_comparison.csv'")
print(f"- RL training history: 'rl_training_history.csv'")