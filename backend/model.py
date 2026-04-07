import pandas as pd
import numpy as np
from env import PortfolioEnv
from stable_baselines3 import PPO

def load_data(filepath="portfolio data.csv"):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Exclude Date and volume from asset calculation
    asset_cols = [c for c in df.columns if c not in ['Date', 'volume']]
    df_prices = df[asset_cols]
    
    # Calculate returns
    df_returns = df_prices.pct_change().fillna(0)
    
    return df_prices, df_returns, asset_cols

class TradingSimulation:
    def __init__(self, data_path="portfolio data.csv", window_size=20, cost=0.001, test_split=0.2):
        self.df_prices, self.df_returns, self.assets = load_data(data_path)
        
        split_idx = int(len(self.df_returns) * (1 - test_split))
        self.train_returns = self.df_returns.iloc[:split_idx]
        self.test_returns = self.df_returns.iloc[split_idx:]
        self.test_prices = self.df_prices.iloc[split_idx:]
        
        self.env_kwargs = {
            "transaction_cost": cost,
            "window_size": window_size,
            "initial_balance": 1000000
        }
        
        self.train_env = PortfolioEnv(self.df_prices.iloc[:split_idx], self.train_returns, **self.env_kwargs)
        self.test_env = PortfolioEnv(self.test_prices, self.test_returns, **self.env_kwargs)
        
        self.model = None
        self.is_trained = False
        
        # Simulation state
        self.current_obs = None
        self.done = False
        self.benchmark_value = 1000000
    
    def train(self, episodes=2):
        # Learn for a given number of total timesteps
        # One episode roughly equals the training dataset length
        total_timesteps = len(self.train_returns) * episodes
        self.model = PPO("MlpPolicy", self.train_env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)
        self.is_trained = True
        self.reset_simulation()
        
    def reset_simulation(self):
        self.current_obs, _ = self.test_env.reset()
        self.done = False
        self.benchmark_value = 1000000
        
    def step_simulation(self):
        if not self.is_trained:
            return None
        if self.done:
            return None
            
        action, _states = self.model.predict(self.current_obs, deterministic=True)
        self.current_obs, reward, self.done, truncated, info = self.test_env.step(action)
        
        # Benchmark step (equal weight market return)
        market_return = np.mean(self.test_returns.iloc[self.test_env.current_step].values)
        self.benchmark_value = self.benchmark_value * (1 + market_return)
        
        return {
            "portfolio_value": self.test_env.portfolio_value,
            "benchmark_value": self.benchmark_value,
            "weights": self.test_env.current_weights.tolist(),
            "returns": reward,
            "timestep": self.test_env.current_step
        }
