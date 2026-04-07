import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    """Custom Environment that follows gym interface for Portfolio Allocation"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df_prices, df_returns, initial_balance=1000000, transaction_cost=0.001, window_size=20):
        super(PortfolioEnv, self).__init__()
        self.df_prices = df_prices
        self.df_returns = df_returns
        self.n_assets = len(df_returns.columns)
        
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        # Action space: weights for each asset 
        # (Stable-baselines3 continuous space)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        # Observation space: Window of returns + current weights
        obs_shape = (self.window_size * self.n_assets) + self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.done = False
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Flatten the history of returns for the window
        history = self.df_returns.iloc[self.current_step - self.window_size:self.current_step].values.flatten()
        obs = np.concatenate([history, self.current_weights])
        return obs.astype(np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}
            
        # Normalize action to sum to 1 (softmax)
        exp_action = np.exp(action)
        new_weights = exp_action / np.sum(exp_action)
        
        # Get market returns for this step
        market_returns = self.df_returns.iloc[self.current_step].values
        
        # Calculate portfolio return
        portfolio_return = np.sum(new_weights * market_returns)
        
        # Calculate transaction costs
        weight_changes = np.abs(new_weights - self.current_weights)
        costs = self.transaction_cost * np.sum(weight_changes)
        
        # Net Return
        net_return = portfolio_return - costs
        
        # Update Portfolio Value
        self.portfolio_value = self.portfolio_value * (1 + net_return)
        self.current_weights = new_weights
        
        reward = portfolio_return  # The reward is the raw portfolio return as specified
        
        self.current_step += 1
        
        # Check if Done
        self.done = self.current_step >= len(self.df_returns) - 1
        
        return self._get_obs(), reward, self.done, False, {"portfolio_value": self.portfolio_value}
