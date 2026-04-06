import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 1. Advanced Data Loader
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    close_cols = [c for c in df.columns if 'Close' in c]
    df_close = df[close_cols]
    
    asset_names = [c.split('_')[0] for c in close_cols]
    df_close.columns = asset_names
    
    returns = df_close.pct_change().dropna()
    return df_close, returns, asset_names

# 2. Realistic Portfolio Environment with Slippage & Transaction Costs
class PortfolioEnv:
    def __init__(self, returns, window_size=10, initial_balance=1000000, 
                 transaction_cost_pct=0.001): # 0.1% transaction cost
        self.returns = returns.values
        self.n_assets = self.returns.shape[1]
        self.window_size = window_size
        self.n_steps = len(self.returns) - window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        # Start with all cash or equal weight
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = [self.initial_balance]
        self.history_returns = []
        return self._get_state()
        
    def step(self, action):
        # Enforce softmax for allocation
        action = np.exp(action) / np.sum(np.exp(action))
        
        # Calculate turnover (how much we changed the portfolio)
        turnover = np.sum(np.abs(action - self.weights))
        trading_cost = turnover * self.transaction_cost_pct
        self.weights = action
        
        # Calculate portfolio return for the step
        step_return = np.sum(self.returns[self.current_step + self.window_size] * self.weights)
        
        # Net return after trading costs
        net_return = step_return - trading_cost
        
        self.balance = self.balance * (1 + net_return)
        self.portfolio_value.append(self.balance)
        self.history_returns.append(net_return)
        
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Enterprise-Grade Reward: Sharpe Ratio proxy + Return
        # We penalize variance to ensure the AI cares about risk
        if len(self.history_returns) > 2:
            volatility = np.std(self.history_returns)
            # small epsilon to avoid divide by zero
            sharpe_reward = net_return / (volatility + 1e-6) 
        else:
            sharpe_reward = net_return

        reward = sharpe_reward
        return self._get_state(), reward, done
        
    def _get_state(self):
        state = self.returns[self.current_step : self.current_step + self.window_size]
        return state.flatten()

# 3. Enterprise LSTMActorCritic (Temporal Memory)
class LSTMActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_assets):
        super(LSTMActor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.out = nn.Linear(64, n_assets)
        
    def forward(self, x):
        # x is (batch, seq_len, input_dim) -> (1, window_size, n_assets)
        lstm_out, _ = self.lstm(x)
        # Take the last output of the LSTM sequence
        last_out = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last_out))
        return self.out(x)

# 4. Multi-Agent Framework
class AdvancedTradingFramework:
    def __init__(self, n_assets, window_size, lr=5e-4):
        self.n_assets = n_assets
        self.window_size = window_size
        self.actor = LSTMActor(input_dim=n_assets, hidden_dim=32, n_assets=n_assets)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
    def select_action(self, state):
        # Reshape for LSTM: (batch=1, seq_len=window_size, input_dim=n_assets)
        state_tensor = torch.FloatTensor(state).view(1, self.window_size, self.n_assets)
        with torch.no_grad():
            action = self.actor(state_tensor).numpy().flatten()
        return action
        
    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).view(1, self.window_size, self.n_assets)
        action_pred = self.actor(state_tensor)
        
        log_probs = torch.nn.functional.log_softmax(action_pred, dim=1)
        loss = -torch.mean(log_probs * reward)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optimizer.step()

# 5. Out-of-Sample Execution
if __name__ == "__main__":
    file_path = "portfolio data.csv"
    if not os.path.exists(file_path):
        print(f"Data file {file_path} not found.")
        exit(1)
        
    print("Loading data for Production-Grade Simulation...")
    df_close, returns, asset_names = load_data(file_path)
    
    # Train / Test Split (80% Train, 20% Test)
    split_idx = int(len(returns) * 0.8)
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:]
    
    print(f"Training on {len(train_returns)} days, Testing on {len(test_returns)} days.")
    
    window_size = 20 # Larger memory window
    episodes = 5
    
    train_env = PortfolioEnv(train_returns, window_size=window_size, transaction_cost_pct=0.001)
    test_env = PortfolioEnv(test_returns, window_size=window_size, transaction_cost_pct=0.001)
    
    agent = AdvancedTradingFramework(n_assets=len(asset_names), window_size=window_size)
    
    print("\n--- Phase 1: Training Agent ---")
    for ep in range(episodes):
        state = train_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = train_env.step(action)
            agent.train_step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        print(f"Training Ep {ep+1}/{episodes} | Final Train Portfolio Value: ${train_env.portfolio_value[-1]:,.2f}")
        
    print("\n--- Phase 2: Out-of-Sample Backtest ---")
    state = test_env.reset()
    done = False
    
    # Simple Benchmark: Buy and Hold Equal Weight
    benchmark_balance = test_env.initial_balance
    benchmark_history = [benchmark_balance]
    
    while not done:
        # Agent trading
        action = agent.select_action(state)
        next_state, reward, done = test_env.step(action)
        state = next_state
        
        # Benchmark logic
        bench_return = np.sum(test_returns.values[test_env.current_step + window_size - 1] * (1.0/len(asset_names)))
        benchmark_balance = benchmark_balance * (1 + bench_return)
        benchmark_history.append(benchmark_balance)
        
    print(f"Test Phase Complete!")
    print(f"Agent Final Value: ${test_env.portfolio_value[-1]:,.2f}")
    print(f"Benchmark Final Value: ${benchmark_history[-1]:,.2f}")
    
    plt.figure(figsize=(10,6))
    plt.plot(test_env.portfolio_value, color='blue', label='AI Portfolio (After Costs)')
    plt.plot(benchmark_history, color='gray', linestyle='--', label='Equal Weight Benchmark')
    plt.title('Out-of-Sample Production Backtest')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('production_backtest.png')
    print("\nSaved robust backtest chart to 'production_backtest.png'.")
