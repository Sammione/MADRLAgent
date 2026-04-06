from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import os

# Import our advanced framework classes
from drl_portfolio import load_data, PortfolioEnv, AdvancedTradingFramework

app = FastAPI(title="MADRL Portfolio API")

# Configure CORS so the React frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    data_path: str = "portfolio data.csv"
    window_size: int = 20
    transaction_cost: float = 0.001  # 0.1% transaction cost
    test_split: float = 0.20 # 20% test split
    episodes: int = 2

class SimulationResponse(BaseModel):
    assets: list[str]
    ai_final_value: float
    benchmark_final_value: float
    ai_portfolio_history: list[float]
    benchmark_history: list[float]
    ai_return_pct: float
    benchmark_return_pct: float

@app.post("/run-simulation", response_model=SimulationResponse)
def run_simulation(req: SimulationRequest):
    if not os.path.exists(req.data_path):
        raise HTTPException(status_code=404, detail=f"Dataset '{req.data_path}' not found!")
        
    try:
        df_close, returns, asset_names = load_data(req.data_path)
        
        # Split Data
        split_idx = int(len(returns) * (1 - req.test_split))
        train_returns = returns.iloc[:split_idx]
        test_returns = returns.iloc[split_idx:]
        
        train_env = PortfolioEnv(train_returns, window_size=req.window_size, transaction_cost_pct=req.transaction_cost)
        test_env = PortfolioEnv(test_returns, window_size=req.window_size, transaction_cost_pct=req.transaction_cost)
        
        agent = AdvancedTradingFramework(n_assets=len(asset_names), window_size=req.window_size)
        
        # Training Phase
        for ep in range(req.episodes):
            state = train_env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done = train_env.step(action)
                agent.train_step(state, action, reward, next_state, done)
                state = next_state
                
        # Testing Phase
        state = test_env.reset()
        done = False
        
        benchmark_balance = test_env.initial_balance
        benchmark_history = [benchmark_balance]
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = test_env.step(action)
            state = next_state
            
            # Benchmark (Equal Weight)
            bench_return = np.sum(test_returns.values[test_env.current_step + req.window_size - 1] * (1.0/len(asset_names)))
            benchmark_balance = benchmark_balance * (1 + bench_return)
            benchmark_history.append(benchmark_balance)
            
        ai_final = float(test_env.portfolio_value[-1])
        bench_final = float(benchmark_history[-1])
        
        ai_return_pct = ((ai_final / test_env.initial_balance) - 1) * 100
        bench_return_pct = ((bench_final / test_env.initial_balance) - 1) * 100
        
        return SimulationResponse(
            assets=list(asset_names),
            ai_final_value=ai_final,
            benchmark_final_value=bench_final,
            ai_portfolio_history=[float(x) for x in test_env.portfolio_value],
            benchmark_history=[float(x) for x in benchmark_history],
            ai_return_pct=ai_return_pct,
            benchmark_return_pct=bench_return_pct
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
