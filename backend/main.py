from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Ensure local imports work correctly for relative path running
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import TradingSimulation

app = FastAPI(title="Real-Time MADRL Portfolio API")

# Configure CORS so the React frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation state (Single-user instance for demonstration)
sim_instance = None

class SetupRequest(BaseModel):
    window_size: int = 20
    transaction_cost: float = 0.001
    test_split: float = 0.20
    episodes: int = 2

@app.post("/setup-and-train")
def setup_and_train(req: SetupRequest):
    global sim_instance
    try:
        # Search for data file depending on where script runs from
        data_path = "portfolio data.csv"
        if not os.path.exists(data_path) and os.path.exists("../portfolio data.csv"):
            data_path = "../portfolio data.csv"
        
        if not os.path.exists(data_path):
             raise Exception(f"Data file not found at {data_path}")

        sim_instance = TradingSimulation(
            data_path=data_path,
            window_size=req.window_size,
            cost=req.transaction_cost,
            test_split=req.test_split
        )
        sim_instance.train(episodes=req.episodes)
        return {"status": "success", "assets": sim_instance.assets}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulate")
def simulate_step():
    global sim_instance
    if sim_instance is None or not sim_instance.is_trained:
        raise HTTPException(status_code=400, detail="Simulation not trained yet.")
        
    res = sim_instance.step_simulation()
    
    if res is None:
        return {"done": True}
        
    res["done"] = False
    return res

@app.get("/reset")
def reset_sim():
    global sim_instance
    if sim_instance:
        sim_instance.reset_simulation()
    return {"status": "reset"}

@app.get("/metrics")
def get_metrics():
    global sim_instance
    if sim_instance is None or not sim_instance.is_trained:
        return {}
    
    ai_final = sim_instance.test_env.portfolio_value
    bench_final = sim_instance.benchmark_value
    init_bal = sim_instance.test_env.initial_balance
    
    return {
        "cumulative_return_ai": ai_final - init_bal,
        "roi_ai": ((ai_final / init_bal) - 1) * 100,
        "roi_benchmark": ((bench_final / init_bal) - 1) * 100
    }
