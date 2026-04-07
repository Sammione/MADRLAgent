import React, { useState, useEffect, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine 
} from 'recharts';
import { Settings, Play, BrainCircuit, Activity, DollarSign, TrendingUp, BarChart3, RefreshCw } from 'lucide-react';
import './App.css';

function App() {
  const [isSimulating, setIsSimulating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [data, setData] = useState([{ day: 'Day 0', agent: 1000000, benchmark: 1000000 }]);
  
  const [metrics, setMetrics] = useState({
    agent: 1000000,
    benchmark: 1000000,
    aiRoi: 0,
    benchRoi: 0,
    alpha: 0
  });

  const [params, setParams] = useState({
    windowSize: 20,
    cost: 0.1,
    testSplit: 20,
    episodes: 2
  });

  // Ref to hold the interval ID so we can clear it
  const intervalRef = useRef(null);
  
  // API Url fallback
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'; // Default locahost for rapid dev, or Render URL if deployed

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const handleStart = async () => {
    // 1. Train and Setup
    setIsTraining(true);
    setData([{ day: 'Day 0', agent: 1000000, benchmark: 1000000 }]); // reset chart
    setMetrics({ agent: 1000000, benchmark: 1000000, aiRoi: 0, benchRoi: 0, alpha: 0 });
    
    if (intervalRef.current) clearInterval(intervalRef.current);
    
    try {
      const resp = await fetch(`${apiUrl}/setup-and-train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          window_size: params.windowSize,
          transaction_cost: params.cost / 100.0,
          test_split: params.testSplit / 100.0,
          episodes: params.episodes
        })
      });
      
      if (!resp.ok) throw new Error("Training failed");
      
      setIsTraining(false);
      setIsSimulating(true);
      
      // 2. Start polling the simulation
      intervalRef.current = setInterval(stepSimulation, 100); 
    } catch (error) {
      console.error(error);
      alert("Error training agent. Is the backend running?");
      setIsTraining(false);
    }
  };

  const stepSimulation = async () => {
    try {
      const resp = await fetch(`${apiUrl}/simulate`);
      const result = await resp.json();
      
      if (result.done) {
        clearInterval(intervalRef.current);
        setIsSimulating(false);
        fetchFinalMetrics();
        return;
      }
      
      // Add data point to chart dynamically
      setData(prevData => {
        const newData = [...prevData, {
          day: `Day ${result.timestep}`,
          agent: Math.round(result.portfolio_value),
          benchmark: Math.round(result.benchmark_value)
        }];
        
        // Keep chart moving by slicing if it gets too long, or just let it grow
        // For standard trading visuals, growing from left to right is nice.
        return newData;
      });
      
      // Update Live Metrics
      const agentVal = result.portfolio_value;
      const benchVal = result.benchmark_value;
      const aiRoi = ((agentVal / 1000000) - 1) * 100;
      const benchRoi = ((benchVal / 1000000) - 1) * 100;
      
      setMetrics({
        agent: agentVal,
        benchmark: benchVal,
        aiRoi: aiRoi,
        benchRoi: benchRoi,
        alpha: aiRoi - benchRoi
      });
      
    } catch (error) {
       console.error("Simulation step failed:", error);
       clearInterval(intervalRef.current);
       setIsSimulating(false);
    }
  };
  
  const fetchFinalMetrics = async () => {
     try {
       const resp = await fetch(`${apiUrl}/metrics`);
       const stats = await resp.json();
       // Optionally update final stats
     } catch (e) {
       console.error(e);
     }
  };
  
  const stopSimulation = () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      setIsSimulating(false);
  };

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="brand">
          <BrainCircuit size={28} />
          <span>MADRL Agent</span>
        </div>
        
        <div className="control-group">
          <label>LSTM/PPO Window (Days): {params.windowSize}</label>
          <input 
            type="range" min="5" max="60" 
            value={params.windowSize} 
            onChange={(e) => setParams({...params, windowSize: parseInt(e.target.value)})} 
            disabled={isSimulating || isTraining}
          />
        </div>

        <div className="control-group">
          <label>Transaction Costs (%): {params.cost}</label>
          <input 
            type="number" step="0.05" min="0" max="2" 
            value={params.cost} 
            onChange={(e) => setParams({...params, cost: parseFloat(e.target.value)})} 
            disabled={isSimulating || isTraining}
          />
        </div>

        <div className="control-group">
          <label>Test Split (%): {params.testSplit}</label>
          <input 
            type="range" min="10" max="50" step="5"
            value={params.testSplit} 
            onChange={(e) => setParams({...params, testSplit: parseInt(e.target.value)})} 
            disabled={isSimulating || isTraining}
          />
        </div>

        <div className="control-group">
          <label>Training Episodes: {params.episodes}</label>
          <input 
            type="range" min="1" max="20" 
            value={params.episodes} 
            onChange={(e) => setParams({...params, episodes: parseInt(e.target.value)})} 
            disabled={isSimulating || isTraining}
          />
        </div>

        {!isSimulating ? (
          <button className="run-btn" onClick={handleStart} disabled={isTraining}>
            {isTraining ? <Activity className="animate-spin" /> : <Play />}
            {isTraining ? 'Training PPO Agent...' : 'Start Simulation'}
          </button>
        ) : (
          <button className="run-btn stop-btn" style={{backgroundColor: '#ef4444'}} onClick={stopSimulation}>
            Stop Simulation
          </button>
        )}
      </aside>

      <main className="main-content">
        <header className="header">
          <h1>PPO Live Execution Terminal</h1>
          <p>Real-Time Out-of-Sample Performance vs Equal-Weight Benchmark</p>
        </header>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-title"><DollarSign size={16}/> Total Agent Value</div>
            <div className="metric-value">${Math.round(metrics.agent).toLocaleString()}</div>
            <div className={`metric-sub ${metrics.aiRoi < 0 ? 'negative' : ''}`}>
              {metrics.aiRoi > 0 ? '+' : ''}{metrics.aiRoi.toFixed(2)}% ROI
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-title"><BarChart3 size={16}/> Benchmark Value</div>
            <div className="metric-value">${Math.round(metrics.benchmark).toLocaleString()}</div>
            <div className={`metric-sub ${metrics.benchRoi < 0 ? 'negative' : ''}`}>
              {metrics.benchRoi > 0 ? '+' : ''}{metrics.benchRoi.toFixed(2)}% ROI
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-title"><TrendingUp size={16}/> Excess Return (Alpha)</div>
            <div className="metric-value">{metrics.alpha > 0 ? '+' : ''}{metrics.alpha.toFixed(2)}%</div>
            <div className="metric-sub">Above market average</div>
          </div>
        </div>

        <div className="chart-container">
          <div className="chart-header">
            <h2 className="chart-title">Real-Time Portfolio Tracking</h2>
            {isSimulating && <RefreshCw size={20} color="#00ff88" className="animate-spin" />}
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis 
                  dataKey="day" 
                  stroke="#94a3b8" 
                  tick={{fill: '#94a3b8', fontSize: 12}} 
                  tickLine={false}
                  axisLine={false}
                  minTickGap={30}
                />
                <YAxis 
                  domain={['dataMin - 10000', 'dataMax + 10000']} 
                  stroke="#94a3b8" 
                  tick={{fill: '#94a3b8', fontSize: 12}} 
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => `$${(value/1000).toFixed(0)}k`}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#141b2d', border: '1px solid #1e293b', borderRadius: '8px' }}
                  itemStyle={{ fontWeight: 600 }}
                  formatter={(value) => `$${value.toLocaleString()}`}
                />
                <ReferenceLine y={1000000} stroke="#475569" strokeDasharray="3 3" />
                <Line 
                  type="monotone" 
                  name="AI Agent"
                  dataKey="agent" 
                  stroke="#00ff88" 
                  strokeWidth={3} 
                  dot={false}
                  isAnimationActive={false} // Disable recharts animation so our live data stream looks smooth
                />
                <Line 
                  type="monotone" 
                  name="Market Benchmark"
                  dataKey="benchmark" 
                  stroke="#ff4444" 
                  strokeWidth={2} 
                  strokeDasharray="5 5"
                  dot={false} 
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
