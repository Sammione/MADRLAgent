import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine 
} from 'recharts';
import { Settings, Play, BrainCircuit, Activity, DollarSign, TrendingUp, BarChart3 } from 'lucide-react';
import './App.css';

function App() {
  const [isSimulating, setIsSimulating] = useState(false);
  const [data, setData] = useState([{ day: 'Day 1', agent: 1000000, benchmark: 1000000 }]);
  
  const [params, setParams] = useState({
    windowSize: 20,
    cost: 0.1,
    testSplit: 20,
    episodes: 2
  });

  const handleSimulate = async () => {
    setIsSimulating(true);
    try {
      // Use Environment Variable, fallback to your direct Render URL
      const apiUrl = import.meta.env.VITE_API_URL || 'https://madrlagent.onrender.com';
      
      const response = await fetch(`${apiUrl}/run-simulation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          window_size: params.windowSize,
          transaction_cost: params.cost / 100.0,
          test_split: params.testSplit / 100.0,
          episodes: params.episodes
        })
      });
      
      if (!response.ok) {
         throw new Error(`Simulation failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Transform the arrays from backend into the format Recharts expects
      const formattedData = result.ai_portfolio_history.map((val, index) => ({
        day: `Day ${index + 1}`,
        agent: Math.round(val),
        benchmark: Math.round(result.benchmark_history[index])
      }));
      
      setData(formattedData);
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend or running simulation!");
    } finally {
      setIsSimulating(false);
    }
  };

  const latestData = data[data.length - 1] || { agent: 1000000, benchmark: 1000000 };
  const aiRoi = ((latestData.agent / 1000000) - 1) * 100;
  const benchRoi = ((latestData.benchmark / 1000000) - 1) * 100;
  const excessReturn = aiRoi - benchRoi;

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="brand">
          <BrainCircuit size={28} />
          <span>MADRL Agent</span>
        </div>
        
        <div className="control-group">
          <label>LSTM Memory Window (Days): {params.windowSize}</label>
          <input 
            type="range" min="5" max="60" 
            value={params.windowSize} 
            onChange={(e) => setParams({...params, windowSize: parseInt(e.target.value)})} 
          />
        </div>

        <div className="control-group">
          <label>Transaction Costs (%): {params.cost}</label>
          <input 
            type="number" step="0.05" min="0" max="2" 
            value={params.cost} 
            onChange={(e) => setParams({...params, cost: parseFloat(e.target.value)})} 
          />
        </div>

        <div className="control-group">
          <label>Test Split (%): {params.testSplit}</label>
          <input 
            type="range" min="10" max="50" step="5"
            value={params.testSplit} 
            onChange={(e) => setParams({...params, testSplit: parseInt(e.target.value)})} 
          />
        </div>

        <div className="control-group">
          <label>Training Episodes: {params.episodes}</label>
          <input 
            type="range" min="1" max="20" 
            value={params.episodes} 
            onChange={(e) => setParams({...params, episodes: parseInt(e.target.value)})} 
          />
        </div>

        <button className="run-btn" onClick={handleSimulate} disabled={isSimulating}>
          {isSimulating ? <Activity className="animate-spin" /> : <Play />}
          {isSimulating ? 'Simulating in Cloud...' : 'Run Live Backtest'}
        </button>
      </aside>

      <main className="main-content">
        <header className="header">
          <h1>Execution Terminal</h1>
          <p>Out-of-Sample Performance vs Equal-Weight Benchmark</p>
        </header>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-title"><DollarSign size={16}/> Total Agent Value</div>
            <div className="metric-value">${latestData.agent.toLocaleString()}</div>
            <div className={`metric-sub ${aiRoi < 0 ? 'negative' : ''}`}>
              {aiRoi > 0 ? '+' : ''}{aiRoi.toFixed(2)}% ROI
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-title"><BarChart3 size={16}/> Benchmark Value</div>
            <div className="metric-value">${latestData.benchmark.toLocaleString()}</div>
            <div className={`metric-sub ${benchRoi < 0 ? 'negative' : ''}`}>
              {benchRoi > 0 ? '+' : ''}{benchRoi.toFixed(2)}% ROI
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-title"><TrendingUp size={16}/> Excess Return (Alpha)</div>
            <div className="metric-value">{excessReturn > 0 ? '+' : ''}{excessReturn.toFixed(2)}%</div>
            <div className="metric-sub">Above market average</div>
          </div>
        </div>

        <div className="chart-container">
          <div className="chart-header">
            <h2 className="chart-title">Cumulative Performance</h2>
            <Settings size={20} color="#94a3b8" style={{cursor: 'pointer'}} />
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
                  domain={['dataMin - 50000', 'dataMax + 50000']} 
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
                  activeDot={{ r: 6, fill: '#00ff88', stroke: '#fff' }}
                />
                <Line 
                  type="monotone" 
                  name="Market Benchmark"
                  dataKey="benchmark" 
                  stroke="#ff4444" 
                  strokeWidth={2} 
                  strokeDasharray="5 5"
                  dot={false} 
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
