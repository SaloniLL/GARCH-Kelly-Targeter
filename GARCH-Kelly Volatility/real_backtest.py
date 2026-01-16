import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    pass
import matplotlib.pyplot as plt

# --- 1. ROBUST DATA FETCHING ---
def fetch_real_data():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD']
    data_storage = {}
    
    print("\n" + "="*40)
    print(" STARTING DATA DOWNLOAD")
    print("="*40)
    
    for t in tickers:
        print(f"Attempting download for {t}...")
        
        # TRY 1: Hourly Data (Last 730 days is the limit for Yahoo)
        df = yf.download(t, period="730d", interval="1h", progress=False)
        
        # Fallback: If Hourly fails (empty), try Daily data
        if df.empty:
            print(f"   ⚠️ Hourly download failed for {t}. Retrying with Daily data...")
            df = yf.download(t, period="2y", interval="1d", progress=False)
        
        if df.empty:
            print(f"   ❌ CRITICAL ERROR: Could not download {t}. Skipping.")
            continue

        # CLEANING: Handle MultiIndex headers (yfinance update fix)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Try to extract the specific ticker level
                df.columns = df.columns.get_level_values(0)
            except:
                pass
        
        # Rename to lowercase for Backtrader
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        
        # Ensure we have data
        if len(df) < 50:
            print(f"   ⚠️ Warning: Very short data history for {t} ({len(df)} rows).")

        # Calculate Log Returns
        # Use 'Close' or 'close' depending on what remains after cleaning
        c_col = 'close' if 'close' in df.columns else 'Close'
        df['log_return'] = np.log(df[c_col] / df[c_col].shift(1))
        df = df.dropna()
        
        clean_name = t.split('-')[0]
        data_storage[clean_name] = df
        print(f"   ✅ Success: {clean_name} ({len(df)} rows)")
        
    if not data_storage:
        raise ValueError("No data could be downloaded. Check internet connection or yfinance version.")
        
    return data_storage

# --- 2. BACKTRADER SETUP ---
class CryptoPandasData(bt.feeds.PandasData):
    lines = ('log_return',)
    params = (
        ('datetime', None),
        ('open', 'open'), ('high', 'high'), ('low', 'low'), 
        ('close', 'close'), ('volume', 'volume'), ('log_return', 'log_return'),
    )

class PortfolioStrategy(bt.Strategy):
    params = (
        ('kelly_fraction', 0.50),  # KEEP HIGH: Aggressive growth
        ('min_alloc', 0.0),        # Allow cash if volatility explodes
        ('max_alloc', 0.70),       # KEEP HIGH: Capture the bull run
        ('warmup_period', 50),     # Lowered back to 50 since we don't need 200 bars for SMA
        ('drift_threshold', 0.05), # 5% buffer to save fees
        ('rebalance_interval', 4),
    )

    def __init__(self):
        self.tickers = [d._name for d in self.datas]
        self.data_feeds = {d._name: d for d in self.datas}
        # REMOVED: All SMA logic

    def next(self):
        if len(self) % self.p.rebalance_interval != 0 or len(self) < self.p.warmup_period:
            return

        # 1. GARCH FORECASTING (Unchanged)
        vols = {}
        for d in self.datas:
            if len(d) < self.p.warmup_period: continue
            returns = np.array(d.log_return.get(size=self.p.warmup_period)) * 100
            try:
                model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', show_batch=False)
                res = model.fit(disp='off')
                vols[d._name] = np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / 100
            except:
                vols[d._name] = np.std(returns / 100) if len(returns) > 1 else 0.01

        if not vols: return

        # 2. KELLY SIZING
        inv_vols = {n: 1.0/v for n,v in vols.items() if v > 0}
        total_inv = sum(inv_vols.values())
        if total_inv == 0: return
        
        # Calculate Aggressive Weights
        # Note: We removed the Shorting logic, so this is purely Long-Only
        weights = {n: inv/total_inv for n,inv in inv_vols.items()}
        kelly_weights = {n: w * (1.0/(vols[n]**2)) * self.p.kelly_fraction for n,w in weights.items()}
        
        # Apply Constraints
        constrained = {n: max(min(w, self.p.max_alloc), self.p.min_alloc) for n,w in kelly_weights.items()}
        
        # Leverage Check (Allow up to 1.2x leverage if cash permits, otherwise normalize)
        total_exposure = sum(constrained.values())
        if total_exposure > 1.0: 
            constrained = {n: w / total_exposure for n,w in constrained.items()}

        # 3. EXECUTION (Long Only)
        portfolio_val = self.broker.getvalue()
        
        for ticker in self.tickers:
            if ticker not in constrained: continue
            
            data = self.data_feeds[ticker]
            current_val = self.getposition(data).size * data.close[0]
            current_weight = current_val / portfolio_val if portfolio_val > 0 else 0
            
            # Simple Drift Check
            if abs(current_weight - constrained[ticker]) > self.p.drift_threshold:
                self.order_target_percent(data=data, target=constrained[ticker])


class BenchmarkStrategy(bt.Strategy):
    def next(self):
        for d in self.datas:
            self.order_target_percent(d, target=0.25)

def setup_cerebro(data_storage, strategy_class):
    cerebro = bt.Cerebro()
    for ticker, df in data_storage.items():
        cerebro.adddata(CryptoPandasData(dataname=df), name=ticker)
    
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    # [CRITICAL] Enable Margin for Shorting
    cerebro.broker.set_coc(True)       # Cheat-on-Close (Trade at close price)
    cerebro.broker.set_shortcash(False) # Shorting adds cash to balance (standard accounting)
    
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    return cerebro


# --- 3. EXECUTION ---
if __name__ == "__main__":
    # A. GET REAL DATA
    try:
        real_data = fetch_real_data()
    except Exception as e:
        print(f"Data Error: {e}")
        exit()

    # B. RUN STRATEGIES
    print("\nRunning GARCH-Kelly on Real History...")
    res_g = setup_cerebro(real_data, PortfolioStrategy).run()[0]
    
    print("Running Benchmark on Real History...")
    res_b = setup_cerebro(real_data, BenchmarkStrategy).run()[0]

    # C. PROCESS RESULTS
    g_ret = pd.Series(res_g.analyzers.returns.get_analysis())
    b_ret = pd.Series(res_b.analyzers.returns.get_analysis())
    
    if g_ret.empty or b_ret.empty:
        print("Error: No returns generated. Strategy failed to trade.")
        exit()

    g_val = (1 + g_ret).cumprod() * 10000
    b_val = (1 + b_ret).cumprod() * 10000

    # D. REGIME STATS
    bench_vol = b_ret.rolling(24).std().fillna(0)
    high_vol = bench_vol > bench_vol.quantile(0.80)
    
    # Align Indexes to handle different lengths
    common_idx = g_ret.index.intersection(b_ret.index)
    g_aligned = g_ret.loc[common_idx]
    b_aligned = b_ret.loc[common_idx]
    high_vol = high_vol.loc[common_idx]

    def get_sharpe(r):
        if len(r) < 2 or r.std() == 0: return 0.0
        # Annualize: If Hourly (8760), If Daily (252)
        # We detect based on data length. 730d hourly ~ 17000 rows. Daily ~ 730.
        factor = np.sqrt(8760) if len(real_data['BTC']) > 2000 else np.sqrt(365)
        return (r.mean()/r.std()) * factor

    print("\n" + "="*45)
    print(f"{'Regime':<15} | {'GARCH Sharpe':<12} | {'Bench Sharpe':<10}")
    print("-" * 45)
    print(f"{'High Vol (20%)':<15} | {get_sharpe(g_aligned[high_vol]):<12.2f} | {get_sharpe(b_aligned[high_vol]):<10.2f}")
    print(f"{'Low Vol (80%)':<15} | {get_sharpe(g_aligned[~high_vol]):<12.2f} | {get_sharpe(b_aligned[~high_vol]):<10.2f}")
    print("="*45)
    
    print(f"Final Value GARCH: ${g_val.iloc[-1]:.2f}")
    print(f"Final Value Bench: ${b_val.iloc[-1]:.2f}")
    print(f"Net Alpha:         ${g_val.iloc[-1] - b_val.iloc[-1]:.2f}")

    # E. PLOT
    plt.figure(figsize=(12, 6))
    g_val.plot(label='GARCH-Kelly', color='orange')
    b_val.plot(label='Benchmark', color='gray', linestyle='--')
    
    # Shade alpha only if indexes align
    try:
        plt.fill_between(g_val.index, g_val, b_val, where=(g_val > b_val), color='green', alpha=0.1, label='Alpha')
    except:
        pass # Skip shading if index mismatch
        
    plt.title("Real History Backtest")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)