import backtrader as bt
import pandas as pd
import numpy as np
from arch import arch_model
from volatility_models import data_storage 

import matplotlib
try:
    matplotlib.use('TkAgg') 
except:

    pass
import matplotlib.pyplot as plt

class CryptoPandasData(bt.feeds.PandasData):
    lines = ('log_return',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('log_return', 'log_return'),
    )

# --- STEP 6-18: CONSOLIDATED STRATEGY ---
class PortfolioStrategy(bt.Strategy):
    params = (
        ('kelly_fraction', 0.20),
        ('min_alloc', 0.05),
        ('max_alloc', 0.30),
        ('warmup_period', 100),
        ('drift_threshold', 0.05),
        ('rebalance_interval', 8),
    )
    def __init__(self):
        self.total_commissions = 0.0
        self.tickers = [d._name for d in self.datas]
        self.data_feeds = {d._name: d for d in self.datas}
        print(f"Strategy initialized with assets: {', '.join(self.tickers)}")
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt} | {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.total_commissions += order.executed.comm
            if order.isbuy():
                self.log(f'BUY: {order.data._name}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
            else:
                self.log(f'SELL: {order.data._name}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')

    def next(self):
        if len(self) % self.p.rebalance_interval != 0 or len(self) < self.p.warmup_period:
            return

        vols = self.calculate_garch_volatilities()
        inv_weights = self.get_inverse_vol_weights(vols)
        target_weights = self.apply_constraints(self.apply_kelly_scaling(inv_weights, vols))

        portfolio_value = self.broker.getvalue()
        for ticker in self.tickers:
            data = self.data_feeds[ticker]
            pos = self.getposition(data).size

            current_val = pos * data.close[0]
            current_weight = current_val / portfolio_value if portfolio_value > 0 else 0
            
            if abs(current_weight - target_weights[ticker]) > self.p.drift_threshold:
                self.order_target_percent(data=data, target=target_weights[ticker])

    def calculate_garch_volatilities(self):
        vols = {}
        for d in self.datas:
            returns = np.array(d.log_return.get(size=self.p.warmup_period)) * 100
            try:
                model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', show_batch=False)
                res = model.fit(disp='off')
                vols[d._name] = np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / 100
            except:
                vols[d._name] = np.std(returns / 100)
        return vols

    def get_inverse_vol_weights(self, vols):
        inv_vols = {name: 1.0 / v for name, v in vols.items() if v > 0}
        total = sum(inv_vols.values())
        return {name: (inv / total) for name, inv in inv_vols.items()}

    def apply_kelly_scaling(self, weights, vols):
        # Kelly Fractioning
        return {n: w * (1.0 / (vols[n]**2)) * self.p.kelly_fraction for n, w in weights.items()}

    def apply_constraints(self, weights):
        constrained = {n: max(min(w, self.p.max_alloc), self.p.min_alloc) for n, w in weights.items()}
        total = sum(constrained.values())
        return {n: w / total for n, w in constrained.items()}


class BenchmarkStrategy(bt.Strategy):
    def __init__(self):
        self.done = False
    def next(self):
        if not self.done:
            for d in self.datas:
                self.order_target_percent(d, target=0.25)
            self.done = True


def setup_cerebro(data_storage, strategy_class):
    cerebro = bt.Cerebro()
    for ticker, df in data_storage.items():
        # --- DIAGNOSTIC PRINT ---
        print(f"[{ticker}] Original Index Head: {df.index[:2]}")

        # --- THE NUCLEAR FIX ---
        # 1. Check if the index is valid datetime; if not, coerce it
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        if df.index[0].year == 1970:
            print(f"[{ticker}] REPAIRING TIMESTAMPS: 1970 epoch detected.")
            # Create a new index starting from Jan 1, 2025
            clean_index = pd.date_range(start='2025-01-01', periods=len(df), freq='h')
            df.index = clean_index
            print(f"[{ticker}] New Start Date: {df.index[0]}")
        
        # 3. Ensure columns are numeric (Data Cleaning)
        cols_to_fix = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Final drop of bad rows
        df = df.dropna()

        data_feed = CryptoPandasData(dataname=df)
        cerebro.adddata(data_feed, name=ticker)
    
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_coc(True)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='all_returns')
    return cerebro



if __name__ == "__main__":

    print("Running GARCH-Kelly Strategy...")
    cerebro_garch = setup_cerebro(data_storage, PortfolioStrategy)
    results_garch = cerebro_garch.run()
    

    print("Running Benchmark Strategy...")
    cerebro_bench = setup_cerebro(data_storage, BenchmarkStrategy)
    results_bench = cerebro_bench.run()


    garch_hrly = pd.Series(results_garch[0].analyzers.all_returns.get_analysis())
    bench_hrly = pd.Series(results_bench[0].analyzers.all_returns.get_analysis())

    garch_value = (1 + garch_hrly).cumprod() * 10000
    bench_value = (1 + bench_hrly).cumprod() * 10000


    def get_regime_stats(returns):
        returns = returns.dropna()
        if len(returns) < 5 or returns.std() == 0: return {'Sharpe': 0.0}
        sharpe = (returns.mean() / returns.std()) * np.sqrt(8760) 
        return {'Sharpe': sharpe}


    bench_vol = bench_hrly.rolling(24).std().dropna()
    aligned_g = garch_hrly.loc[bench_vol.index]
    aligned_b = bench_hrly.loc[bench_vol.index]
    
    vol_thresh = bench_vol.quantile(0.80) 
    high_vol = bench_vol > vol_thresh

    print("\n" + "="*45)
    print(f"{'Regime':<15} | {'GARCH Sharpe':<12} | {'Bench Sharpe':<10}")
    print("-" * 45)
    print(f"{'High Vol (20%)':<15} | {get_regime_stats(aligned_g[high_vol])['Sharpe']:<12.2f} | {get_regime_stats(aligned_b[high_vol])['Sharpe']:<10.2f}")
    print(f"{'Low Vol (80%)':<15} | {get_regime_stats(aligned_g[~high_vol])['Sharpe']:<12.2f} | {get_regime_stats(aligned_b[~high_vol])['Sharpe']:<10.2f}")
    print("="*45)

    print("\n" + "="*45)
    print(f"{'Final Value':<15} | ${garch_value.iloc[-1]:<11.2f} | ${bench_value.iloc[-1]:<10.2f}")
    print(f"{'Net Alpha':<15} | ${garch_value.iloc[-1] - bench_value.iloc[-1]:<11.2f} | {'N/A':<10}")
    print("="*45)

    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    

    garch_value.plot(label='GARCH-Kelly', color='darkorange', linewidth=2)
    bench_value.plot(label='Benchmark', color='gray', linestyle='--', alpha=0.7)
    

    plt.fill_between(garch_value.index, garch_value, bench_value, 
                     where=(garch_value > bench_value), color='green', alpha=0.1, label='Alpha')

    plt.title("Backtest Comparison: GARCH-Kelly vs Benchmark")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()


    plt.savefig('backtest_results.png')
    print("Plot saved to 'backtest_results.png'")
    
    
    print("Attempting to open plot window...")

    plt.show(block=True)
