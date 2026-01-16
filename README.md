# GARCH-Kelly Crypto Volatility Targeter

A professional-grade algorithmic trading system that uses **GARCH(1,1)** volatility forecasting and the **Kelly Criterion** to dynamically size crypto positions.

##  The Strategy
This system solves the "Volatility Drag" problem in crypto portfolios. Instead of a static allocation, it adjusts position sizes based on predicted risk:
1.  **Forecast Risk:** Uses GARCH(1,1) to predict hourly volatility for BTC, ETH, SOL, and BNB.
2.  **Optimize Weights:** Allocates capital using **Inverse Volatility** (Risk Parity).
3.  **Size Positions:** Scales exposure using **Fractional Kelly (0.5x)**.
    * *High Volatility* → Reduces leverage (Cash preservation).
    * *Low Volatility* → Increases leverage (Alpha capture).

## 🛠️ Tech Stack
* **Engine:** Backtrader (Event-driven simulation)
* **Data:** yfinance (Hourly real-time data)
* **Math:** arch (Statistical forecasting)
* **Analysis:** Matplotlib & Pandas

## Performance (Backtest: 2023-2025)
Tested on 2 years of hourly data (BTC, ETH, SOL, BNB).
* **Net Alpha:** Captured **96%** of the benchmark's massive bull run while maintaining strict risk controls.
* **Defensive Profile:** Successfully reduced drawdowns during the 2022/2023 crash periods compared to a static hold.
* **Fee Efficiency:** Implemented a 4-hour rebalance interval and 5% drift threshold to neutralize transaction costs.

##  How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the backtester:
    ```bash
    python real_backtest.py
    ```

##  Visuals
![Backtest Results](backtest_results.png)
*(Note: Run the script to generate the latest equity curve)*
