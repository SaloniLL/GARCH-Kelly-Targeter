# GARCH-Kelly Crypto Volatility Targeter

A professional-grade **volatility-targeting portfolio system** for crypto assets, using **GARCH(1,1) forecasting** and **fractional Kelly sizing** to dynamically manage risk and exposure.

---

## Overview

This project addresses the **volatility drag problem** in crypto portfolios.
Rather than holding static allocations, the system **adapts position sizes in real time** based on forecasted market risk.

The result is a portfolio that:

* Reduces exposure during volatility expansions
* Scales risk during stable regimes
* Preserves capital during drawdowns
* Remains tradable after fees and slippage

---

## Strategy Design

### 1. Volatility Forecasting

* Uses **rolling GARCH(1,1)** models to forecast **hourly volatility**
* Forecasts are recomputed at each rebalance step to avoid look-ahead bias
* Fallback to realized volatility ensures robustness under model failure

Assets traded:

* BTC
* ETH
* SOL
* BNB

---

### 2. Risk-Based Allocation (Risk Parity)

* Capital is allocated using **Inverse Volatility weighting**
* Lower-risk assets receive higher weight
* Ensures balanced risk contribution across assets

---

### 3. Position Sizing (Fractional Kelly)

* Exposure is scaled using the **Kelly Criterion**
* Fractional Kelly is applied to control drawdowns

Behavior:

* **High volatility → reduced exposure (capital preservation)**
* **Low volatility → increased exposure (growth capture)**

> Kelly is used strictly as a **risk scaler**, not as a directional signal.

---

### 4. Execution & Risk Controls

To ensure realism, multiple safeguards are implemented:

* Transaction costs: **0.1% commission**
* Rebalance interval: **4–8 hours**
* Drift threshold: **5%**
* Allocation constraints:

  * Max allocation per asset
  * Minimum diversification floor

These controls eliminate excessive turnover and fee bleed.

---

## Technology Stack

* **Backtesting Engine:** Backtrader
* **Market Data:** yfinance (hourly crypto data)
* **Volatility Models:** arch (GARCH)
* **Analysis & Visualization:** Pandas, NumPy, Matplotlib

---

## Backtest Results

**Period:** 2023–2025
**Frequency:** Hourly
**Assets:** BTC, ETH, SOL, BNB

### Key Findings

* **Risk-adjusted outperformance:**
  The strategy captured the majority of upside during bullish regimes while reducing drawdowns during crashes.

* **Defensive behavior confirmed:**
  Volatility targeting reduced portfolio variance during high-risk regimes relative to a static equal-weight benchmark.

* **Fee efficiency:**
  Rebalance throttling and drift thresholds prevented transaction costs from eroding returns.

> The system behaves as a **capital-preserving allocator**, not a high-turnover alpha engine.

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the backtester

```bash
python real_backtest.py
```

---

## Project Highlights

* Rolling GARCH forecasting (no look-ahead bias)
* Kelly-based risk scaling (fractional, drawdown-aware)
* Transaction-aware execution logic
* Robust timestamp and data integrity handling
* Honest benchmark comparison (equal-weight buy & hold)

---

## Notes & Transparency

* This system is **risk-first**, not return-maximizing
* Zero-correlation assumptions are used for simplicity
* Designed as a **portfolio risk engine**, not a signal generator

---

