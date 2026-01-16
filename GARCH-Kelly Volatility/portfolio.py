import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from volatility_models import *

def VerifySys(data_storage,res):
    required_tickers = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
    # 1. Check data_storage
    print("--- Data Check ---")
    for ticker in required_tickers:
        if ticker in data_storage:
            rows = len(data_storage[ticker])
            print(f"{ticker} OHLCV: Found ({rows} rows)")
        else:
            print(f"{ticker} OHLCV: Missing!")

    print("\n--- Model Check ---")
    for ticker in required_tickers:
        if ticker in res:
            if hasattr(res[ticker], 'summary'):
                print(f"{ticker} GARCH: Fitted and ready")
            else:
                print(f"{ticker} GARCH: Object found but not fitted correctly")
        else:
            print(f"{ticker} GARCH: Model missing!")

def generate_step1_forecasts(res, data_storage):
    """
    Task 1: Extract variance from fitted GARCH models
    Task 2: Convert variance to volatility (Standard Deviation)
    Task 3: Calculate percentage volatility and USD ranges
    """
    forecast_report = {}
    
    for ticker, model_fit in res.items():

        forecast_obj = model_fit.forecast(horizon=1, reindex=False)
        forecasted_var = forecast_obj.variance.iloc[-1, 0] #
        

        forecasted_vol_raw = np.sqrt(forecasted_var)
        vol_decimal = forecasted_vol_raw / 100 
        
        last_price = data_storage[ticker]['close'].iloc[-1] #
        vol_pct = vol_decimal * 100
        usd_range = last_price * vol_decimal
        
        forecast_report[ticker] = {
            'forecast_vol_pct': vol_pct,
            'expected_move_usd': usd_range,
            'last_price': last_price,
            'upper_band': last_price + usd_range,
            'lower_band': last_price - usd_range
        }
    
    return forecast_report

forecast_report = generate_step1_forecasts(res,data_storage )

def compute_risk_signals(forecast_report):
    """
    Task 1: Calculate Inverse Volatility (1/sigma)
    Task 2: Sum the inverses to find the total signal strength
    Task 3: Normalize signals into percentage weights
    """

    raw_inverses = {ticker: 1 / data['forecast_vol_pct'] 
                    for ticker, data in forecast_report.items()}
    

    total_inverse_vol = sum(raw_inverses.values())
    

    initial_allocations = {}
    for ticker, inv_vol in raw_inverses.items():
        weight = inv_vol / total_inverse_vol
        initial_allocations[ticker] = weight
        
    return initial_allocations, total_inverse_vol

weights, total_inv = compute_risk_signals(forecast_report)

def apply_fractional_kelly(initial_weights, forecast_report, kelly_fraction=0.5):
    """
    Task 1: Set the Kelly fraction (k)
    Task 2: Compute raw Kelly allocation (f* = mu / sigma^2)
    Task 3: Scale initial allocation by the Kelly factor
    """
    kelly_scaled_positions = {}
    
    for ticker, weight in initial_weights.items():

        vol = forecast_report[ticker]['forecast_vol_pct'] / 100
        variance = vol ** 2
        

        raw_kelly_f = 1 / variance

        final_kelly_factor = raw_kelly_f * kelly_fraction
        
        kelly_scaled_positions[ticker] = {
            'initial_weight': weight,
            'kelly_factor': final_kelly_factor,
            'scaled_allocation': weight * final_kelly_factor
        }
        
    return kelly_scaled_positions

kelly_positions = apply_fractional_kelly(weights, forecast_report, kelly_fraction=0.5)

def apply_constraints(kelly_results, min_alloc=0.05, max_alloc=0.40):
    """
    Task 1: Set maximum allocation (cap) per coin
    Task 2: Set minimum allocation (floor) per coin
    Task 3: Re-normalize to sum to 100%
    """

    raw_weights = {ticker: data['scaled_allocation'] for ticker, data in kelly_results.items()}
    total_raw = sum(raw_weights.values())

    constrained_weights = {t: (w / total_raw) for t, w in raw_weights.items()}

    for ticker in constrained_weights:
        if constrained_weights[ticker] > max_alloc:
            constrained_weights[ticker] = max_alloc
        elif constrained_weights[ticker] < min_alloc:
            constrained_weights[ticker] = min_alloc

    new_total = sum(constrained_weights.values())
    for ticker in constrained_weights:
        constrained_weights[ticker] = constrained_weights[ticker] / new_total

    return constrained_weights


final_constrained_weights = apply_constraints(kelly_positions, min_alloc=0.05, max_alloc=0.40)

def compute_portfolio_metrics(final_weights, forecast_report, total_capital=10000):
    """
    Task 1: Compute portfolio volatility (assuming zero correlation)
    Task 2: Compute expected USD range for the total portfolio
    Task 3: Compute 95% Confidence Interval (Value at Risk proxy)
    """

    portfolio_variance = 0
    for ticker, weight in final_weights.items():
        vol = forecast_report[ticker]['forecast_vol_pct'] / 100
        portfolio_variance += (weight**2) * (vol**2)
    
    portfolio_vol = np.sqrt(portfolio_variance)
    portfolio_vol_pct = portfolio_vol * 100

    expected_move_usd = total_capital * portfolio_vol
    

    var_95_usd = expected_move_usd * 1.96

    metrics = {
        'portfolio_vol_pct': portfolio_vol_pct,
        'expected_move_usd': expected_move_usd,
        'var_95_usd': var_95_usd
    }  
    return metrics


portfolio_risk = compute_portfolio_metrics(final_constrained_weights, forecast_report)

def generate_actionable_report(weights, initial_weights, kelly_results, forecast_report):
    """
    Task 1: Assemble all metrics into a structured Pandas DataFrame.
    Task 2: Format and print the consolidated table to the console.
    """
    report_data = []

    for ticker in weights.keys():
        report_data.append({
            'Ticker': ticker,
            'Last Price': forecast_report[ticker]['last_price'],
            'Forecast Vol (%)': forecast_report[ticker]['forecast_vol_pct'],
            'Raw Risk Factor': 1 / forecast_report[ticker]['forecast_vol_pct'],
            'Kelly Allocation (%)': kelly_results[ticker]['scaled_allocation'] * 100,
            'Final Allocation (%)': weights[ticker] * 100
        })


    output_df = pd.DataFrame(report_data)


    print("\n" + "="*85)
    print("STRATEGIC PORTFOLIO ALLOCATION REPORT")
    print("="*85)
    

    formatted_df = output_df.copy()
    formatted_df['Last Price'] = formatted_df['Last Price'].map('${:,.2f}'.format)
    formatted_df['Forecast Vol (%)'] = formatted_df['Forecast Vol (%)'].map('{:.3f}%'.format)
    formatted_df['Raw Risk Factor'] = formatted_df['Raw Risk Factor'].map('{:.4f}'.format)
    formatted_df['Kelly Allocation (%)'] = formatted_df['Kelly Allocation (%)'].map('{:,.2f}%'.format)
    formatted_df['Final Allocation (%)'] = formatted_df['Final Allocation (%)'].map('{:.2f}%'.format)

    print(formatted_df.to_string(index=False))
    print("="*85)
    
    return output_df


final_report = generate_actionable_report(
    final_constrained_weights, 
    weights, 
    kelly_positions, 
    forecast_report
)


def plot_portfolio_dashboard(weights, forecast_report, data_storage, res):
    """
    Task 1: Plot bar chart of suggested allocation per coin
    Task 2: Overlay realized vs forecast volatility for each coin
    Task 3: Highlight USD expected range per coin
    """
    tickers = list(weights.keys())
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)


    ax1 = fig.add_subplot(gs[0, 0])
    alloc_pcts = [weights[t] * 100 for t in tickers]
    bars = ax1.bar(tickers, alloc_pcts, color=['#f2a900', '#3c3c3d', '#00ffbd', '#f3ba2f'], alpha=0.8)
    ax1.set_title("Target Portfolio Allocation (%)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Percentage Weight")
    ax1.set_ylim(0, 50) # To accommodate the 40% cap

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom')

    ax2 = fig.add_subplot(gs[0, 1])
    usd_ranges = [forecast_report[t]['expected_move_usd'] for t in tickers]
    ax2.bar(tickers, usd_ranges, color='tab:red', alpha=0.6)
    ax2.set_title("Expected 1-Hour USD Range (1-Sigma)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("USD Range (+/-)")
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :])
    ticker_to_plot = 'BTC/USDT'
    realized_vol = data_storage[ticker_to_plot]['log_return'].abs().dropna()
    forecast_vol = (res[ticker_to_plot].conditional_volatility / 100).loc[realized_vol.index]
    
    ax3.plot(realized_vol.index[-100:], realized_vol.tail(100), color='silver', label='Realized Vol (|Log Returns|)')
    ax3.plot(forecast_vol.index[-100:], forecast_vol.tail(100), color='tab:orange', label='GARCH Forecast', linewidth=2)
    ax3.set_title(f"Volatility Tracking: {ticker_to_plot} (Last 100 Hours)", fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

plot_portfolio_dashboard(final_constrained_weights, forecast_report, data_storage, res)