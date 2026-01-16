import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import ccxt
from arch import arch_model
import time
tickers = ['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT']
def HistoricalOHLCV(symbols,timeframe='1h',days=90):
    exchange=ccxt.binance({'enableRateLimit':True})
    ohlcv_vault = {}
    for symbol in symbols:
        since=exchange.milliseconds()-(days*24*60*60*1000)
        all_ohlcv=[]
        while True:
            try:
                ohlcv=exchange.fetch_ohlcv(symbol,timeframe,since)
                if not ohlcv:
                    break
                since=ohlcv[-1][0]+1
                all_ohlcv.extend(ohlcv)
                if since >= exchange.milliseconds() or len(ohlcv)<100:
                    break
            except Exception as e:
                print(f'Error: {e}')
                break
        df=pd.DataFrame(all_ohlcv,columns=['timestamp','open','high','low','close','volume'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
        ohlcv_vault[symbol]=df
    return ohlcv_vault
data_storage = HistoricalOHLCV(tickers)
def Wrangler(data_storage,window=24):
    for ticker,df in data_storage.items():
        #Log Returns
        df['log_return']=np.log(df['close']/df['close'].shift(1))
        df['log_return'] = df['log_return'].fillna(0)
        #Rolling std. dev
        df['volatility'] = df['log_return'].rolling(window=window).std()
    return data_storage
data_storage = Wrangler(data_storage,window=24)
def GARCH(data_storage):
    results = {}
    for ticker,df in data_storage.items():
        returns = df['log_return'].dropna()*100
        model = arch_model(returns,p=1,q=1,vol='Garch',dist='normal')
        results[ticker]=model.fit(disp='off')
    return results
res = GARCH(data_storage)
def GARCHForecast(res,data_storage):
    forecast_results = {}
    for ticker,model_fit in res.items():
        forecasts = model_fit.forecast(horizon=1,reindex=False)
        forecasted_var=forecasts.variance.iloc[-1,0]
        forecasted_vol = (forecasted_var ** 0.5)/100
        last_price = data_storage[ticker]['close'].iloc[-1]
        forecast_results[ticker] = {
            'ticker': ticker,
            'forecasted_vol_pct': forecasted_vol * 100,
            'expected_range_usd': last_price * forecasted_vol,
            'last_price': last_price
        }
    return pd.DataFrame(forecast_results)

def RMSE(res,data_storage):
    rmse_results={}
    for ticker,model_fit in res.items():
        forecasted_vol = model_fit.conditional_volatility/100
        actual_returns = data_storage[ticker]['log_return'].abs().dropna()
        forecasted_vol=forecasted_vol.loc[actual_returns.index]
        mse=np.mean((forecasted_vol-actual_returns)**2)
        rmse = np.sqrt(mse)
        rmse_results[ticker]=rmse
    return rmse_results

def PlotResults(res,data_storage):
    tickers = list(res.keys())
    fig,axes=plt.subplots(nrows=len(tickers),ncols=1,figsize=(15,5*len(tickers)))
    if len(tickers) == 1:
        axes = [axes]
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        realized = data_storage[ticker]['log_return'].abs().dropna()
        forecasted=(res[ticker].conditional_volatility/100)
        ax.plot(realized.index, realized, color='silver', label='Realized Vol (|Log Returns|)', alpha=0.6)
        ax.plot(forecasted.index, forecasted, color='tab:red', label='GARCH(1,1) Forecast', linewidth=2)
        ax.set_title(f"{ticker}: Realized vs. Forecasted Volatility", fontsize=14, fontweight='bold')
        ax.set_ylabel("Volatility Magnitude")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

forecast_results = GARCHForecast(res,data_storage)

def GenRiskSignals(forecast_results):
    risk_signals = {}
    raw_weights = {ticker:1/ data['forecasted_vol_pct'] for ticker, data in forecast_results.items()}
    total_weight = sum(raw_weights.values())
    for ticker, weight in raw_weights.items():
        risk_signals[ticker] = {
            'risk_factor_raw':weight,
            'suggested_allocation_pct': (weight/total_weight)*100
        }
    return risk_signals

risk_factors = GenRiskSignals(forecast_results)







    



