from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import pandas as pd

class TwoEmaCrossLongOnly(IStrategy):
    # --- Required settings ---
    minimal_roi = {"0": 1.0}  # Not used, as we apply custom exit
    stoploss = -1.0           # Custom stoploss logic is used
    trailing_stop = False     # Trailing is handled manually
    timeframe = '1h'    
    can_short = False

    # --- Strategy Parameters ---
    fast_ema = 15
    slow_ema = 30
    atr_period = 30
    sl_coef = 1
    tp_coef = 5
    rsi_period = 14
    rsi_threshold = 65

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['ema_fast'] = ta.EMA(df['close'], timeperiod=self.fast_ema)
        df['ema_slow'] = ta.EMA(df['close'], timeperiod=self.slow_ema)
        df['rsi'] = ta.RSI(df['close'], timeperiod=self.rsi_period)

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum.reduce([high_low, high_close, low_close])
        df['atr'] = pd.Series(tr).rolling(window=self.atr_period).mean()

        # Plotting data for TP, SL, entry, and exit
        df['entry'] = np.nan
        df['exit'] = np.nan
        df['tp'] = np.nan
        df['sl'] = np.nan

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        entry_condition = (df['ema_fast'] > df['ema_slow']) & (df['rsi'] > self.rsi_threshold)
        df.loc[entry_condition, 'enter_long'] = 1
        df.loc[entry_condition, 'entry'] = df['close']
        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # We exit manually via custom_exit
        df['exit_long'] = 0
        return df

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return 1

        ema = df.iloc[-1]['ema_slow']
        if current_rate <= ema:
            return 0.01  # trigger stoploss now
        return 1

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None

        if not trade.enter_tag:
            atr = df.iloc[-1]['atr']
            sl = trade.open_rate - self.sl_coef * atr
            tp = trade.open_rate + self.tp_coef * atr
            trade.enter_tag = f"{sl},{tp}"
            df.loc[df.index[-1], 'tp'] = tp
            df.loc[df.index[-1], 'sl'] = sl

        sl, tp = map(float, trade.enter_tag.split(','))
        if current_rate <= sl:
            df.loc[df.index[-1], 'exit'] = current_rate
            return 'atr_stoploss', 0.01
        elif current_rate >= tp:
            df.loc[df.index[-1], 'exit'] = current_rate
            return 'atr_takeprofit', 0.01

        return None
    plot_config = {
    'main_plot': {
            'close': {'color': 'black'},
            'ema_fast': {'color': 'blue'},
            'ema_slow': {'color': 'red'},
            'entry': {'color': 'green', 'type': 'scatter', 'plotly': {'mode': 'markers'}},
            'exit': {'color': 'red', 'type': 'scatter', 'plotly': {'mode': 'markers'}},
            'tp': {'color': 'gold', 'type': 'scatter', 'plotly': {'mode': 'markers'}},
            'sl': {'color': 'orange', 'type': 'scatter', 'plotly': {'mode': 'markers'}}
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'purple'}
            },
            "ATR": {
                'atr': {'color': 'green'}
            }
        }
        }