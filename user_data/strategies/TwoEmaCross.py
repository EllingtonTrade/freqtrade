from freqtrade.strategy import IStrategy, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import pandas as pd
from typing import Optional

class TwoEmaCrossLongOnly(IStrategy):
    timeframe = '1h'
    can_short = False
    use_custom_stoploss = True
    use_custom_exit = True

    # Strategy Parameters
    fast_ema_period = 15
    slow_ema_period = 30
    atr_period = 30
    sl_coef = 1
    tp_coef = 5
    rsi_period = 14
    rsi_threshold = 65.0

    # Minimal ROI and Stoploss (dummy values as we use custom stoploss)
    minimal_roi = {"0": 100}
    stoploss = -0.99

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_stop = {}


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA Indicators
        dataframe['fast_ema'] = ta.EMA(dataframe['close'], timeperiod=int(self.fast_ema_period))
        dataframe['slow_ema'] = ta.EMA(dataframe['close'], timeperiod=int(self.slow_ema_period))

        # ATR Indicator
        atr = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=int(self.atr_period))
        dataframe['atr'] = atr

        # RSI Indicator
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=int(self.rsi_period))

        # EMA Difference for Crossover Detection
        dataframe['ema_diff'] = dataframe['fast_ema'] - dataframe['slow_ema']
        dataframe['ema_diff_prev'] = dataframe['ema_diff'].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry on EMA crossover (fast crosses above slow) and RSI above threshold
        dataframe['enter_long'] = (
            (dataframe['ema_diff'] > 0) &
            (dataframe['rsi'] > self.rsi_threshold)
        ).astype('int')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit on opposite EMA crossover or price below slow EMA
        dataframe['exit_long'] = (
            (dataframe['ema_diff'] < 0) |
            (dataframe['close'] < dataframe['slow_ema'])
        ).astype('int')

        return dataframe

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        # Access the stored TP from the trade metadata
        tp_price = self.custom_stop.get(pair, {}).get('tp_price', None)

        # Check if TP is set and hit
        if tp_price is not None:
            if current_rate >= tp_price:
                return 'TP hit'  # Signal to close the trade

        return None

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        # Get the analyzed dataframe for the pair and timeframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return -0.99

        # Get the latest data for ATR and Slow EMA
        last_row = dataframe.iloc[-1]
        atr_value = last_row['atr']
        slow_ema = last_row['slow_ema']

        if atr_value == 0 or np.isnan(atr_value) or np.isnan(slow_ema):
            return -0.99

        # Calculate initial SL and TP based on ATR
        sl_price = trade.open_rate - (self.sl_coef * atr_value)
        tp_price = trade.open_rate + (self.tp_coef * atr_value)

        # Dynamic SL based on Slow EMA
        if slow_ema > sl_price:
            sl_price = slow_ema

        # Take profit condition
        if current_rate >= tp_price:
            return 0  # Close the trade when TP is hit

        # Calculate the stop loss percentage
        sl_percentage = (sl_price / current_rate) - 1
        return max(sl_percentage, -0.99)

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float] = None,
                            max_stake: Optional[float] = None, leverage: float = 1.0,
                            entry_tag: Optional[str] = None, side: str = 'long',
                            **kwargs) -> float:
        wallet_balance = self.wallets.get_total(self.config['stake_currency'])
        risk_amount = wallet_balance * (2 / 100)

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        # Get the ATR value
        atr_value = dataframe['atr'].iloc[-1]
        stop_loss_distance = (self.sl_coef * atr_value) / current_rate
        position_size = risk_amount / stop_loss_distance

        # Calculate ATR-based TP and store it in the trade metadata
        tp_price = current_rate + (self.tp_coef * atr_value)
        self.custom_stop[pair] = {'tp_price': tp_price}

        # Apply min and max stake limits
        stake_amount = max(min(position_size, max_stake if max_stake else float('inf')), min_stake if min_stake else 0)
        return stake_amount

    plot_config = {
        'main_plot': {
            'fast_ema': {'color': 'blue'},
            'slow_ema': {'color': 'red'}
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
