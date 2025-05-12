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

    fast_ema_period = DecimalParameter(5, 50, default=10, space='buy')
    slow_ema_period = DecimalParameter(10, 100, default=20, space='buy')
    atr_period = DecimalParameter(10, 50, default=30, space='buy')
    sl_coef = DecimalParameter(0.5, 3.0, default=1.5, space='buy')
    tp_coef = DecimalParameter(1.0, 5.0, default=3.0, space='buy')
    rsi_threshold = DecimalParameter(30.0, 70.0, default=50.0, space='buy')
    rsi_period = DecimalParameter(7, 21, default=14, space='buy')
    trailing_sl_threshold = DecimalParameter(0.5, 3.0, default=1.0, space='buy')

    minimal_roi = {"0": 100}
    stoploss = -0.99

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['fast_ema'] = ta.EMA(dataframe['close'], timeperiod=int(self.fast_ema_period.value))
        dataframe['slow_ema'] = ta.EMA(dataframe['close'], timeperiod=int(self.slow_ema_period.value))
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=int(self.rsi_period.value))
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=int(self.atr_period.value))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = (
            (dataframe['fast_ema'] > dataframe['slow_ema']) &
            (dataframe['rsi'] > self.rsi_threshold.value)
        ).astype('int')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return -0.99

        last_row = dataframe.iloc[-1]
        atr_value = last_row['atr']
        if atr_value == 0 or np.isnan(atr_value):
            return -0.99

        sl_price = trade.open_rate - (self.sl_coef.value * atr_value)
        tp_price = trade.open_rate + (self.tp_coef.value * atr_value)

        if current_profit > 0.01:
            new_sl = current_rate - (self.trailing_sl_threshold.value * atr_value)
            if new_sl > sl_price:
                sl_price = new_sl

        sl_percentage = (sl_price / current_rate) - 1
        return max(sl_percentage, -0.99)
