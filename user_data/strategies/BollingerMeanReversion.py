from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class BollingerMeanReversion(IStrategy):
    # Strategy info
    timeframe = '4h'
    minimal_roi = {
        "0": 0.1  # 2% target
    }
    stoploss = -0.1  # 2% stop loss
    risk_reward_ratio = 1

    # Plot BB bands
    plot_config = {
    'main_plot': {
        'bb_upperband': {
            'color': 'red',
            'linewidth': 1
        },
        'bb_middleband': {
            'color': 'black',
            'linewidth': 1
        },
        'bb_lowerband': {
            'color': 'blue',
            'linewidth': 1
        },
        'close': {
            'color': 'grey',
            'linewidth': 1
        },
    },
    'subplots': {
        "RSI": {
            'rsi': {
                'color': 'purple',
                'linewidth': 1
                },
            },
      }
    }

    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate Bollinger Bands
        upper, middle, lower = ta.BBANDS(
        dataframe['close'],
        timeperiod=50,
        nbdevup=2.0,
        nbdevdn=2.0,
        matype=0
)
        dataframe['bb_upperband'] = upper
        dataframe['bb_middleband'] = middle
        dataframe['bb_lowerband'] = lower


        # Optional: RSI for filtering
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=10)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upperband']) |
                (dataframe['rsi'] > 70)
            ),
            'exit_long'] = 1
        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['low'] < dataframe['bb_lowerband']) &
               ( (dataframe['rsi'] >30) &
                (dataframe['rsi']< 70))
            ),
            'enter_long'] = 1
        return dataframe

