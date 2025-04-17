from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class SupportResistanceStrategy(IStrategy):
    timeframe = '5m'  
    stoploss = -0.1 

    plot_config = {
        'main_plot': {
            'support': {
                'color': 'green',
                'style': '-',
            },
            'resistance': {
                'color': 'red',
                'style': '-',
            },
        },
    }
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['support'] = ta.SMA(dataframe['close'], timeperiod=50)
        dataframe['resistance'] = ta.SMA(dataframe['close'], timeperiod=200)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] < dataframe['support']),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['resistance']),
            'sell'] = 1
        return dataframe
