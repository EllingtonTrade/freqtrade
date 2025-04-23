from freqtrade.strategy import IStrategy, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import pandas as pd


class AroonRsiStrategy(IStrategy):
    timeframe = '1h'
    can_short = True
    use_custom_stoploss = True
    use_custom_exit = True
    custom_stop = {}

    # === Hyperoptable parameters ===
    trailing_sl_threshold = DecimalParameter(0.0, 4.0, decimals=1, default=1.2, space='buy')
    sl_coef = DecimalParameter(0.0, 4.0, decimals=1, default=1.9, space='buy')
    tp_threshold = DecimalParameter(0.5, 4.0, decimals=1, default=1.0, space='buy')
    
    risk_exposure = 0.05 # 1%

    minimal_roi = { "0": 100 }
    stoploss = -0.99

    plot_config = {
        'main_plot': {
            'ema_35': {'color': 'orange', 'linewidth': 1},
            'bb_upper': {'color': 'red', 'linewidth': 1},
            'bb_middle': {'color': 'black', 'linewidth': 1},
            'bb_lower': {'color': 'blue', 'linewidth': 1},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'purple', 'linewidth': 1}
            },
            "Aroon": {
                'aroon_up': {'color': 'green', 'linewidth': 1},
                'aroon_down': {'color': 'red', 'linewidth': 1}
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        aroondown, aroonup = ta.AROON(dataframe['high'], dataframe['low'], timeperiod=14)
        dataframe['aroon_up'] = aroonup
        dataframe['aroon_down'] = aroondown

        bb = ta.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = bb

        dataframe['avg_range'] = (dataframe['high'] - dataframe['low']).rolling(window=30).mean()
        dataframe['higher_low'] = dataframe['low'] > dataframe['low'].shift(1)
        dataframe['bb_break_low'] = dataframe['close'] < dataframe['bb_lower']
        dataframe['trailing_sl'] = np.nan  # Placeholder for plot/debug

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = (
            (dataframe['rsi'] > 70) &
            (dataframe['aroon_up'] > 80) &
            (dataframe['aroon_down'] < 50)
        ).astype('int')
        dataframe['enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df = dataframe[dataframe['date'] <= current_time]

        if df.empty:
            return None

        last_row = df.iloc[-1]
        avg_range = last_row['avg_range']
        if avg_range == 0 or np.isnan(avg_range):
            return None

        min_profit = self.tp_threshold.value * avg_range / trade.open_rate

        if current_profit >= min_profit and last_row['close'] > last_row['bb_upper']:
            return 'bb_tp'

        return None

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df = dataframe[dataframe['date'] <= current_time].copy()

        if df.empty:
            return -0.99

        entry_candle = df[df['date'] <= trade.open_date_utc].iloc[-1]
        avg_range = entry_candle['avg_range']
        if avg_range == 0 or np.isnan(avg_range):
            return -0.99

        sl_price = trade.open_rate - self.sl_coef.value * avg_range

        profit_trigger = self.trailing_sl_threshold.value * avg_range / trade.open_rate
        if current_profit < profit_trigger:
            sl_percentage = (sl_price / current_rate) - 1
            return max(sl_percentage, -0.99)

        df = df[df['date'] >= trade.open_date_utc + pd.Timedelta(minutes=self.timeframe_to_minutes(self.timeframe))].copy()
        df['higher_low'] = df['low'] > df['low'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        trailing_candidates = df[df['higher_low']]['prev_low']

        if not trailing_candidates.empty:
            new_sl = trailing_candidates.max() - (self.sl_coef.value * avg_range)
            if new_sl > sl_price:
                sl_price = new_sl

        sl_percentage = (sl_price / current_rate) - 1
        return max(sl_percentage, -0.99)

    def timeframe_to_minutes(self, tf: str) -> int:
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 1440
        return 0

    def custom_stake_amount(self, pair: str, current_time, current_rate, proposed_stake: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df = dataframe[dataframe['date'] <= current_time]

        if df.empty:
            return proposed_stake

        last_row = df.iloc[-1]
        avg_range = last_row['avg_range']
        if avg_range == 0 or np.isnan(avg_range):
            return proposed_stake

        balance = self.wallets.get_total(self.config['stake_currency'])

        stake = (self.risk_exposure * balance * current_rate) / avg_range
        return stake
