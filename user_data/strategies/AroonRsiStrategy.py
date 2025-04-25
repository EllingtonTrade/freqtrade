from freqtrade.strategy import IStrategy, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import pandas as pd
from typing import Optional



import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class AroonRsiStrategy(IStrategy):
    timeframe = '1h'
    can_short = True
    use_custom_stoploss = True
    use_custom_exit = True
    custom_stop = {}

    trailing_sl_threshold = DecimalParameter(0.0, 4.0, decimals=1, default=2.9, space='buy') #0.6
    sl_coef = DecimalParameter(0.0, 4.0, decimals=1, default=2.6, space='buy') #2.2
    tp_threshold = DecimalParameter(0.5, 4.0, decimals=1, default=3.3, space='buy') #3.8
    
    risk_exposure = 0.005

    minimal_roi = {"0": 100}
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
            },
            "buy_sell_plot": {
                "enter_long": {"color": "green", "marker": "v", "markersize": 10},
                "exit_long": {"color": "red", "marker": "^", "markersize": 10},
            },
            "stop_loss_roi_plot": {
                "stop_loss": {"color": "orange", "linewidth": 1, "linestyle": "--"},
                "roi": {"color": "purple", "linewidth": 1, "linestyle": ":"}
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
        dataframe['trailing_sl'] = np.nan

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = (
            (dataframe['rsi'] > 70) &
            (dataframe['aroon_up'] > 80) &
            (dataframe['aroon_down'] < 50)
        ).astype('int')

        dataframe['enter_short'] = (
            (dataframe['rsi'] < 30) &
            (dataframe['aroon_up'] < 50) &
            (dataframe['aroon_down'] > 80)
        ).astype('int')

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

        if trade.is_short:
            if current_profit >= min_profit and current_rate < last_row['bb_lower']:
                return 'short_tp_bb'
        else:
            if current_profit >= min_profit and current_rate > last_row['bb_upper']:
                return 'long_tp_bb'

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

        if trade.is_short:
            sl_price = entry_candle['high'] + self.sl_coef.value * avg_range
        else:
            sl_price = entry_candle['low'] - self.sl_coef.value * avg_range

        profit_trigger = self.trailing_sl_threshold.value * avg_range / trade.open_rate
        if current_profit < profit_trigger:
            sl_percentage = (sl_price / current_rate - 1) if trade.is_short else (sl_price / current_rate) - 1
            return max(sl_percentage, -0.99)

        df = df[df['date'] >= trade.open_date_utc + pd.Timedelta(minutes=self.timeframe_to_minutes(self.timeframe))].copy()
        df['higher_low'] = df['low'] > df['low'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        trailing_candidates = df[df['higher_low']]['prev_low']

        if not trailing_candidates.empty:
            if trade.is_short:
                new_sl = trailing_candidates.min() + (self.sl_coef.value * avg_range)
                if new_sl < sl_price:
                    sl_price = new_sl
            else:
                new_sl = trailing_candidates.max() - (self.sl_coef.value * avg_range)
                if new_sl > sl_price:
                    sl_price = new_sl

        sl_percentage = (sl_price / current_rate - 1) if trade.is_short else (sl_price / current_rate) - 1
        return max(sl_percentage, -0.99)

    def timeframe_to_minutes(self, tf: str) -> int:
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 1440
        return 0



    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float] = None,
                            max_stake: Optional[float] = None, leverage: float = 1.0,
                            entry_tag: Optional[str] = None, side: str = 'long',
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        df = dataframe[dataframe['date'] <= current_time]

        if df.empty:
            logger.debug(f"[{pair}] No data available up to {current_time}. Using proposed stake: {proposed_stake}")
            return proposed_stake

        last_row = df.iloc[-1]
        avg_range = last_row.get('avg_range', None)
        if avg_range is None or avg_range == 0 or np.isnan(avg_range):
            logger.debug(f"[{pair}] Invalid avg_range ({avg_range}). Using proposed stake: {proposed_stake}")
            return proposed_stake

        entry_price = current_rate
        stop_loss_distance = self.sl_coef.value * avg_range
        loss_per_unit = stop_loss_distance / entry_price

        if loss_per_unit <= 0:
            logger.debug(f"[{pair}] Non-positive loss per unit ({loss_per_unit}). Using proposed stake: {proposed_stake}")
            return proposed_stake

        wallet_balance = self.wallets.get_total(self.config['stake_currency'])

        stake_amount = (self.risk_exposure * wallet_balance) / loss_per_unit

        # Respect min/max stake limits
        if min_stake is not None:
            stake_amount = max(stake_amount, min_stake)
        if max_stake is not None:
            stake_amount = min(stake_amount, max_stake)

        logger.debug(
            f"[{pair}] custom_stake_amount calculation:\n"
            f"  Current Time: {current_time}\n"
            f"  Entry Price: {entry_price}\n"
            f"  Avg Range: {avg_range}\n"
            f"  Stop Loss Distance: {stop_loss_distance}\n"
            f"  Loss per Unit: {loss_per_unit}\n"
            f"  Wallet Balance: {wallet_balance}\n"
            f"  Calculated Stake Amount: {stake_amount}\n"
            f"  Min Stake: {min_stake}, Max Stake: {max_stake}"
        )

        return stake_amount


