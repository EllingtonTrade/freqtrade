from freqtrade.strategy import IStrategy,DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import numpy as np
## HATEF terst strategy
def view(df, output_path=None):
    """
    Visualizes the candlestick chart with EMA lines and entry/exit points.
    
    :param df: DataFrame containing the candlestick data ('date', 'open', 'high', 'low', 'close', 'ema7', 'ema17', 'enter_long', 'exit_long')
    :param output_path: The file path to save the generated chart (if None, the chart won't be saved to a file)
    """
    
    df['date'] = pd.to_datetime(df['date'])

    # Initialize the figure with a candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    )])

    # Add EMA7 line (Blue)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ema7'],
        mode='lines',
        name='EMA7',
        line=dict(color='blue')
    ))

    # Add EMA17 line (Red)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ema17'],
        mode='lines',
        name='EMA17',
        line=dict(color='red')
    ))

    # Add entry points (green triangles down)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'].where(df['enter_long'] == 1),
        mode='markers',
        name='Enter Long',
        marker=dict(color='green', size=7, symbol='triangle-down')
    ))

    # Add exit points (red triangles up)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'].where(df['exit_long'] == 1),
        mode='markers',
        name='Exit Long',
        marker=dict(color='red', size=7, symbol='triangle-up')
    ))

    # Update layout of the chart
    fig.update_layout(
        title='Candlestick Chart with EMA and Trade Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=700
    )

    # Save the plot if output_path is provided
    if output_path:
        fig.write_html(output_path)
        print(f"Chart saved to {output_path}")

    # Show the plot
    fig.show()

# Example usage:
# df should have 'date', 'open', 'high', 'low', 'close', 'ema7', 'ema17', 'enter_long', 'exit_long' columns
# view(df, output_path='candlestick_with_ema_and_trades.html')
import numpy as np  
def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err;


def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared) 
    # coefs[0] = slope,  coefs[1] = intercept 
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
   
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 



def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)

    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)
def chunk_trendlines(df, chunk_size: int):
    """
    Fit trendlines to chunks of data.
    """
    support_lines = []
    resist_lines = []

    for i in range(0, len(df) , chunk_size):
        chunk = df.iloc[i:min(i + chunk_size , len(df))]
        if len(chunk) < chunk_size:
            break
        support_coefs, resist_coefs = fit_trendlines_high_low(chunk['high'].values, chunk['low'].values, chunk['close'].values)
        start_date = chunk['date'].iloc[0]
        end_date = chunk['date'].iloc[-1]
        support_lines.append((start_date, end_date, tuple(support_coefs)))
        resist_lines.append((start_date, end_date, tuple(resist_coefs)))

    return support_lines, resist_lines
   
class AwesomeStrategy(IStrategy):
    """
    EMA Crossover Strategy with EMA lengths of 7 and 17.
    """

    # Define the minimal ROI and stoploss
    minimal_roi = {
        "0": 100.0
    }
    can_short = False
    use_custom_stoploss = True
    use_custom_exit = True
    custom_stop = {}
    trailing_sl_threshold = DecimalParameter(0.0, 4.0, decimals=1, default=1.2, space='buy')
    sl_coef = DecimalParameter(0.0, 4.0, decimals=1, default=1.9, space='buy')
    tp_threshold = DecimalParameter(0.5, 4.0, decimals=1, default=1.0, space='buy')
    
    risk_exposure = 0.05 # 1%

    plot_config = {
        "main_plot": {
            "ema7": {"color": "blue", "linewidth": 2},
            "ema17": {"color": "red", "linewidth": 2},
        },
        "subplots": {
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
    stoploss = -1.0

    # Define the timeframe
    timeframe = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add EMA indicators to the dataframe.
        """
        # Calculate EMA7
        dataframe['ema7'] = ta.EMA(dataframe['close'], timeperiod=3)
        # Calculate EMA17
        dataframe['ema17'] = ta.EMA(dataframe['close'], timeperiod=20)
        dataframe['avg_range'] = (dataframe['high'] - dataframe['low']).rolling(window=30).mean()
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions: when EMA7 crosses above EMA17.
        """
        # dataframe['support'] = np.where(dataframe['low'] == dataframe['low'].rolling(5, center=True).min(), dataframe['low'], 0)

        # # Resistance level: Local maxima in the 'High' price over a rolling window
        # dataframe['resistance'] = np.where(dataframe['high'] == dataframe['high'].rolling(5, center=True).max(), dataframe['high'], 0)
        
        # support_lines, resist_lines= chunk_trendlines(dataframe,180)
        # view(dataframe, support_lines=support_lines, resist_lines=resist_lines)
        # input("Press any key to continue...")
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['ema7'], dataframe['ema17'])
            ),
            'enter_long'
        ] = 1
        # dataframe.to_csv(r"C:\Users\HATEF\Desktop\plots\a.csv", index=False)
  # or dataframe.to_pickle('your_dataframe.pkl')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions: when EMA7 crosses below EMA17.
        """
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['ema7'], dataframe['ema17'])
            ),
            'exit_long'
        ] = 1
        # view(dataframe, output_path="C:\\Users\\HATEF\\Desktop\\plots\\candlestick_with_ema_and_trades.html")
        return dataframe
    
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
