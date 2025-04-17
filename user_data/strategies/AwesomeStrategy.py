from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
import numpy as np
def view(df, support_lines=None, resist_lines=None):

    import plotly.graph_objects as go

    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure(data=[
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )
    ])

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['support'].where(df['support'] > 0),
        mode='markers',
        name='Support',
        marker=dict(color='green', size=5, symbol='triangle-down')
    ))

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['resistance'].where(df['resistance'] > 0),
        mode='markers',
        name='Resistance',
        marker=dict(color='red', size=5, symbol='triangle-up')
    ))

    # Draw trendlines if provided
    if support_lines:
        for start_date, end_date, (slope, intercept) in support_lines:
            range_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if len(range_df) > 0:
                x_vals = range_df['date']
                x_idx = np.arange(len(x_vals))
                y_vals = slope * x_idx + intercept
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(color='lightblue', dash='dot'),
                    name='Support Trendline'
                ))

    if resist_lines:
        for start_date, end_date, (slope, intercept) in resist_lines:
            range_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if len(range_df) > 0:
                x_vals = range_df['date']
                x_idx = np.arange(len(x_vals))
                y_vals = slope * x_idx + intercept
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(color='orange', dash='dot'),
                    name='Resistance Trendline'
                ))

    fig.update_layout(
        title='Candlestick Chart with Support/Resistance & Trendlines',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=700
    )

    fig.write_html("C:\\Users\\HATEF\\Desktop\\plots\\candlestick_with_sr.html")


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
        "0": 0.1
    }
    stoploss = -0.1

    # Define the timeframe
    timeframe = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add EMA indicators to the dataframe.
        """
        # Calculate EMA7
        dataframe['ema7'] = ta.EMA(dataframe['close'], timeperiod=7)
        # Calculate EMA17
        dataframe['ema17'] = ta.EMA(dataframe['close'], timeperiod=17)
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions: when EMA7 crosses above EMA17.
        """
        dataframe['support'] = np.where(dataframe['low'] == dataframe['low'].rolling(5, center=True).min(), dataframe['low'], 0)

        # Resistance level: Local maxima in the 'High' price over a rolling window
        dataframe['resistance'] = np.where(dataframe['high'] == dataframe['high'].rolling(5, center=True).max(), dataframe['high'], 0)
        
        support_lines, resist_lines= chunk_trendlines(dataframe,180)
        view(dataframe, support_lines=support_lines, resist_lines=resist_lines)
        input("Press any key to continue...")
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['ema7'], dataframe['ema17'])
            ),
            'enter_long'
        ] = 1
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
        return dataframe
