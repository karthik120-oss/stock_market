import pandas as pd
import json
from tabulate import tabulate
import yfinance as yf
from datetime import date, timedelta
import traceback

# Function to read stock symbols from a JSON file
def read_stock_symbols_from_json(filename):
    with open(filename, "r") as file:
        stock_data = json.load(file)
    return stock_data

def check_five_day_uptrend(stock_data):
    """Check if current price is higher than the past 5 trading days"""
    last_six_days = stock_data[-6:]['Close'].values  # Get last 6 days including today
    current_price = last_six_days[-1]
    past_five_days = last_six_days[:-1]  # Exclude today's price
    return all(current_price > price for price in past_five_days)

def detect_bearish_harami(stock_data):
    """
    Detect Bearish Harami pattern in the last two candlesticks
    with confirmation of prior uptrend
    Returns True if pattern is found after uptrend, False otherwise
    """
    if len(stock_data) < 7:  # Need at least 7 days (5 for trend + 2 for pattern)
        return False
        
    # Check if there was an uptrend in the previous 5 days
    last_seven_days = stock_data[-7:].copy()
    prior_five_days = last_seven_days[:-2]['Close'].values
    is_uptrend = all(prior_five_days[i] < prior_five_days[i+1] for i in range(len(prior_five_days)-1))
    
    if not is_uptrend:
        return False
    
    # Get the last two candles
    last_two_days = last_seven_days[-2:].copy()
    
    # First day (previous day)
    prev_open = last_two_days['Open'].iloc[0].item()
    prev_close = last_two_days['Close'].iloc[0].item()
    prev_body = abs(prev_close - prev_open)
    
    # Second day (current day)
    curr_open = last_two_days['Open'].iloc[1].item()
    curr_close = last_two_days['Close'].iloc[1].item()
    curr_body = abs(curr_close - curr_open)
    
    # Bearish Harami conditions after uptrend
    is_bearish_harami = (
        prev_close > prev_open and      # Previous day is bullish
        curr_close < curr_open and      # Current day is bearish
        curr_body < prev_body and       # Current body is smaller
        curr_open <= prev_close and     # Current body is contained
        curr_close >= prev_open and     # within previous body
        is_uptrend                      # Pattern appears after uptrend
    )
    
    return is_bearish_harami

# Function to get 30-day moving average, RSI, MACD, and current price, then make buy/sell recommendation
def get_stock_recommendation(stock_symbol):
    try:
        # Fetch 60 days of data for better RSI calculation
        stock_data = yf.download(stock_symbol, period="60d", interval="1d", progress=False, auto_adjust=False)

        # Filter out weekends
        stock_data = stock_data[stock_data.index.dayofweek < 5]

        # Check if there are enough trading days of data
        if len(stock_data) < 30:  # Still check for 30 days minimum
            return {
                "Stock": stock_symbol,
                "Current Price": float('nan'),
                "Today Open": float('nan'),
                "Previous Close": float('nan'),
                "30-Day MA": float('nan'),
                "RSI": float('nan'),
                "MACD": float('nan'),
                "Recommendation": "HOLD"
            }

        # Calculate 30-day EMA instead of SMA
        moving_avg_30 = stock_data['Close'].ewm(span=30, adjust=False).mean().iloc[-1].item()
        
        # Get current price and other values
        current_price = stock_data['Close'].iloc[-1].item()
        today_open = stock_data['Open'].iloc[-1].item()
        previous_close = stock_data['Close'].iloc[-2].item()
        
        # Calculate RSI using exponential moving averages
        delta = stock_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        # Avoid division by zero
        rs = avg_gain.replace(0, 0.00001) / avg_loss.replace(0, 0.00001)
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        rsi = stock_data['RSI'].iloc[-1].item()

        # Calculate MACD with signal line and histogram
        exp12 = stock_data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = stock_data['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        macd = macd_line.iloc[-1].item()
        macd_signal = signal_line.iloc[-1].item()
        macd_hist = macd_histogram.iloc[-1].item()

        # Check for 5-day uptrend
        five_day_uptrend = check_five_day_uptrend(stock_data)

        # Check for Bearish Harami pattern
        bearish_harami = detect_bearish_harami(stock_data)

        # Determine buy or sell recommendation
        recommendation = None
        if bearish_harami:
            recommendation = "SELL (Bearish Harami Pattern)"
        elif (current_price > moving_avg_30 and
              rsi < 70 and rsi > 40 and
              macd > macd_signal):  # Changed to compare MACD and signal line
            recommendation = "BUY"
            if five_day_uptrend:
                recommendation += " (Strong Buy - 5 Day Uptrend)"
        elif (current_price < moving_avg_30 and
              rsi > 30 and rsi < 60 and
              macd < macd_signal):  # Changed to compare MACD and signal line
            recommendation = "SELL"
        elif five_day_uptrend:
            recommendation = "WATCH (Uptrend Present - Monitor for Buy Signal)"
        else:
            recommendation = "HOLD (No Clear Trend)"

        return {
            "Stock": stock_symbol,
            "Current Price": current_price,
            "Today Open": today_open,
            "Previous Close": previous_close,
            "30-Day MA": moving_avg_30,
            "RSI": rsi,
            "MACD": macd,
            "MACD Signal": macd_signal,
            "MACD Histogram": macd_hist,
            "Recommendation": recommendation
        }

    except Exception as e:
        print(f"Failed to get data for {stock_symbol}: {e}")
        traceback.print_exc()
        return {
            "Stock": stock_symbol,
            "Current Price": float('nan'),
            "Today Open": float('nan'),
            "Previous Close": float('nan'),
            "30-Day MA": float('nan'),
            "RSI": float('nan'),
            "MACD": float('nan'),
            "Recommendation": "HOLD"
        }

# Track each stock and provide recommendations
def track_stocks(stock_file):
    # Read the stock symbols from the JSON file
    stocks = read_stock_symbols_from_json(stock_file)
    
    recommendations = []
    for stock_name, stock_symbol in stocks.items():
        recommendation = get_stock_recommendation(stock_symbol)
        recommendation["Company"] = stock_name  # Include the company name in the output
        recommendations.append(recommendation)
    
    # Print a message after downloading all data
    print("All stock data downloaded successfully.")
    
    # Create a DataFrame for better readability
    recommendations_df = pd.DataFrame(recommendations)
    
    # Reorder columns to include 30-Day MA
    recommendations_df = recommendations_df[['Company', 'Current Price', '30-Day MA',
                                          'RSI', 'MACD', 'Recommendation']]
    
    # Add a serial number column starting from 1
    recommendations_df.index = recommendations_df.index + 1
    recommendations_df.index.name = 'S.No'
    
    # Print the DataFrame as a table
    print(tabulate(recommendations_df, headers='keys', tablefmt='grid'))

# Specify the JSON file that contains the stock symbols and company names
stock_file = "stocks.json"  # Change this to your actual file name

# Run the tracking function
track_stocks(stock_file)
