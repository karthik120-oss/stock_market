import pandas as pd
import json
from tabulate import tabulate
import yfinance as yf
from datetime import date, timedelta
import traceback
import logging

# Constants for EMA periods
VOLUME_EMA_SHORT = 5   # Changed back to 5 days for more responsive volume signals
VOLUME_EMA_LONG = 20   # Changed back to 20 days for volume trend
PRICE_EMA = 30        # Keep 30 days for price trend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read stock symbols from a JSON file
def read_stock_symbols_from_json(filename):
    with open(filename, "r") as file:
        stock_data = json.load(file)
    return stock_data

def is_today_data_available(stock_data):
    today = date.today()
    latest_date = stock_data.index[-1].date()
    return latest_date == today

def get_latest_trading_date(stock_data):
    return stock_data.index[-1].strftime('%Y-%m-%d')

# Function to analyze volume trends
def analyze_volume_trend(stock_data, use_today=True):
    # Filter out weekends
    stock_data = stock_data[stock_data.index.dayofweek < 5]
    
    # Use either latest data or previous day's data
    analysis_data = stock_data if use_today else stock_data[:-1]
    last_30_days_data = analysis_data[-30:].copy()
    
    # Calculate volume EMA with shorter periods for more responsive signals
    volume_ema_short = last_30_days_data['Volume'].ewm(span=VOLUME_EMA_SHORT, adjust=False).mean()
    volume_ema_long = last_30_days_data['Volume'].ewm(span=VOLUME_EMA_LONG, adjust=False).mean()
    
    volume_trend = {
        "Volume Trend": "N/A",
        "Price Trend": "N/A",
        "Volume EMA Signal": "N/A",
        "Analysis": "N/A"
    }

    if len(last_30_days_data) >= 2:
        # Calculate average volume and price for first 15 days and last 15 days
        first_half_volume = last_30_days_data['Volume'].iloc[:15].mean().item()
        second_half_volume = last_30_days_data['Volume'].iloc[15:].mean().item()
        first_half_price = last_30_days_data['Close'].iloc[:15].mean().item()
        second_half_price = last_30_days_data['Close'].iloc[15:].mean().item()

        # Check volume EMA crossover
        current_short_ema = volume_ema_short.iloc[-1].item()
        current_long_ema = volume_ema_long.iloc[-1].item()
        prev_short_ema = volume_ema_short.iloc[-2].item()
        prev_long_ema = volume_ema_long.iloc[-2].item()

        volume_trend["Volume Trend"] = "Increases" if second_half_volume > first_half_volume else "Decreases"
        volume_trend["Price Trend"] = "Increases" if second_half_price > first_half_price else "Decreases"
        
        # Determine EMA signal using scalar values
        if current_short_ema > current_long_ema and prev_short_ema <= prev_long_ema:
            volume_trend["Volume EMA Signal"] = "Bullish Crossover"
        elif current_short_ema < current_long_ema and prev_short_ema >= prev_long_ema:
            volume_trend["Volume EMA Signal"] = "Bearish Crossover"
        else:
            volume_trend["Volume EMA Signal"] = "No Crossover"

        # Enhanced analysis incorporating EMA signals
        if (volume_trend["Volume Trend"] == "Increases" and 
            volume_trend["Price Trend"] == "Increases" and 
            volume_trend["Volume EMA Signal"] in ["Bullish Crossover", "No Crossover"]):
            volume_trend["Analysis"] = "Strong Bullish"
        elif (volume_trend["Volume Trend"] == "Increases" and 
              volume_trend["Price Trend"] == "Decreases"):
            volume_trend["Analysis"] = "Caution – weak hands buying"
        elif (volume_trend["Volume Trend"] == "Decreases" and 
              volume_trend["Price Trend"] == "Increases"):
            volume_trend["Analysis"] = "Bearish"
        elif (volume_trend["Volume Trend"] == "Decreases" and 
              volume_trend["Price Trend"] == "Decreases" and 
              volume_trend["Volume EMA Signal"] in ["Bearish Crossover", "No Crossover"]):
            volume_trend["Analysis"] = "Strong Bearish"
        else:
            volume_trend["Analysis"] = "Neutral"

    return volume_trend

def check_five_day_uptrend(stock_data):
    """Check if the close price is higher than the open price for the past 5 trading days"""
    last_six_days = stock_data[-6:]  # Get last 6 days including today
    past_five_days = last_six_days[:-1]  # Exclude today's price

    return all(past_five_days['Close'].values[i] > past_five_days['Open'].values[i] for i in range(len(past_five_days)))

def detect_bearish_harami(stock_data):
    """
    Detect Bearish Harami pattern in the last two candlesticks
    Returns True if pattern is found, False otherwise
    """
    return False  # Functionality removed

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(stock_data, window=20, num_std_dev=2):
    stock_data['20_MA'] = stock_data['Close'].rolling(window=window).mean()
    stock_data['20_STD'] = stock_data['Close'].rolling(window=window).std()
    stock_data['Upper Band'] = stock_data['20_MA'] + (stock_data['20_STD'] * num_std_dev)
    stock_data['Lower Band'] = stock_data['20_MA'] - (stock_data['20_STD'] * num_std_dev)
    return stock_data

# Function to get 30-day moving average, RSI, MACD, Bollinger Bands, and current price, then make buy/sell recommendation
def get_stock_recommendation(stock_symbol):
    try:
        # Fetch historical stock data from Yahoo Finance
        stock_data = yf.download(stock_symbol, period="60d", interval="1d", progress=False, auto_adjust=False)

        # Filter out weekends
        stock_data = stock_data[stock_data.index.dayofweek < 5]

        # Check if today's data is available and get latest date
        use_today = is_today_data_available(stock_data)
        latest_date = get_latest_trading_date(stock_data)
        
        if use_today:
            last_close = stock_data['Close'].iloc[-1].item()
            previous_close = stock_data['Close'].iloc[-2].item()
        else:
            last_close = stock_data['Close'].iloc[-2].item()
            previous_close = stock_data['Close'].iloc[-3].item()

        # Create a proper copy of the data to avoid SettingWithCopyWarning
        analysis_data = stock_data.copy() if use_today else stock_data[:-1].copy()
        last_30_days_data = analysis_data[-30:].copy()

        # Calculate RSI using exponential moving average
        delta = last_30_days_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()  # Changed to EMA
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()  # Changed to EMA
        rs = gain / loss
        last_30_days_data['RSI'] = 100 - (100 / (1 + rs))
        rsi = last_30_days_data['RSI'].iloc[-1].item()

        # Calculate MACD using EMAs (correct - already using EMA)
        exp1 = analysis_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = analysis_data['Close'].ewm(span=26, adjust=False).mean()
        macd = (exp1 - exp2).iloc[-1].item()

        # Analyze volume trend
        volume_trend = analyze_volume_trend(stock_data, use_today)

        # Get price data based on availability
        if use_today:
            last_close = stock_data['Close'].iloc[-1].item()
            previous_close = stock_data['Close'].iloc[-2].item()
            today_open = stock_data['Open'].iloc[-1].item()
        else:
            last_close = stock_data['Close'].iloc[-2].item()
            previous_close = stock_data['Close'].iloc[-3].item()
            today_open = None

        # Use all data including today for calculations
        last_30_days_data = stock_data[-30:].copy()

        # Calculate 30-day exponential moving average (EMA)
        moving_avg_30 = last_30_days_data['Close'].ewm(span=30, adjust=False).mean().iloc[-1].item()

        # Calculate Bollinger Bands
        stock_data = calculate_bollinger_bands(stock_data)
        upper_band = stock_data['Upper Band'].iloc[-1].item()
        lower_band = stock_data['Lower Band'].iloc[-1].item()

        # Check for 5-day uptrend
        five_day_uptrend = check_five_day_uptrend(stock_data)

        # Determine buy or sell recommendation using last closing price
        recommendation = None
        if last_close < moving_avg_30 and rsi < 30 and macd < 0 and volume_trend["Analysis"] == "Bearish":
            recommendation = "SELL"
        elif last_close > moving_avg_30 and rsi > 40 and rsi < 70 and macd > 0 and volume_trend["Analysis"] == "Bullish":
            recommendation = "BUY"
            if five_day_uptrend:
                recommendation += " (Strong Buy - 5 Day Uptrend)"
        elif five_day_uptrend:
            recommendation = "WATCH (Uptrend Present - Monitor for Buy Signal)"
        else:
            recommendation = "HOLD (No Clear Trend)"

        if recommendation is None:
            recommendation = "HOLD (Lack of clear trend indicators)"

        return {
            "Stock": stock_symbol,
            "Last Close": last_close,
            "Previous Close": previous_close,
            "Today Open": today_open,
            "30-Day Moving Average": moving_avg_30,
            "RSI": rsi,
            "MACD": macd,
            "Upper Bollinger Band": upper_band,
            "Lower Bollinger Band": lower_band,
            "Volume Analysis": volume_trend["Analysis"],
            "Recommendation": recommendation
        }

    except Exception as e:
        logging.error(f"Failed to get data for {stock_symbol}: {e}")
        traceback.print_exc()
        return {
            "Stock": stock_symbol,
            "Last Close": float('nan'),
            "Previous Close": float('nan'),
            "Today Open": float('nan'),
            "30-Day Moving Average": float('nan'),
            "RSI": float('nan'),
            "MACD": float('nan'),
            "Upper Bollinger Band": float('nan'),
            "Lower Bollinger Band": float('nan'),
            "Volume Analysis": "N/A",
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
    
    # Create a DataFrame for better readability
    recommendations_df = pd.DataFrame(recommendations)
    
    # Reorder columns to show only needed columns
    recommendations_df = recommendations_df[['Company', 'Last Close', '30-Day Moving Average', 
                                          'RSI', 'MACD', 'Volume Analysis', 'Recommendation']]
    
    # Add a serial number column starting from 1
    recommendations_df.index = recommendations_df.index + 1
    recommendations_df.index.name = 'S.No'
    
    # Print the DataFrame as a table
    print(tabulate(recommendations_df, headers='keys', tablefmt='grid'))

# Specify the JSON file that contains the stock symbols and company names
stock_file = "stocks.json"  # Change this to your actual file name

# Run the tracking function
track_stocks(stock_file)
