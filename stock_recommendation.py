import pandas as pd
import json
from tabulate import tabulate
import yfinance as yf
from datetime import date, timedelta
import traceback
import logging

# Constants for EMA periods
VOLUME_EMA_SHORT = 25   # Changed back to 25 days for more responsive volume signals
VOLUME_EMA_LONG = 50   # Changed back to 50 days for volume trend
PRICE_EMA = 60        # Keep 60 days for price trend

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
    
    # Get full 60 days of data for analysis
    last_60_days_data = analysis_data[-60:].copy()  # Ensure we get exactly 60 days
    
    if len(last_60_days_data) < 60:
        logging.warning(f"Less than 60 days of data available. Using {len(last_60_days_data)} days.")
    
    # Calculate Chaikin Money Flow Multiplier
    high_low = last_60_days_data['High'] - last_60_days_data['Low']
    close_low = last_60_days_data['Close'] - last_60_days_data['Low']
    high_close = last_60_days_data['High'] - last_60_days_data['Close']
    
    # Avoid division by zero
    high_low = high_low.replace(0, 1e-5)
    
    multiplier = ((2 * close_low) - high_close) / high_low
    
    # Calculate Money Flow Volume
    money_flow_volume = multiplier * last_60_days_data['Volume']
    
    # Calculate Chaikin Oscillator using full 60 days
    ema_short = money_flow_volume.ewm(span=5, adjust=False).mean()
    ema_long = money_flow_volume.ewm(span=20, adjust=False).mean()
    chaikin_oscillator = ema_short - ema_long
    
    volume_trend = {
        "Volume Trend": "N/A",
        "Price Trend": "N/A",
        "Chaikin Signal": "N/A",
        "Analysis": "N/A"
    }

    if len(last_60_days_data) >= 2:
        current_chaikin = chaikin_oscillator.iloc[-1].item()
        prev_chaikin = chaikin_oscillator.iloc[-2].item()
        
        # Determine price trend using full 60-day period
        first_half_price = last_60_days_data['Close'].iloc[:30].mean().item()
        second_half_price = last_60_days_data['Close'].iloc[30:].mean().item()
        volume_trend["Price Trend"] = "Increases" if second_half_price > first_half_price else "Decreases"
        
        # Determine Chaikin signal
        if current_chaikin > 0 and prev_chaikin <= 0:
            volume_trend["Chaikin Signal"] = "Bullish Crossover"
        elif current_chaikin < 0 and prev_chaikin >= 0:
            volume_trend["Chaikin Signal"] = "Bearish Crossover"
        else:
            volume_trend["Chaikin Signal"] = "No Crossover"

        # Volume trend based on Chaikin Oscillator value
        volume_trend["Volume Trend"] = "Increases" if current_chaikin > 0 else "Decreases"

        # Enhanced analysis incorporating Chaikin Oscillator
        if current_chaikin > 0 and volume_trend["Price Trend"] == "Increases":
            volume_trend["Analysis"] = "Strong Bullish"
        elif current_chaikin > 0 and volume_trend["Price Trend"] == "Decreases":
            volume_trend["Analysis"] = "Potential Reversal - High Volume but Declining Price"
        elif current_chaikin < 0 and volume_trend["Price Trend"] == "Increases":
            volume_trend["Analysis"] = "Weak Bullish"
        elif current_chaikin < 0 and volume_trend["Price Trend"] == "Decreases":
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
def calculate_bollinger_bands(stock_data, window=60, num_std_dev=2):
    stock_data['60_MA'] = stock_data['Close'].rolling(window=window).mean()
    stock_data['60_STD'] = stock_data['Close'].rolling(window=window).std()
    stock_data['Upper Band'] = stock_data['60_MA'] + (stock_data['60_STD'] * num_std_dev)
    stock_data['Lower Band'] = stock_data['60_MA'] - (stock_data['60_STD'] * num_std_dev)
    return stock_data

# Function to get 60-day moving average, RSI, MACD, Bollinger Bands, and current price, then make buy/sell recommendation
def get_stock_recommendation(stock_symbol):
    try:
        # Initialize recommendation to None
        recommendation = None

        # Fetch historical stock data from Yahoo Finance
        stock_data = yf.download(stock_symbol, period="90d", interval="1d", progress=False, auto_adjust=False)

        # Use the last close price instead
        current_price = stock_data['Close'].iloc[-1].item()  # Use the last close price

        # Filter out weekends
        stock_data = stock_data[stock_data.index.dayofweek < 5]

        # Log the latest available date
        latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
        today = date.today().strftime('%Y-%m-%d')
        # logging.info(f"Stock: {stock_symbol}")
        # logging.info(f"Latest data date: {latest_date}")
        # logging.info(f"Current date: {today}")

        # Check if data is stale (more than 5 days old)
        if (date.today() - stock_data.index[-1].date()).days > 5:
            logging.warning(f"Warning: Data for {stock_symbol} might be stale. Latest data is from {latest_date}")

        # Check if today's data is available
        use_today = is_today_data_available(stock_data)

        # Determine last_close and previous_close based on data availability
        if use_today:
            last_close = stock_data['Close'].iloc[-1].item()  # Use today's close price
            previous_close = stock_data['Close'].iloc[-2].item()  # Use yesterday's close price
        else:
            last_close = stock_data['Close'].iloc[-2].item()  # Use yesterday's close price
            previous_close = stock_data['Close'].iloc[-3].item()  # Use day before yesterday's close price

        # Create a proper copy of the data to avoid SettingWithCopyWarning
        analysis_data = stock_data.copy() if use_today else stock_data[:-1].copy()
        last_60_days_data = analysis_data[-60:].copy()

        # Calculate RSI using exponential moving average
        delta = last_60_days_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()  # Changed to EMA
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()  # Changed to EMA
        rs = gain / loss
        last_60_days_data['RSI'] = 100 - (100 / (1 + rs))
        rsi = last_60_days_data['RSI'].iloc[-1].item()

        # Calculate MACD using EMAs (correct - already using EMA)
        exp1 = analysis_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = analysis_data['Close'].ewm(span=26, adjust=False).mean()
        macd = (exp1 - exp2).iloc[-1].item()

        # Analyze volume trend
        volume_trend = analyze_volume_trend(stock_data, use_today)

        # Get price data based on availability
        today_open = stock_data['Open'].iloc[-1].item() if use_today else None

        # Use all data including today for calculations
        last_60_days_data = stock_data[-60:].copy()

        # Calculate 60-day exponential moving average (EMA)
        moving_avg_60 = last_60_days_data['Close'].ewm(span=60, adjust=False).mean().iloc[-1].item()

        # Calculate Bollinger Bands
        stock_data = calculate_bollinger_bands(stock_data, window=60)
        upper_band = stock_data['Upper Band'].iloc[-1].item()
        lower_band = stock_data['Lower Band'].iloc[-1].item()

        # Calculate short-term and long-term EMAs for volume
        stock_data['VOLUME_EMA_SHORT'] = stock_data['Volume'].ewm(span=5, adjust=False).mean()
        stock_data['VOLUME_EMA_LONG'] = stock_data['Volume'].ewm(span=20, adjust=False).mean()

        # Check for BUY signal based on EMA crossover
        if stock_data['VOLUME_EMA_SHORT'].iloc[-1] > stock_data['VOLUME_EMA_LONG'].iloc[-1] and \
           stock_data['VOLUME_EMA_SHORT'].iloc[-2] <= stock_data['VOLUME_EMA_LONG'].iloc[-2]:
            recommendation = "BUY (Volume EMA Crossover)"

        # Check for SELL signal based on EMA crossover
        if stock_data['VOLUME_EMA_SHORT'].iloc[-1] < stock_data['VOLUME_EMA_LONG'].iloc[-1] and \
           stock_data['VOLUME_EMA_SHORT'].iloc[-2] >= stock_data['VOLUME_EMA_LONG'].iloc[-2]:
            recommendation = "SELL (Volume EMA Crossover)"

        # Check for 5-day uptrend
        five_day_uptrend = check_five_day_uptrend(stock_data)

        # Enhanced recommendation logic with more specific conditions
        if recommendation is None:
            if last_close < moving_avg_60 and rsi < 30 and macd < 0 and volume_trend["Analysis"] in ["Strong Bearish", "Bearish"]:
                recommendation = "SELL"
            elif last_close > moving_avg_60 and rsi > 40 and rsi < 70 and macd > 0:
                if volume_trend["Analysis"] == "Strong Bullish":
                    recommendation = "STRONG BUY"
                elif volume_trend["Analysis"] == "Weak Bullish":
                    if five_day_uptrend:
                        recommendation = "BUY (Weak Volume but Price Uptrend)"
                    else:
                        recommendation = "ACCUMULATE (Weak Volume Signal)"
            elif last_close > moving_avg_60 and rsi > 30 and macd > 0 and volume_trend["Analysis"] == "Weak Bullish":
                recommendation = "WATCH (Price Above MA with Weak Volume)"
            elif five_day_uptrend and volume_trend["Analysis"] in ["Strong Bullish", "Weak Bullish"]:
                recommendation = "ACCUMULATE (Uptrend with Volume Support)"
            else:
                if last_close > moving_avg_60 and rsi > 45:
                    recommendation = "HOLD (Above MA - Monitor for Strength)"
                else:
                    recommendation = "HOLD (Wait for Clear Signals)"

        if recommendation is None:
            recommendation = "HOLD (Lack of clear trend indicators)"

        return {
            "Stock": stock_symbol,
            "Current Price": current_price,  # Updated to use the close price
            "Last Close": last_close,  # Always use the correct last_close value
            "Previous Close": previous_close,
            "Today Open": today_open,
            "60-Day Moving Average": moving_avg_60,
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
            "Current Price": float('nan'),
            "Last Close": float('nan'),
            "Previous Close": float('nan'),
            "Today Open": float('nan'),
            "60-Day Moving Average": float('nan'),
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
    
    # Create Signal Alert column
    recommendations_df['Signal Alert'] = recommendations_df.apply(
        lambda x: x['Company'] if any(signal in x['Recommendation'] 
        for signal in ['BUY', 'SELL']) else '', axis=1)
    
    # Reorder columns to show only needed columns
    recommendations_df = recommendations_df[['Company', 'Current Price', '60-Day Moving Average', 
                                          'RSI', 'MACD', 'Volume Analysis', 'Recommendation',
                                          'Signal Alert']]
    
    # Add a serial number column starting from 1
    recommendations_df.index = recommendations_df.index + 1
    recommendations_df.index.name = 'S.No'
    
    # Print the DataFrame as a table
    print(tabulate(recommendations_df, headers='keys', tablefmt='grid'))

# Specify the JSON file that contains the stock symbols and company names
stock_file = "stocks.json"  # Change this to your actual file name

# Run the tracking function
track_stocks(stock_file)
