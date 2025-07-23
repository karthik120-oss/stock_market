#! /usr/bin/env python3
import pandas as pd
import json
from tabulate import tabulate
import yfinance as yf
from datetime import date, timedelta
import traceback
import logging
import numpy as np
from typing import Dict, Tuple



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced Technical Indicators
def calculate_stochastic_rsi(close_prices: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
    """Calculate Stochastic RSI - more sensitive than regular RSI"""
    # First calculate regular RSI
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=rsi_period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Then calculate Stochastic of RSI
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100
    return stoch_rsi

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R - momentum indicator"""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) - measures market volatility"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index (CCI)"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(period).mean()
    mean_deviation = typical_price.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

def calculate_enhanced_signal_confidence(indicators: Dict) -> Tuple[str, float]:
    """Calculate confidence score based on multiple enhanced indicators"""
    signals = []
    
    # Stochastic RSI signals (more sensitive)
    stoch_rsi = indicators.get('stoch_rsi')
    if stoch_rsi is not None and not pd.isna(stoch_rsi):
        if stoch_rsi < 20:
            signals.append(('buy', 0.9))
        elif stoch_rsi < 30:
            signals.append(('buy', 0.7))
        elif stoch_rsi > 80:
            signals.append(('sell', 0.9))
        elif stoch_rsi > 70:
            signals.append(('sell', 0.7))
    
    # Williams %R signals
    williams_r = indicators.get('williams_r')
    if williams_r is not None and not pd.isna(williams_r):
        if williams_r < -80:
            signals.append(('buy', 0.8))
        elif williams_r > -20:
            signals.append(('sell', 0.8))
    
    # CCI signals
    cci = indicators.get('cci')
    if cci is not None and not pd.isna(cci):
        if cci < -100:
            signals.append(('buy', 0.7))
        elif cci > 100:
            signals.append(('sell', 0.7))
    
    # Regular RSI (for confirmation)
    rsi = indicators.get('rsi')
    if rsi is not None and not pd.isna(rsi):
        if rsi < 30:
            signals.append(('buy', 0.6))
        elif rsi > 70:
            signals.append(('sell', 0.6))
    
    # MACD signal
    macd_line = indicators.get('macd')
    macd_signal = indicators.get('macd_signal', 0)
    if macd_line is not None and not pd.isna(macd_line):
        if macd_line > macd_signal:
            signals.append(('buy', 0.7))
        else:
            signals.append(('sell', 0.7))
    
    # Volume confirmation
    volume_trend = indicators.get('volume_trend', '')
    if volume_trend == "Strong Bullish":
        signals.append(('buy', 0.8))
    elif volume_trend == "Strong Bearish":
        signals.append(('sell', 0.8))
    elif volume_trend == "Weak Bullish":
        signals.append(('buy', 0.4))
    elif volume_trend == "Weak Bearish":
        signals.append(('sell', 0.4))
    
    # Calculate consensus
    if not signals:
        return 'hold', 0.5
    
    buy_signals = [conf for sig, conf in signals if sig == 'buy']
    sell_signals = [conf for sig, conf in signals if sig == 'sell']
    
    buy_strength = sum(buy_signals)
    sell_strength = sum(sell_signals)
    total_signals = len(signals)
    
    if buy_strength > sell_strength:
        confidence = min(buy_strength / total_signals, 1.0)
        return 'buy', confidence
    elif sell_strength > buy_strength:
        confidence = min(sell_strength / total_signals, 1.0)
        return 'sell', confidence
    else:
        return 'hold', 0.5

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
    
    # Get full 30 days of data for analysis
    last_30_days_data = analysis_data[-30:].copy()  # Ensure we get exactly 30 days
    
    if len(last_30_days_data) < 30:
        logging.warning(f"Less than 30 days of data available. Using {len(last_30_days_data)} days.")
    
    # Calculate Chaikin Money Flow Multiplier
    high_low = last_30_days_data['High'] - last_30_days_data['Low']
    close_low = last_30_days_data['Close'] - last_30_days_data['Low']
    high_close = last_30_days_data['High'] - last_30_days_data['Close']
    
    # Avoid division by zero
    high_low = high_low.replace(0, 1e-5)
    
    multiplier = ((2 * close_low) - high_close) / high_low
    
    # Calculate Money Flow Volume
    money_flow_volume = multiplier * last_30_days_data['Volume']
    
    # Calculate Chaikin Oscillator using full 30 days
    ema_short = money_flow_volume.ewm(span=5, adjust=False).mean()
    ema_long = money_flow_volume.ewm(span=20, adjust=False).mean()
    chaikin_oscillator = ema_short - ema_long
    
    volume_trend = {
        "Volume Trend": "N/A",
        "Price Trend": "N/A",
        "Chaikin Signal": "N/A",
        "Analysis": "N/A"
    }

    if len(last_30_days_data) >= 2:
        current_chaikin = chaikin_oscillator.iloc[-1].item()
        prev_chaikin = chaikin_oscillator.iloc[-2].item()
        
        # Determine price trend using full 30-day period
        first_half_price = last_30_days_data['Close'].iloc[:15].mean().item()
        second_half_price = last_30_days_data['Close'].iloc[15:].mean().item()
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
def calculate_bollinger_bands(stock_data, window=20, num_std_dev=2):
    stock_data['20_MA'] = stock_data['Close'].rolling(window=window).mean()
    stock_data['20_STD'] = stock_data['Close'].rolling(window=window).std()
    stock_data['Upper Band'] = stock_data['20_MA'] + (stock_data['20_STD'] * num_std_dev)
    stock_data['Lower Band'] = stock_data['20_MA'] - (stock_data['20_STD'] * num_std_dev)
    return stock_data

# Function to get 30-day moving average, RSI, MACD, Bollinger Bands, and current price, then make buy/sell recommendation
def get_stock_recommendation(stock_symbol):
    try:
        # Initialize recommendation to None
        recommendation = None

        # Fetch historical stock data from Yahoo Finance
        stock_data = yf.download(stock_symbol, period="60d", interval="1d", progress=False, auto_adjust=False)

        # Check if stock_data is empty
        if stock_data.empty:
            logging.error(f"No data available for {stock_symbol}. The DataFrame is empty.")
            return {
                "Stock": stock_symbol,
                "Current Price": float('nan'),
                "Last Close": float('nan'),
                "Previous Close": float('nan'),
                "Today Open": float('nan'),
                "30-Day Moving Average": float('nan'),
                "RSI": float('nan'),
                "MACD": float('nan'),
                "Upper Bollinger Band": float('nan'),
                "Lower Bollinger Band": float('nan'),
                "Volume Analysis": "N/A",
                "Recommendation": "HOLD (No Data Available)"
            }

        # Flatten multi-level column names if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]

        # Log the flattened column names for debugging
        #logging.info(f"Flattened stock data columns for {stock_symbol}: {stock_data.columns.tolist()}")

        # Log the structure of the stock_data DataFrame for debugging
        #logging.info(f"Stock data columns for {stock_symbol}: {stock_data.columns.tolist()}")
        # logging.info(f"Stock data sample for {stock_symbol}:\n{stock_data.head()}")

        # Ensure required columns are present in the stock_data DataFrame
        required_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
        for column in required_columns:
            if column not in stock_data.columns:
                logging.error(f"Missing required column '{column}' in stock data for {stock_symbol}")
                return {
                    "Stock": stock_symbol,
                    "Current Price": float('nan'),
                    "Last Close": float('nan'),
                    "Previous Close": float('nan'),
                    "Today Open": float('nan'),
                    "30-Day Moving Average": float('nan'),
                    "RSI": float('nan'),
                    "MACD": float('nan'),
                    "Upper Bollinger Band": float('nan'),
                    "Lower Bollinger Band": float('nan'),
                    "Volume Analysis": "N/A",
                    "Recommendation": "HOLD (Missing Data)"
                }

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
        
        # Calculate Enhanced Technical Indicators
        try:
            stoch_rsi = calculate_stochastic_rsi(analysis_data['Close'])
            stoch_rsi_value = stoch_rsi.iloc[-1] if not stoch_rsi.empty else None
            
            williams_r = calculate_williams_r(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            williams_r_value = williams_r.iloc[-1] if not williams_r.empty else None
            
            atr = calculate_atr(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            atr_value = atr.iloc[-1] if not atr.empty else None
            
            cci = calculate_cci(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            cci_value = cci.iloc[-1] if not cci.empty else None
            
            # Determine volatility level
            volatility_level = "High" if atr_value and atr_value > atr.rolling(20).mean().iloc[-1] else "Normal"
            
        except Exception as e:
            logging.warning(f"Error calculating enhanced indicators for {stock_symbol}: {e}")
            stoch_rsi_value = None
            williams_r_value = None
            atr_value = None
            cci_value = None
            volatility_level = "Unknown"

        # Analyze volume trend
        volume_trend = analyze_volume_trend(stock_data, use_today)

        # Get price data based on availability
        today_open = stock_data['Open'].iloc[-1].item() if use_today else None

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

        # Function to analyze candlestick patterns
        candlestick_pattern = analyze_candlestick_patterns(stock_data)

        # Analyze candlestick patterns for 9-day and 21-day periods
        candlestick_trend = analyze_candlestick_trend(stock_data)

        # Enhanced recommendation logic using confidence scoring
        enhanced_indicators = {
            'stoch_rsi': stoch_rsi_value,
            'williams_r': williams_r_value,
            'cci': cci_value,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': 0,  # Simplified for now
            'volume_trend': volume_trend["Analysis"]
        }
        
        # Get enhanced signal and confidence
        signal_type, confidence = calculate_enhanced_signal_confidence(enhanced_indicators)
        
        # Enhanced recommendation logic with confidence scoring
        if recommendation is None:
            # Priority 1: Candlestick patterns with volume confirmation
            if candlestick_trend == "Bearish Crossover":
                recommendation = f"SELL (Bearish Crossover - Confidence: {confidence:.1%})"
            elif candlestick_trend == "Bullish Crossover":
                recommendation = f"BUY (Bullish Crossover - Confidence: {confidence:.1%})"
            elif candlestick_pattern == "Bearish Engulfing" and volume_trend["Analysis"] in ["Strong Bearish", "Bearish"]:
                recommendation = f"SELL (Bearish Engulfing - Confidence: {confidence:.1%})"
            elif candlestick_pattern == "Bullish Engulfing" and volume_trend["Analysis"] in ["Strong Bullish", "Weak Bullish"]:
                recommendation = f"BUY (Bullish Engulfing - Confidence: {confidence:.1%})"
            
            # Priority 2: Enhanced signal-based recommendations
            elif signal_type == 'buy':
                if confidence > 0.8:
                    recommendation = f"STRONG BUY (Confidence: {confidence:.1%})"
                elif confidence > 0.6:
                    recommendation = f"BUY (Confidence: {confidence:.1%})"
                else:
                    recommendation = f"WEAK BUY (Confidence: {confidence:.1%})"
            elif signal_type == 'sell':
                if confidence > 0.8:
                    recommendation = f"STRONG SELL (Confidence: {confidence:.1%})"
                elif confidence > 0.6:
                    recommendation = f"SELL (Confidence: {confidence:.1%})"
                else:
                    recommendation = f"WEAK SELL (Confidence: {confidence:.1%})"
            
            # Priority 3: Traditional logic with confidence context
            elif last_close < moving_avg_30 and rsi < 30 and macd < 0:
                recommendation = f"SELL (Traditional Signals - Confidence: {confidence:.1%})"
            elif last_close > moving_avg_30 and rsi > 40 and rsi < 70 and macd > 0:
                if volume_trend["Analysis"] == "Strong Bullish":
                    recommendation = f"STRONG BUY (Volume Confirmed - Confidence: {confidence:.1%})"
                else:
                    recommendation = f"BUY (Above MA - Confidence: {confidence:.1%})"
            elif five_day_uptrend and volume_trend["Analysis"] in ["Strong Bullish", "Weak Bullish"]:
                recommendation = f"ACCUMULATE (Uptrend - Confidence: {confidence:.1%})"
            else:
                recommendation = f"HOLD (Mixed Signals - Confidence: {confidence:.1%})"

        if recommendation is None:
            recommendation = f"HOLD (Insufficient Data - Confidence: {confidence:.1%})"

        return {
            "Stock": stock_symbol,
            "Current Price": current_price,  # Updated to use the close price
            "Last Close": last_close,  # Always use the correct last_close value
            "Previous Close": previous_close,
            "Today Open": today_open,
            "30-Day Moving Average": moving_avg_30,
            "RSI": rsi,
            "MACD": macd,
            "Upper Bollinger Band": upper_band,
            "Lower Bollinger Band": lower_band,
            "Volume Analysis": volume_trend["Analysis"],
            "Recommendation": recommendation,
            # Enhanced Technical Indicators
            "Stochastic RSI": stoch_rsi_value if stoch_rsi_value is not None else float('nan'),
            "Williams %R": williams_r_value if williams_r_value is not None else float('nan'),
            "CCI": cci_value if cci_value is not None else float('nan'),
            "ATR": atr_value if atr_value is not None else float('nan'),
            "Volatility Level": volatility_level,
            "Signal Confidence": f"{confidence:.1%}" if 'confidence' in locals() else "N/A"
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
            "30-Day Moving Average": float('nan'),
            "RSI": float('nan'),
            "MACD": float('nan'),
            "Upper Bollinger Band": float('nan'),
            "Lower Bollinger Band": float('nan'),
            "Volume Analysis": "N/A",
            "Recommendation": "HOLD"
        }

# Function to analyze candlestick patterns
def analyze_candlestick_patterns(stock_data):
    """
    Analyze the last two candlesticks for patterns like Bearish Engulfing or Bullish Engulfing.
    Returns "Bearish Engulfing", "Bullish Engulfing", or None.
    """
    if len(stock_data) < 2:
        return None

    # Ensure prev_candle and last_candle are extracted as single rows
    last_candle = stock_data.iloc[-1]
    prev_candle = stock_data.iloc[-2]

    # Convert to dictionaries to avoid Series ambiguity
    last_candle = last_candle.to_dict()
    prev_candle = prev_candle.to_dict()

    # Bearish Engulfing Pattern
    if (
        prev_candle['Close'] > prev_candle['Open'] and  # Previous candle is bullish
        last_candle['Open'] > last_candle['Close'] and  # Last candle is bearish
        last_candle['Open'] > prev_candle['Close'] and  # Last candle opens above previous close
        last_candle['Close'] < prev_candle['Open']      # Last candle closes below previous open
    ):
        return "Bearish Engulfing"

    # Bullish Engulfing Pattern
    if (
        prev_candle['Open'] > prev_candle['Close'] and  # Previous candle is bearish
        last_candle['Close'] > last_candle['Open'] and  # Last candle is bullish
        last_candle['Open'] < prev_candle['Close'] and  # Last candle opens below previous close
        last_candle['Close'] > prev_candle['Open']      # Last candle closes above previous open
    ):
        return "Bullish Engulfing"

    return None

def analyze_candlestick_trend(stock_data):
    """
    Analyze candlestick trends for 9-day and 21-day periods.
    Returns "Bullish Crossover", "Bearish Crossover", or None.
    """
    if len(stock_data) < 21:
        return None

    # Calculate 9-day and 21-day moving averages of the close price
    stock_data['9_MA'] = stock_data['Close'].rolling(window=9).mean()
    stock_data['21_MA'] = stock_data['Close'].rolling(window=21).mean()

    # Check for crossover in the last two days
    if (
        stock_data['9_MA'].iloc[-2] <= stock_data['21_MA'].iloc[-2] and
        stock_data['9_MA'].iloc[-1] > stock_data['21_MA'].iloc[-1]
    ):
        return "Bullish Crossover"

    if (
        stock_data['9_MA'].iloc[-2] >= stock_data['21_MA'].iloc[-2] and
        stock_data['9_MA'].iloc[-1] < stock_data['21_MA'].iloc[-1]
    ):
        return "Bearish Crossover"

    return None

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
    if recommendations:
        try:
            df = pd.DataFrame(recommendations)
            if 'Signal Alert' in df.columns:
                df = df.drop(columns=['Signal Alert'])
            
            # Select only essential columns for clean output
            essential_columns = [
                'Company', 
                'Recommendation', 
                'Current Price'
            ]
            
            # Select only columns that exist in the DataFrame
            available_columns = [col for col in essential_columns if col in df.columns]
            if len(available_columns) > 0 and len(df) > 0:
                df = df.loc[:, available_columns]
                
                # Add a serial number column starting from 1
                df.index = range(1, len(df) + 1)
                df.index.name = 'S.No'
                
                # Print the DataFrame as a table
                print(tabulate(df, headers='keys', tablefmt='grid'))
            else:
                print("No valid recommendations available.")
                return
        except Exception as e:
            print(f"Error processing recommendations: {str(e)}")
            return
    else:
        print("No recommendations generated.")
        return

def analyze_stocks_with_patterns(stock_file):
    # Read the stock symbols from the JSON file
    stocks = read_stock_symbols_from_json(stock_file)
    
    pattern_recommendations = []
    for stock_name, stock_symbol in stocks.items():
        try:
            # Fetch data and analyze patterns
            stock_data = yf.download(stock_symbol, period="60d", interval="1d", progress=False, auto_adjust=False)
            
            if stock_data.empty:
                continue
                
            # Flatten multi-level column names if present
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = [col[0] for col in stock_data.columns]
                
            # Analyze candlestick patterns
            candlestick_pattern = analyze_candlestick_patterns(stock_data)
            candlestick_trend = analyze_candlestick_trend(stock_data)
            
            # Check if stock has any of the specified patterns (only crossovers)
            if candlestick_trend in ["Bearish Crossover", "Bullish Crossover"]:
                
                # Calculate pivot points
                previous_row = stock_data.iloc[-2]
                high = previous_row['High']
                low = previous_row['Low']
                close = previous_row['Close']

                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                s1 = 2 * pivot - high
                r2 = pivot + (high - low)
                s2 = pivot - (high - low)
                r3 = high + 2 * (pivot - low)
                s3 = low - 2 * (high - pivot)
                
                pattern_recommendations.append({
                    "Company": stock_name,
                    "Symbol": stock_symbol,
                    "Pattern": candlestick_trend,
                    "Current Price": stock_data['Close'].iloc[-1],
                    "Pivot": pivot,
                    "R1": r1,
                    "R2": r2,
                    "R3": r3,
                    "S1": s1,
                    "S2": s2,
                    "S3": s3
                })
                
        except Exception as e:
            print(f"Error analyzing {stock_symbol}: {str(e)}")
            continue
    
    if pattern_recommendations:
        print("\n🔍 CROSSOVER PATTERN ANALYSIS")
        print("=" * 50)
        
        for idx, stock in enumerate(pattern_recommendations, 1):
            pattern_emoji = "🔴" if "Bearish" in stock["Pattern"] else "🟢"
            print(f"\n{pattern_emoji} Stock #{idx}: {stock['Company']} ({stock['Symbol']})")
            print("-" * 50)
            print(f"Pattern Detected: {stock['Pattern']}")
            print(f"Current Price: ₹{stock['Current Price']:.2f}")
            print("\n📊 Price Levels:")
            print(f"Pivot Point:  ₹{stock['Pivot']:.2f}")
            print("\n🔺 Resistance Levels:")
            print(f"R1: ₹{stock['R1']:.2f}")
            print(f"R2: ₹{stock['R2']:.2f}")
            print(f"R3: ₹{stock['R3']:.2f}")
            print("\n🔻 Support Levels:")
            print(f"S1: ₹{stock['S1']:.2f}")
            print(f"S2: ₹{stock['S2']:.2f}")
            print(f"S3: ₹{stock['S3']:.2f}")
            
            # Add trading insight based on current price vs pivot
            print("\n📈 Trading Insight:")
            if stock['Current Price'] > stock['Pivot']:
                if "Bearish" in stock["Pattern"]:
                    print("Price is above pivot point but showing bearish crossover - Watch for potential reversal")
                else:
                    print("Price is above pivot point and showing bullish crossover - Uptrend likely to continue")
            else:
                if "Bearish" in stock["Pattern"]:
                    print("Price is below pivot point and showing bearish crossover - Downtrend likely to continue")
                else:
                    print("Price is below pivot point but showing bullish crossover - Watch for potential reversal")
            
            print("\n" + "=" * 50)
    else:
        print("\nNo stocks found with crossover patterns.")

if __name__ == "__main__":
    # Specify the JSON file that contains the stock symbols and company names
    stock_file = "stocks.json"
    
    # Only run the pattern analysis
    analyze_stocks_with_patterns(stock_file)
