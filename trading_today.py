#! /usr/bin/env python3
"""
Enhanced Intraday Trading Analysis Tool

This script provides comprehensive technical analysis with advanced indicators specifically
optimized for intraday trading. It includes both traditional and cutting-edge indicators
to provide high-accuracy trading signals.

NEW INTRADAY INDICATORS ADDED:
==============================

1. VWAP (Volume Weighted Average Price) - CRITICAL for intraday
   - Shows the true average price weighted by volume
   - Price above VWAP = Bullish, below VWAP = Bearish

2. Supertrend Indicator - EXCELLENT for trend following
   - Uses ATR-based dynamic support/resistance
   - Very reliable for intraday trend direction

3. Money Flow Index (MFI) - Volume-weighted RSI
   - Combines price momentum with volume
   - More reliable than regular RSI for intraday

4. Parabolic SAR - Trend reversal detection
   - Identifies potential trend reversals
   - Excellent for stop-loss placement

5. Multiple RSI Timeframes (5, 9, 14, 21 periods)
   - Faster signals for intraday trading
   - RSI(5) and RSI(9) are very sensitive to short-term moves

6. Ultimate Oscillator - Multi-timeframe momentum
   - Uses 7, 14, and 28-period averages
   - Reduces false signals compared to single-period oscillators

7. Awesome Oscillator - Momentum indicator
   - Shows market momentum changes
   - Good for confirming trend direction

8. Elder's Force Index - Price + Volume combination
   - Measures the power used to move price
   - Positive = buying pressure, Negative = selling pressure

9. MACD Histogram - Better timing than MACD line
   - Shows the difference between MACD and signal line
   - Earlier signals than traditional MACD crossovers

10. Rate of Change (ROC) - Momentum measurement
    - Shows percentage change in price
    - Good for identifying momentum shifts

CONFIDENCE SCORING:
==================
The system now uses a sophisticated confidence scoring algorithm that weighs:
- VWAP position (highest weight for intraday)
- Supertrend direction (very high confidence)
- Multiple RSI timeframes for confirmation
- Volume-based indicators (MFI, Elder's Force Index)
- Trend reversal signals (Parabolic SAR)

Only signals with 60%+ confidence are highlighted in the intraday summary.
"""

import pandas as pd
import json
from tabulate import tabulate
import yfinance as yf
from datetime import date, timedelta
import traceback
import logging
import numpy as np
from typing import Dict, Tuple
import time
import sys
import os
import argparse
import re
import warnings


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced data fetching with fallback options
def fetch_stock_data_robust(symbol, period="25d", max_retries=3, delay_between_requests=0.5):
    """
    Robust stock data fetching with rate limiting, retry strategies and fallback options
    """
    # Add delay to respect rate limits
    time.sleep(delay_between_requests)
    
    for attempt in range(max_retries):
        try:
            # Let yfinance handle its own sessions (no custom session)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, auto_adjust=False, timeout=30)
            
            if not data.empty:
                return data
            else:
                logging.warning(f"Empty data received for {symbol} on attempt {attempt + 1}")
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limiting error
            if "429" in error_msg or "too many requests" in error_msg:
                logging.warning(f"Rate limited for {symbol}, waiting longer...")
                rate_limit_wait = 5 + (attempt * 3)  # Longer wait for rate limits
                time.sleep(rate_limit_wait)
            else:
                logging.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
            
            if attempt < max_retries - 1:
                # Exponential backoff with longer delays for rate limiting
                if "429" in error_msg or "too many requests" in error_msg:
                    wait_time = 10 + (attempt * 5)  # Much longer for rate limits
                else:
                    wait_time = (2 ** attempt) + (attempt * 0.5)
                
                time.sleep(wait_time)
            else:
                logging.error(f"All attempts failed for {symbol}")
    
    # Return empty DataFrame if all attempts failed
    return pd.DataFrame()

# Alternative data source function (placeholder for future implementation)
def fetch_from_alternative_source(symbol):
    """
    Placeholder for alternative data sources like Alpha Vantage, Quandl, etc.
    This can be implemented as a fallback when Yahoo Finance fails
    """
    # This is a placeholder - you can implement alternative APIs here
    return pd.DataFrame()

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

# NEW INTRADAY-FOCUSED INDICATORS
def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP) - crucial for intraday trading"""
    typical_price = (high + low + close) / 3
    price_volume = typical_price * volume
    cumulative_pv = price_volume.cumsum()
    cumulative_volume = volume.cumsum()
    vwap = cumulative_pv / cumulative_volume
    return vwap

def calculate_money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index (MFI) - volume-weighted RSI"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = []
    negative_flow = []
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    
    positive_flow = pd.Series([0] + positive_flow, index=typical_price.index)
    negative_flow = pd.Series([0] + negative_flow, index=typical_price.index)
    
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

def calculate_parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Calculate Parabolic SAR - excellent for intraday trend reversal detection"""
    sar = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    af = af_start
    ep = 0  # Extreme Point
    
    # Initialize
    sar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_increment, af_max)
            
            if sar.iloc[i] > low.iloc[i]:
                trend.iloc[i] = -1
                sar.iloc[i] = ep
                af = af_start
                ep = low.iloc[i]
            else:
                trend.iloc[i] = 1
        else:  # Downtrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_increment, af_max)
            
            if sar.iloc[i] < high.iloc[i]:
                trend.iloc[i] = 1
                sar.iloc[i] = ep
                af = af_start
                ep = high.iloc[i]
            else:
                trend.iloc[i] = -1
    
    return sar

def calculate_awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """Calculate Awesome Oscillator - momentum indicator"""
    median_price = (high + low) / 2
    ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    return ao

def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """Calculate Supertrend indicator - excellent for intraday trend following"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    for i in range(len(close)):
        if i == 0:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if close.iloc[i] <= supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
    
    return supertrend, direction

def calculate_elder_force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Calculate Elder's Force Index - combines price and volume"""
    force_index = (close - close.shift(1)) * volume
    return force_index.ewm(span=period).mean()

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV)"""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    """Calculate Rate of Change (ROC)"""
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    return roc

def calculate_ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                                period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """Calculate Ultimate Oscillator - momentum indicator using multiple timeframes"""
    bp = close - low.rolling(2).min()  # Buying Pressure
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    
    avg7 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
    avg14 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
    avg28 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
    
    uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
    return uo

def calculate_support_resistance_levels(stock_data):
    """
    Calculate support and resistance levels using pivot points methodology
    Returns dictionary with pivot, resistance, and support levels
    """
    try:
        # Use previous day's data for pivot calculation
        previous_row = stock_data.iloc[-2] if len(stock_data) >= 2 else stock_data.iloc[-1]
        high = previous_row['High']
        low = previous_row['Low']
        close = previous_row['Close']
        
        # Calculate pivot point
        pivot = (high + low + close) / 3
        
        # Calculate resistance levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        # Calculate support levels
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3
        }
    except Exception as e:
        logging.warning(f"Error calculating support/resistance levels: {e}")
        return {
            "pivot": None,
            "r1": None,
            "r2": None,
            "r3": None,
            "s1": None,
            "s2": None,
            "s3": None
        }

def calculate_intraday_rsi_multiple(close: pd.Series) -> Dict[str, pd.Series]:
    """Calculate RSI for multiple periods for intraday sensitivity"""
    rsi_periods = {'RSI_5': 5, 'RSI_9': 9, 'RSI_14': 14, 'RSI_21': 21}
    rsi_dict = {}
    
    for name, period in rsi_periods.items():
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs = gain / loss
        rsi_dict[name] = 100 - (100 / (1 + rs))
    
    return rsi_dict

def calculate_macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD with histogram for better signal timing"""
    exp1 = close.ewm(span=fast).mean()
    exp2 = close.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_enhanced_signal_confidence(indicators: Dict) -> Tuple[str, float]:
    """Calculate confidence score based on multiple enhanced indicators - optimized for intraday trading"""
    signals = []
    
    # VWAP signals (critical for intraday)
    current_price = indicators.get('current_price')
    vwap = indicators.get('vwap')
    if current_price is not None and vwap is not None and not pd.isna(vwap):
        if current_price > vwap:
            signals.append(('buy', 0.9))  # High confidence when above VWAP
        else:
            signals.append(('sell', 0.9))  # High confidence when below VWAP
    
    # Supertrend signals (excellent for intraday)
    supertrend_direction = indicators.get('supertrend_direction')
    if supertrend_direction is not None:
        if supertrend_direction == 1:
            signals.append(('buy', 0.95))  # Very high confidence
        elif supertrend_direction == -1:
            signals.append(('sell', 0.95))  # Very high confidence
    
    # Multiple RSI timeframes for faster signals
    rsi_5 = indicators.get('rsi_5')
    rsi_9 = indicators.get('rsi_9')
    rsi_14 = indicators.get('rsi_14')
    
    if rsi_5 is not None and not pd.isna(rsi_5):
        if rsi_5 < 25:
            signals.append(('buy', 0.85))
        elif rsi_5 > 75:
            signals.append(('sell', 0.85))
    
    if rsi_9 is not None and not pd.isna(rsi_9):
        if rsi_9 < 30:
            signals.append(('buy', 0.75))
        elif rsi_9 > 70:
            signals.append(('sell', 0.75))
    
    # Money Flow Index (volume-weighted RSI)
    mfi = indicators.get('mfi')
    if mfi is not None and not pd.isna(mfi):
        if mfi < 20:
            signals.append(('buy', 0.8))
        elif mfi > 80:
            signals.append(('sell', 0.8))
        elif mfi < 30:
            signals.append(('buy', 0.6))
        elif mfi > 70:
            signals.append(('sell', 0.6))
    
    # Parabolic SAR (trend reversal)
    sar_signal = indicators.get('sar_signal')
    if sar_signal is not None:
        if sar_signal == 'bullish_reversal':
            signals.append(('buy', 0.9))
        elif sar_signal == 'bearish_reversal':
            signals.append(('sell', 0.9))
    
    # MACD Histogram for timing
    macd_histogram = indicators.get('macd_histogram')
    macd_prev_histogram = indicators.get('macd_prev_histogram')
    if macd_histogram is not None and macd_prev_histogram is not None:
        if macd_histogram > 0 and macd_prev_histogram <= 0:
            signals.append(('buy', 0.8))  # Bullish crossover
        elif macd_histogram < 0 and macd_prev_histogram >= 0:
            signals.append(('sell', 0.8))  # Bearish crossover
    
    # Ultimate Oscillator
    uo = indicators.get('ultimate_oscillator')
    if uo is not None and not pd.isna(uo):
        if uo < 30:
            signals.append(('buy', 0.7))
        elif uo > 70:
            signals.append(('sell', 0.7))
    
    # Awesome Oscillator
    ao = indicators.get('awesome_oscillator')
    ao_prev = indicators.get('ao_prev')
    if ao is not None and ao_prev is not None:
        if ao > 0 and ao_prev <= 0:
            signals.append(('buy', 0.75))
        elif ao < 0 and ao_prev >= 0:
            signals.append(('sell', 0.75))
    
    # Elder's Force Index
    efi = indicators.get('elder_force_index')
    if efi is not None and not pd.isna(efi):
        if efi > 0:
            signals.append(('buy', 0.6))
        elif efi < 0:
            signals.append(('sell', 0.6))
    
    # Rate of Change
    roc = indicators.get('rate_of_change')
    if roc is not None and not pd.isna(roc):
        if roc > 2:
            signals.append(('buy', 0.65))
        elif roc < -2:
            signals.append(('sell', 0.65))
    
    # Stochastic RSI signals (more sensitive)
    stoch_rsi = indicators.get('stoch_rsi')
    if stoch_rsi is not None and not pd.isna(stoch_rsi):
        if stoch_rsi < 20:
            signals.append(('buy', 0.7))
        elif stoch_rsi < 30:
            signals.append(('buy', 0.5))
        elif stoch_rsi > 80:
            signals.append(('sell', 0.7))
        elif stoch_rsi > 70:
            signals.append(('sell', 0.5))
    
    # Williams %R signals
    williams_r = indicators.get('williams_r')
    if williams_r is not None and not pd.isna(williams_r):
        if williams_r < -80:
            signals.append(('buy', 0.6))
        elif williams_r > -20:
            signals.append(('sell', 0.6))
    
    # CCI signals
    cci = indicators.get('cci')
    if cci is not None and not pd.isna(cci):
        if cci < -100:
            signals.append(('buy', 0.6))
        elif cci > 100:
            signals.append(('sell', 0.6))
    
    # Volume confirmation
    volume_trend = indicators.get('volume_trend', '')
    if volume_trend == "Strong Bullish":
        signals.append(('buy', 0.7))
    elif volume_trend == "Strong Bearish":
        signals.append(('sell', 0.7))
    elif volume_trend == "Weak Bullish":
        signals.append(('buy', 0.3))
    elif volume_trend == "Weak Bearish":
        signals.append(('sell', 0.3))
    
    # Calculate consensus with improved weighting
    if not signals:
        return 'hold', 0.5
    
    buy_signals = [conf for sig, conf in signals if sig == 'buy']
    sell_signals = [conf for sig, conf in signals if sig == 'sell']
    
    buy_strength = sum(buy_signals)
    sell_strength = sum(sell_signals)
    total_signals = len(signals)
    
    # Enhanced confidence calculation
    net_strength = abs(buy_strength - sell_strength)
    max_possible_strength = max(buy_strength + sell_strength, 1)
    
    if buy_strength > sell_strength:
        confidence = min((net_strength / max_possible_strength) * 1.2, 1.0)
        return 'buy', confidence
    elif sell_strength > buy_strength:
        confidence = min((net_strength / max_possible_strength) * 1.2, 1.0)
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
    
    # Get 20 days of data for analysis (optimized for intraday trading)
    last_20_days_data = analysis_data[-20:].copy()  # More responsive for intraday
    
    if len(last_20_days_data) < 20:
        logging.warning(f"Less than 20 days of data available. Using {len(last_20_days_data)} days.")
    
    # Calculate Chaikin Money Flow Multiplier
    high_low = last_20_days_data['High'] - last_20_days_data['Low']
    close_low = last_20_days_data['Close'] - last_20_days_data['Low']
    high_close = last_20_days_data['High'] - last_20_days_data['Close']
    
    # Avoid division by zero
    high_low = high_low.replace(0, 1e-5)
    
    multiplier = ((2 * close_low) - high_close) / high_low
    
    # Calculate Money Flow Volume
    money_flow_volume = multiplier * last_20_days_data['Volume']
    
    # Calculate Chaikin Oscillator using 20 days (optimized for intraday)
    ema_short = money_flow_volume.ewm(span=5, adjust=False).mean()
    ema_long = money_flow_volume.ewm(span=15, adjust=False).mean()  # Reduced from 20 to 15
    chaikin_oscillator = ema_short - ema_long
    
    volume_trend = {
        "Volume Trend": "N/A",
        "Price Trend": "N/A",
        "Chaikin Signal": "N/A",
        "Analysis": "N/A"
    }

    if len(last_20_days_data) >= 2:
        current_chaikin = chaikin_oscillator.iloc[-1].item()
        prev_chaikin = chaikin_oscillator.iloc[-2].item()
        
        # Determine price trend using 20-day period (split into two 10-day periods)
        first_half_price = last_20_days_data['Close'].iloc[:10].mean().item()
        second_half_price = last_20_days_data['Close'].iloc[10:].mean().item()
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

# Function to get 20-day moving average, RSI, MACD, Bollinger Bands, and current price, then make buy/sell recommendation
def get_stock_recommendation(stock_symbol):
    try:
        # Initialize recommendation to None
        recommendation = None

        # Fetch historical stock data with robust error handling
        stock_data = fetch_stock_data_robust(stock_symbol, period="60d", delay_between_requests=2.0)
        
        # If no data available, try alternative source
        if stock_data.empty:
            stock_data = fetch_from_alternative_source(stock_symbol)

        # Check if stock_data is empty
        if stock_data.empty:
            logging.error(f"No data available for {stock_symbol}. The DataFrame is empty.")
            return {
                "Stock": stock_symbol,
                "Current Price": float('nan'),
                "Last Close": float('nan'),
                "Previous Close": float('nan'),
                "Today Open": float('nan'),
                "20-Day Moving Average": float('nan'),
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
                    "20-Day Moving Average": float('nan'),
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
        last_20_days_data = analysis_data[-20:].copy()

        # Calculate RSI using exponential moving average (optimized for intraday)
        delta = last_20_days_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()  # Changed to EMA
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()  # Changed to EMA
        rs = gain / loss
        last_20_days_data['RSI'] = 100 - (100 / (1 + rs))
        rsi = last_20_days_data['RSI'].iloc[-1].item()

        # Calculate MACD using EMAs (correct - already using EMA)
        exp1 = analysis_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = analysis_data['Close'].ewm(span=26, adjust=False).mean()
        macd = (exp1 - exp2).iloc[-1].item()
        
        # Calculate Enhanced Technical Indicators
        try:
            # Original indicators
            stoch_rsi = calculate_stochastic_rsi(analysis_data['Close'])
            stoch_rsi_value = stoch_rsi.iloc[-1] if not stoch_rsi.empty else None
            
            williams_r = calculate_williams_r(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            williams_r_value = williams_r.iloc[-1] if not williams_r.empty else None
            
            atr = calculate_atr(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            atr_value = atr.iloc[-1] if not atr.empty else None
            
            cci = calculate_cci(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            cci_value = cci.iloc[-1] if not cci.empty else None
            
            # NEW INTRADAY INDICATORS
            # VWAP - Critical for intraday trading
            vwap = calculate_vwap(analysis_data['High'], analysis_data['Low'], analysis_data['Close'], analysis_data['Volume'])
            vwap_value = vwap.iloc[-1] if not vwap.empty else None
            
            # Money Flow Index - Volume-weighted RSI
            mfi = calculate_money_flow_index(analysis_data['High'], analysis_data['Low'], analysis_data['Close'], analysis_data['Volume'])
            mfi_value = mfi.iloc[-1] if not mfi.empty else None
            
            # Parabolic SAR - Trend reversal detection
            sar = calculate_parabolic_sar(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            sar_value = sar.iloc[-1] if not sar.empty else None
            sar_signal = None
            if len(sar) >= 2:
                if current_price > sar_value and analysis_data['Close'].iloc[-2] <= sar.iloc[-2]:
                    sar_signal = 'bullish_reversal'
                elif current_price < sar_value and analysis_data['Close'].iloc[-2] >= sar.iloc[-2]:
                    sar_signal = 'bearish_reversal'
            
            # Supertrend - Excellent for intraday trend following
            supertrend, supertrend_direction = calculate_supertrend(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            supertrend_value = supertrend.iloc[-1] if not supertrend.empty else None
            supertrend_dir = supertrend_direction.iloc[-1] if not supertrend_direction.empty else None
            
            # Awesome Oscillator - Momentum indicator
            ao = calculate_awesome_oscillator(analysis_data['High'], analysis_data['Low'])
            ao_value = ao.iloc[-1] if not ao.empty else None
            ao_prev = ao.iloc[-2] if len(ao) >= 2 else None
            
            # Elder's Force Index - Price and volume combination
            efi = calculate_elder_force_index(analysis_data['Close'], analysis_data['Volume'])
            efi_value = efi.iloc[-1] if not efi.empty else None
            
            # On-Balance Volume
            obv = calculate_obv(analysis_data['Close'], analysis_data['Volume'])
            obv_value = obv.iloc[-1] if not obv.empty else None
            
            # Rate of Change
            roc = calculate_rate_of_change(analysis_data['Close'])
            roc_value = roc.iloc[-1] if not roc.empty else None
            
            # Ultimate Oscillator
            uo = calculate_ultimate_oscillator(analysis_data['High'], analysis_data['Low'], analysis_data['Close'])
            uo_value = uo.iloc[-1] if not uo.empty else None
            
            # Multiple RSI timeframes for intraday sensitivity
            rsi_multiple = calculate_intraday_rsi_multiple(analysis_data['Close'])
            rsi_5_value = rsi_multiple['RSI_5'].iloc[-1] if not rsi_multiple['RSI_5'].empty else None
            rsi_9_value = rsi_multiple['RSI_9'].iloc[-1] if not rsi_multiple['RSI_9'].empty else None
            rsi_14_value = rsi_multiple['RSI_14'].iloc[-1] if not rsi_multiple['RSI_14'].empty else None
            rsi_21_value = rsi_multiple['RSI_21'].iloc[-1] if not rsi_multiple['RSI_21'].empty else None
            
            # MACD with Histogram for better timing
            macd_line, macd_signal_line, macd_histogram = calculate_macd_histogram(analysis_data['Close'])
            macd_histogram_value = macd_histogram.iloc[-1] if not macd_histogram.empty else None
            macd_prev_histogram = macd_histogram.iloc[-2] if len(macd_histogram) >= 2 else None
            
            # Determine volatility level
            volatility_level = "High" if atr_value and atr_value > atr.rolling(20).mean().iloc[-1] else "Normal"
            
        except Exception as e:
            logging.warning(f"Error calculating enhanced indicators for {stock_symbol}: {e}")
            # Set default values for all indicators
            stoch_rsi_value = None
            williams_r_value = None
            atr_value = None
            cci_value = None
            vwap_value = None
            mfi_value = None
            sar_value = None
            sar_signal = None
            supertrend_value = None
            supertrend_dir = None
            ao_value = None
            ao_prev = None
            efi_value = None
            obv_value = None
            roc_value = None
            uo_value = None
            rsi_5_value = None
            rsi_9_value = None
            rsi_14_value = None
            rsi_21_value = None
            macd_histogram_value = None
            macd_prev_histogram = None
            volatility_level = "Unknown"

        # Analyze volume trend
        volume_trend = analyze_volume_trend(stock_data, use_today)

        # Get price data based on availability
        today_open = stock_data['Open'].iloc[-1].item() if use_today else None

        # Use all data including today for calculations
        last_20_days_data = stock_data[-20:].copy()

        # Calculate 20-day exponential moving average (EMA) - better for intraday
        moving_avg_20 = last_20_days_data['Close'].ewm(span=20, adjust=False).mean().iloc[-1].item()

        # Calculate Bollinger Bands
        stock_data = calculate_bollinger_bands(stock_data)
        upper_band = stock_data['Upper Band'].iloc[-1].item()
        lower_band = stock_data['Lower Band'].iloc[-1].item()

        # Check for 5-day uptrend
        five_day_uptrend = check_five_day_uptrend(stock_data)

        # Function to analyze candlestick patterns
        candlestick_pattern = analyze_candlestick_patterns(stock_data)



        # Enhanced recommendation logic using confidence scoring with all intraday indicators
        enhanced_indicators = {
            # VWAP and current price (critical for intraday)
            'current_price': current_price,
            'vwap': vwap_value,
            
            # Supertrend (excellent for intraday)
            'supertrend_direction': supertrend_dir,
            
            # Multiple RSI timeframes
            'rsi_5': rsi_5_value,
            'rsi_9': rsi_9_value,
            'rsi_14': rsi_14_value,
            'rsi_21': rsi_21_value,
            
            # Money Flow Index
            'mfi': mfi_value,
            
            # Parabolic SAR
            'sar_signal': sar_signal,
            
            # MACD Histogram
            'macd_histogram': macd_histogram_value,
            'macd_prev_histogram': macd_prev_histogram,
            
            # Ultimate Oscillator
            'ultimate_oscillator': uo_value,
            
            # Awesome Oscillator
            'awesome_oscillator': ao_value,
            'ao_prev': ao_prev,
            
            # Elder's Force Index
            'elder_force_index': efi_value,
            
            # Rate of Change
            'rate_of_change': roc_value,
            
            # Original indicators
            'stoch_rsi': stoch_rsi_value,
            'williams_r': williams_r_value,
            'cci': cci_value,
            'rsi': rsi,
            'macd': macd,
            'volume_trend': volume_trend["Analysis"]
        }
        
        # Get enhanced signal and confidence
        signal_type, confidence = calculate_enhanced_signal_confidence(enhanced_indicators)
        
        # Enhanced recommendation logic with confidence scoring
        if recommendation is None:
            # Priority 1: Candlestick patterns with volume confirmation
            if candlestick_pattern == "Bearish Engulfing" and volume_trend["Analysis"] in ["Strong Bearish", "Bearish"]:
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
            elif last_close < moving_avg_20 and rsi < 30 and macd < 0:
                recommendation = f"SELL (Traditional Signals - Confidence: {confidence:.1%})"
            elif last_close > moving_avg_20 and rsi > 40 and rsi < 70 and macd > 0:
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
            "Current Price": current_price,
            "Last Close": last_close,
            "Previous Close": previous_close,
            "Today Open": today_open,
            "20-Day Moving Average": moving_avg_20,
            "Recommendation": recommendation,
            "Signal Confidence": f"{confidence:.1%}" if 'confidence' in locals() else "N/A",
            
            # Key Intraday Indicators (Most Important)
            "VWAP": vwap_value if vwap_value is not None else float('nan'),
            "Supertrend": supertrend_value if supertrend_value is not None else float('nan'),
            "Supertrend Signal": "Bullish" if supertrend_dir == 1 else "Bearish" if supertrend_dir == -1 else "N/A",
            "Money Flow Index": mfi_value if mfi_value is not None else float('nan'),
            "Parabolic SAR": sar_value if sar_value is not None else float('nan'),
            "SAR Signal": sar_signal if sar_signal else "No Signal",
            
            # Multiple RSI Timeframes
            "RSI (5-period)": rsi_5_value if rsi_5_value is not None else float('nan'),
            "RSI (9-period)": rsi_9_value if rsi_9_value is not None else float('nan'),
            "RSI (14-period)": rsi_14_value if rsi_14_value is not None else float('nan'),
            "RSI (21-period)": rsi_21_value if rsi_21_value is not None else float('nan'),
            
            # Advanced Momentum Indicators
            "Ultimate Oscillator": uo_value if uo_value is not None else float('nan'),
            "Awesome Oscillator": ao_value if ao_value is not None else float('nan'),
            "Elder Force Index": efi_value if efi_value is not None else float('nan'),
            "Rate of Change": roc_value if roc_value is not None else float('nan'),
            "MACD Histogram": macd_histogram_value if macd_histogram_value is not None else float('nan'),
            
            # Volume Indicators
            "On-Balance Volume": obv_value if obv_value is not None else float('nan'),
            "Volume Analysis": volume_trend["Analysis"],
            
            # Traditional Indicators
            "RSI": rsi,
            "MACD": macd,
            "Upper Bollinger Band": upper_band,
            "Lower Bollinger Band": lower_band,
            "Stochastic RSI": stoch_rsi_value if stoch_rsi_value is not None else float('nan'),
            "Williams %R": williams_r_value if williams_r_value is not None else float('nan'),
            "CCI": cci_value if cci_value is not None else float('nan'),
            "ATR": atr_value if atr_value is not None else float('nan'),
            "Volatility Level": volatility_level
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



# NEW: Intraday Trading Summary Function
def get_intraday_trading_summary(stock_file):
    """Generate focused intraday trading recommendations with key signals"""
    stocks = read_stock_symbols_from_json(stock_file)
    
    print(f"📊 Analyzing {len(stocks)} stocks with fresh data and rate limiting...")
    print("⏱️  This may take a few minutes to fetch and analyze all stocks...")
    
    intraday_recommendations = []
    for idx, (stock_name, stock_symbol) in enumerate(stocks.items(), 1):
        try:
            recommendation_data = get_stock_recommendation(stock_symbol)
            
            # Only include stocks with strong signals for intraday trading
            confidence_str = recommendation_data.get("Signal Confidence", "0%")
            confidence_val = float(confidence_str.replace("%", "")) / 100 if confidence_str != "N/A" else 0
            
            if confidence_val >= 0.6:  # Only show high-confidence signals
                # Calculate support and resistance levels (reuse data from recommendation if possible)
                try:
                    stock_data = fetch_stock_data_robust(stock_symbol, period="60d", delay_between_requests=0.5)
                    if not stock_data.empty:
                        # Flatten multi-level column names if present
                        if isinstance(stock_data.columns, pd.MultiIndex):
                            stock_data.columns = [col[0] for col in stock_data.columns]
                        
                        support_resistance = calculate_support_resistance_levels(stock_data)
                    else:
                        support_resistance = {"pivot": None, "r1": None, "r2": None, "r3": None, 
                                            "s1": None, "s2": None, "s3": None}
                except Exception as e:
                    logging.warning(f"Error fetching data for support/resistance calculation for {stock_symbol}: {e}")
                    support_resistance = {"pivot": None, "r1": None, "r2": None, "r3": None, 
                                        "s1": None, "s2": None, "s3": None}
                
                intraday_data = {
                    "Company": stock_name,
                    "Symbol": stock_symbol,
                    "Current Price": recommendation_data["Current Price"],
                    "VWAP": recommendation_data["VWAP"],
                    "Supertrend Signal": recommendation_data["Supertrend Signal"],
                    "SAR Signal": recommendation_data["SAR Signal"],
                    "Money Flow Index": recommendation_data["Money Flow Index"],
                    "RSI (5)": recommendation_data["RSI (5-period)"],
                    "RSI (9)": recommendation_data["RSI (9-period)"],
                    "Recommendation": recommendation_data["Recommendation"],
                    "Confidence": confidence_str,
                    "Support_Resistance": support_resistance
                }
                
                # Add trading signals
                signals = []
                current_price = recommendation_data["Current Price"]
                vwap = recommendation_data["VWAP"]
                
                if not pd.isna(vwap):
                    if current_price > vwap:
                        signals.append("Above VWAP ✓")
                    else:
                        signals.append("Below VWAP ✗")
                
                supertrend_signal = recommendation_data["Supertrend Signal"]
                if supertrend_signal == "Bullish":
                    signals.append("Supertrend: Bullish ✓")
                elif supertrend_signal == "Bearish":
                    signals.append("Supertrend: Bearish ✗")
                
                sar_signal = recommendation_data["SAR Signal"]
                if "bullish" in sar_signal.lower():
                    signals.append("SAR: Bullish Reversal ✓")
                elif "bearish" in sar_signal.lower():
                    signals.append("SAR: Bearish Reversal ✗")
                
                intraday_data["Key Signals"] = " | ".join(signals) if signals else "Mixed signals"
                intraday_recommendations.append(intraday_data)
                
        except Exception as e:
            logging.error(f"Error analyzing {stock_symbol}: {e}")
            continue
    
    if intraday_recommendations:
        print("\n" + "="*80)
        print("🚀 INTRADAY TRADING OPPORTUNITIES (High Confidence Signals)")
        print("="*80)
        
        for idx, stock in enumerate(intraday_recommendations, 1):
            # Determine signal emoji based on recommendation
            if any(word in stock["Recommendation"].upper() for word in ["BUY", "STRONG BUY"]):
                emoji = "🟢"  # Green circle for BUY
                action_color = "GREEN"
            elif any(word in stock["Recommendation"].upper() for word in ["SELL", "STRONG SELL"]):
                emoji = "🔴"  # Red circle for SELL
                action_color = "RED"
            else:
                emoji = "⚖️"  # Scale for neutral/hold
                action_color = "YELLOW"
            
            print(f"\n{emoji} #{idx}: {stock['Company']} ({stock['Symbol']})")
            print("-" * 60)
            print(f"Current Price: ₹{stock['Current Price']:.2f}")
            if not pd.isna(stock['VWAP']):
                vwap_status = "Above" if stock['Current Price'] > stock['VWAP'] else "Below"
                print(f"VWAP: ₹{stock['VWAP']:.2f} ({vwap_status})")
            
            print(f"Recommendation: {stock['Recommendation']}")
            print(f"Confidence: {stock['Confidence']}")
            print(f"Key Signals: {stock['Key Signals']}")
            
            # Add Support and Resistance Levels
            sr_levels = stock.get('Support_Resistance', {})
            if sr_levels and sr_levels.get('pivot') is not None:
                print("\n📊 Support & Resistance Levels:")
                print(f"Pivot Point:  ₹{sr_levels['pivot']:.2f}")
                print("\n🔺 Resistance Levels:")
                print(f"R1: ₹{sr_levels['r1']:.2f}")
                print(f"R2: ₹{sr_levels['r2']:.2f}")
                print(f"R3: ₹{sr_levels['r3']:.2f}")
                print("\n🔻 Support Levels:")
                print(f"S1: ₹{sr_levels['s1']:.2f}")
                print(f"S2: ₹{sr_levels['s2']:.2f}")
                print(f"S3: ₹{sr_levels['s3']:.2f}")
                
                # Add trading insight based on current price vs pivot
                current_price = stock['Current Price']
                pivot = sr_levels['pivot']
                print(f"\n📈 Price vs Pivot:")
                if current_price > pivot:
                    distance = ((current_price - pivot) / pivot) * 100
                    print(f"Above Pivot (+{distance:.1f}%) - Bullish bias")
                else:
                    distance = ((pivot - current_price) / pivot) * 100
                    print(f"Below Pivot (-{distance:.1f}%) - Bearish bias")
            
            # Add quick stats
            print("\n📊 Quick Stats:")
            if not pd.isna(stock['Money Flow Index']):
                mfi = stock['Money Flow Index']
                mfi_status = "Overbought" if mfi > 80 else "Oversold" if mfi < 20 else "Normal"
                print(f"Money Flow Index: {mfi:.1f} ({mfi_status})")
            
            if not pd.isna(stock['RSI (5)']):
                rsi5 = stock['RSI (5)']
                rsi5_status = "Overbought" if rsi5 > 75 else "Oversold" if rsi5 < 25 else "Normal"
                print(f"RSI (5-period): {rsi5:.1f} ({rsi5_status})")
            
            print("-" * 60)
        
        print(f"\n📋 Total High-Confidence Opportunities: {len(intraday_recommendations)}")
        print("="*80)
    else:
        print("\n⚠️  No high-confidence intraday trading opportunities found at this time.")
        print("Consider waiting for better setups or checking lower timeframes.")

def parse_analysis_file_and_check_levels(file_path):
    """
    Parse a previous analysis file and check current stock prices against 
    the stored pivot points and support/resistance levels based on recommendation
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📄 Reading analysis from: {file_path}")
    print("🎯 Will only check current prices for stocks found in this file")
    print("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Parse each stock entry
        stock_entries = re.split(r'📈|📉|⚖️', content)[1:]  # Split by emojis and remove first empty element
        
        results = []
        
        for entry in stock_entries:
            if not entry.strip():
                continue
                
            # Extract stock information using regex
            stock_match = re.search(r'#\d+:\s*([^(]+)\(([^)]+)\)', entry)
            if not stock_match:
                continue
                
            company_name = stock_match.group(1).strip()
            symbol = stock_match.group(2).strip()
            
            # Extract stored data
            price_match = re.search(r'Current Price:\s*₹([\d,.]+)', entry)
            recommendation_match = re.search(r'Recommendation:\s*([^\n\r]+)', entry)
            pivot_match = re.search(r'Pivot Point:\s*₹([\d,.]+)', entry)
            
            # Extract resistance levels
            r1_match = re.search(r'R1:\s*₹([\d,.]+)', entry)
            r2_match = re.search(r'R2:\s*₹([\d,.]+)', entry)
            r3_match = re.search(r'R3:\s*₹([\d,.]+)', entry)
            
            # Extract support levels
            s1_match = re.search(r'S1:\s*₹([\d,.]+)', entry)
            s2_match = re.search(r'S2:\s*₹([\d,.]+)', entry)
            s3_match = re.search(r'S3:\s*₹([\d,.]+)', entry)
            
            if not all([price_match, recommendation_match, pivot_match]):
                continue
            
            stored_price = float(price_match.group(1).replace(',', ''))
            recommendation = recommendation_match.group(1).strip()
            pivot = float(pivot_match.group(1).replace(',', ''))
            
            # Get resistance and support levels
            r1 = float(r1_match.group(1).replace(',', '')) if r1_match else None
            r2 = float(r2_match.group(1).replace(',', '')) if r2_match else None
            r3 = float(r3_match.group(1).replace(',', '')) if r3_match else None
            s1 = float(s1_match.group(1).replace(',', '')) if s1_match else None
            s2 = float(s2_match.group(1).replace(',', '')) if s2_match else None
            s3 = float(s3_match.group(1).replace(',', '')) if s3_match else None
            
            # Fetch current price
            try:
                print(f"🔄 Fetching current price for {symbol}...")
                current_data = yf.download(symbol, period="1d", interval="1m", progress=False)
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1].item()  # Ensure it's a scalar, not pandas object
                else:
                    print(f"⚠️  Could not fetch current data for {symbol}")
                    continue
                    
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error fetching data for {symbol}: {e}")
                continue
            
            # Determine recommendation type
            is_buy_recommendation = any(word in recommendation.upper() for word in ['BUY', 'ACCUMULATE'])
            is_sell_recommendation = any(word in recommendation.upper() for word in ['SELL'])
            
            # Analyze current position vs levels
            analysis_result = {
                'company': company_name,
                'symbol': symbol,
                'stored_price': stored_price,
                'current_price': current_price,
                'recommendation': recommendation,
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3,
                'is_buy_rec': is_buy_recommendation,
                'is_sell_rec': is_sell_recommendation,
                'price_change': ((current_price - stored_price) / stored_price) * 100
            }
            
            results.append(analysis_result)
        
        if not results:
            print("❌ No valid stock entries found in the file")
            return
        
        # Store original count before filtering
        total_stocks_found = len(results)
        
        # Filter results to show only stocks that have reached significant levels
        filtered_results = []
        
        for stock in results:
            show_stock = False
            
            if stock['is_buy_rec']:
                # For BUY: Show if above pivot OR above R1
                if (stock['current_price'] > stock['pivot'] or 
                    (stock['r1'] and stock['current_price'] > stock['r1'])):
                    show_stock = True
                    
            elif stock['is_sell_rec']:
                # For SELL: Show if below pivot OR below S1
                if (stock['current_price'] < stock['pivot'] or 
                    (stock['s1'] and stock['current_price'] < stock['s1'])):
                    show_stock = True
            else:
                # For neutral recommendations, always show
                show_stock = True
                
            if show_stock:
                filtered_results.append(stock)
        
        # Display results
        showing_stocks = len(filtered_results)
        
        print(f"\n🔍 LEVEL CHECK ANALYSIS")
        print(f"📊 Found {total_stocks_found} stocks in analysis file")
        print(f"🎯 Showing {showing_stocks} stocks that have reached key levels")
        
        if showing_stocks == 0:
            print("\n✨ No stocks have reached their key levels yet")
            print("💡 For BUY recommendations: Waiting for price above Pivot or R1")
            print("💡 For SELL recommendations: Waiting for price below Pivot or S1")
            print("=" * 80)
            return
            
        print("=" * 80)
        
        results = filtered_results  # Use filtered results for display
        
        for idx, stock in enumerate(results, 1):
            print(f"\n📊 #{idx}: {stock['company']} ({stock['symbol']})")
            print("-" * 60)
            print(f"Stored Price: ₹{stock['stored_price']:.2f}")
            print(f"Current Price: ₹{stock['current_price']:.2f}")
            
            # Price change indicator
            change_symbol = "🔴" if stock['price_change'] < 0 else "🟢"
            print(f"Price Change: {change_symbol} {stock['price_change']:+.2f}%")
            print(f"Original Recommendation: {stock['recommendation']}")
            
            print(f"\n📍 Pivot Point: ₹{stock['pivot']:.2f}")
            
            # Check against levels based on recommendation type
            if stock['is_buy_rec']:
                print("\n🔺 BUY RECOMMENDATION - Checking resistance breakouts:")
                
                if stock['current_price'] > stock['pivot']:
                    distance = ((stock['current_price'] - stock['pivot']) / stock['pivot']) * 100
                    print(f"✅ Above Pivot (+{distance:.1f}%) - Bullish confirmed")
                else:
                    distance = ((stock['pivot'] - stock['current_price']) / stock['pivot']) * 100
                    print(f"❌ Below Pivot (-{distance:.1f}%) - Caution advised")
                
                # Check resistance levels
                for level_name, level_value in [('R1', stock['r1']), ('R2', stock['r2']), ('R3', stock['r3'])]:
                    if level_value:
                        if stock['current_price'] > level_value:
                            distance = ((stock['current_price'] - level_value) / level_value) * 100
                            print(f"🚀 Above {level_name} (₹{level_value:.2f}) +{distance:.1f}% - Strong breakout!")
                        else:
                            distance = ((level_value - stock['current_price']) / level_value) * 100
                            print(f"⏳ Below {level_name} (₹{level_value:.2f}) -{distance:.1f}% - Target ahead")
            
            elif stock['is_sell_rec']:
                print("\n🔻 SELL RECOMMENDATION - Checking support breakdowns:")
                
                if stock['current_price'] < stock['pivot']:
                    distance = ((stock['pivot'] - stock['current_price']) / stock['pivot']) * 100
                    print(f"✅ Below Pivot (-{distance:.1f}%) - Bearish confirmed")
                else:
                    distance = ((stock['current_price'] - stock['pivot']) / stock['pivot']) * 100
                    print(f"❌ Above Pivot (+{distance:.1f}%) - Caution advised")
                
                # Check support levels
                for level_name, level_value in [('S1', stock['s1']), ('S2', stock['s2']), ('S3', stock['s3'])]:
                    if level_value:
                        if stock['current_price'] < level_value:
                            distance = ((level_value - stock['current_price']) / level_value) * 100
                            print(f"📉 Below {level_name} (₹{level_value:.2f}) -{distance:.1f}% - Support broken!")
                        else:
                            distance = ((stock['current_price'] - level_value) / level_value) * 100
                            print(f"🛡️ Above {level_name} (₹{level_value:.2f}) +{distance:.1f}% - Support holding")
            
            else:
                print("\n⚖️ HOLD/NEUTRAL RECOMMENDATION - General level analysis:")
                if stock['current_price'] > stock['pivot']:
                    distance = ((stock['current_price'] - stock['pivot']) / stock['pivot']) * 100
                    print(f"📈 Above Pivot (+{distance:.1f}%) - Bullish bias")
                else:
                    distance = ((stock['pivot'] - stock['current_price']) / stock['pivot']) * 100
                    print(f"📉 Below Pivot (-{distance:.1f}%) - Bearish bias")
            
            print("-" * 60)
        
        # Summary (inside the try block where total_stocks_found is available)
        buy_stocks = [s for s in results if s['is_buy_rec']]
        sell_stocks = [s for s in results if s['is_sell_rec']]
        
        print(f"\n📋 SUMMARY (Stocks at Key Levels):")
        print(f"🟢 BUY stocks above Pivot/R1: {len(buy_stocks)}")
        print(f"🔴 SELL stocks below Pivot/S1: {len(sell_stocks)}")
        print(f"⚖️ Other recommendations: {len(results) - len(buy_stocks) - len(sell_stocks)}")
        print(f"\n💡 Total stocks checked: {total_stocks_found} | Showing actionable: {len(results)}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error parsing file: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Intraday Trading Analysis')
    parser.add_argument('--check-levels', '-c', type=str, metavar='FILE_PATH',
                       help='Check current prices against support/resistance levels from a previous analysis file. Only fetches current prices for stocks that are present in the input file.')
    
    args = parser.parse_args()
    
    # If check-levels option is provided, run the level checking function
    if args.check_levels:
        print("🔍 CHECKING CURRENT PRICES AGAINST PREVIOUS ANALYSIS")
        print("=" * 80)
        parse_analysis_file_and_check_levels(args.check_levels)
        sys.exit(0)
    
    # Original functionality - generate new analysis
    # Create filename with format day_month_year.txt (e.g., 30_july_2025.txt)
    current_date = date.today()
    filename = f"{current_date.day}_{current_date.strftime('%B_%Y').lower()}.txt"
    
    # Redirect stdout to file
    original_stdout = sys.stdout
    
    try:
        with open(filename, 'w', encoding='utf-8') as output_file:
            sys.stdout = output_file
            
            # Specify the JSON file that contains the stock symbols and company names
            stock_file = "stocks.json"
            
            print("ENHANCED INTRADAY TRADING ANALYSIS")
            print("=" * 50)
            
            print("⚠️  Note: Analysis fetches fresh data with automatic rate limiting.")
            print("   This may take 5-10 minutes to complete all stocks.")
            print("=" * 50)
            
            # Run the new intraday trading summary with error handling
            try:
                print("\n🚀 STARTING INTRADAY ANALYSIS...")
                get_intraday_trading_summary(stock_file)
            except KeyboardInterrupt:
                print("\n❌ Analysis interrupted by user.")
                print("   Run again to retry the analysis.")
            except Exception as e:
                print(f"❌ Error in intraday analysis: {e}")
                print("   Try running again later.")
            

            
            # Provide troubleshooting tips
            print("\n💡 TIPS & TRADING GUIDE")
            print("=" * 50)
            print("✅ The script includes automatic rate limiting")
            print("✅ Fresh data is fetched for each analysis")  
            print("🔄 Run the script when you need updated analysis")
            print("⏰ Wait 10-15 minutes between full runs to avoid rate limits")
            print("\n📊 SUPPORT & RESISTANCE LEVELS GUIDE:")
            print("🔺 Resistance (R1, R2, R3): Price levels where selling pressure may increase")
            print("🔻 Support (S1, S2, S3): Price levels where buying interest may emerge")
            print("📍 Pivot Point: Key reference level - above = bullish bias, below = bearish bias")
            print("💡 Use these levels for entry/exit points and stop-loss placement")
            print("⚠️  Higher numbered levels (R3/S3) are stronger but less frequently tested")
            print("\n🆕 NEW FEATURE:")
            print("💡 Use --check-levels to analyze current prices against previous analysis:")
            print("   python trading_today.py --check-levels path/to/previous_analysis.txt")
            print("=" * 50)
            
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"Analysis complete! Output saved to: {filename}")
        print("\n💡 TIP: To check current prices against these levels later, use:")
        print(f"   python trading_today.py --check-levels {filename}")
