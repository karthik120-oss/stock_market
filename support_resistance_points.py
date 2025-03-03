import yfinance as yf
import pandas as pd
import numpy as np
import json
from tabulate import tabulate

# Function to read stock symbols from a JSON file
def read_stock_symbols_from_json(filename):
    with open(filename, "r") as file:
        stock_data = json.load(file)
    return stock_data

def fetch_data(stock_symbol, period, interval):
    data = yf.download(stock_symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    return data

def calculate_pivot_points(data):
    # Use the second most recent row of data (previous day's prices)
    previous_row = data.iloc[-2]
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

    return pivot.item(), s1.item(), s2.item(), s3.item(), r1.item(), r2.item(), r3.item()

# Example Usage
stock_file = "stocks.json"
stocks = read_stock_symbols_from_json(stock_file)

period = '1y'
interval = '1d'

results = []

for stock_name, stock_symbol in stocks.items():
    data = fetch_data(stock_symbol, period, interval)
    pivot, s1, s2, s3, r1, r2, r3 = calculate_pivot_points(data)
    
    result = {
        "Stock": stock_name,
        "Pivot": pivot,
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "R1": r1,
        "R2": r2,
        "R3": r3
    }
    
    results.append(result)

# Create a DataFrame for better readability
results_df = pd.DataFrame(results)

# Add a serial number column starting from 1
results_df.index = results_df.index + 1
results_df.index.name = 'S.No'

# Ensure all values are properly formatted as strings
results_df = results_df.map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

# Print the DataFrame as a table
print(tabulate(results_df, headers='keys', tablefmt='grid'))
