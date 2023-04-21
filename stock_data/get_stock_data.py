import yfinance as yf
import pandas as pd

# Get the Apple stock data
aapl = yf.Ticker("AAPL")

# Get the historical price data for the last 252 trading days
hist = aapl.history(period="2y")
hist = hist.tail(500)

# Save the data to a CSV file
hist.to_csv("aapl_stock_data_500.csv")
