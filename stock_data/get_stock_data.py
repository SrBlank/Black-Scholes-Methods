import yfinance as yf
import pandas as pd

# Get the Apple stock data
aapl = yf.Ticker("GOOGL")

# Get the historical price data for the last 252 trading days
hist = aapl.history(period="1y")
hist = hist.tail(252)

# Save the data to a CSV file
hist.to_csv("googl_stock_data.csv")
