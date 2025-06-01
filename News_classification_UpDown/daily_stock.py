import yfinance as yf
import datetime
import csv
import pandas as pd

start_date = "2025-04-01"
end_date = "2025-05-31"

twii = yf.download('^TWII', start=start_date, end=end_date)
print(twii.head())
twii['Change'] = twii['Close'].diff()/twii['Close'].shift(1)*100

def classify_movement(change):
    if change > 0.1:
        return 'bullish'
    elif change < -0.1:
        return 'bearish'
    else:
        return 'flat'

twii['Movement'] = twii['Change'].apply(classify_movement)
print(twii[['Close', 'Change', 'Movement']].head())


