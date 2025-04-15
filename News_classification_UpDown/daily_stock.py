import yfinance as yf
import datetime
import csv

# 計算昨天的日期
yesterday = datetime.date.today() - datetime.timedelta(days=1)
filename = f"stock_{yesterday}.csv"  # 用日期命名檔案

# 指數名稱與代碼
indices = {
    'Dow Jones': '^DJI',
    'S&P 500': '^GSPC',
    'Nasdaq': '^IXIC'
}

# 準備要寫入CSV的資料
records = []

for name, symbol in indices.items():
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=yesterday, end=yesterday + datetime.timedelta(days=1))
    if not hist.empty:
        close_price = hist['Close'][0]
        records.append([name, symbol, yesterday.strftime('%Y-%m-%d'), round(close_price, 2)])
    else:
        records.append([name, symbol, yesterday.strftime('%Y-%m-%d'), 'No data'])

# 寫入 CSV
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Index Name', 'Symbol', 'Date', 'Close Price'])
    writer.writerows(records)

print(f"資料已儲存為：stock_{filename}")
