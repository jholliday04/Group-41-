 # Get stock details
import yfinance as yf
import pandas as pd
stock = yf.Ticker("AAPL")
print(stock.info)  

# Sample ESG dataset
data = {"Company": ["Tesla", "Apple"], "ESG Score": [85, 78]}
df = pd.DataFrame(data)
print(df)   
from sklearn.linear_model import LinearRegression

#Dowloading data
tickers = ["GOOGL", "AMZN", "TSLA", "GS", "^GDAXI", "BNDX"]
data = yf.download(tickers, start="2020-01-01", end="2024-12-31")["Adj Close"]
data.to_csv("prices.csv")

# Example: ESG score impact on stock performance
X = df[["ESG Score"]]
y = [120, 150]  # Hypothetical stock returns
model = LinearRegression().fit(X, y)
print(model.predict([[90]]))  # Predict return for ESG score 90

#Calc Daily Return
df = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
returns = df.pct_change().dropna()
returns.to_csv("daily_returns.csv")

#Visualising Data

#Line Plot of Prices
df.plot(figsize=(12, 6), title="Adjusted Closing Prices")

#Line Plot of Daily Returns
returns.plot(figsize=(12, 6), title="Daily Returns")

#Histogram of Returns
returns.hist(bins=50, figsize=(12, 8))

#Corrolation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Daily Returns")

#Stat Analysis
print(returns.describe())
print("Standard Deviation:\n", returns.std())
print("Skewness:\n", returns.skew())
print("Kurtosis:\n", returns.kurt())
