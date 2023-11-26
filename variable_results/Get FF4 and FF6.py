import pandas as pd

# Load the provided data files
path_3_factors = 'ff3.csv'
path_5_factors = 'ff5.csv'
path_monthly_returns = 'monthly_returns.csv'

# Read the Fama-French 3-factor and 5-factor data
ff3_data = pd.read_csv(path_3_factors, header=0)
ff5_data = pd.read_csv(path_5_factors, header=0)

# Read the monthly returns data
monthly_returns = pd.read_csv(path_monthly_returns)

# Define the lookback period for momentum
lookback_period = 6  # months

# Step 1: Calculate the WML (Winners Minus Losers) factor
# Convert 'Date' column to datetime and set as the index
monthly_returns['Date'] = pd.to_datetime(monthly_returns['Date'])
monthly_returns.set_index('Date', inplace=True)

# Calculate momentum for each stock
momentum = monthly_returns.rolling(window=lookback_period).sum()

# Transpose the DataFrame to have dates as rows and tickers as columns for ranking
momentum_transposed = momentum.transpose()

# Categorize stocks into winners and losers for each date
winners = momentum_transposed.apply(lambda x: x.nlargest(int(len(x) * 0.1)).index)
losers = momentum_transposed.apply(lambda x: x.nsmallest(int(len(x) * 0.1)).index)

# Calculate WML for each date
wml = {}
for date in momentum.index:
    winner_returns = monthly_returns.loc[date, winners[date]]
    loser_returns = monthly_returns.loc[date, losers[date]]
    wml[date] = winner_returns.mean() - loser_returns.mean()

wml_series = pd.Series(wml)
wml_series.index = wml_series.index.strftime('%Y%m')
wml_series = wml_series.reset_index()
wml_series.columns = ['Date', 'WML']
print(wml_series)

# Merge 
ff4_data = ff3_data.copy()
ff6_data = ff5_data.copy()

ff4_data['Date'] = ff4_data['Date'].astype(str)
ff6_data['Date'] = ff6_data['Date'].astype(str)


ff4_data = pd.merge(ff4_data, wml_series, on='Date', how='left')
ff6_data = pd.merge(ff6_data, wml_series, on='Date', how='left')
ff4_data.set_index('Date', inplace=True)
ff6_data.set_index('Date', inplace=True)

# Print the resulting DataFrame
print(ff4_data)
print(ff6_data)

ff4_data.to_csv('North_America_FF4.csv')
ff6_data.to_csv('North_America_FF6.csv')