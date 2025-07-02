import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf

# Load and cache stock price data from yfinance
@st.cache_data(ttl=3600)
def load_data(tickers, period="1y", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval, progress=False)
    
    price_type = None
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            adj_close = data.xs('Adj Close', axis=1, level=0)
            price_type = 'Adjusted Close'
        elif 'Adj_Close' in data.columns.levels[0]:
            adj_close = data.xs('Adj_Close', axis=1, level=0)
            price_type = 'Adjusted Close'
        elif 'Close' in data.columns.levels[0]:
            adj_close = data.xs('Close', axis=1, level=0)
            price_type = 'Close'
        else:
            raise KeyError("No 'Adj Close' or 'Close' columns found in downloaded data.")
    else:
        if 'Adj Close' in data.columns:
            adj_close = data['Adj Close']
            price_type = 'Adjusted Close'
        elif 'Adj_Close' in data.columns:
            adj_close = data['Adj_Close']
            price_type = 'Adjusted Close'
        elif 'Close' in data.columns:
            adj_close = data['Close']
            price_type = 'Close'
        else:
            raise KeyError("No 'Adj Close' or 'Close' columns found in downloaded data.")

    adj_close.dropna(axis=0, how='any', inplace=True)
    return adj_close, price_type


# Calculate expected annual returns and covariance matrix from price data
def calculate_returns_and_covariance(data):
    daily_returns = data.pct_change().dropna()
    expected_annual_returns = daily_returns.mean() * 252  # 252 trading days/year
    cov_matrix = daily_returns.cov() * 252
    return expected_annual_returns, cov_matrix

# Compute portfolio return and volatility given weights
def portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Objective function to minimize volatility
def minimize_volatility(weights, expected_returns, cov_matrix, target_return):
    return portfolio_performance(weights, expected_returns, cov_matrix)[1]

# Optimize portfolio weights for given target return
def optimize_portfolio(expected_returns, cov_matrix, target_return):
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights: np.dot(weights, expected_returns) - target_return}
    )
    result = minimize(minimize_volatility, initial_weights,
                      args=(expected_returns, cov_matrix, target_return),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        st.error("Optimization failed. Please try a different target return.")
        return None
    return result.x

# Main Streamlit app
st.title("Optimal Portfolio Allocation Using yfinance Data")

# Define default tickers (S&P 500 top holdings or a small subset for demo)
default_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "BRK-B", "JNJ", "V", "NVDA"
]

# User input: select tickers (multiselect)
tickers = st.multiselect(
    "Select Stock Tickers for Portfolio",
    options=default_tickers,
    default=default_tickers
)

if len(tickers) < 2:
    st.warning("Please select at least two tickers.")
else:
    # Load adjusted close price data for selected tickers
    data ,price_type = load_data(tickers)

    # Calculate annualized returns and covariance matrix
    expected_annual_returns, cov_matrix = calculate_returns_and_covariance(data)

    # Set slider limits dynamically based on data
    min_return = float(expected_annual_returns.min())
    max_return = float(expected_annual_returns.max())
    default_return = (min_return + max_return) / 2

    # User input: target return slider (%)
    target_return_pct = st.slider(
        "Target Portfolio Return (%)",
        min_value=round(min_return * 100, 2),
        max_value=round(max_return * 100, 2),
        value=round(default_return * 100, 2),
        step=0.01
    )
    target_return = target_return_pct / 100  # Convert to decimal

    # Run optimization
    optimal_weights = optimize_portfolio(expected_annual_returns, cov_matrix, target_return)

    if optimal_weights is not None:
        optimal_return, optimal_volatility = portfolio_performance(optimal_weights, expected_annual_returns, cov_matrix)

        # Show optimal weights
        st.subheader("Optimal Portfolio Weights")
        optimal_weights_df = pd.DataFrame(optimal_weights, index=expected_annual_returns.index, columns=["Weight"])
        optimal_weights_df = optimal_weights_df[optimal_weights_df.Weight > 0.001]  # Filter small weights
        st.dataframe(optimal_weights_df.style.format({"Weight": "{:.4f}"}))

        # Show portfolio performance
        st.subheader("Portfolio Performance")
        st.write(f"Expected Annual Return: **{optimal_return * 100:.2f}%**")
        st.write(f"Expected Annual Volatility (Risk): **{optimal_volatility * 100:.2f}%**")
    else:
        st.info("Adjust the target return slider to a feasible value.")
