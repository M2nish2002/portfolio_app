import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('sp500_stocks.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    adj_close_prices = df.pivot(index='Date', columns='Symbol', values='Adj Close')
    adj_close_prices.dropna(inplace=True)
    return adj_close_prices

# Portfolio optimization functions
def calculate_returns_and_covariance(data):
    daily_returns = data.pct_change().dropna()
    expected_annual_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    return expected_annual_returns, cov_matrix

def portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def minimize_volatility(weights, expected_returns, cov_matrix, target_return):
    portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
    return portfolio_volatility

def optimize_portfolio(expected_returns, cov_matrix, target_return):
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights: np.dot(weights, expected_returns) - target_return}
    )
    result = minimize(minimize_volatility, initial_weights, args=(expected_returns, cov_matrix, target_return),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Streamlit app
st.title("Optimal Portfolio Allocation")

# Load data
data = load_data()

# Calculate expected returns and covariance matrix
expected_annual_returns, cov_matrix = calculate_returns_and_covariance(data)

# User input for target return
target_return = st.slider("Target Portfolio Return (%)", min_value=0.0, max_value=0.5, step=0.01) / 100  # Convert to decimal

# Run optimization
optimal_weights = optimize_portfolio(expected_annual_returns, cov_matrix, target_return)
optimal_return, optimal_volatility = portfolio_performance(optimal_weights, expected_annual_returns, cov_matrix)

# Display results
st.subheader("Optimal Portfolio Weights")
optimal_weights_df = pd.DataFrame(optimal_weights, index=expected_annual_returns.index, columns=["Weight"])
st.write(optimal_weights_df)

st.subheader("Portfolio Performance")
st.write(f"Expected Return: {optimal_return * 100:.2f}%")
st.write(f"Expected Volatility (Risk): {optimal_volatility * 100:.2f}%")