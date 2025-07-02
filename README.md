# Intelligent Portfolio Optimization Engine

A Streamlit app for optimal portfolio allocation using modern portfolio theory and Monte Carlo simulation.  
It dynamically fetches stock price data from Yahoo Finance using the `yfinance` Python library and optimizes portfolio weights for a user-selected target return.

---

## Features

- Fetches historical price data (Adjusted Close or Close prices) for selected stocks via `yfinance`.
- Calculates expected annual returns and covariance matrix based on daily returns.
- Uses `scipy.optimize.minimize` to minimize portfolio volatility for a target return.
- Interactive Streamlit UI with:
  - Stock ticker multiselect.
  - Target portfolio return slider with dynamic range.
  - Displays optimal portfolio weights and performance metrics.
  - Informs user about the price type (Adjusted Close or Close) used.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/portfolio-optimizer.git
   cd portfolio-optimizer