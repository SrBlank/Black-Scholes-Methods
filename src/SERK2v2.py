"""
Application of Stabalized Runge-Kutta Method of Order 2 (SERK2v2) on
Black-Scholes Partial Differntial Equation for American Call Options

Authors: Saad Rafiq and Josh Avery

Description: 
This code contains functions to calculate the Black-Scholes PDE using the SERK2v2 
method to estimate American call option values. It takes in several parameters, 
including the initial stock price, strike price, number of discretized stock price
steps, number of discretized time steps, time to maturity, number of SERK2v2 stages,
historical stock prices, risk-free interest rate, and stock price volatility.
The code also includes functions to estimate the annualized return and volatility 
of a stock given its historical prices. When executed, the code generates a plot of
the option values and prints the option value at the strike price.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Parameters:
S (float): initial stock price
K (float): strike price
N (int): number of discretized stock price steps
M (int): number of discretized time steps
T (float): time to maturity
s_stages (int): number of SERK2v2 stages
stock_prices (ndarray): historical stock prices
r (float): risk-free interest rate
sigma (float): stock price volatility
"""
# Parameters
S = 100 #stock_prices[-1]
K = 100
N = 200
M = 200
T = 1
s_stages = 4 #20

# Load historical stock data here
# For example, load data from a CSV file:
# stock_data = pd.read_csv('stock_data.csv')
# stock_prices = stock_data['Close'].values
# T = len(stock_prices) / 252  # Assuming 252 trading days in a year

# Synthetic data for demonstration purposes
np.random.seed(0)
stock_prices = np.linspace(0, 200, N + 1)

# Estimate parameters from historical data
r = .05 #calculate_annualized_return(stock_prices, T)
sigma = .20 #calculate_annualized_volatility(stock_prices, T)

def black_scholes_rhs(stock_prices, g_prev, r, sigma, dS, dT):
    """
    Calculates the right-hand side of the Black-Scholes PDE.
    
    Parameters:
    stock_prices (ndarray): array of stock prices
    g_prev (ndarray): previous option values
    r (float): risk-free interest rate
    sigma (float): stock price volatility
    dS (float): discretized stock price step
    dT (float): discretized time step
    
    Returns:
    ndarray: right-hand side values for the Black-Scholes PDE
    """
    # Calculate terms for the Black-Scholes PDE
    diffusion_term = 0.5 * sigma**2 * stock_prices[1:-1]**2 * (g_prev[:-2] - 2 * g_prev[1:-1] + g_prev[2:]) / dS**2
    drift_term = (r - 0.5 * sigma**2) * stock_prices[1:-1] * (g_prev[2:] - g_prev[:-2]) / (2 * dS)
    discount_term = r * g_prev[1:-1]

    # Combine terms to calculate the right-hand side of the Black-Scholes PDE
    g_interior = g_prev[1:-1] + dT * (diffusion_term + drift_term - discount_term)

    return g_interior


def SERK2v2(g_prev, stock_prices, r, sigma, dS, dT, s_stages, time, m, K):
    """
    Solves the Black-Scholes PDE using the SERK2v2 method.
    
    Parameters:
    g_prev (ndarray): previous option values
    stock_prices (ndarray): array of stock prices
    r (float): risk-free interest rate
    sigma (float): stock price volatility
    dS (float): discretized stock price step
    dT (float): discretized time step
    s_stages (int): number of SERK2v2 stages
    time (ndarray): array of time steps
    m (int): current time step index
    K (float): strike price
    
    Returns:
    ndarray: option values at the current time step
    """
    alpha = 1 / (0.4*s_stages**2)
    g = np.zeros((s_stages + 1, len(stock_prices)))
    g[0] = g_prev

    for j in range(1, s_stages + 1):
        c = 2 if j % 2 == 0 else 1
        g_interior = black_scholes_rhs(stock_prices, g[j - 1], r, sigma, dS, dT) * c * alpha

        # Apply early exercise constraint
        g_interior = np.maximum(g_interior, stock_prices[1:-1] - K)

        # Handle boundary conditions
        g0 = 0
        gN = 2 * g_prev[-1] - g_prev[-2]  # Fixed boundary condition

        g[j] = np.concatenate(([g0], g_interior, [gN]))

    return np.mean(g, axis=0)


def black_scholes_american_call_option(S, K, T, r, sigma, N, M, s_stages):
    """
    Calculates American call option values using the Black-Scholes PDE and SERK2v2 method.
    
    Parameters:
    S (float): initial stock price
    K (float): strike price
    T (float): time to maturity
    r (float): risk-free interest rate
    sigma (float): stock price volatility
    N (int): number of discretized stock price steps
    M (int): number of discretized time steps
    s_stages (int): number of SERK2v2 stages
    
    Returns:
    ndarray: option values at each stock price and time step
    """
    # Discretize stock prices and time
    dS = 2 * S / N
    dT = T / M
    #stock_prices = np.linspace(0, S, N + 1)
    #stock_prices = np.linspace(0, 200, N + 1)
    time = np.linspace(0, T, M + 1)

    # Initialize the option values matrix
    V = np.zeros((N + 1, M + 1))

    # Set option values at maturity (boundary condition)
    V[:, -1] = np.maximum(stock_prices - K, 0)
    
    # Time-stepping loop (backwards in time)
    for m in range(M - 1, -1, -1):
        # Compute option values at the current time step using the SERK2v2 method
        V[:, m] = SERK2v2(V[:, m + 1], stock_prices, r, sigma, dS, dT, s_stages, time, m, K)

    return V

def calculate_annualized_return(stock_prices, T):
    """
    Calculates the annualized return of a stock given its historical prices.

    Parameters:
    stock_prices (ndarray): historical stock prices
    T (float): time period in years

    Returns:
    float: annualized return
    """
    return_rate = (stock_prices[-1] / stock_prices[0])**(1 / T) - 1
    return return_rate

def calculate_annualized_volatility(stock_prices, T):
    """
    Calculates the annualized volatility of a stock given its historical prices.

    Parameters:
    stock_prices (ndarray): historical stock prices
    T (float): time period in years

    Returns:
    float: annualized volatility
    """
    log_returns = np.log(stock_prices[1:] / stock_prices[:-1])
    return np.sqrt(252) * log_returns.std()

if __name__ == "__main__":
    # Compute the option values
    V = black_scholes_american_call_option(S, K, T, r, sigma, N, M, s_stages)

    # Plot the option values
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, S, N + 1), V[:, 0], label='Option Value')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    plt.title('American Call Option Value at t=0')
    plt.legend()
    plt.show()

    # Print the option value at S=K
    print(f"Option value at S=K: {V[N//2, 0]}")