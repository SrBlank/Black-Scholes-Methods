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
from mpl_toolkits.mplot3d import Axes3D

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

# Calculates the annualized return of a stock given its historical prices.
def calculate_annualized_return(stock_prices, T):
    return_rate = (stock_prices[-1] / stock_prices[0])**(1 / T) - 1
    return return_rate


# Calculates the annualized volatility of a stock given its historical prices.
def calculate_annualized_volatility(stock_prices, T):
    log_returns = np.log(stock_prices[1:] / stock_prices[:-1])
    return np.sqrt(252) * log_returns.std()


def plot_3d_option_values(stock_prices, strike_prices, option_values):
    X, Y = np.meshgrid(stock_prices, strike_prices)
    Z = option_values

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel('Option Value')

    plt.show()

def plot_2d_option_values(stock_prices, option_values, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, option_values, label='Option Value')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    plt.title('American Call Option Value at t=0')
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_option_value_table(stock_prices, option_values, filename):
    # Set up the figure and axis for the table
    fig, ax = plt.subplots(figsize=(20, 10))  # Increase the figure size
    ax.axis('tight')
    ax.axis('off')

    # Create a table using stock prices and option values
    table_data = [[stock_price, option_value] for stock_price, option_value in zip(stock_prices, option_values)]
    table = ax.table(cellText=table_data, colLabels=['Stock Price', 'Option Value'], cellLoc='center', loc='center')

    # Adjust the table properties
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Save the table to a high-resolution PNG file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


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

# Import stock data
stock_data = pd.read_csv('../stock_data/aapl_stock_data.csv')
stock_prices = stock_data['Close'].values
# Synthetic data 
#np.random.seed(0)
#stock_prices = np.linspace(0, 200, N + 1)

# Parameters
S = stock_prices[-1]
K = 140
N = len(stock_prices) - 1 #200
M = len(stock_prices) - 1 #200
T = len(stock_prices) / 252  #1 # Assuming 252 trading days in a year
s_stages = 20
r = calculate_annualized_return(stock_prices, T)
sigma = calculate_annualized_volatility(stock_prices, T)

if __name__ == "__main__":
    strike_prices = np.linspace(50, 150, 21)
    option_values = np.zeros((len(strike_prices), N + 1))

    #for i, K in enumerate(strike_prices):
    V = black_scholes_american_call_option(S, K, T, r, sigma, N, M, s_stages)
        # removes outliers, negative option values
    V[V < 0] = 0
    #    option_values[i] = V[:, 0]
    
    #plot_3d_option_values(stock_prices, strike_prices, option_values)
    create_option_value_table(stock_prices, V[:, 0], '../results/aapl_option_value_table.png')
    plot_2d_option_values(stock_prices, V[:, 0], '../results/aapl_2d_option_value_graph.png')

    print(f"Option value at S=K: {V[N//2, 0]}")

