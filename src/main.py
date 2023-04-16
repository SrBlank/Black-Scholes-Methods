# Authors: Saad Rafiq and Josh Avery
# Numerical Analysis II
# Desc: Implementation of the binomial tree, and crank-nicholson methods

import numpy as np
import scipy.linalg as sla
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time
import functools


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


class BinomialTree:

    def __int__(self,
                expiration_time: np.datetime64, stock_price: float,
                strike_price: float, risk_free_rate: float, sigma: float,
                dividend_rate: float, tree_height: int,
                start_time: np.datetime64 = np.datetime64('now')):

        self.T: np.timedelta64 = expiration_time - start_time
        self.S = stock_price
        self.K = strike_price
        self.r = risk_free_rate
        self.sigma = sigma
        self.q = dividend_rate
        self.n = tree_height

        if not self.T/self.n < ((self.sigma**2)/((self.r-self.q)**2)):
            raise ValueError("The time step can not be less than sigma^2 / (r-q)^2. " +
                             "Increase the height of the tree.")

    def american_put(self) -> float:
        # Requirement: dt < sigma^2/(r-q)^2 or else p is not on (0,1).
        # T... expiration time
        # S... (initial) stock price
        # K... strike price
        # q... dividend yield
        # n... height of the binomial tree

        dt = self.T / self.n
        up = np.exp(self.sigma * np.sqrt(dt))
        p0 = (up * np.exp(-self.q * dt) - np.exp(-self.r * dt)) / (up**2 - 1)
        p1 = np.exp(-self.r * dt) - p0

        # create a pandas dataframe to store the option prices at each time step
        p = pd.DataFrame(np.zeros((self.n+1, self.n+1)))

        # initial values at time T (i.e. the exercise price)
        for i in range(self.n+1):
            p.iloc[self.n, i] = max(self.K - self.S * up**(2*i - self.n), 0)

        # move to earlier times
        for j in range(self.n-1, -1, -1):
            for i in range(j+1):
                # binomial value
                p.iloc[j, i] = p0 * p.iloc[j+1, i+1] + p1 * p.iloc[j+1, i]
                # exercise value
                exercise = self.K - self.S * up**(2*i - j)
                p.iloc[j, i] = max(p.iloc[j, i], exercise)

        return p.iloc[0, 0]

    def american_call(self) -> float:
        # Requirement: dt < sigma^2/(r-q)^2 or else p is not on (0,1).
        # T... expiration time
        # S... (initial) stock price
        # K... strike price
        # q... dividend yield
        # n... height of the binomial tree

        dt = self.T / self.n
        up = np.exp(self.sigma * np.sqrt(dt))
        p0 = (up * np.exp(-self.q * dt) - np.exp(-self.r * dt)) / (up**2 - 1)
        p1 = np.exp(-self.r * dt) - p0

        # create a pandas dataframe to store the option prices at each time step
        p = pd.DataFrame(np.zeros((self.n+1, self.n+1)))

        # initial values at time T (i.e. the exercise price)
        for i in range(self.n+1):
            p.iloc[self.n, i] = max(self.S * up**(2*i - self.n) - self.K, 0)

        # move to earlier times
        for j in range(self.n-1, -1, -1):
            for i in range(j+1):
                # binomial value
                p.iloc[j, i] = p0 * p.iloc[j+1, i+1] + p1 * p.iloc[j+1, i]
                # exercise value
                exercise = self.K - self.S * up**(2*i - j)
                p.iloc[j, i] = max(p.iloc[j, i], exercise)

        return p.iloc[0, 0]


def is_invertible(A: np.ndarray) -> bool:
    sign, log_det = np.linalg.slogdet(A)
    if log_det > -np.inf:
        return True
    else:
        return False


def check_multiplicities(A: np.ndarray, only_1s: bool = False) -> bool:
    # Calculate eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Get unique eigenvalues and their counts (algebraic multiplicities)
    unique_eigenvalues, algebraic_multiplicities = np.unique(eigenvalues, return_counts=True)

    if only_1s:
        mag_1_eigens = unique_eigenvalues[np.abs(unique_eigenvalues) == 1.0]
        for i, eigenvalue in enumerate(mag_1_eigens):
            # Calculate eigenvectors corresponding to the current eigenvalue
            eigenvectors_for_eigenvalue = eigenvectors[:, np.isclose(eigenvalues, eigenvalue)]

            # Calculate the geometric multiplicity as the rank of the eigenvectors matrix
            geometric_multiplicity = np.linalg.matrix_rank(eigenvectors_for_eigenvalue)

            # Check if the algebraic and geometric multiplicities are the same
            if algebraic_multiplicities[i] != geometric_multiplicity:
                return False
    else:
        for i, eigenvalue in enumerate(unique_eigenvalues):
            # Calculate eigenvectors corresponding to the current eigenvalue
            eigenvectors_for_eigenvalue = eigenvectors[:, np.isclose(eigenvalues, eigenvalue)]

            # Calculate the geometric multiplicity as the rank of the eigenvectors matrix
            geometric_multiplicity = np.linalg.matrix_rank(eigenvectors_for_eigenvalue)

            # Check if the algebraic and geometric multiplicities are the same
            if algebraic_multiplicities[i] != geometric_multiplicity:
                return False

    return True

class CrankNicholson:
    """
    Crank-Nicholson Method for American Options.
    Averages the implicit and explicit finite-difference representations of the B.S.M. equation.
    M_j+i * v_j+1 + r_j+1 = M_j * v_j + r_j
    v_j+1 = M^-1_j+1 * M_j * v_j * (r_j - r_j+1)
    """

    def __init__(self,
                 expiration_time_years_from_now: float, sigma: float,
                 min_stock_price: float, max_stock_price: float,
                 strike_price: float, risk_free_rate: float,
                 dividend_rate: float,
                 n_s: int, n_t: int):
        self.n_t = n_t
        self.T = expiration_time_years_from_now
        self.dT: float = self.T / n_t

        self.n_s = n_s
        self.dS: float = (max_stock_price - min_stock_price) / n_s
        self.S_min = min_stock_price
        self.S_max = max_stock_price
        self.stock_prices = np.linspace(self.S_min, self.S_max, self.n_s)

        self.K = strike_price
        self.r = risk_free_rate
        self.sigma = sigma
        self.q = dividend_rate

    @timer
    def american_call(self) -> np.ndarray:
        # Imposing Linear Boundary Conditions
        # Compose A using alpha and beta definitions

        # Calculate alpha and beta vectors
        first_term = (0.5 * (self.sigma ** 2) * (self.stock_prices ** 2)) / (self.dS ** 2)
        second_term = (self.r - self.q) * self.stock_prices / (2 * self.dS)

        alpha = first_term - second_term
        beta = first_term + second_term

        # Set alpha and beta entries to first_term if the corresponding entry is negative
        alpha = np.where(alpha < 0, first_term, alpha)
        beta = np.where(beta < 0, first_term, beta)

        # Create an empty matrix A of size n x n
        A = np.zeros((self.n_s, self.n_s))

        # Set the first row
        A[0, 0] = self.r

        # Set the triple-diagonal elements
        for i in range(1, self.n_s - 1):
            A[i, i - 1] = -alpha[i]
            A[i, i] = self.r + alpha[i] + beta[i]
            A[i, i + 1] = -beta[i]

        # Implement Linear Boundary Conditions
        A[-1, -3] = 0
        A[-1, -2] = (self.r - self.q) * self.stock_prices[-1] / self.dS
        A[-1, -1] = self.r - A[-1, -2]

        # Check Stability: is the spectral radius of B < 1?
        # Calculate the eigenvalues of A
        eig_A = sla.eigvals(A)

        # Use the formula provided in Windcliff, Forsyth,and Vetzal (2003) to get the eigenvalues of B (Eq. 5.4)
        eig_B = (1 - eig_A*.5*self.dT)/(1 + eig_A*.5*self.dT)

        # Calculate the spectral radius of A (maximum absolute eigenvalue)
        spectral_radius_B = np.max(np.abs(eig_B))

        # Create an n_s x n_s identity matrix
        I_n = np.identity(self.n_s)

        if spectral_radius_B == 1:
            # Make B = (I + .5A * dtau)^-1 * (I - .5A * dtau)
            B = sla.inv(I_n + 0.5 * self.dT * A) * (I_n - 0.5 * self.dT * A)
            if not check_multiplicities(B, only_1s=True):
                raise ValueError("The method is not stable. Algebraic and geometric multiplicities of B's"
                                 "magnitude-1 eigenvalues are not equal.")

        elif spectral_radius_B > 1:
            raise ValueError("The method is not stable. Spectral radius of B must be less than 1.")
        else:
            # We're good, make B = (I + .5A * dtau)^-1 * (I - .5A * dtau)
            B = np.matmul(sla.inv(I_n + 0.5 * self.dT * A), (I_n - 0.5 * self.dT * A))

        # Define the matrix which will hold our option values over the pricing grid:
        V = np.zeros((self.n_s, self.n_t + 1))

        # The value of the option at the initial time is just the payoff:
        V[:, 0] = np.maximum(self.stock_prices - self.K, 0)

        # The value of the option at expiry is the stock's price discounted with the dividends missed
        #   minus the strike price
        V[:, -1] = np.maximum(self.stock_prices * np.exp(-self.q * self.T) - self.K, 0)

        # We have v_(k+1) = B * v_(k).
        for i in range(self.n_t):
            V[:, i+1] = np.matmul(B, V[:, i])
            # Check for Early Exercise
            V[:, i+1] = np.maximum(V[:, i+1], self.stock_prices * np.exp(-self.q * (i * self.dT)) -
                                   self.K * np.exp(-self.r * (i * self.dT)))  # Consider early exercise
        return V

    def american_put(self):
        # Imposing Linear Boundary Conditions
        # Compose A using alpha and beta definitions

        # Calculate alpha and beta vectors
        first_term = (0.5 * (self.sigma ** 2) * (self.stock_prices ** 2)) / (self.dS ** 2)
        second_term = (self.r - self.q) * self.stock_prices / (2 * self.dS)

        alpha = first_term - second_term
        beta = first_term + second_term

        # Set alpha and beta entries to first_term if the corresponding entry is negative
        alpha = np.where(alpha < 0, first_term, alpha)
        beta = np.where(beta < 0, first_term, beta)

        # Create an empty matrix A of size n x n
        A = np.zeros((self.n_s, self.n_s))

        # Set the first row
        A[0, 0] = self.r

        # Set the triple-diagonal elements
        for i in range(1, self.n_s - 1):
            A[i, i - 1] = -alpha[i]
            A[i, i] = self.r + alpha[i] + beta[i]
            A[i, i + 1] = -beta[i]

        # Implement Linear Boundary Conditions
        A[-1, -3] = 0
        A[-1, -2] = (self.r - self.q) * self.stock_prices[-1] / self.dS
        A[-1, -1] = self.r - A[-1, -2]

        # Check Stability: is the spectral radius of B < 1?
        # Calculate the eigenvalues of A
        eig_A = sla.eigvals(A)

        # Use the formula provided in Windcliff, Forsyth,and Vetzal (2003) to get the eigenvalues of B (Eq. 5.4)
        eig_B = (1 - eig_A * .5 * self.dT) / (1 + eig_A * .5 * self.dT)

        # Calculate the spectral radius of A (maximum absolute eigenvalue)
        spectral_radius_B = np.max(np.abs(eig_B))

        # Create an n_s x n_s identity matrix
        I_n = np.identity(self.n_s)

        if spectral_radius_B == 1:
            # Make B = (I + .5A * dtau)^-1 * (I - .5A * dtau)
            B = sla.inv(I_n + 0.5 * self.dT * A) * (I_n - 0.5 * self.dT * A)
            if not check_multiplicities(B, only_1s=True):
                raise ValueError("The method is not stable. Algebraic and geometric multiplicities of B's"
                                 "magnitude-1 eigenvalues are not equal.")

        elif spectral_radius_B > 1:
            raise ValueError("The method is not stable. Spectral radius of B must be less than 1.")
        else:
            # We're good, make B = (I + .5A * dtau)^-1 * (I - .5A * dtau)
            B = np.matmul(sla.inv(I_n + 0.5 * self.dT * A), (I_n - 0.5 * self.dT * A))

        # Define the matrix which will hold our option values over the pricing grid:
        V = np.zeros((self.n_s, self.n_t + 1))

        # The value of the option at the initial time is just the payoff:
        V[:, 0] = np.maximum(self.K - self.stock_prices, 0)

        # The value of the option at expiry is the strike price minus the discounted stock price
        V[:, -1] = np.maximum(self.K * np.exp(-self.r * self.T) - self.stock_prices * np.exp(-self.q * self.T), 0)

        # We have v_(k+1) = B * v_(k).
        for i in range(self.n_t):
            V[:, i + 1] = np.matmul(B, V[:, i])
            # Check for Early Exercise
            V[:, i + 1] = np.maximum(V[:, i + 1], self.K * np.exp(-self.r * (i * self.dT)) -
                                     self.stock_prices * np.exp(-self.q * (i * self.dT)))  # Consider early exercise
        return V


if __name__ == "__main__":
    test_cranky = CrankNicholson(expiration_time_years_from_now=10,
                                 sigma=0.5,
                                 min_stock_price=0.0, max_stock_price=1000.0,
                                 strike_price=10,
                                 risk_free_rate=0.04,
                                 dividend_rate=0.01,
                                 n_s=1000,
                                 n_t=10*365)  # 1-Day time steps

    prices = test_cranky.american_call()
    np.savetxt("Crank_Test_Results.csv", prices, delimiter=",")
