# Authors: Saad Rafiq and Josh Avery
# Numerical Analysis II
# Desc: Implementation of the binomial tree, and crank-nicholson methods

import numpy as np
import scipy.linalg as sla
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import functools
from typing import Union, Tuple


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


def matrix_power(matrix, k):
    if k == 0:
        return np.identity(matrix.shape[0])
    elif k % 2 == 0:
        half_power = matrix_power(matrix, k // 2)
        return np.matmul(half_power, half_power)
    else:
        half_power = matrix_power(matrix, (k - 1) // 2)
        return np.matmul(matrix, np.matmul(half_power, half_power))


def infinity_norm(matrix):
    return np.max(np.sum(np.abs(matrix), axis=1))


def infinity_norm_of_power(matrix, k):
    matrix_k = matrix_power(matrix, k)
    return infinity_norm(matrix_k)


class BinomialTree:

    def __init__(self,
                 expiration_time: np.datetime64,
                 stock_price: float,
                 strike_price: float,
                 risk_free_rate: float,
                 sigma: float,
                 dividend_rate: float,
                 tree_height: int,
                 start_time: np.datetime64 = np.datetime64('now')):

        # Subtract two dates to get a timedelta64 object
        x = expiration_time - start_time
        # Convert the timedelta64 object to a float value in seconds
        seconds = x / np.timedelta64(1, 's')

        # Time in Years
        self.T: float = seconds / (60.0 * 60.0 * 24.0 * 365.25)
        self.S = stock_price
        self.K = strike_price
        self.r = risk_free_rate
        self.sigma = sigma
        self.q = dividend_rate
        self.n = tree_height

        if not self.T/self.n < ((self.sigma**2)/((self.r-self.q)**2)):
            raise ValueError("The time step can not be less than sigma^2 / (r-q)^2. " +
                             "Increase the height of the tree.")

    @timer
    def american_put(self,
                     return_time: bool = False,
                     return_matrix: bool = False) -> Union[float, np.ndarray,
                                                           Tuple[float, float], Tuple[np.ndarray, float]]:
        # Requirement: dt < sigma^2/(r-q)^2 or else p is not on (0,1).
        # T... expiration time
        # S... (initial) stock price
        # K... strike price
        # q... dividend yield
        # n... height of the binomial tree

        tic = None
        if return_time:
            tic = time.perf_counter()

        dt: float = self.T / self.n
        up = np.exp(self.sigma * np.sqrt(dt))
        p0 = (up * np.exp(-self.q * dt) - np.exp(-self.r * dt)) / (up**2 - 1)
        p1 = np.exp(-self.r * dt) - p0

        # create an ndarray to store the option prices at each time step
        p = np.zeros((self.n+1, self.n+1))

        # initial values at time T (i.e. the exercise price)
        for i in range(self.n+1):
            p[self.n, i] = max(self.K - self.S * np.exp((i - self.n) * self.sigma * np.sqrt(dt)), 0)

        # move to earlier times
        for j in range(self.n-1, -1, -1):
            for i in range(j+1):
                # binomial value
                p[j, i] = p0 * p[j+1, i+1] + p1 * p[j+1, i]
                # exercise value
                exercise = self.K - self.S * np.exp((i - j) * self.sigma * np.sqrt(dt))
                p[j, i] = max(p[j, i], exercise)

        if return_time:
            toc = time.perf_counter()
            if return_matrix:
                return p, toc - tic
            else:
                return p[0, 0], toc - tic
        elif return_matrix:
            return p
        else:
            return p[0, 0]

    @timer
    def american_call(self,
                      return_time: bool = False,
                      return_matrix: bool = False) -> Union[float, np.ndarray,
                                                            Tuple[float, float], Tuple[np.ndarray, float]]:
        # Requirement: dt < sigma^2/(r-q)^2 or else p is not on (0,1).
        # T... expiration time
        # S... (initial) stock price
        # K... strike price
        # q... dividend yield
        # n... height of the binomial tree

        tic = None
        if return_time:
            tic = time.perf_counter()

        dt = self.T / self.n
        up = np.exp(self.sigma * np.sqrt(dt))
        p0 = (up * np.exp(-self.q * dt) - np.exp(-self.r * dt)) / (up**2 - 1)
        p1 = np.exp(-self.r * dt) - p0

        # create an ndarray to store the option prices at each time step
        p = np.zeros((self.n+1, self.n+1))

        # initial values at time T (i.e. the exercise price)
        for i in range(self.n+1):
            p[self.n, i] = max(self.S * np.exp((i - self.n) * self.sigma * np.sqrt(dt)) - self.K, 0)

        # move to earlier times
        for j in range(self.n-1, -1, -1):
            for i in range(j+1):
                # binomial value
                p[j, i] = p0 * p[j+1, i+1] + p1 * p[j+1, i]
                # exercise value
                exercise = self.K - self.S * np.exp((i - j) * self.sigma * np.sqrt(dt))
                p[j, i] = max(p[j, i], exercise)

        if return_time:
            toc = time.perf_counter()
            if return_matrix:
                return p, toc - tic
            else:
                return p[0, 0], toc - tic
        elif return_matrix:
            return p
        else:
            return p[0, 0]


class CrankNicholson:
    """
    Crank-Nicholson Method for American Options.
    Averages the implicit and explicit finite-difference representations of the B.S.M. equation.
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
    def american_call(self, return_time: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        # Imposing Linear Boundary Conditions
        # Compose A using alpha and beta definitions

        tic = None
        if return_time:
            tic = time.perf_counter()

        # Calculate alpha and beta vectors
        first_term = (0.5 * (self.sigma ** 2) * (self.stock_prices ** 2)) / (self.dS ** 2)
        second_term = (self.r - self.q) * self.stock_prices / (2 * self.dS)

        alpha = first_term - second_term
        beta = first_term + second_term

        # Set alpha and beta entries to their adjusted versions as in the paper
        alpha = np.where(beta < 0, first_term - (2 * second_term), alpha)
        beta = np.where(alpha < 0, first_term + (2 * second_term), beta)

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
            if spectral_radius_B < 1.1:
                B = sla.inv(I_n + 0.5 * self.dT * A) * (I_n - 0.5 * self.dT * A)
                norm = infinity_norm_of_power(B, self.n_t)
                if norm > 1:
                    # Otherwise we're fine by the author's def. of stability
                    raise ValueError("The method is not stable." +
                                     "Spectral radius of B is " + str(spectral_radius_B) +
                                     ", but must be less than 1." +
                                     " The infinity norm of B^n_T is: " + str(norm) + ". The method is not stable.")
                else:
                    warnings.warn("1 < rho(B) < 1.1 but ||B^n_T|| < 1. Borderline Instability.")
            else:
                raise ValueError("The method is not stable. Spectral radius of B must be less than 1. (rho(B) > 1.1)")
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
        if return_time:
            toc = time.perf_counter()
            return V, toc-tic
        else:
            return V

    @timer
    def american_put(self, return_time: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        # Imposing Linear Boundary Conditions
        # Compose A using alpha and beta definitions

        tic = None
        if return_time:
            tic = time.perf_counter()

        # Calculate alpha and beta vectors
        first_term = (0.5 * (self.sigma ** 2) * (self.stock_prices ** 2)) / (self.dS ** 2)
        second_term = (self.r - self.q) * self.stock_prices / (2 * self.dS)

        alpha = first_term - second_term
        beta = first_term + second_term

        # Set alpha and beta entries to their adjusted versions as in the paper
        alpha = np.where(beta < 0, first_term - (2 * second_term), alpha)
        beta = np.where(alpha < 0, first_term + (2 * second_term), beta)

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
            if spectral_radius_B < 1.1:
                B = sla.inv(I_n + 0.5 * self.dT * A) * (I_n - 0.5 * self.dT * A)
                norm = infinity_norm_of_power(B, self.n_t)
                if norm > 1:
                    # Otherwise we're fine by the author's def. of stability
                    raise ValueError("The method is not stable." +
                                     "Spectral radius of B is " + str(spectral_radius_B) +
                                     ", but must be less than 1." +
                                     " The infinity norm of B^n_T is: " + str(norm) + ". The method is not stable.")
                else:
                    warnings.warn("1 < rho(B) < 1.1 but ||B^n_T|| < 1. Borderline Instability.")
            else:
                raise ValueError("The method is not stable. Spectral radius of B must be less than 1. (rho(B) > 1.1)")
        else:
            # We're good, make B = (I + .5A * dtau)^-1 * (I - .5A * dtau)
            B = np.matmul(sla.inv(I_n + 0.5 * self.dT * A), (I_n - 0.5 * self.dT * A))

        # Define the matrix which will hold our option values over the pricing grid:
        V = np.zeros((self.n_s, self.n_t))

        # The value of the option at the initial time is just the payoff:
        V[:, 0] = np.maximum(self.K - self.stock_prices, 0)

        # The value of the option at expiry is the strike price minus the discounted stock price
        V[:, -1] = np.maximum(self.K * np.exp(-self.r * self.T) - self.stock_prices * np.exp(-self.q * self.T), 0)

        # We have v_(k+1) = B * v_(k).
        for i in range(self.n_t-1):
            V[:, i + 1] = np.matmul(B, V[:, i])
            # Check for Early Exercise
            V[:, i + 1] = np.maximum(V[:, i + 1], self.K * np.exp(-self.r * (i * self.dT)) -
                                     self.stock_prices * np.exp(-self.q * (i * self.dT)))  # Consider early exercise
        if return_time:
            toc = time.perf_counter()
            return V, toc-tic
        else:
            return V


def test_crank_nicholson(save: bool = False):
    # CN Test
    goog_times = []
    apple_times = []
    stock_step_list = []
    time_step_list = []
    counter = 0
    stock_granularities = [1000, 2000, 3000, 4000, 5000, 50000]
    #                   Monthly Weekly Daily  Hourly
    time_granularities = [2 * 12, 2 * 52, 2 * 365, 2 * 365 * 24]
    temp_time: float = 0.0
    for stock_steps in stock_granularities:
        for time_steps in time_granularities:
            counter += 1
            stock_step_list.append(stock_steps)
            time_step_list.append(time_steps)
            print("N_S: " + str(stock_steps) + " N_T: " + str(time_steps))
            # Google Stats
            test_cranky = CrankNicholson(expiration_time_years_from_now=2,
                                         sigma=35.9,
                                         min_stock_price=0.0, max_stock_price=500.0,
                                         strike_price=105,
                                         risk_free_rate=0.03585,
                                         dividend_rate=0.0001,
                                         n_s=stock_steps,
                                         n_t=time_steps)
            prices, temp_time = test_cranky.american_call(return_time=True)
            goog_times.append(temp_time)
            if save and (counter == len(stock_granularities) * len(time_granularities)):
                np.savetxt("Crank_Test_GOOG_Results_Call_" + str(stock_steps) + "_nS_" + str(time_steps) + "_nt.csv",
                           prices, delimiter=",")

            # Apple Stats
            test_cranky = CrankNicholson(expiration_time_years_from_now=2,
                                         sigma=32.6,
                                         min_stock_price=0.0, max_stock_price=500.0,
                                         strike_price=170,
                                         risk_free_rate=0.03585,
                                         dividend_rate=0.055,
                                         n_s=stock_steps,
                                         n_t=time_steps)
            prices, temp_time = test_cranky.american_call(return_time=True)
            apple_times.append(temp_time)
            if save and (counter == len(stock_granularities) * len(time_granularities)):
                np.savetxt("Crank_Test_APPL_Results_Call_" + str(stock_steps) + "_nS_" + str(time_steps) + "_nt.csv",
                           prices, delimiter=",")

    time_data = np.array([stock_step_list, time_step_list, goog_times, apple_times]).T

    np.savetxt('Crank_Time_Results_Call.csv', time_data, delimiter=',', header='n_S, n_t, Google Time, Apple Time',
               fmt='%.8f')

    goog_times = []
    apple_times = []
    stock_step_list = []
    time_step_list = []
    counter = 0
    temp_time: float = 0.0
    for stock_steps in stock_granularities:
        for time_steps in time_granularities:
            counter += 1
            stock_step_list.append(stock_steps)
            time_step_list.append(time_steps)
            print("N_S: " + str(stock_steps) + " N_T: " + str(time_steps))
            # Google Stats
            test_cranky = CrankNicholson(expiration_time_years_from_now=2,
                                         sigma=35.9,
                                         min_stock_price=0.0, max_stock_price=500.0,
                                         strike_price=105,
                                         risk_free_rate=0.03585,
                                         dividend_rate=0.0001,
                                         n_s=stock_steps,
                                         n_t=time_steps)
            prices, temp_time = test_cranky.american_put(return_time=True)
            goog_times.append(temp_time)
            if save and (counter == len(stock_granularities) * len(time_granularities)):
                np.savetxt("Crank_Test_GOOG_Results_Put_" + str(stock_steps) + "_nS_" + str(time_steps) + "_nt.csv",
                           prices, delimiter=",")

            # Apple Stats
            test_cranky = CrankNicholson(expiration_time_years_from_now=2,
                                         sigma=32.6,
                                         min_stock_price=0.0, max_stock_price=500.0,
                                         strike_price=170,
                                         risk_free_rate=0.03585,
                                         dividend_rate=0.055,
                                         n_s=stock_steps,
                                         n_t=time_steps)
            prices, temp_time = test_cranky.american_put(return_time=True)
            apple_times.append(temp_time)
            if save and (counter == len(stock_granularities) * len(time_granularities)):
                np.savetxt("Crank_Test_APPL_Results_Put" + str(stock_steps) + "_nS_" + str(time_steps) + "_nt.csv",
                           prices, delimiter=",")

    time_data = np.array([stock_step_list, time_step_list, goog_times, apple_times]).T

    np.savetxt('Crank_Time_Results_Put.csv', time_data, delimiter=',', header='n_S, n_t, Google Time, Apple Time',
               fmt='%.8f')


def test_binomial(save: bool = False):
    # Binomial Test
    goog_times = []
    apple_times = []
    height_list = []
    counter = 0
    stock_prices = [104.70, 167.63]
    tree_heights = [20, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    temp_time: float = 0.0

    for height in tree_heights:
        counter += 1
        height_list.append(height)
        print("Height: " + str(height))
        # Google Stats
        test_binom = BinomialTree(expiration_time=np.datetime64('2025-04-19'),
                                  stock_price=stock_prices[0],
                                  strike_price=105.,
                                  risk_free_rate=0.03585,
                                  sigma=35.9,
                                  dividend_rate=0.0,
                                  tree_height=height,
                                  start_time=np.datetime64('2023-04-19'))

        prices, temp_time = test_binom.american_call(return_time=True, return_matrix=True)
        goog_times.append(temp_time)
        if save and (counter == len(tree_heights)):
            np.savetxt("Binomial_Test_GOOG_Results_Call_" + str(stock_prices[0]) + "_Price_" +
                       str(height) + "_Height.csv",
                       prices, delimiter=",")
        # Apple Stats
        test_binom = BinomialTree(expiration_time=np.datetime64('2025-04-19'),
                                  stock_price=stock_prices[1],
                                  strike_price=170.,
                                  risk_free_rate=0.03585,
                                  sigma=32.6,
                                  dividend_rate=0.055,
                                  tree_height=height,
                                  start_time=np.datetime64('2023-04-19'))

        prices, temp_time = test_binom.american_call(return_time=True, return_matrix=True)
        apple_times.append(temp_time)
        if save and (counter == 8):
            np.savetxt("Binomial_Test_APPL_Results_Call_" + str(stock_prices[1]) + "_Price_" +
                       str(height) + "_Height.csv",
                       prices, delimiter=",")

    time_data = np.array([height_list, goog_times, apple_times]).T

    np.savetxt('Binomial_Time_Results_Call.csv', time_data, delimiter=',',
               header='Tree Height,Google Time,Apple Time',
               fmt='%.8f')

    goog_times = []
    apple_times = []
    height_list = []
    counter = 0
    temp_time: float = 0.0
    for height in tree_heights:
        counter += 1
        height_list.append(height)
        print("Height: " + str(height))
        # Google Stats
        test_binom = BinomialTree(expiration_time=np.datetime64('2025-04-19'),
                                  stock_price=stock_prices[0],
                                  strike_price=105.,
                                  risk_free_rate=0.03585,
                                  sigma=35.9,
                                  dividend_rate=0.0,
                                  tree_height=height,
                                  start_time=np.datetime64('2023-04-19'))

        prices, temp_time = test_binom.american_put(return_time=True, return_matrix=True)
        goog_times.append(temp_time)
        if save and (counter == len(tree_heights)):
            np.savetxt("Binomial_Test_GOOG_Results_Put_" + str(stock_prices[0]) + "_Price_" +
                       str(height) + "_Height.csv",
                       prices, delimiter=",")

        # Apple Stats
        test_binom = BinomialTree(expiration_time=np.datetime64('2025-04-19'),
                                  stock_price=stock_prices[1],
                                  strike_price=170.,
                                  risk_free_rate=0.03585,
                                  sigma=32.6,
                                  dividend_rate=0.055,
                                  tree_height=height,
                                  start_time=np.datetime64('2023-04-19'))

        prices, temp_time = test_binom.american_put(return_time=True, return_matrix=True)
        apple_times.append(temp_time)
        if save and (counter == 8):
            np.savetxt("Binomial_Test_APPL_Results_Put_" + str(stock_prices[1]) + "_Price_" +
                       str(height) + "_Height.csv",
                       prices, delimiter=",")

    time_data = np.array([height_list, goog_times, apple_times]).T

    np.savetxt('Binomial_Time_Results_Put.csv', time_data, delimiter=',',
               header='Tree Height,Google Time,Apple Time',
               fmt='%.8f')


if __name__ == "__main__":
    test_binomial(save=False)
    test_crank_nicholson(save=False)