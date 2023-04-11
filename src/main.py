# Authors: Saad Rafiq and Josh Avery
# Numerical Analysis II
# Desc: Implementation of the binomial tree, crank-nicholson, and
#   runge kutta method on the black scholes equation

import numpy as np
import pandas as pd

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
        # Requirement: deltaT < sigma^2/(r-q)^2 or else p is not on (0,1).
        # T... expiration time
        # S... stock price
        # K... strike price
        # q... dividend yield
        # n... height of the binomial tree

        deltaT = self.T / self.n
        up = np.exp(self.sigma * np.sqrt(deltaT))
        p0 = (up * np.exp(-self.q * deltaT) - np.exp(-self.r * deltaT)) / (up**2 - 1)
        p1 = np.exp(-self.r * deltaT) - p0

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
        # Requirement: deltaT < sigma^2/(r-q)^2 or else p is not on (0,1).
        # T... expiration time
        # S... stock price
        # K... strike price
        # q... dividend yield
        # n... height of the binomial tree

        deltaT = self.T / self.n
        up = np.exp(self.sigma * np.sqrt(deltaT))
        p0 = (up * np.exp(-self.q * deltaT) - np.exp(-self.r * deltaT)) / (up**2 - 1)
        p1 = np.exp(-self.r * deltaT) - p0

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


if __name__ == "__main__":
    pass
