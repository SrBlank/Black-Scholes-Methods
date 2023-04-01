# Authors: Saad Rafiq and Josh Avery
# Numerical Analysis II
# Desc: Implementation of the bionomial tree, crank-nicolson, and
#   runge kutta method on the black scholes equation

import numpy as np

# V: option price as a function of S 
# S: stock price
# t: time to maturity
# r: risk free interest rate, .035 -> 3.5%
# sigma: volatility
def black_scholes_pde(V, S, t, r, sigma):
    def f(x):
        return V(x[1], x[0])
    
    V_t = np.gradient(V(S,t), t)
    V_s = np.gradient(V(S,t), S)
    V_ss = np.gradient(V_s, S)

    pde = V_t + (.5)*(S**2)*(sigma**2)*(V_ss) + r*S*V_s - r*V # r*f([t,S]) 

    return pde

def f(initial):
    pass

# a: alpha
#   1/(.4s^2)
# s: is the degree? large number 10 - 250
def runge_kutta(y0: int, h: int, a: int):
    g = []
    g[0] = y0
    g[1] = g[0] + a*h*f(g[0])

    m = 5 # temp placeholder
    s = 10 # temp placeholder

    # i = 2,...,m
    for i in range(2, m):
        g[i] = 2*g[i-1] + g[i-2] + 2*a*h*f(g[i-1])
    g[m+1] = g[m] + a*h*f(g[m])

    # i = m + 2,...,2m
    for i in range(m+2, 2*m):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])
    g[2*m+1] = g[m] + a*h*f(g[m])

    # i = 2m + 2,...,3m
    for i in range(2*m+2, 3*m):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])
    g[3*m+1] = g[m] + a*h*f(g[m])

    # i = 3m + 2,...,4m
    for i in range(3*m+2, 4*m):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])
    g[4*m+1] = g[m] + a*h*f(g[m])

    # i = 4m + 2,...,5m
    for i in range(4*m+2, 5*m):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])
    g[5*m+1] = g[m] + a*h*f(g[m])

    # i = 5m + 2,...,6m
    for i in range(5*m+2, 6*m):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])
    g[6*m+1] = g[m] + a*h*f(g[m])

    # i = 6m + 2,...,7m
    for i in range(6*m+2, 7*m):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])
    g[7*m+1] = g[m] + a*h*f(g[m])

    # i = 7m + 2,...,8m
    for i in range(7*m+2, 8*m):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])
    g[8*m+1] = g[m] + a*h*f(g[m])

    # i = 8m + 2,...,9m
    for i in range(8*m+2, 9*m):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])
    g[9*m+1] = g[m] + a*h*f(g[m])
    
    # i = 9m + 2,...,s
    for i in range(9*m+2, s):
        g[i] = 2*g[i-1] - g[i-2] + 2*a*h*f(g[i-1])

if __name__ == "__main__":
    pass
