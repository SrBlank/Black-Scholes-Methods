# A Comparative Study of Numerical Methods for Solving the Black-Scholes Partial Differential Equation for American Options Pricing

## Introduction
Financial options are derivatives whose values are based on the prices of underlying assets.
European options and American options are two types of financial options with different
characteristics. European options can only be exercised at the expiration date, while American
options can be exercised at any time before the expiration date. This difference has significant
implications for pricing and hedging strategies.


The Black-Scholes model is a widely used mathematical model for pricing financial options. The
Black-Scholes model assumes that the underlying asset follows a geometric Brownian motion
and that the market is efficient. Black and Scholes showed in their seminal paper (Black &
Scholes, 1973) that the pricing distribution of the underlying asset follows a lognormal
distribution. Under these assumptions, the value of a European call option can be determined by
solving the Black-Scholes partial differential equation (PDE)


where V is the value of the option, S is the price of the underlying asset, t is time, σ is the
volatility of the underlying asset, and r is the risk-free interest rate.
However, the Black-Scholes model assumes that the option can only be exercised at expiration,
which is not the case for American options. The problem of pricing American options requires
finding the optimal exercise time, which is the time at which the option holder should exercise
the option to maximize their profit. This problem is more complicated than pricing European
options because the option holder can exercise the option at any time before expiration, which
creates the possibility of early exercise.


The optimal exercise problem for American options can be formulated as an optimal stopping
problem, where the option holder must decide when to stop waiting and exercise the option. An
American option's value is modeled as the maximum of the value gained from immediate
exercise or waiting until a later time to exercise. The challenge in solving this problem is that the
optimal exercise time depends on the future price of the underlying asset, which is uncertain.
Numerical methods such as finite difference methods, spectral methods, and Monte Carlo
methods have been used to approximate solutions to the Black-Scholes PDE for American
options. These methods allow for the incorporation of the possibility of early exercise into the
pricing model by simulating the future prices of the underlying asset and determining the optimal
exercise time at each time step. The choice of numerical method depends on the characteristics 

of the problem, such as the complexity of the underlying asset price dynamics and the desired
accuracy of the solution.
In our project, we will explore the use of numerical methods to solve the problem of pricing
American options, focusing on the optimal exercise problem. We will compare and contrast the
performance of finite difference methods and lattice methods in approximating the solution to
the Black-Scholes PDE for American options with various boundary conditions. Our goal is to
provide insights into each method's strengths and weaknesses and identify the circumstances
under which each method is most appropriate while recording which experiments yield the most
valuable exercises.


## Methods


We chose to use finite difference methods and a lattice method to approximate solutions to the
Black-Scholes PDE.


To approximate using finite difference methods, we will use both an explicit Runge-Kutta
method as formulated by (Martin-Vaquero, Khaliq, & Kleefeld, 2014), and the Crank-Nicolson
method as formulated in their paper (Crank & Nicolson, 1946). Runge-Kutta is a time-stepping
technique that approximates the solution of the PDE at each time step based on its previous
value, while Crank-Nicolson is a time-stepping technique that takes into account the midpoint
between two-time steps to calculate the solution. We will implement both methods in MATLAB
or Python and compare their accuracy, efficiency, and computational complexity.


In addition to finite difference methods, we will also use the binomial tree method to
approximate the Black-Scholes equation, as proposed by (Cox, Ross, & Rubinstein, 1979). The
binomial tree method is a discrete-time model that divides the time horizon of the option into
smaller time steps and models the evolution of the underlying asset price over time as a binomial
tree. It is one of the most widely used lattice methods to price American options. We will
implement this method in MATLAB or Python and compare its accuracy, efficiency, and
computational complexity with the finite difference methods.


By comparing the results obtained from different numerical methods, we aim to gain insights
into their relative strengths and limitations for option pricing in the financial markets. We will
evaluate the performance of each method by comparing its results with the closed-form solution
of the Black-Scholes equation, when available, and by conducting sensitivity analyses on
different input parameters, such as volatility and interest rates