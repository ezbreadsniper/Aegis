# Mathematical Foundations: Fractal and Quantum Optimization

## 1. Fractal Market Analysis (FMA)

### 1.1. The Fractional Ornstein-Uhlenbeck (fOU) Process
The price dynamics $X_t$ are modeled using a fractional Ornstein-Uhlenbeck process, which incorporates mean reversion and long-range dependence:
$$dX_t = \theta(\mu - X_t)dt + \sigma dB_t^H$$
where:
- $\theta > 0$ is the rate of mean reversion.
- $\mu$ is the long-term mean price.
- $\sigma > 0$ is the volatility.
- $B_t^H$ is a **Fractional Brownian Motion (fBm)** with Hurst exponent $H \in (0, 1)$.

### 1.2. Hurst Exponent and Market Regime
The Hurst exponent $H$ characterizes the "memory" of the market:
- **$H = 0.5$**: Standard Brownian Motion (Random Walk).
- **$H > 0.5$**: Persistent (Trending) regime. The market has positive correlation; past increases imply future increases.
- **$H < 0.5$**: Anti-persistent (Mean-reverting) regime. The market has negative correlation; past increases imply future decreases.

### 1.3. Multi-Scale Volatility
Volatility is modeled as a multi-scale fractal process:
$$\sigma(t) = \sum_{i=1}^N a_i \phi_i(t)$$
where $\phi_i(t)$ are basis functions representing different time scales (e.g., 1m, 15m, 1h, 1d).

***

## 2. Quantum-Inspired Optimization (QIO)

### 2.1. Quadratic Unconstrained Binary Optimization (QUBO)
The order routing problem is mapped to a QUBO formulation, which is the standard input for quantum-inspired solvers:
$$\min_{x \in \{0, 1\}^n} x^T Q x + c^T x$$
where:
- $x$ is a binary vector representing order allocation decisions.
- $Q$ is a matrix representing the costs and constraints (e.g., slippage, fees, liquidity).
- $c$ is a vector representing linear costs.

### 2.2. The Routing Cost Function
The cost function for routing an order of size $S$ across $K$ liquidity pools is:
$$C(x) = \sum_{k=1}^K (f_k x_k + s_k(x_k S)^2) + \lambda (\sum_{k=1}^K x_k - 1)^2$$
where:
- $f_k$ is the fixed fee for pool $k$.
- $s_k$ is the slippage coefficient for pool $k$.
- $\lambda$ is a penalty parameter to ensure the total order size is filled.

### 2.3. Quantum-Inspired Parallel Annealing
The solver uses **Parallel Annealing** to find the global minimum of $C(x)$ by simulating multiple "replicas" of the system at different "temperatures" and allowing them to exchange states. This enables the solver to escape local minima and find the near-optimal routing in microseconds.
