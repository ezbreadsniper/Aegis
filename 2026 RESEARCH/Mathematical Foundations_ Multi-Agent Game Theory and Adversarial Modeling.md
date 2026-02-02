# Mathematical Foundations: Multi-Agent Game Theory and Adversarial Modeling

## 1. Multi-Agent Game Theory (MAGT)

### 1.1. The Stochastic Game Framework
The Polymarket ecosystem is modeled as a **Stochastic Game** $(S, A, P, R, \gamma)$, where:
- $S$ is the state space (order book, sentiment, macro data).
- $A = A_1 \times A_2 \times \dots \times A_n$ is the joint action space of all $n$ agents (bots and humans).
- $P(s' | s, a)$ is the transition probability to state $s'$ given state $s$ and joint action $a$.
- $R_i(s, a)$ is the reward function for agent $i$.
- $\gamma$ is the discount factor.

### 1.2. Nash Equilibrium and GTO
A **Nash Equilibrium** is a joint strategy $\pi^* = (\pi_1^*, \dots, \pi_n^*)$ such that for every agent $i$:
$$V_i^{\pi^*}(s) \geq V_i^{(\pi_i, \pi_{-i}^*)}(s) \quad \forall \pi_i, \forall s$$
In prediction markets, a **Game Theory Optimal (GTO)** strategy is a Nash Equilibrium strategy that is unexploitable by any other agent.

### 1.3. Counterfactual Regret Minimization (CFR)
The GTO policy is solved using **CFR**, which iteratively minimizes the "regret" of not having taken a particular action in the past:
$$R_i^T(a) = \sum_{t=1}^T (r_i^t(a) - r_i^t(a_i^t))$$
where $r_i^t(a)$ is the reward agent $i$ would have received at time $t$ by taking action $a$.

***

## 2. Adversarial Modeling

### 2.1. Adversarial Attack Formulation
An adversarial attack on a trading bot's model $f(s)$ is a small perturbation $\delta$ to the state $s$ such that:
$$\max_{\|\delta\| \leq \epsilon} \mathcal{L}(f(s + \delta), y)$$
where $\mathcal{L}$ is the loss function and $y$ is the target output. In HFT, $\delta$ represents artificial price movements or "spoofed" orders.

### 2.2. Adversarial Training (Robustness)
The bot's models are made robust through **Adversarial Training**, which involves solving a min-max problem:
$$\min_{\theta} \mathbb{E}_{(s, y) \sim \mathcal{D}} \left[ \max_{\|\delta\| \leq \epsilon} \mathcal{L}(f_\theta(s + \delta), y) \right]$$
This ensures the bot's policy remains stable even in the presence of adversarial actors.

### 2.3. Exploitation of Predictable Bots
The bot identifies and exploits "weak" bots by modeling their policy $\pi_{weak}$ and finding the **Best Response** $\pi_{BR}$:
$$\pi_{BR} = \arg\max_{\pi_i} V_i^{(\pi_i, \pi_{weak})}(s)$$
This allows the bot to generate "adversarial" orders that trigger predictable, loss-making responses from the weak bot.
