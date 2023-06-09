# Normalization Effects on Physics-Informed Neural Networks

Infinite-width neural netwoks have been getting considerable attention, as they have been shown to behave in a deterministic way under gradient descent. For this deterministic behaviour to appear, the outputs of the inner layer has to be scaled in proportion to the number of nodes in that layer. For a neural network with two hidden layers, this would look like this:

$$
g^{N_{1}, N_{2}}(x) = \mathbf{W}^{(3)} \cdot \frac{1}{N_{2}^{\gamma_2}} \boldsymbol{\sigma}\left(\mathbf{W}^{(2)} \cdot \frac{1}{N_{1}^{\gamma_1}}\boldsymbol{\sigma} \left(\mathbf{W}^{(1)} \cdot \mathbf{x}\right)\right)
$$

The exponents $\gamma_1, \gamma_2$ are hyperparameters that one can tweak.

For the case $\gamma_1 = \gamma_2 = 1$, the network's evolution has been connected to mean-field theory, an approximation ethod used in statistical physics.

The case $\gamma_1 = \gamma_2 = \frac{1}{2}$ has been linked to kernel methods, Gaussian processes and kernel regression.

The case $\gamma_1, \gamma_2 \in \left(\frac{1}{2}, 1\right)$ has only been considered recently by Yu and Spiliopoulos (2022), who derived an asymptotic expasion in terms of $\gamma_2$ as $N_2 \to \infty$. They found that the network output's variance is decreasing in $\gamma_2$, therefore has the best accuracy when $\gamma_2 = 1$. Additionally, the effect of $\gamma_1$ on the output is smaller than $\gamma_2$.

My thesis investigated the effects of varying normalization exponents in the physics-informed setting, e.g. approximating the solution of a partial differential equation (PDE) by a neural network as the width tends to infinity.

I trained 2-layer neural networks for the combinations $\{\gamma_1, \gamma_2\} = \{0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}^2$.

For the Poisson equation, I found a very large difference between the "$\gamma_1 \approx  0.5$ and $\gamma_2 \approx 1.0$" region of the scaling landscape, and the rest, while 4 out of the 5 lowest errors had $\gamma_2 = 0.7$.

| $\gamma_2$ \ $\gamma_1$ | $0.5$   | $0.6$             | $0.7$         | $0.8$            | $0.9$            | $1.0$   |
|------------------------------------------------------------------------------------------------|---------|-------------------|---------------|------------------|------------------|---------|
| $0.5$                                                                                          | 7.35e-8 | 1.39e-7           | 3.98e-8       | 2.86e-8          | 4.41e-8          | 2.32e-8 |
| $0.6$                                                                                          | 3.76e-8 | 6.24e-8           | 7.66e-8       | 3.07e-8          | 3.14e-8          | 3.02e-8 |
| $0.7$                                                                                          | 3.01e-8 | **6.17e-10** | **5e-9** | **2.28e-8** | **1.78e-8** | 5.42e-8 |
| $0.8$                                                                                          | 2.87e-4 | 4.98e-5           | 3.86e-7       | 1.07e-7          | 7.92e-8          | 4.58e-8 |
| $0.9$                                                                                          | 6.97e-3 | 1.14e-2           | 9.68e-2       | 5.57e-1          | **6e-9**    | 2.38e-8 |
| $1.0$                                                                                          | 1.2e-1  | 2.12e-1           | 3.32e-1       | 2.17e-1          | 1.63e-7          | 7.04e-7 |

Table: MSE losses for the Poisson equation

The Burgers equation had vastly different dependencies on the exponents. Once again the exponent $0.7$ has the best performance, but for $\gamma_1$ this time. The network's performance seems to show some robustness in terms of $\gamma_2$, whether it is a good performance or not.

| $\gamma_2$ \ $\gamma_1$ | $0.5$   | $0.6$            | $0.7$            | $0.8$    | $0.9$   | $1.0$    |
|------------------------------------------------------------------------------------------------|---------|------------------|------------------|----------|---------|----------|
| $0.5$                                                                                          | 5.51e-2 | **2.06e-3**   | 2.29e-2          | 1.270e-1 | 1.4e-1  | 1.28e-1  |
| $0.6$                                                                                          | 1.3e-1  | **1.33e-3** | 9.73e-2          | 1.272e-1 | 1.25e-1 | 1.226e-1 |
| $0.7$                                                                                          | 1.46e-1 | 4.21e-2          | **2.6e-3**  | 1.212e-1 | 1.35e-1 | 1.217e-1 |
| $0.8$                                                                                          | 1.25e-1 | 1.2e-1           | **1.12e-3** | 8.31e-2  | 1.23e-1 | 1.228e-1 |
| $0.9$                                                                                          | 1.22e-1 | 1.14e-1          | **1.16e-3** | 1.21e-2  | 1.19e-1 | 1.14e-1  |
| $1.0$                                                                                          | 8.59e-2 | 7.12e-2          | 7.03e-3          | 2.07e-2  | 1.08e-1 | 1.04e-1  |

Table: MSE loss for the Burgers equation