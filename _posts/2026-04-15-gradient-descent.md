---
layout: post
title: "Gradient Descent — Learning by Sliding Downhill"
date: 2026-04-15
category: ml
category-label: Machine Learning
excerpt: "How gradient descent optimizes neural networks, starting from first principles of calculus. Covers batch, stochastic, and mini-batch variants along with the mathematics of learning rates and convergence."
---

Every time a neural network "learns," it is running some variant of gradient descent. The idea is disarmingly simple: measure how wrong the model is, figure out which direction to adjust the parameters to reduce that error, and take a small step in that direction. Repeat until the error is small enough. But the simplicity hides rich mathematical structure worth understanding properly.

## The Setup

Suppose we have a model with parameters $\boldsymbol{\theta} = (\theta_1, \theta_2, \ldots, \theta_d)$ and a loss function $L(\boldsymbol{\theta})$ that measures how poorly the model performs. Our goal is to find:

<div class="math-block">
$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta})$$
</div>

For most neural networks, $L$ is non-convex and high-dimensional, so we cannot solve this analytically. Instead, we use an iterative approach.

## The Gradient

The gradient of $L$ with respect to $\boldsymbol{\theta}$ is the vector of partial derivatives:

<div class="math-block">
$$\nabla_{\boldsymbol{\theta}} L = \left(\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \ldots, \frac{\partial L}{\partial \theta_d}\right)$$
</div>

The gradient has a crucial geometric meaning: it points in the direction of steepest *ascent* of the loss function. Therefore, the negative gradient $-\nabla_{\boldsymbol{\theta}} L$ points in the direction of steepest *descent* — exactly where we want to go to reduce the loss.

## The Update Rule

Gradient descent updates the parameters by taking a step in the negative gradient direction:

<div class="math-block">
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)$$
</div>

where $\eta > 0$ is the **learning rate** — a scalar that controls the step size. This single equation is the heart of nearly all neural network training.

## A Concrete Example

Let us work through a simple 1D example. Consider the loss function $L(\theta) = (\theta - 3)^2$. The minimum is obviously at $\theta = 3$, but let us see how gradient descent finds it.

The gradient is:

<div class="math-block">
$$\frac{dL}{d\theta} = 2(\theta - 3)$$
</div>

Starting at $\theta_0 = 0$ with learning rate $\eta = 0.1$:

<div class="math-block">
$$\theta_1 = 0 - 0.1 \cdot 2(0 - 3) = 0 + 0.6 = 0.6$$
</div>

<div class="math-block">
$$\theta_2 = 0.6 - 0.1 \cdot 2(0.6 - 3) = 0.6 + 0.48 = 1.08$$
</div>

<div class="math-block">
$$\theta_3 = 1.08 - 0.1 \cdot 2(1.08 - 3) = 1.08 + 0.384 = 1.464$$
</div>

Each step moves $\theta$ closer to 3. Notice how the steps get smaller as we approach the minimum — because the gradient magnitude $\|2(\theta - 3)\|$ decreases. This natural deceleration is a feature, not a bug.

## The Learning Rate Matters — A Lot

The learning rate $\eta$ is arguably the most important hyperparameter in deep learning. Consider what happens with different choices:

### Too small ($\eta \to 0$)

Steps are tiny, convergence is extremely slow. For the quadratic example with $L(\theta) = (\theta - 3)^2$, the convergence rate is $(1 - 2\eta)^t$. With $\eta = 0.001$, after 1000 steps the error is still $(0.998)^{1000} \approx 0.135$ of its initial value.

### Too large ($\eta$ large)

Steps overshoot the minimum. For our quadratic, if $\eta > 0.5$, the update factor $(1 - 2\eta)$ becomes negative — the parameter oscillates around the minimum. If $\eta > 1$, we get $\|1 - 2\eta\| > 1$ and the process *diverges*: the loss increases at every step.

### Just right

For a quadratic $L(\theta) = a\theta^2$, the optimal learning rate is $\eta = \frac{1}{2a}$, which gives exact convergence in one step. In general, the optimal rate depends on the curvature (second derivative) of the loss surface — a connection that leads to second-order methods like Newton's method.

## From One Parameter to Millions

In a real neural network, $\boldsymbol{\theta}$ might have millions of components. The update rule remains the same:

<div class="math-block">
$$\theta_j^{(t+1)} = \theta_j^{(t)} - \eta \cdot \frac{\partial L}{\partial \theta_j} \quad \text{for } j = 1, 2, \ldots, d$$
</div>

Each parameter gets its own gradient component telling it which direction to move and by how much. The gradient is computed efficiently using **backpropagation**, which is just the chain rule applied systematically through the network's computational graph.

## Three Flavors of Gradient Descent

In practice, the loss function is computed over a dataset of $N$ training examples:

<div class="math-block">
$$L(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \ell(\boldsymbol{\theta}; x_i, y_i)$$
</div>

where $\ell(\boldsymbol{\theta}; x_i, y_i)$ is the loss on the $i$-th example. This sum gives rise to three variants.

### 1. Batch Gradient Descent

Computes the gradient using the entire dataset:

<div class="math-block">
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}_t; x_i, y_i)$$
</div>

This gives the exact gradient but is prohibitively expensive for large datasets. Processing millions of examples for a single parameter update is wasteful.

### 2. Stochastic Gradient Descent (SGD)

Uses a single randomly chosen example per update:

<div class="math-block">
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}_t; x_i, y_i)$$
</div>

This is very fast per step but noisy — the gradient from a single example is a poor estimate of the true gradient. However, this noise can actually help escape local minima and saddle points.

### 3. Mini-batch Gradient Descent

The practical sweet spot. Uses a small random subset (mini-batch) of size $B$:

<div class="math-block">
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}_t; x_i, y_i)$$
</div>

Typical batch sizes are 32, 64, 128, or 256. This balances gradient quality against computational cost and also exploits GPU parallelism efficiently.

## Why the Noise in SGD Actually Helps

At first glance, SGD seems like a worse version of batch gradient descent — noisier and less accurate. But the noise serves a purpose. The loss landscape of a deep neural network is riddled with:

- **Saddle points:** Points where the gradient is zero but the point is neither a minimum nor a maximum. In high dimensions, saddle points vastly outnumber local minima. The noise in SGD helps escape them.
- **Sharp minima:** Narrow valleys where the loss is low but the model generalizes poorly. SGD's noise tends to bounce out of sharp minima and settle in flat, wide minima — which correspond to better generalization.

## Momentum — Remembering Where You Have Been

Vanilla gradient descent can oscillate in ravines — narrow valleys where the gradient is steep across the valley but shallow along it. Momentum fixes this by accumulating a velocity vector:

<div class="math-block">
$$v_{t+1} = \beta \cdot v_t + \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)$$
</div>

<div class="math-block">
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot v_{t+1}$$
</div>

The coefficient $\beta$ (typically 0.9) controls how much history to keep. The velocity builds up in directions with consistent gradients and cancels out in directions that oscillate. Think of a ball rolling downhill — it accumulates speed along the slope direction.

## A Note on Convergence

For convex functions with Lipschitz-continuous gradients (meaning the gradient does not change too abruptly), gradient descent with learning rate $\eta = \frac{1}{L}$ (where $L$ is the Lipschitz constant) converges at a rate of:

<div class="math-block">
$$L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*) \leq \frac{L \|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{2t}$$
</div>

This is $O(1/t)$ convergence — to halve the error you need to double the iterations. For strongly convex functions the rate improves to linear (exponential) convergence $O(\rho^t)$ with $\rho < 1$.

Neural network loss surfaces are non-convex, so these theoretical guarantees do not directly apply. In practice, gradient descent works remarkably well on neural networks — a phenomenon that is still an active area of theoretical research.

## The Complete Picture

Modern deep learning uses many enhancements beyond vanilla gradient descent: Adam combines momentum with per-parameter adaptive learning rates, learning rate schedules warm up and then decay the step size, and gradient clipping prevents explosions in recurrent networks. But at the foundation of all of them sits the same elegant idea — compute the gradient, step opposite to it, repeat.

Gradient descent is the engine that turns data into learned representations. Understanding its mechanics — the update rule, the role of the learning rate, the interplay between noise and convergence — is not just academic. It is the difference between a model that trains smoothly and one that diverges, stalls, or memorizes the training set.
