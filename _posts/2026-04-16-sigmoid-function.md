---
layout: post
title: "The Sigmoid Function — Anatomy of a Curve That Powers Neural Networks"
date: 2026-04-16
category: math
category-label: Mathematics
excerpt: "A thorough look at the sigmoid activation function — its mathematical definition, derivative, geometric intuition, and why it was the default choice for early neural networks."
---

If you have spent any time studying neural networks, you have encountered the sigmoid function. It was the default activation function for decades before ReLU took over. But even today, sigmoid appears in logistic regression, output layers for binary classification, and inside gating mechanisms like LSTMs and GRUs. Understanding it deeply — its shape, its derivative, its flaws — gives you real intuition about how networks learn.

## Definition

The sigmoid function, often denoted $\sigma(x)$, maps any real number to the interval $(0, 1)$. Its formula is:

<div class="math-block">
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
</div>

Here $e$ is Euler's number ($\approx 2.71828$) and $x$ is the input — in a neural network context this is typically the weighted sum $z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$.

## Geometric Intuition

The sigmoid curve is an S-shaped (sigmoidal) curve with three distinct regions:

- **Left tail** ($x \to -\infty$): The output approaches $0$. The exponential $e^{-x}$ grows extremely large, making the denominator huge and the fraction tiny.
- **Center** ($x \approx 0$): The output is near $0.5$ and the curve is steepest. At exactly $x = 0$, we get $\sigma(0) = \frac{1}{1+1} = 0.5$.
- **Right tail** ($x \to +\infty$): The output approaches $1$. The exponential $e^{-x}$ vanishes, leaving $\frac{1}{1+0} = 1$.

This "squashing" behavior is what makes sigmoid useful — it compresses any real-valued input into a probability-like output between 0 and 1.

## The Derivative — An Elegant Result

One of the most beautiful properties of the sigmoid is that its derivative can be expressed entirely in terms of itself. Let us derive it step by step.

Starting with:

<div class="math-block">
$$\sigma(x) = (1 + e^{-x})^{-1}$$
</div>

Applying the chain rule:

<div class="math-block">
$$\frac{d\sigma}{dx} = -1 \cdot (1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^{2}}$$
</div>

Now we rewrite this as a product:

<div class="math-block">
$$\frac{d\sigma}{dx} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}}$$
</div>

The first factor is $\sigma(x)$. For the second, observe that:

<div class="math-block">
$$\frac{e^{-x}}{1 + e^{-x}} = \frac{(1 + e^{-x}) - 1}{1 + e^{-x}} = 1 - \sigma(x)$$
</div>

Therefore:

<div class="math-block">
$$\boxed{\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))}$$
</div>

This is computationally wonderful. Once you compute the forward pass $\sigma(x)$, you get the derivative for free — just multiply the output by one minus itself. No additional exponentials needed during backpropagation.

## Maximum Value of the Derivative

The derivative $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ is a product of two values that always sum to 1. By the AM-GM inequality, this product is maximized when both factors are equal, i.e., $\sigma(x) = 0.5$, which happens at $x = 0$:

<div class="math-block">
$$\max \sigma'(x) = 0.5 \times 0.5 = 0.25$$
</div>

This maximum of $0.25$ is critically important — it means the gradient is *always* at most one quarter. In a deep network with $n$ sigmoid layers, the gradient flowing backward gets multiplied by at most $0.25$ at each layer:

<div class="math-block">
$$\text{gradient} \leq (0.25)^n$$
</div>

For $n = 10$ layers, this is $(0.25)^{10} \approx 9.5 \times 10^{-7}$. The gradient practically vanishes. This is the infamous **vanishing gradient problem**.

## Sigmoid as a Probability

In logistic regression, we model the probability that input $\mathbf{x}$ belongs to class 1 as:

<div class="math-block">
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$
</div>

Since $\sigma$ outputs values in $(0, 1)$, this is a valid probability. The decision boundary is the set of points where $\sigma(\mathbf{w}^T \mathbf{x} + b) = 0.5$, which corresponds to $\mathbf{w}^T \mathbf{x} + b = 0$ — a hyperplane in input space.

## Relationship to the Logit Function

The inverse of the sigmoid is the **logit** function. If $p = \sigma(x)$, then solving for $x$:

<div class="math-block">
$$p = \frac{1}{1 + e^{-x}} \implies e^{-x} = \frac{1-p}{p} \implies x = \ln\left(\frac{p}{1-p}\right) = \text{logit}(p)$$
</div>

The quantity $\frac{p}{1-p}$ is called the **odds**, and $x$ is the log-odds. This connection is fundamental in statistics — logistic regression is literally a linear model in the log-odds space.

## Sigmoid in the Softmax Family

For binary classification, the softmax function over two classes reduces exactly to sigmoid. If we have logits $z_1$ and $z_0$ for two classes:

<div class="math-block">
$$\text{softmax}(z_1) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \frac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)$$
</div>

This is why binary classifiers only need a single output neuron with sigmoid, rather than two outputs with softmax — they are mathematically equivalent.

## Why Sigmoid Lost Its Crown

Despite its elegance, sigmoid has been largely replaced by ReLU ($\max(0, x)$) as the default hidden-layer activation. The reasons are:

1. **Vanishing gradients:** As shown above, gradients shrink exponentially in deep networks.
2. **Non-zero centered outputs:** Sigmoid outputs are always positive (between 0 and 1), which means all gradients for weights in the next layer have the same sign. This creates inefficient zig-zagging during optimization.
3. **Computational cost:** Computing $e^{-x}$ is more expensive than $\max(0, x)$.

However, sigmoid remains essential in output layers for binary classification, in gating mechanisms (LSTM forget gates, GRU update gates), and in attention mechanisms where a soft binary decision is needed.

## Summary

The sigmoid function packs a remarkable amount of mathematical depth into a simple formula. Its self-referential derivative $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ is both computationally efficient and the root cause of vanishing gradients. Understanding this duality — the elegance and the limitation — is essential for anyone building or debugging deep learning models.
