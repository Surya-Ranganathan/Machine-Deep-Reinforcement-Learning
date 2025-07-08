# Gradient Descent

Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent (i.e., the negative of the gradient).

## Mathematical Definition
Let’s say you have a function:

𝑓(𝑥)
You want to find the value of 
𝑥
x where 
𝑓
(
𝑥
)
f(x) is minimum.

The update rule in gradient descent is:

𝑥
=
𝑥
−# 🧠 What is Gradient Descent?

**Gradient Descent** is an optimization algorithm used to **minimize** a function by iteratively moving in the direction of the **steepest descent** — the negative of the gradient.

---

## 📉 Simple Intuition

Imagine you're standing on a hill in the fog, trying to reach the **lowest point (the valley)**. You can't see far, but you can feel the slope. At every step, you:

- Check the slope (gradient)
- Take a small step **downhill**
- Repeat this until you can't go lower

This is exactly how gradient descent works.

---

## 🧾 Mathematical Definition

Given a function:
\[
f(x)
\]

You want to find \( x \) such that \( f(x) \) is **minimized**.

The gradient descent update rule is:

\[
x = x - \alpha \cdot f'(x)
\]

Where:
- \( x \) is the parameter you're updating
- \( \alpha \) is the **learning rate** (step size)
- \( f'(x) \) is the **derivative** (gradient) of the function

---

## 📍 Example: Minimize \( f(x) = x^2 \)

- Function: \( f(x) = x^2 \)
- Derivative: \( f'(x) = 2x \)
- Update rule: \( x = x - \alpha \cdot 2x \)

Over iterations, this moves \( x \) closer to **0**, the minimum of the function.

---

## 🔍 Applications

Gradient Descent is widely used in:

- **Machine Learning** – To optimize loss functions in models like linear regression, logistic regression, and neural networks.
- **Deep Learning** – In backpropagation to adjust neural network weights.
- **Engineering & Control Systems**
- **Scientific Optimization Problems**

---

## ⚠️ Key Concepts

| Term | Description |
|------|-------------|
| **Gradient** | The slope or rate of change of a function |
| **Learning Rate (α)** | Controls how big each step is |
| **Convergence** | When the updates become minimal and the solution stabilizes |
| **Local Minimum** | A point where \( f(x) \) is lower than nearby points, but not necessarily the lowest overall |

---

## 📈 Visualization

Gradient Descent paths can be plotted on functions like \( f(x) = x^2 \) to see how the value of \( x \) approaches the minimum with each iteration.

---

## 🧠 Summary

Gradient Descent helps find the **minimum** of a function by using its derivative to take small, iterative steps downhill. It’s a cornerstone of modern machine learning and optimization.

---
𝛼
⋅
𝑓
′
(
𝑥
)
x=x−α⋅f 
′
 (x)
Where:

𝑥
x is the parameter you're adjusting

𝑓
′
(
𝑥
)
f 
′
 (x) is the derivative of the function (gradient)

𝛼
α is the learning rate (how big a step you take)

