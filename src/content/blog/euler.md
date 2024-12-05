---
author: Shrey Vishen
pubDatetime: 2024-12-04T23:38:06.000+00:00
title: Understanding Euler's Identity
slug: "eitopiisneg1"
featured: true
tags:
  - euler's identity
  - taylor series
  - complex analysis
description: "Understanding Euler's Identity with Taylor Polynomials: An introduction to complex analysis"
---

## Introduction

Euler's identity,

$$
e^{i\pi} + 1 = 0,
$$

is often celebrated as one of the most elegant equations in mathematics. It connects five fundamental constants: $e$ (the base of natural logarithms), $i$ (the imaginary unit), $\pi$ (a cornerstone of geometry), and the numbers $1$ and $0$. Discovered by Leonhard Euler in the 18th century, this identity bridges algebra, geometry, and calculus in a profound way.

In this paper, we will:

1. Explore the mathematical foundations behind Euler’s identity, such as Taylor series and complex numbers.
2. Derive key expansions for $e^x$, $\sin(x)$, and $\cos(x)$.
3. Prove Euler’s formula and its special case $e^{i\pi} = -1$.

This journey reveals the surprising beauty of mathematics hidden within its symbols.

### Taylor Polynomials

#### Intuition behind Taylor Polynomials

The intuition behind Taylor Polynomials is that they help us approximate functions. Specifically, they help us approximate functions that can be hard or tedious to compute, such as trigonometric or logarithmic functions, by using polynomial functions, which are easier to compute. To understand how they do this, let's take a function $ f(x) = \ln(x+1) $. Now let's say we are trying to approximate this function with a polynomial $ P(x) = c_0 + c_1{x} + c_2{x^2} $. Let's "center" our approximation around $ x = 0 $.

Now, if we want to approximate $ f(x) $, we can start out by saying that at $ x = 0 $, $ f(0) = P(0) $, and since $ f(0) = \ln(1) = 0 $, we get that $ c_0 = 0 $. To make our approximation better, we can match the first derivatives of the function and polynomial, so the polynomial can shift itself to better fit the function.

Thus, we get

$$
f'(0) = P'(0),
$$

and we can compute $ f'(0) = \left. \frac{d}{dx} \ln(x+1) \right|_{x=0} = \frac{1}{x+1}\Big|_{x=0} = 1 $.

Taking the derivative of $ P'(x) = c_1 + 2c_2x $, we get $ P'(0) = c_1 $, and since $ P'(0) = 1 $, we have $ c_1 = 1 $.

Let's do this again for the second derivative. We now get

$$
f''(0) = P''(0),
$$

so

$$
f''(0) = \left. \frac{d^2}{dx^2} \ln(x+1) \right|_{x=0} = -\frac{1}{{(1+x)}^2}\Big|_{x=0} = -1.
$$

Taking the second derivative of $ P(x) $, we have $ P''(x) = 2c_2 $, so $ P''(0) = 2c_2 = -1 $, giving $ c_2 = -\frac{1}{2} $.

Thus, our polynomial becomes

$$
P(x) = x - \frac{x^2}{2}.
$$

#### Definition and General Form

Let's now say we are approximating some function $ f(x) $ around some center $ x = c $. We represent our polynomial as

$$
T_n(x) = \sum_{i=0}^{n} c_n{(x-c)^n}.
$$

Now we just have to compute $ c_n $ for any given $ n $. Starting with $ c_0 $, this is essentially like taking the 0th derivative of our function and polynomial, which computes to $ f^{(0)}(c) = c_0 $, as stated before. For the first derivative, we have $ f'(c) = c_1 $.

For the second derivative, since the power multiplies our constant by two, we get

$$
f''(c) = 2c_2.
$$

Let's think of an arbitrary coefficient $ c_n $, where its value will help set the nth derivative of the function and polynomial equal. We get

$$
f^{(n)}(c) = \frac{d}{dx^n} c_n{(x-c)^n}.
$$

To compute $ \frac{d}{dx^n} c_n{(x-c)^n} $, we take the first derivative:

$$
\frac{d}{dx} c_n{(x-c)^n} = n c_n{(x-c)^{n-1}},
$$

and the second derivative:

$$
\frac{d^2}{dx^2} c_n{(x-c)^n} = n(n-1) c_n{(x-c)^{n-2}}.
$$

Continuing this process, we see that it looks a lot like a bunch of descending consecutive integers being multiplied, so we get

$$
\frac{d^n}{dx^n} c_n{(x-c)^n} = n(n-1)(n-2)\dots(2)(1) c_n{(x-c)^{n-n}} = n! c_n.
$$

Thus, to compute any $ c_n $, we have to compute

$$
c_n = \frac{f^{(n)}(c)}{n!}.
$$

Substituting this back into our polynomial definition, we get the definition of the Taylor series:

$$
T_n(x) = \sum_{i=0}^{n} \frac{f^{(n)}(c)}{n!}(x-c)^n.
$$

### Computing Taylor Polynomials

#### Taylor Series for $ e^x $

To help us prove Euler's identity, we need to use the Taylor polynomials of $ e^x $, $ \sin(x) $, and $ \cos(x) $. You will see why later, but for now, let's just go with the flow!

Let's compute the Taylor approximation of $ e^x $ around $ x=0 $. By using a polynomial with a never-ending number of terms, we can approximate $ e^x $ closer and closer until it equals the function. This means:

$$
f(x) = \lim_{n \to \infty} T_n(x)
$$

Thus, we can replace our function with a series of polynomial terms, which will be handy later.

Now, we compute derivatives and observe the pattern. For $ e^x $:

$$
e^x \big|_{x=0} = 1,
$$

$$
\frac{d}{dx}e^x \big|_{x=0} = e^x \big|_{x=0} = 1,
$$

$$
\frac{d^2}{dx^2}e^x \big|_{x=0} = e^x \big|_{x=0} = 1,
$$

$$
\text{and so on.}
$$

Essentially, any $ n $-th derivative of $ e^x $ at $ x=0 $ is $ 1 $.

Thus, the Taylor polynomial becomes:

$$
e^x = T(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots = \sum_{n=0}^{\infty} \frac{x^n}{n!}
$$

#### Taylor Series for $ \sin(x) $

Now we do the same for $ \sin(x) $:

$$
\sin(x) \big|_{x=0} = 0,
$$

$$
\frac{d}{dx}\sin(x) \big|_{x=0} = \cos(x) \big|_{x=0} = 1,
$$

$$
\frac{d^2}{dx^2}\sin(x) \big|_{x=0} = \frac{d}{dx}\cos(x) \big|_{x=0} = -\sin(x) \big|_{x=0} = 0,
$$

$$
\frac{d^3}{dx^3}\sin(x) \big|_{x=0} = \frac{d}{dx}(-\sin(x)) \big|_{x=0} = -\cos(x) \big|_{x=0} = -1,
$$

$$
\text{and so on.}
$$

Essentially, the $ n $-th derivative of $ \sin(x) $ at $ x=0 $ is $ (-1)^{n} $ for odd $ n $.

Thus, the Taylor polynomial becomes:

$$
\sin(x) = T(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n+1}}{(2n+1)!}
$$

#### Taylor Series for $ \cos(x) $

Now we do the same for $ \cos(x) $:

$$
\cos(x) \big|_{x=0} = 1,
$$

$$
\frac{d}{dx}\cos(x) \big|_{x=0} = -\sin(x) \big|_{x=0} = 0,
$$

$$
\frac{d^2}{dx^2}\cos(x) \big|_{x=0} = -\frac{d}{dx}\sin(x) \big|_{x=0} = -\cos(x) \big|_{x=0} = -1,
$$

$$
\frac{d^3}{dx^3}\cos(x) \big|_{x=0} = \frac{d}{dx}(-\cos(x)) \big|_{x=0} = \sin(x) \big|_{x=0} = 0,
$$

$$
\text{and so on.}
$$

Essentially, the $ n $-th derivative of $ \cos(x) $ at $ x=0 $ is $ (-1)^{n} $ for even $ n $.

Thus, the Taylor polynomial becomes:

$$
\sin(x) = T(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \dots = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n}}{(2n)!}
$$

### Bringing it all together

#### Complex Numbers

I won't spend too much time on this, but the number we defined as $ i $ in Euler's identity is $ i = \sqrt{-1} $. One important thing to note is the following powers of $ i $:

$$
i^1 = i,
$$

$$
i^2 = -1,
$$

$$
i^3 = i \cdot i^2 = -i,
$$

$$
i^4 = 1.
$$

These powers repeat in a cycle of length 4. Specifically, for any integer $ n $, we have the following periodicity:

$$
i^{4n} = 1, \quad i^{4n+1} = i, \quad i^{4n+2} = -1, \quad i^{4n+3} = -i.
$$

This cyclical pattern is important in many applications involving complex numbers, which will be important next.

#### Euler's Formula

Let's take our Taylor expansion for $ e^x $ and substitute $ ix $ for $ x $, to get $ e^{ix} $. Substituting, the Taylor expansion for this is:

$$
e^{ix} = T(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots = \sum_{n=0}^{\infty} \frac{i^n x^n}{n!}.
$$

We can split this into a real part, where $ n $ is even, and an imaginary part, where $ n $ is odd, to get:

$$
e^{ix} = \sum_{n=0}^{\infty} \frac{i^{2n} x^{2n}}{(2n)!} + \sum_{n=0}^{\infty} \frac{i^{2n+1} x^{2n+1}}{(2n+1)!}.
$$

Let's look at the real and imaginary parts separately. For the real part, we substitute $ i^{2n} = (-1)^n $, so we get:

$$
\sum_{n=0}^{\infty} \frac{i^{2n} x^{2n}}{(2n)!} = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}.
$$

Meanwhile, for the imaginary part, we can substitute $ i^{2n+1} = (-1)^n \cdot i $, to get:

$$
\sum_{n=0}^{\infty} \frac{i^{2n+1} x^{2n+1}}{(2n+1)!} = i \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}.
$$

Finally, we get:

$$
e^{ix} = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} + i \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}.
$$

The first term is just the expansion of $ \cos(x) $, and the second term is the expansion of $ \sin(x) $ times $ i $. So we prove Euler's formula:

$$
e^{ix} = \cos(x) + i \sin(x).
$$

#### Substitution

Now we see why we used Taylor polynomials, so that we can prove Euler's identity by matching like terms in the respective expansions. Now, in the identity, we can plug in $ x = \pi $, to get:

$$
e^{i \pi} = \cos(\pi) + i \sin(\pi) = -1 + i \cdot 0 = -1.
$$

This means we have proven Euler's Identity:

$$
e^{i \pi} + 1 = 0.
$$

### Conclusion

Euler's identity beautifully connects five fundamental mathematical constants: $ e, i, \pi, 1, $ and $ 0 $, in a single equation. This result is a culmination of concepts from calculus, Taylor series, and complex numbers, showcasing the inherent harmony of mathematics.
