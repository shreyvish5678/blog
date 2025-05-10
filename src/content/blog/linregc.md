---
author: Shrey Vishen
pubDatetime: 2025-05-04T11:03:00.000+00:00
title: "ML in C: Linear Regression"
slug: "linregc"
featured: true
tags:
  - linear-regression
  - c
description: "Implementing Linear Regression from scratch in C, gradient descent and closed form"
---

## Introduction

In this series, I am going to be moving away from Deep Learning algorithms, and go back to my roots. Specifically we will be exploring classical ML Algorithms in detail and implementing them from scratch in C. Today we will start simple and build a Linear Regression algorithm from scratch in C.

## Gradient Descent Math

Assume we have some dependent variable $y$, and some independent variable $x$, such that, $y = f(x)$, or the variable $y$ is influenced by $x$ in some way. For the purposes of this problem, we assume that this function $f(x)$ is a linear function. So we write:

$y = wx + b + \epsilon$, where $w$ is the weight, or a constant for how much $y$ changes as $x$ changes. Here, $b$ is a bias or correction term and then $\epsilon$ is some error term, since this relationship isn't perfectly modeled linearly and is assumed to be gaussian, or $\epsilon \sim \mathcal{N}(0, \sigma^2)$

Now if we have multiple features, $x_1, x_2, ..., x_n$, then our equation becomes: $y = w_1x_1 + w_2x_2 + ... + w_dx_d + b + \epsilon$. We can further make this easier, by saying that there is some vector $\mathbf{w} = [w_1, ..., w_d]$, and $\mathbf{x} = [x_1, ..., x_d]$, where they are in the set of all vectors size $d$, which is the number of features or, $\mathbf{w} \in \mathbb{R}^d$ and $\mathbf{x} \in \mathbb{R}^d$. Now we can compute their dot product, which multiplies every corresponding element in the weight vector with the feature vector, or we get: $y = \mathbf{w}^{\top} \mathbf{x} + b + \epsilon$, where we have to transpose the weight vector for the dot product to work.

The goal now is to find a $\mathbf{w}$ and $b$ such that we minimize: $\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$, where $\hat{y}_i = \mathbf{w}^{\top} \mathbf{x}_i + b$. What we are doing here is going through all the samples in our dataset (of which there are $n$) and computing what is called the mean squared error which is the squared difference between the true value and our predicted value. We square it because it's easier to differentiate and therefore optimize with gradient descent, and it penalizes larger errors more heavily, since squaring amplifies them. Now, time for some calculus!

We need to find the partial derivative of this loss function with respect to our weights and bias so let's get started. First we have:

$\frac{\partial{\mathcal{L}}}{\partial{\mathbf{w}}} = \frac{\partial}{\partial{\mathbf{w}}}(\frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b)^2)$

$\frac{\partial}{\partial{\mathbf{w}}}(\frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b)^2) = \frac{2}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b) \cdot \frac{\partial}{\partial{\mathbf{w}}}(- \mathbf{w}^{\top} \mathbf{x}_i) = \frac{2}{n} \sum_{i=1}^{n} (\mathbf{w}^{\top} \mathbf{x}_i + b - y_i) \mathbf{x}_i$

What we can do here is say that the features are a matrix of size samples by features or $\mathbf{x} \in \mathbb{R}^{n \times d}$, and likewise say $\mathbf{y} \in \mathbb{R}^n$, so this becomes: $\frac{\partial{\mathcal{L}}}{\partial{\mathbf{w}}} = \mathbf{x}^{\top} \frac{2}{n}(\mathbf{x} \mathbf{w} + b - \mathbf{y})$, where $\mathbf{x} \in \mathbb{R}^{n \times d}$, $\mathbf{y} \in \mathbb{R}^n$, $\mathbf{w} \in \mathbb{R}^d$, $b \in \mathbb{R}$, $\frac{\partial{\mathcal{L}}}{\partial{\mathbf{w}}} \in \mathbb{R}^d$. Now for the bias:

$\frac{\partial{\mathcal{L}}}{\partial{b}} = \frac{\partial}{\partial{b}}(\frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b)^2)$

$\frac{\partial}{\partial{b}}(\frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b)^2) = \frac{2}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^{\top} \mathbf{x}_i - b) \cdot \frac{\partial}{\partial{b}}(-b) = \frac{2}{n} \sum_{i=1}^{n} (\mathbf{w}^{\top} \mathbf{x}_i + b - y_i)$

If we say that the error $\mathbf{e} = \mathbf{x} \mathbf{w} + b - \mathbf{y}$, $\mathbf{e} \in \mathbb{R}^n$, we get the bias error in a cleaner way: $\frac{2}{n} \sum_{i=1}^{n} \mathbf{e}_i$

## Gradient Descent in Numpy

Now we can write up this algorithm in basic numpy. First let's create a linreg function and initialize all our values:

```python
import numpy as np

def linreg(num_samples, num_features, learning_rate, num_iterations):
    weights = np.random.rand(num_features)
    X = np.random.rand(num_samples, num_features)
    y = X @ weights + np.random.randn(num_samples) * 0.1
    weights_learned = np.zeros(num_features)
    bias = 0
```

Here, we are creating our weights we are trying to learn, $\mathbf{w_{\text{actual}}} \in \mathbb{R}^d$, our features: $\mathbf{x} \in \mathbb{R}^{n \times d}$, our bias is set to 0, then we get our labels: $\mathbf{y} = \mathbf{x} \mathbf{w_{\text{actual}}} + b + \epsilon$. Our goal is to learn the weights we have initialized through gradient descent. Now let's create a loop for our iterations, where we will calculate our gradients, then use the learning rate to update our weights and do this for our iterations. We have:

```python
for _ in range(num_iterations):
    #get prediction and error
    #get the gradients
    #update weights and bias
```

To get the prediction we can do:

```python
y_pred = X @ weights_learned + bias
error = y_pred - y
```

Here we use the formula $\mathbf{e} = \mathbf{x} \mathbf{w} + b - \mathbf{y}$ to get our error.

Now we use our formulas to get the gradients so for the weights we have $\frac{\partial{\mathcal{L}}}{\partial{\mathbf{w}}} = \mathbf{x}^{\top} \frac{2}{n}(\mathbf{e})$, and $\frac{\partial{\mathcal{L}}} = \frac{2}{n} \sum_{i=1}^{n} \mathbf{e}_i$, which becomes

```python
gradients = 2 * X.T @ error / num_samples
bias_gradient = 2 * np.sum(error) / num_samples
```

Now we can use the gradient descent rule: $w = w - l \cdot \frac{\partial{\mathcal{L}}}{w}$, to perform gradient descent on our weights and bias to get:

```python
weights_learned -= learning_rate * gradients
bias -= learning_rate * bias_gradient
```

Here is the full loop and function in python:

```python
def linreg(num_samples, num_features, learning_rate, num_iterations):
    weights = np.random.rand(num_features)
    X = np.random.rand(num_samples, num_features)
    y = X @ weights + np.random.randn(num_samples) * 0.1
    weights_learned = np.zeros(num_features)
    bias = 0

    for _ in range(num_iterations):
        y_pred = X @ weights_learned + bias
        error = y_pred - y
        gradients = 2 * X.T @ error / num_samples
        bias_gradient = 2 * np.sum(error) / num_samples
        weights_learned -= learning_rate * gradients
        bias -= learning_rate * bias_gradient

    print("True weights:", weights)
    print("Learned weights:", weights_learned)
    print("Learned bias:", bias)
```

Running `linreg(1000, 10, 0.01, 10000)` we get:

```
True weights: [0.85343845 0.49349678 0.37013164 0.36190771 0.78289762 0.07346953
 0.69709517 0.51353253 0.92723031 0.01299727]
Learned weights: [0.85936534 0.49433907 0.37394154 0.37123143 0.78007247 0.07445098
 0.69582561 0.52409001 0.91319363 0.00376582]
Learned bias: -0.00655331919084931
```

This is pretty good evidence that our code works. Now time for some C!

## Gradient Descent in C

Firstly we will have to recreate some numpy functions, mainly one to randomly generate a number from 0-1, one to sample from the gaussian distribution and many matrix helper functions like multiply and transpose. Let's start simple and create a function to generate a number from 0-1

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double randp() {
    return (double)rand() / RAND_MAX;
}
```

Here we import some standard libraries, and make a basic function to randomly generate a number from 0-1. Now we can create a function to sample from the gaussian distribution. We will use the Box-Muller transform to do this, which is a way to generate a normally distributed random number from two uniform random numbers from 0-1. The formula is: $\sqrt{-2 \log(u_1)} \cdot \cos(2 \pi u_2)$, where $u_1$ and $u_2$ are two uniform random numbers from 0-1. We can implement this in C:

```c
#define PI 3.14159265358979323846

double randn() {
    double u1 = randp();
    double u2 = randp();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}
```

Now we want to create a function to multiply two matrices, but first let us define a `struct Matrix` in C, to make it easier to deal with them, with 2 integer variables to store size, and a pointer to store the data of the matrix:

```c
struct Matrix {
    int rows;
    int cols;
    double* data;
};
```

Now here is an overview of matrix multiplication:

1. We have two matrices, $A, B$, where the size of $A$ is $r_a \times c_a$, and $B$ is $r_b \times c_b$. Matrix multiplication works by taking two matrices assuming an initial condition where $c_a = r_b$ and returning a matrix of size $r_a \times c_b$, call this matrix $C$

2. To compute one element in $C$, let's say at row $i$ and column $j$, you want to multiply all the corresponding elements in the $i$th row of $A$ with the $j$th column of $B$, or compute the dot product between the row vector at index $i$ of $A$ and column vector of index $j$ at $B$

3. We mathematically write this as: $C_{ij} = \sum_{k=0}^{c_a} A_{ik} \cdot B_{kj}$

Now to write this in C, we define the function and check whether $c_a = r_b$:

```c
struct Matrix matrix_multiplication(struct Matrix a, struct Matrix b) {
    if (a.cols != b.rows) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return (struct Matrix){0, 0, NULL};
    }

    ...
}
```

If it doesn't we print an error and return an empty matrix. Now we create our result matrix with size $r_a \times c_b$:

```c
struct Matrix result = {a.rows, b.cols, (double*)malloc(a.rows * b.cols * sizeof(double))};
```

Now we iterate through all the elements of our new result matrix:

```c
for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < b.cols; j++) {
          int sum = 0;
          ...
          result.data[i * b.cols + j] = sum;
      }
  }
```

So we iterate through our matrix, calculate the value at that index $i, j$ and set it for our new matrix. To calculate it, we iterate from $0 - c_a$ with $k$, and get $A_{ik}$ and $B_{kj}$. Now since the pointers point to a one dimensional array, we have to do a workaround:

If we have a one dimensional array pretending to be a matrix so the matrix is size $N \times M$, so the 1D array is size $NM$, to get the value at index $i, j$ of our matrix, this is the same as value $i \cdot M + j$, here we assume row-major order. So let's implement this now:

```c
for (int k = 0; k < a.cols; k++) {
    sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
}
```

Now we have the full function for matrix multiplication:

```c
struct Matrix matrix_multiplication(struct Matrix a, struct Matrix b) {
    if (a.cols != b.rows) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return (struct Matrix){0, 0, NULL};
    }

    struct Matrix result = {a.rows, b.cols, (double*)malloc(a.rows * b.cols * sizeof(double))};
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            double sum = 0;
            for (int k = 0; k < a.cols; k++) {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            result.data[i * b.cols + j] = sum;
        }
    }

    return result;
}
```

Now we can define transpose mathematically as well here: $A^{\top}_{ij} = A_{ji}$, and use a similar approach to create a transpose method.

```c
struct Matrix transpose(struct Matrix a) {
    struct Matrix result = {a.cols, a.rows, (double*)malloc(a.cols * a.rows * sizeof(double))};
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.data[j * a.rows + i] = a.data[i * a.cols + j];
        }
    }
    return result;
}
```

Here we have `j * a.rows + i` on the right, because if the size of $A$ is $N \times M$, the size of $R$ is $M \times N$, so we swap while computing the index, and also swap the rows and columns in the struct define.

Now we just recreate what we did with numpy, firstly, initializing our weights, features and labels:

```c
void LinearRegression(int num_samples, int num_features, double lr, int iterations) {
    struct Matrix weight = {num_features, 1, (double*)malloc(num_features * sizeof(double))};
    for (int i = 0; i < weight.rows; i++) {
        weight.data[i] = randp();
    }

    struct Matrix x = {num_samples, num_features, (double*)malloc(num_samples * num_features * sizeof(double))};
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            x.data[i * x.cols + j] = randp();
        }
    }

    struct Matrix y = matrix_multiplication(x, weight);
    for (int i = 0; i < y.rows; i++) {
        y.data[i] += randn() * 0.1;
    }

    struct Matrix weights_learned = {num_features, 1, (double*)malloc(num_features * sizeof(double))};
    for (int i = 0; i < weights_learned.rows; i++) {
        weights_learned.data[i] = 0;
    }

    double bias = 0;
    ...
}
```

Here we use our pre-made `randp` function, and use it to fill up our weight and feature matrices. Then we use our matrix multiplication function to get our labels, and use `randn` to add a little bit of noise to our labels just like we did in numpy with `np.random.randn()`.

Now we can write up the code to perform one iteration of gradient descent. Firstly, we compute our predicted labels by doing matrix multiplcation between features and weights and then adding a bias to every term, like so:

```c
struct Matrix y_pred = matrix_multiplication(x, weights_learned);
for (int i = 0; i < y.rows; i++) {
    y_pred.data[i] += bias;
}
```

Now we get our error, by computing the difference in our predicted and actual labels across every element:

```c
struct Matrix error = {y.rows, 1, (double*)malloc(y.rows * sizeof(double))};
for (int i = 0; i < y.rows; i++) {
    error.data[i] = y_pred.data[i] - y.data[i];
}
```

Now we use our formula for the gradients, $\frac{\partial{\mathcal{L}}}{\partial{\mathbf{w}}} = \mathbf{x}^{\top} \frac{2}{n}(\mathbf{e})$, and $\frac{\partial{\mathcal{L}}} = \frac{2}{n} \sum_{i=1}^{n} \mathbf{e}_i$, so first we have the weight gradients, where we first transpose our feature matrix and multiply by the error matrix, then scale all elements by $2/n$:

```c
struct Matrix x_t = transpose(x);
struct Matrix gradient = matrix_multiplication(x_t, error);
for (int i = 0; i < gradient.rows; i++) {
    gradient.data[i] *= 2.0 / num_samples;
}
```

Now we get our bias gradient, by adding up all the elements in our error, and scaling by $2/n$:

```c
double error_sum = 0;
for (int i = 0; i < error.rows; i++) {
  error_sum += error.data[i];
}
double bias_gradient = 2 * error_sum / num_samples;
```

We apply our gradients to our weights and biases with the gradient descent update rule, $w = w - l \cdot \frac{\partial{\mathcal{L}}}{w}$:

```c
for (int i = 0; i < weights_learned.rows; i++) {
    weights_learned.data[i] -= lr * gradient.data[i];
}
bias -= lr * bias_gradient;
```

This is the full C code for gradient descent linear regression:

```c
void LinearRegression(int num_samples, int num_features, double lr, int iterations) {
    struct Matrix weight = {num_features, 1, (double*)malloc(num_features * sizeof(double))};
    for (int i = 0; i < weight.rows; i++) {
        weight.data[i] = randp();
    }

    struct Matrix x = {num_samples, num_features, (double*)malloc(num_samples * num_features * sizeof(double))};
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            x.data[i * x.cols + j] = randp();
        }
    }

    struct Matrix y = matrix_multiplication(x, weight);
    for (int i = 0; i < y.rows; i++) {
        y.data[i] += randn() * 0.1;
    }

    struct Matrix weights_learned = {num_features, 1, (double*)malloc(num_features * sizeof(double))};
    for (int i = 0; i < weights_learned.rows; i++) {
        weights_learned.data[i] = 0;
    }

    double bias = 0;

    for (int iter = 0; iter < iterations; iter++) {
        struct Matrix y_pred = matrix_multiplication(x, weights_learned);
        for (int i = 0; i < y.rows; i++) {
            y_pred.data[i] += bias;
        }

        struct Matrix error = {y.rows, 1, (double*)malloc(y.rows * sizeof(double))};
        for (int i = 0; i < y.rows; i++) {
            error.data[i] = y_pred.data[i] - y.data[i];
        }

        struct Matrix x_t = transpose(x);
        struct Matrix gradient = matrix_multiplication(x_t, error);
        for (int i = 0; i < gradient.rows; i++) {
            gradient.data[i] *= 2.0 / num_samples;
        }

        double error_sum = 0;
        for (int i = 0; i < error.rows; i++) {
            error_sum += error.data[i];
        }
        double bias_gradient = 2 * error_sum / num_samples;

        for (int i = 0; i < weights_learned.rows; i++) {
            weights_learned.data[i] -= lr * gradient.data[i];
        }
        bias -= lr * bias_gradient;
    }

    printf("Actual weights:\n");
    for (int i = 0; i < weight.rows; i++) {
        printf("%f ", weight.data[i]);
    }
    printf("\nLearned weights:\n");
    for (int i = 0; i < weights_learned.rows; i++) {
        printf("%f ", weights_learned.data[i]);
    }
    printf("\nLearned bias: %f\n", bias);
}
```

Running `LinearRegression(1000, 10, 0.01, 10000);` we get:

```
Actual weights:
0.840188 0.394383 0.783099 0.798440 0.911647 0.197551 0.335223 0.768230 0.277775 0.553970
Learned weights:
0.841056 0.414686 0.793716 0.789665 0.909581 0.186562 0.324858 0.745904 0.281371 0.544149
Learned bias: 0.019330
```

Showing our code works pretty well! Now onto closed form linear regression.

## Closed Form Math

There is a closed form derivation of linear regression, so let's solve that:

For now we will ignore the bias. Define our matrices we have: $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, and $w \in $\mathbb{R}^d$, where: $y^{\hat} = Xw$, to minimize $L(w) = ||Xw - y||^2 = (Xw - y)^{\top} (Xw - y)$

Derivation:

1. $L(w) = ||Xw - y||^2 = (Xw - y)^{\top} (Xw - y)$
2. Expand it: $L(w) = w^{\top} X^{\top} Xw - 2y^{\top} Xw + y^{\top} y$
3. Take the gradient with respect to $w$: $\nabla_{w} L = 2X^{\top} Xw - 2X^{\top} y$
4. Set the gradient to 0, as that's the optimal solution: $2X^{\top} Xw = 2X^{\top} y$
5. Solve for $w$: $w = (X^{\top} X)^{-1} X^{\top} y$, where $X^{-1}$, is the inverse of that matrix

Now our bias can just be a correction term, taking the mean of the differences between the actual labels and predicted labels so: $b = \frac{\sum{y - Xw}}{n}$

So we have: $w = (X^{\top} X)^{-1} X^{\top} y$, $b = \frac{\sum{y - Xw}}{n}$

## Closed Form in Numpy

This is very straightforward, firstly let's borrow our initialization code from gradient descent:

```python
def linreg_closed(num_samples, num_features):
    weights = np.random.rand(num_features)
    X = np.random.rand(num_samples, num_features)
    y = X @ weights + np.random.randn(num_samples) * 0.1
```

First we compute the transpose, and apply our respective formulas:

```python
X_transpose = X.T
weights_learned = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
bias = np.mean(y - X @ weights_learned)
```

This is the full function in numpy:

```python
def linreg_closed(num_samples, num_features):
    weights = np.random.rand(num_features)
    X = 2 * np.random.rand(num_samples, num_features)
    y = X @ weights + np.random.randn(num_samples) * 0.1

    X_transpose = X.T
    weights_learned = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    bias = np.mean(y - X @ weights_learned)

    print("True weights:", weights)
    print("Learned weights:", weights_learned)
    print("Learned bias:", bias)
```

Running `linreg_closed(1000, 10)` we get:

```
True weights: [0.26775505 0.03225343 0.74154335 0.20135775 0.97358949 0.49574533
 0.93035554 0.2090532  0.9050105  0.64912548]
Learned weights: [0.26577245 0.03873288 0.73850193 0.20191474 0.97023463 0.49543545
 0.93384334 0.2037116  0.90201497 0.65225537]
Learned bias: -6.41897512583971e-05
```

This shows our solution is pretty accurate! Now let's code this up in C

## Closed Form in C

First let's create our function and copy the initilization code:

````c
void LinearRegressionClosed(int num_samples, int num_features) {
    struct Matrix weight = {num_features, 1, (double*)malloc(num_features * sizeof(double))};
    for (int i = 0; i < weight.rows; i++) {
        weight.data[i] = randp();
    }

    struct Matrix x = {num_samples, num_features, (double*)malloc(num_samples * num_features * sizeof(double))};
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            x.data[i * x.cols + j] = randp();
        }
    }

    struct Matrix y = matrix_multiplication(x, weight);
    for (int i = 0; i < y.rows; i++) {
        y.data[i] += randn() * 0.1;
    }
    ...
}

Now we first want to get our features transposed and multiply with our features:

```c
struct Matrix x_t = transpose(x);
struct Matrix x_t_x = matrix_multiplication(x_t, x);
struct Matrix x_t_x_inv = inverse(x_t_x);
````

The inverse function is a placeholder we will create later. Now we just apply our formula and get our learned weights along with our predicted labels:

```c
struct Matrix x_t_y = matrix_multiplication(x_t, y);
struct Matrix weights_learned = matrix_multiplication(x_t_x_inv, x_t_y);
struct Matrix x_w_c = matrix_multiplication(x, weights_learned);
```

Now we calculate the mean of differences between the actual and predicted, and set as our bias:

```c
double diff = 0;
for (int i = 0; i < x_w_c.rows; i++) {
    diff += y.data[i] - x_w_c.data[i];
}
double bias = diff / num_samples;
```

Here is the full function in C for closed form:

```c
void LinearRegressionClosed(int num_samples, int num_features) {
    struct Matrix weight = {num_features, 1, (double*)malloc(num_features * sizeof(double))};
    for (int i = 0; i < weight.rows; i++) {
        weight.data[i] = randp();
    }

    struct Matrix x = {num_samples, num_features, (double*)malloc(num_samples * num_features * sizeof(double))};
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            x.data[i * x.cols + j] = randp();
        }
    }

    struct Matrix y = matrix_multiplication(x, weight);
    for (int i = 0; i < y.rows; i++) {
        y.data[i] += randn() * 0.1;
    }

    struct Matrix x_t = transpose(x);
    struct Matrix x_t_x = matrix_multiplication(x_t, x);
    struct Matrix x_t_x_inv = inverse(x_t_x);

    struct Matrix x_t_y = matrix_multiplication(x_t, y);
    struct Matrix weights_learned = matrix_multiplication(x_t_x_inv, x_t_y);
    struct Matrix x_w_c = matrix_multiplication(x, weights_learned);

    double diff = 0;
    for (int i = 0; i < x_w_c.rows; i++) {
        diff += y.data[i] - x_w_c.data[i];
    }
    double bias = diff / num_samples;

    printf("Actual weights:\n");
    for (int i = 0; i < weight.rows; i++) {
        printf("%f ", weight.data[i]);
    }
    printf("\nLearned weights:\n");
    for (int i = 0; i < weights_learned.rows; i++) {
        printf("%f ", weights_learned.data[i]);
    }
    printf("\nLearned bias: %f\n", bias);
}
```

Now let's code up our inverse function. We want to write a function that returns the inverse of a matrix given the matrix. But what is the inverse of a matrix. Before getting into the inverse itself, here are the conditions of a inverse:

1. Square, $A$ must be $A \in \mathbb{R}^{n \times n}$
2. Non-zero determinant
3. All rows and columns need to be linearly indepedent

Now imagine you have a square matrix $A$, with the inverse denoted as $A^{-1}$. The inverse is defined as such: $A \cdot A^{-1} = I$, where $I$ is the identity matrix, where the values across the diagonal are equal to 1, and the rest are equal to 0. So let's define our function:

```c
struct Matrix inverse(struct Matrix orig) {
    struct Matrix a = {orig.rows, orig.cols, (double*)malloc(orig.rows * orig.cols * sizeof(double))};
    for (int i = 0; i < orig.rows; i++) {
        for (int j = 0; j < orig.cols; j++) {
            a.data[i * orig.cols + j] = orig.data[i * orig.cols + j];
        }
    }

    if (a.rows != a.cols) {
        fprintf(stderr, "Matrix is not square, cannot compute inverse.\n");
        return (struct Matrix){0, 0, NULL};
    }
    ...
}
```

So here we first copy over our matrix to another matrix, because while creating the inverse, the passed in matrix is changed, so we need to avoid that. Then we check whether it is a square matrix, and if not return a null matrix. Now we create our identity matrix:

```c
int n = a.rows;
struct Matrix result = {n, n, (double*)malloc(n * n * sizeof(double))};
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        if (i == j) {
            result.data[i * n + j] = 1.0;
        } else {
            result.data[i * n + j] = 0.0;
        }
    }
}
```

This creates a result matrix the same size as our initial matrix and then sets every value on the diagonal to 1, and the rest to 0. Now our goal is essentially to swap the original and result matrix, so our result becomes the inverse, and our original becomes the identity matrix. We first create a loop to go through every value in the original diagonal, called the pivot. Then we divide every value in that row by the pivot, so that the pivot turns into 1, and the rest just get scaled by the pivot. This is the code for it:

```c
for (int i = 0; i < n; i++) {
    double pivot = a.data[i * n + i];
    for (int j = 0; j < n; j++) {
        a.data[i * n + j] /= pivot;
        result.data[i * n + j] /= pivot;
    }
    ...
}
```

Now for every other row in the matrix, we subtract a multiple of the pivot row to make the other values in the column to 0, which makes the original into the identity matrix and the result into the inverse. Combining all this we get our inverse function:

```c
struct Matrix inverse(struct Matrix orig) {
    struct Matrix a = {orig.rows, orig.cols, (double*)malloc(orig.rows * orig.cols * sizeof(double))};
    for (int i = 0; i < orig.rows; i++) {
        for (int j = 0; j < orig.cols; j++) {
            a.data[i * orig.cols + j] = orig.data[i * orig.cols + j];
        }
    }

    if (a.rows != a.cols) {
        fprintf(stderr, "Matrix is not square, cannot compute inverse.\n");
        return (struct Matrix){0, 0, NULL};
    }

    int n = a.rows;
    struct Matrix result = {n, n, (double*)malloc(n * n * sizeof(double))};
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                result.data[i * n + j] = 1.0;
            } else {
                result.data[i * n + j] = 0.0;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        double pivot = a.data[i * n + i];
        for (int j = 0; j < n; j++) {
            a.data[i * n + j] /= pivot;
            result.data[i * n + j] /= pivot;
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = a.data[k * n + i];
                for (int j = 0; j < n; j++) {
                    a.data[k * n + j] -= factor * a.data[i * n + j];
                    result.data[k * n + j] -= factor * result.data[i * n + j];
                }
            }
        }
    }

    return result;
}
```

So we are now done! Running `LinearRegressionClosed(1000, 10);` we get:

```
Actual weights:
0.327250 0.473494 0.533105 0.935250 0.396167 0.273517 0.242439 0.634404 0.295955 0.479751
Learned weights:
0.341502 0.484882 0.528503 0.929786 0.388703 0.272344 0.246667 0.624750 0.283870 0.487815
Learned bias: -0.000499
```

Showing our solution is pretty close.

## Conclusion

That is all for today! I am going to try to continue this series over the summer and build other algorithms in C, I had a lot of fun doing it with Linear Regression. See you guys next time!
