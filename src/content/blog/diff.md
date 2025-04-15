---
author: Shrey Vishen
pubDatetime: 2025-04-14T12:30:04.000+00:00
title: An Introduction to Diffusion Models Part 1
slug: "diffusion-models"
featured: true
tags:
  - pytorch
  - diffusion
  - noise
description: "An Introduction to Diffusion Models Part 1: The Mathematics Behind Diffusion Models"
---

## Introduction

Diffusion Models are the cornerstone of Image Generation in modern Generative AI. From being used in applications like Video generation in Runway, to high quality open-source generation such as Stable Diffusion and FLUX. With the advent of GPT 4o's new autoregressive image gen, it's time to look at diffusion models and why they aren't going away anytime soon. This is a very surface level introduction to diffusion models, from data setup to training and testing. We will be training the model on the CelebA dataset, an image dataset with over 200k+ images. For part 1, we will just be covering the theory, so if you aren't a math person (you can still explore this), or want to jump straight to the code, you can skip to part 2 (when it is out). The code will be available on my GitHub as well.

## Groundwork

Let's say our goal is to generate a sample, such that it mimicks a given dataset that could be comprised of anything. So how do diffusion models, or more specifically, the diffusion process acheive this?

Here's a high level overview of how it works:

- We take a sample from our dataset, calling it the "clean" image
- We label this image as being at timestep 0, meaning this is the original image
- To go to timestep 1, we take this image, and add some "noise"
- You can think of noise as a bunch of randomly generated pixels for now
- We keep adding noise to the image until a max timestep, say 1000
- Now our image looks like pure randomness
- How can we mathematically express this?

Let's say we are given the clean image at timestep 0, calling it $x_0$. Now we can try to get to $x_1$.

$x_1 = x_0 + \beta \epsilon$

Here, $\beta$ is just some small value, so we add a little bit of noise to our original sample. But this doesn't work. What we are doing here is just taking the original image, and adding some noise to it, disregarding the information of the clean image. Also this means we are adding the same amount of noise very early on and later on. We want to add less noise initially and gradually start adding more and more noise. So what we could do is instead of having one $\beta$, we have multiple across our timesteps, and this value increases as we go through our timesteps, so we get:

$x_1 = (1 - \beta_t)x_0 + \beta_t \epsilon$.

From here, we will replace $\beta_t$ with $1 - \alpha_t$ as needed, so we essentially have a property: $\beta_t = 1 - \alpha_t$.

But where are we getting $\epsilon$ from, which denotes our noise? We commonly get it from what is called a gaussian distribution, which is very common in applications involving random errors and uncertainty. It has the shape of a bell curve, with a mean of 0, and a standard deviation of 1 (so variance is also 1). The distribution has essentially, an infinite number of values, so to get one value, we randomly pick a value, with the probability of our value being in an interval, being equal to the area under that interval, so values closer to 0 are more likely to get chosen, but values further away can still be chosen. So we have: $\epsilon \sim \mathcal{N}(0, 1)$

So now along with the weighted sum we get to go from $x_t$ given $x_{t-1}$, we do:

$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t} \epsilon$, we will explain the square root later.

Here $\beta_t$ scales positively linear as the timesteps increase, so $1 - \alpha_t$ does as well.

To get the noise schedule, we take in some pre-initialized values, like our maximum beta value, minimum beta value and number of timesteps. With this we can craft an equation as such:

$\beta_t = \beta_{\text{min}} + \frac{t}{T} \cdot (\beta_{\text{max}} - \beta_{\text{min}})$

Here both the timesteps are 0-indexed, so if the max timesteps is 1000, $T = 999$, and $t = 0, 1, 2, ..., 999$

The equation above tells us one possibility of $x_t$, that is given $x_{t-1}$, give me one possible configuration of $x_t$. To get all the possibilities of $x_t$, let's express it as a distribution (will make sense later), more specifically a gaussian distribution. So we have:

$x_t \sim \mathcal{N}(\mu_{x_t}, \text{Var}(x_t))$

Now we need to find $\mu_{x_t}$ and $\text{Var}(x_t)$, in terms of our known variables, which are $x_{t-1}$ and $b_t$.

To calculate $\mu_{x_t}$, we essentially have to calculate the expected value of $x_t$, as $\mu_{x_t} = \mathbb{E}[x_t]$. We get that:

$\mathbb{E}[x_t] = \mathbb{E}[\sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t} \epsilon]$. Then:

$\mathbb{E}[\sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t} \epsilon] = \mathbb{E}[\sqrt{\alpha_t}x_{t-1}] + \mathbb{E}[\sqrt{1 - \alpha_t} \epsilon]$.

Since the first part is deterministic, and the second part, the expected value of $\epsilon$, is the mean of the gaussian distribution, we get that:

$\mathbb{E}[\sqrt{\alpha_t}x_{t-1}] + \mathbb{E}[\sqrt{1 - \alpha_t} \epsilon] = \sqrt{\alpha_t}x_{t-1}$. So we get:

$\mu_{x_t} = \sqrt{\alpha_t}x_{t-1}$.

Now for the other part, we need the variance, so let's find $\text{Var}(x_t)$, which we get

$\text{Var}(x_t) = \text{Var}(\sqrt{{\alpha_t}}x_{t-1}) + \text{Var}(\sqrt{{1 - \alpha_t}} \epsilon)$.

Since the first part is deterministic, there is no variance so that part is 0, and the second part, we get:

$\text{Var} = \text{Var}(\sqrt{{1 - \alpha_t}} \epsilon)$. $\text{Var} = 1 - \alpha_t$.

For variance, if we take a constant outside, we square it, because for the standard deviation we would multiply it, because for standard deviation we can multiply it, or simply: $\text{Var}(aX) = a^2\text{Var}(X)$. So finally we get:

$x_t \sim \mathcal{N}(\sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)$.

Now you realize why we added that square root. Here what is happening as we are progressively noising our clean sample, the variance of our next distribution scales linearly instead of quadratically, and this leads to better model generalization and training.

So why did we do all this math? Now what we can do is model this process as what is called a markov chain. We want to use this markov chain and prove the following:

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$, where we have

$\bar{\alpha}_t = \alpha_1 \cdot \alpha_2 \cdot ... \cdot \alpha_t = \prod_{n=1}^{t} \alpha_n$

Once we prove this, for sampling we can directly compute with one formula as the cumulative alpha products can be pre-computed, and allow for faster forward diffusion rather than looping through.

## Forward Process with Markov Chains

Let's say we have a multiple number of "states", like $x_0, x_1, ..., x_{t-1}$. Using these we use a conditional probability distribution, which is basically the distribution of a given variable, given some variables that are already there, or the probability distribution of one event, given that certain events have happened. This is written as $P(x_t \mid x_0, x_1, ..., x_{t-1})$, telling us what is the distribution of $x_t$, given $x_0, x_1, ..., x_{t-1}$.

The special property of markov chains is that the next state only depends on the current state, which in our case is $x_{t-1}$, which means we can discard the other states. This is written as $P(x_t \mid x_0, x_1, ..., x_{t-1}) = P(x_t \mid x_{t-1})$, or that the distribution of $x_t$ only depends on $x_{t-1}$. For diffusion models this is just:

$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)$

This is also known as the transition distribution, or the distribution of $x_t$ given $x_{t-1}$.

Now for the equation we introducted earlier, let's prove it. We can use induction to get this:

Base case ($t = 1$):

$q(x_1 \mid x_0) = \mathcal{N}(x_1; \sqrt{\alpha_1}x_{0}, 1 - \alpha_1)$

Assume for $t - 1$:

$q(x_{t-1} \mid x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0,1 - \bar{\alpha}_{t-1})$

Then we get:

$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)$

Using these we use the Chapman-Kolmogorov equation to get:

$q(x_t \mid x_0) = \int{q(x_t \mid x_{t-1}) \cdot q(x_{t-1} \mid x_0) dx_{t-1}}$

Here we are convolving 2 gaussian distributions, and the result will also be a gaussian distribution.

First let's find the mean:

We want to find $\mathbb{E}[x_t \mid x_0]$. So we get:

$\mathbb{E}[x_t \mid x_{t - 1}] = \mathbb{E}[\sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t} \epsilon] = \sqrt{\alpha_t}x_{t-1}$

Now we find:

$\mathbb{E}[\sqrt{\alpha_t}x_{t-1} \mid x_0] = \sqrt{\alpha_t}\mathbb{E}[x_{t-1} \mid x_0] = \sqrt{\alpha_t}\mathbb{E}[\mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0,1 - \bar{\alpha}_{t-1})] = \sqrt{\alpha_t} \cdot \bar{\alpha}_{t-1}x_0 = \bar{\alpha}_{t}x_0$.

Now we want to find $\text{Var}(x_t \mid x_0)$. This equals

$\mathbb{E}[\text{Var}(x_t \mid x_{t-1}) \mid x_0] + \text{Var}(\mathbb{E}[(x_t \mid x_{t-1})] \mid x_0)$

$\text{Var}(x_t \mid x_{t-1}) = \text{Var}(\mathcal{N}(\sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)) = 1 - \alpha_t$

So since there is no $x_0$ dependent term this means,

$\mathbb{E}[\text{Var}(x_t \mid x_{t-1}) \mid x_0] = 1 - \alpha_t$

For the second part, we first do

$\mathbb{E}[(x_t \mid x_{t-1})] = \mathbb{E}[\mathcal{N}(\sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)] = \sqrt{\alpha_t}x_{t-1}$, and then we do:

$\text{Var}(\sqrt{\alpha_t}x_{t-1} \mid x_0) = \sqrt{\alpha_t}^2\text{Var}(x_{t-1} \mid x_0) = \alpha_t\text{Var}(\mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0,1 - \bar{\alpha}_{t-1})) = \alpha_t(1 - \bar{\alpha}_{t-1})$

Now we add: $\alpha_t(1 - \bar{\alpha}_{t-1}) + 1 - \alpha_t = 1 - \alpha_t + \alpha_t + \bar{\alpha}_t = 1 - \bar{\alpha}_t$

Due to induction, we have proven that,

$q(x_t \mid x_0) = \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_0,1 - \bar{\alpha}_{t})$, and likewise:

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$, Where:

$\bar{\alpha}_t = \alpha_1 \cdot \alpha_2 \cdot ... \cdot \alpha_t = \prod_{n=1}^{t} \alpha_n$

$\epsilon \sim \mathcal{N}(0, 1)$

$\beta_t = \beta_{\text{min}} + \frac{t}{T} \cdot (\beta_{\text{max}} - \beta_{\text{min}})$

We are able to do the process above, as if you have any distribution sampled: $x \sim \mathcal{N}(\mu, \sigma^2)$, this is the same as:

$x \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$.

## Training

The training breakdown for a diffusion model is very surpisingly easy. Here are some pre-requistes to it:

- So far, we have a clean sample and use some noise to corrupt the sample further and further on every timestep
- Now if we want to retrieve the clean sample, we can do so, but that would require us to know the noise at every timestep
- So instead, we build a neural network to predict the noise in the backward process, more on that later

Training process:

1. Sample an image from the dataset: $x_0 \sim X$
2. Sample a timestep: $t \sim \mathcal{U}(0, T)$
3. Sample noise: $\epsilon \sim \mathcal{N}(0, 1)$
4. Get the corrupt sample: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$
5. Get the model prediction: $\epsilon_{\theta} = f_{\theta}(x_t, t)$, accepts the noisy sample and timestep
6. Compute the loss: $\mathcal{L} = ||\epsilon - \epsilon_{\theta}||^2$
7. Backpropagate this loss into the model, update the model weights with an optimizer
8. Do steps 1-7 until satisfied, then return $f_{\theta}$

Now that we have the model, let's learn about the backward process

## Backward Process

Let's say we want to get the image at $x_{t-1}$. For this we need the image at $x_t$ and $x_0$. Since during sampling we will not be given the clean sample, in our derivation we will remove it. But for now we want to find $q(x_{t-1} \mid x_t, x_0)$, this means find the image at t-1, given the images at t and 0. Using bayes' rule we get:

$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) \cdot q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$.

By the markov property, we can discard $x_0$, since $x_t$ only depends on the previous state, $x_{t-1}$.

So we get:

$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}) \cdot q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$.

We know every part of this equation, as we have:

$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)$

$q(x_{t-1} \mid x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0,1 - \bar{\alpha}_{t-1})$

$q(x_t \mid x_0) = \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}}x_0,1 - \bar{\alpha}_{t})$

Setup $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \mu_{t-1}, \sigma^2_{t-1})$

Here $\mu_{t-1}$ and $\sigma^2_{t-1}$ are the mean and variance of the distribution.

Write each of the 3 distributions in their explicit forms:

$q(x_t \mid x_{t-1}) = \frac{1}{\sqrt{2\pi(1 - \alpha_t)}} e^{-\frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{2(1 - \alpha_t)}}$

$q(x_{t-1} \mid x_0) = \frac{1}{\sqrt{2\pi(1 - \bar{\alpha}_{t-1})}} e^{-\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2}{2(1 - \bar{\alpha}_{t-1})}}$

$q(x_t \mid x_0) = \frac{1}{\sqrt{2\pi(1 - \bar{\alpha}_{t})}} e^{-\frac{(x_t - \sqrt{\bar{\alpha}_{t}}x_0)^2}{2(1 - \bar{\alpha}_{t})}}$

We add up the exponent parts:

$-\frac{1}{2}[\frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{1 - \alpha_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1 - \bar{\alpha}_{t-1}} - \frac{(x_t - \sqrt{\bar{\alpha}_{t}}x_0)^2}{1 - \bar{\alpha}_{t}}]$

Since the third term doesn't involve $x_{t-1}$, we can ignore it. Simplifying the first two terms we get:

$-\frac{1}{2}[\frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{1 - \alpha_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1 - \bar{\alpha}_{t-1}}] = -\frac{1}{2}[\frac{x_t^2 - 2\sqrt{\alpha_t}x_tx_{t-1} + \alpha_tx_{t-1}^2}{1 - \alpha_t} + \frac{x_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}}x_0x_{t-1} + \bar{\alpha}_{t-1}x_0^2}{1 - \bar{\alpha}_{t-1}}]$

We write the term above as: $-\frac{1}{2}[\frac{(x_{t-1} - \mu_{t-1})^2}{\sigma^2_{t-1}}]$, so to solve we set them equal and get:

$\frac{(x_{t-1} - \mu_{t-1})^2}{\sigma^2_{t-1}} = \frac{x_t^2 - 2\sqrt{\alpha_t}x_tx_{t-1} + \alpha_tx_{t-1}^2}{1 - \alpha_t} + \frac{x_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}}x_0x_{t-1} + \bar{\alpha}_{t-1}x_0^2}{1 - \bar{\alpha}_{t-1}}$

Looking only at the terms involving $x_{t-1}^2$, we get:

$\frac{x_{t-1}2}{\sigma^2_{t-1}} = (\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})x_{t-1}^2$

Cancelling the $x_{t-1}^2$ terms, we get:

$\frac{1}{\sigma^2_{t-1}} = \frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{(1 - \bar{\alpha}_{t-1})\alpha_t + 1 - \alpha_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}$ = $\frac{\alpha_t - \bar{\alpha}_t + 1 - \alpha_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}$ = $\frac{1 - \bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}$

Solving for variance, we get:

$\sigma^2_{t-1} = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$

Now for the mean, only looking at the $x_{t-1}$ terms, we get:

$\frac{-2x_{t-1}\mu_{t-1}}{\sigma^2_{t-1}} = -2[\frac{\sqrt{\alpha_t}x_tx_{t-1}}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0x_{t-1}}{1 - \bar{\alpha}_{t-1}}]$

Cancelling the $x_{t-1}$ terms, we get:

$\frac{\mu_{t-1}}{\sigma^2_{t-1}} = \frac{\sqrt{\alpha_t}x_t}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1 - \bar{\alpha}_{t-1}}$

Solving for the mean, we get:

$\mu_{t-1} = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}[\frac{\sqrt{\alpha_t}x_t}{1-\alpha_t} + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}x_0}{1 - \bar{\alpha}_{t-1}}] = \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t}x_t + \frac{(1- \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t}x_0$

So our final answer is:

$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t}x_t + \frac{(1- \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t}x_0, \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t})$

Now during inference, we will not have $x_0$, but we can write it in terms of $x_t$. Recalling the equation for $x_t$ in terms of $x_0$, we get:

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$

Rearranging this we get:

$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\ \epsilon}{\sqrt{\bar{\alpha}_t}}$

Simplifying we get:

$q(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}}, \epsilon), \frac{(1 - \bar{\alpha}_{t-1})(1 - \alpha_t)}{1 - \bar{\alpha}_t})$

## Sampling

Now that we have the backward process, we can write out the gaussian as:

$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}} \epsilon) + \frac{(1 - \bar{\alpha}_{t-1})(1 - \alpha_t)}{1 - \bar{\alpha}_t} \cdot z$, where $z \sim \mathcal{N}(0, 1)$

This is what is known as a stochastic or non-deterministic sampling process. This is because we have randomness in the generation of our sample, with variable $z$, which means that different generations can lead to different outputs. Now we can sample using this equation, but have to change one thing. Since we do not know the added noise $\epsilon$, we can replace it with the model prediction $f_{\theta}(x_t, t) = \epsilon_{\theta}$, and plug that in for our noise.

So our final equation is:

$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}) + \frac{(1 - \bar{\alpha}_{t-1})(1 - \alpha_t)}{1 - \bar{\alpha}_t} \cdot z$, where $z \sim \mathcal{N}(0, 1)$, and $\epsilon_{\theta} = f_{\theta}(x_t, t)$

Here is the sampling process:

1. Sample $x_t \sim \mathcal{N}(0, 1)$
2. Iterate steps 3 - 5 from T to 0, where T is the max timestep
3. Sample $z \sim \mathcal{N}(0, 1)$
4. Get the model prediction: $\epsilon_{\theta} = f_{\theta}(x_t, t)$
5. Get the new sample: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}) + \frac{(1 - \bar{\alpha}_{t-1})(1 - \alpha_t)}{1 - \bar{\alpha}_t} \cdot z$
6. Return $x_0$

This sample process is good and works in many cases, but there are some issues:

1. You have to sample across all timesteps, and in cases where you have many, it is very slow & inefficient
2. The process is stochastic, so different outputs for the same input, which is not always desired
3. This makes it bad for image editing and inversion, and more

So we will look at another technique called DDIM, which is a deterministic sampling method, and it allows for faster sampling, allowing it to be more efficient.

## DDIM Sampling

Alright, let's go way back to the forward process equation, which is:

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$.

Now we are trying to get $x_{t-1}$

Let's write: $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\ \epsilon$

If we remember:

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$, so

$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\ \epsilon}{\sqrt{\bar{\alpha}_t}}$

Now plugging this into the equation for $x_{t-1}$, we get:

$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\ \epsilon}{\sqrt{\bar{\alpha}_t}} + \sqrt{1 - \bar{\alpha}_{t-1}}\ \epsilon$

For both the $\epsilon$ terms, replace them with $f_{\theta}(x_t, t) = \epsilon_{\theta}$, and we get:

$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\ \epsilon_{\theta}}{\sqrt{\bar{\alpha}_t}} + \sqrt{1 - \bar{\alpha}_{t-1}}\ \epsilon_{\theta} = \frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} x_t + (\sqrt{1 - \bar{\alpha}_{t-1}} - \sqrt{\frac{\bar{\alpha}_{t-1}(1 - \bar{\alpha}_t)}{\bar{\alpha}_t}}) \epsilon_{\theta}$

One more interesting aspect, is that during sampling, we do not have to sample at every timestep, and this is because since the process is deterministic, we are traversing the same path, just in fewer and bigger steps. This can sometimes cause a slight loss in quality, but it is negligible in many cases and is outweighed by the major speedup.

Sampling process for DDIM
Let's say we will have only $S$ sampling steps, instead of $T$, here $S$ is one less than the number of sampling steps. So we can compute the equivalent timestep and sample using that

1. Sample $x_t \sim \mathcal{N}(0, 1)$
2. Iterate steps 3 - 6 from 0 to S, with the current iteration being $s$
3. Calculate: $t = \frac{T \cdot (S - s)}{S}$
4. Get the model prediction: $\epsilon_{\theta} = f_{\theta}(x_t, t)$
5. Get the clean sample prediction $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_{\theta}}{\sqrt{\bar{\alpha}_t}}$
6. Get $x_t = \sqrt{\bar{\alpha}_t} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon_{\theta}$
7. Return $x_0$

## Conclusion

Wow, that was a lot! Don't worry I was very confused writing this as well, but the main thing to keep in mind is the difference between distributions and samples, as a distribution just shows you all the possible samples. I know I said it was a surface-level introduction, but this covers a lot of the mathematical background behind diffusion models, and I hope you learned something new. In part 2, we will be looking at the code and how to implement this in PyTorch with the CelebA dataset, so stay tuned for that!
