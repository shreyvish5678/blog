---
author: Shrey Vishen
pubDatetime: 2025-04-15T00:58:04.000+00:00
title: An Introduction to Diffusion Models Part 2
slug: "diffusion-models-p2"
featured: true
tags:
  - pytorch
  - diffusion
  - noise
description: "An Introduction to Diffusion Models Part 2: Implementing a Diffusion Model"
---

## Introduction

This is a part 2 post about diffusion models and how they work, specifically centered more on the implementation part. If you are curious about the ground work and mathematics behind diffusion models, I recommend checking out part one [here](https://sv-blog.vercel.app/posts/diffusion-models/). In this post, we will firstly implement the forward and backward diffusion processes involvied in diffusion models, and then implement the model architecture, along with training and sampling.

## Forward Diffusion Process

Here is a recap of the forward diffusion process from part 1:
$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$

Now let's implement this!

Firstly, we want to define our noise schedule, so let's pick our beta min and beta max:

```python
import torch
beta_min = 1e-4
beta_max = 0.02
max_timesteps = 1000
```

We have picked our beta values and timesteps, so now what we want to do is create a linear beta schedule, and all it does is create a line that goes from beta min to beta max on the y-scale and from 0 to max_timesteps on the x-scale. We don't need to worry about the formula, pytorch handles this with a built in function called `linspace`. But let's still recall the formula:

$\beta_t = \beta_{\text{min}} + \frac{t}{T} \cdot (\beta_{\text{max}} - \beta_{\text{min}})$

Now we implement:

```python
beta_schedule = torch.linspace(beta_min, beta_max, max_timesteps)
```

Now we want to get our alpha values, which are just 1 - beta.

```python
alpha_schedule = 1 - beta_schedule
```

Now let's bring up the formula for $\bar{\alpha}_t$ from part 1:
$\bar{\alpha}_t = \alpha_1 \cdot \alpha_2 \cdot ... \cdot \alpha_t = \prod_{n=1}^{t} \alpha_n$

So what we need is a tensor where at index $t$ we have $\bar{\alpha}_t$, instead of $\alpha_t$. There is a neat function in pytorch called `cumprod`, doing exactly this, so we can abstract and get:

```python
alpha_cumprod = torch.cumprod(alpha_schedule, dim=0)
```

The dimension 0 means we are doing it along the first axis, which is the time axis. Here is the full code for the schedule:

```python
import torch
beta_min = 1e-4
beta_max = 0.02
max_timesteps = 1000
beta_schedule = torch.linspace(beta_min, beta_max, max_timesteps)
alpha_schedule = 1 - beta_schedule
alpha_cumprod = torch.cumprod(alpha_schedule, dim=0)
```

Now let's create a function that takes in our clean sample and our timestep, and returns the noised version. Right now this is just a filler:

```python
def forward_diffusion(x_0, t):
    return None
```

Firstly, what we want to do is sample noise from the normal distribution, with the size of this noise tensor being the exact same as our clean sample tensor. So we do:

```python
noise = torch.randn_like(x_0)
```

The `randn_like` function samples a tensor from the normal distribution with the same shape as the input tensor. Bringing back the formula we have:

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$

Now firstly, we get the $\bar{\alpha}_t$ value from our `alpha_cumprod` tensor, and use this to get the two coefficients, one for the clean sample and one for the noise. We can do this by indexing into the tensor with the timestep `t`:

```python
alpha_t = alpha_cumprod[t]
sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1)
sqrt_1_minus_alpha_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1, 1)
```

The `.view(-1, 1, 1, 1)` is used to make sure the alpha value tensors have the same number of dimensions as the clean sample and noise tensors. Now we can plug these values into our formula:

```python
x_t = sqrt_alpha_t * x_0 + sqrt_1_minus_alpha_t * noise
```

This completes our full function:

```python
def forward_diffusion(x_0, t):
    noise = torch.randn_like(x_0)
    alpha_t = alpha_cumprod[t]
    sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1)
    sqrt_1_minus_alpha_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1, 1)
    x_t = sqrt_alpha_t * x_0 + sqrt_1_minus_alpha_t * noise
    return x_t, noise
```

Here is a sample output of using this function for noising an image:

<img src="@assets/diff_assets/forward_process.gif" alt="forward_diffusion" width="256" height="256" />

## Model Training

The model training code is pretty complex, but most of it is not in the context of diffusion models, involving things like setting up configurations, dataloaders, etc, so I won't do into that. There is also a lot of device parameter handling, which I won't include in the code here, but all of it is available on the github I will post (when it's out). I'll also skip gradient clipping, epochs, batch size, etc. So with all of that, let's recap the training process:

1. Sample an image from the dataset: $x_0 \sim X$
2. Sample a timestep: $t \sim \mathcal{U}(0, T)$
3. Sample noise: $\epsilon \sim \mathcal{N}(0, 1)$
4. Get the corrupt sample: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\ \epsilon$
5. Get the model prediction: $\epsilon_{\theta} = f_{\theta}(x_t, t)$, accepts the noisy sample and timestep
6. Compute the loss: $\mathcal{L} = ||\epsilon - \epsilon_{\theta}||^2$
7. Backpropagate this loss into the model, update the model weights with an optimizer
8. Do steps 1-7 until satisfied, then return $f_{\theta}$

Firstly, we sample our image $x_0 \sim X$, which is a tensor of shape (batch_size, channels, height, width), and we are looping through the dataloader, so it is something like:

```python
for images in dataloader:
    x_0 = images.copy()
```

Again we wont worry about data types or device handling. Now we sample our timestep with the formula $t \sim \mathcal{U}(0, T)$, so we want to select a random integer within these bounds with each having equal probability. We can do this with the `randint` function in pytorch:

```python
t = torch.randint(0, max_timesteps, (1,))
```

We assume a batch size of 1, since we don't care about batch size for this example. Now we pass this into our forward diffusion function, and get the noised sample and noise:

```python
x_t, noise = forward_diffusion(x_0, t)
```

Now we want our model prediction, so we will assume we have a pre-intialized and pre-configured `model`, and pass in our 2 values to get our $\epsilon_{\theta} = f_{\theta}(x_t, t)$:

```python
pred_noise = model(x_t, t)
```

Now we want to compute our loss, which is the mean squared error between the predicted noise and the actual noise. Pytorch has a built in function for this called `nn.MSELoss`, so we can do:

```python
loss = nn.MSELoss()(pred_noise, noise)
```

Now we backpropagate and update model weights with optimizer (doesn't really matter for diffusion):

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

This is the full training loop:

```python
for images in dataloader:
    x_0 = images.copy()
    t = torch.randint(0, max_timesteps, (1,))
    x_t, noise = forward_diffusion(x_0, t)
    pred_noise = model(x_t, t)
    loss = nn.MSELoss()(pred_noise, noise)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Now that we have trained the model, we can create our sampling function with DDIM

## Sampling

Recall the math behind sampling from part 1:

1. Sample $x_t \sim \mathcal{N}(0, 1)$
2. Iterate steps 3 - 6 from 0 to S, with the current iteration being $s$
3. Calculate: $t = \frac{T \cdot (S - s)}{S}$
4. Get the model prediction: $\epsilon_{\theta} = f_{\theta}(x_t, t)$
5. Get the clean sample prediction $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_{\theta}}{\sqrt{\bar{\alpha}_t}}$
6. Get $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\ \epsilon_{\theta}$
7. Return $x_0$

Let's implement a function. Our function will take the following parameters: `img_size`, `max_timesteps`, `sample_timesteps`, which are all self-explanatory. SO let's create the filler function:

```python
def sample_ddim(img_size, max_timesteps, sample_timesteps):
    return None
```

Firstly, we want to sample our $x_t$ from the normal distribution, so we do:

```python
x_t = torch.randn(*img_size)
```

This essentially creates a tensor with the shape given in the `img_size` tuple, and our cleaned sample will have the same shape. Now we want to create a tensor that stores our timesteps $t$, and uses the parameters `max_timesteps` and `sample_timesteps` to create a linear space between 0 and $T$ with $S$ points. We can do this with the `linspace` function:

```python
t_schedule = torch.linspace(0, max_timesteps, sample_timesteps).flip(0).long()
```

The `.flip(0)` is used to reverse the order of the tensor, so we are going from $T$ to 0, and the `.long()` is used to convert the tensor to long integers, since we will be using this as an index. Now we can create a loop to iterate through our pre-created timesteps, and the first thing we want to do is get the current timestep $t$ from our `t_schedule` tensor. We can do this with a simple indexing operation. We also want to get our next timestep, since it is used in the sampling formula so we index to the next element

```python
for i in range(len(t_schedule) - 1)
    t = t_schedule[i].view(-1)
    t_next = t_schedule[i + 1].view(-1)
```

Now we get the following values from our pre-computed `alpha_cumprod` tensor: $\sqrt{\bar{\alpha}_t}$, $\sqrt{1 - \bar{\alpha}_t}$, $\sqrt{\bar{\alpha}_{t-1}}$, and $\sqrt{1 - \bar{\alpha}_{t-1}}$. We can do this with a simple indexing operation:

```python
    sqrt_alpha_t = torch.sqrt(alpha_cumprod[t])
    sqrt_1_minus_alpha_t = torch.sqrt(1 - alpha_cumprod[t])
    sqrt_alpha_t_next = torch.sqrt(alpha_cumprod[t_next])
    sqrt_1_minus_alpha_t_next = torch.sqrt(1 - alpha_cumprod[t_next])
```

Now we want to get our model prediction, so we pass in our $x_t$ and $t$ into the model:

```python
    pred_noise = model(x_t, t)
```

Now we get our clean sample prediction $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_{\theta}}{\sqrt{\bar{\alpha}_t}}$:

```python
    x_0_hat = (x_t - sqrt_1_minus_alpha_t * pred_noise) / sqrt_alpha_t
```

Now we can get our $x_{t-1}$ with the formula $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\ \epsilon_{\theta}$:

```python
    x_t_next = sqrt_alpha_t_next * x_0_hat + sqrt_1_minus_alpha_t_next * pred_noise
```

This is the full function:

```python
def sample_ddim(img_size, max_timesteps, sample_timesteps):
    x_t = torch.randn(*img_size)
    t_schedule = torch.linspace(0, max_timesteps, sample_timesteps).flip(0).long()
    for i in range(len(t_schedule) - 1):
        t = t_schedule[i].view(-1)
        t_next = t_schedule[i + 1].view(-1)
        sqrt_alpha_t = torch.sqrt(alpha_cumprod[t])
        sqrt_1_minus_alpha_t = torch.sqrt(1 - alpha_cumprod[t])
        sqrt_alpha_t_next = torch.sqrt(alpha_cumprod[t_next])
        sqrt_1_minus_alpha_t_next = torch.sqrt(1 - alpha_cumprod[t_next])
        pred_noise = model(x_t, t)
        x_0_hat = (x_t - sqrt_1_minus_alpha_t * pred_noise) / sqrt_alpha_t
        x_t_next = sqrt_alpha_t_next * x_0_hat + sqrt_1_minus_alpha_t_next * pred_noise
    return x_0_hat
```

Here is a generated image from the model:

<img src="@assets/diff_assets/sample.png" alt="sample_ddim" width="256" height="256" />

Here is a GIF showing the denoising process in action:

<img src="@assets/diff_assets/sample.gif" alt="sample_ddim" width="256" height="256" />

## Conclusion

This concludes the second part of the introduction to diffusion models. In this blog, we implemented all the math from part 1 into functions utilizing pytorch. If I come back to this, I might explore more complex models involving conditioning, look at low-level implementations, or dive more into the model and other parts themselves. I hope you enjoyed this 2 part series, and can find the github repo [here](https://github.com/shreyvish5678/Diffusion-Explained). If you have any questions, feel free to reach out to me on [Twitter](https://x.com/SVishen7235). Thank you for reading!
