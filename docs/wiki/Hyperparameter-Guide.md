# Hyperparameter Guide for PRISMT

A didactic guide for experimental neuroscientists: what each hyperparameter does, why it matters, and how changing it affects your model.

---

## Quick Navigation

- [Introduction](#introduction)
- [Learning and Optimization](#learning-and-optimization): lr, scheduler, warmup_ratio
- [Data and Memory](#data-and-memory): batch_size
- [Model Size and Capacity](#model-size-and-capacity): hidden_dim, num_heads, num_layers, ff_dim
- [Regularization](#regularization): dropout, weight_decay
- [Constraints and Tips](#constraints-and-tips)

---

## Introduction

### What are hyperparameters?

When you train a neural network, the model learns **weights** (numbers that get updated from your data). But before training starts, you must choose **hyperparameters**: settings that control *how* the model learns and *how big* the model is. Unlike weights, hyperparameters are not learned from data—you (or Optuna) set them.

### Why do they matter?

Poor hyperparameters can lead to:
- **Underfitting**: The model is too simple or learns too slowly; validation accuracy stays low.
- **Overfitting**: The model memorizes the training set; validation accuracy drops even as training accuracy rises.
- **Instability**: Training loss explodes or oscillates; the model never converges.

Good hyperparameters help the model generalize: perform well on new data it has never seen.

### What does Optuna do?

Instead of guessing hyperparameters by hand, **Optuna** automatically searches over many combinations. It runs many "trials," each with different hyperparameters, and keeps track of which combinations work best. You can think of it as a smart grid search that learns from past trials to focus on promising regions.

---

## Learning and Optimization

### Learning rate (lr)

| Property | Value |
|---------|-------|
| **Type** | Float (log scale) |
| **Optuna range** | 1e-5 to 5e-3 |
| **Typical good values** | 1e-4 to 5e-4 |

**What it is:** The step size used when updating the model weights. Each training step moves the weights a little in the direction that reduces the loss.

**What it controls:** How fast the model learns.

**How changing it helps:**
- **Too high**: Updates are too large; the model overshoots good solutions. Loss may spike or never settle.
- **Too low**: Updates are tiny; learning is very slow. You may need many more epochs.
- **Just right**: The model converges steadily to a good solution.

**Practical tip:** If your training loss is noisy or explodes, try lowering the learning rate. If it decreases very slowly, try raising it. Optuna will search this range for you.

---

### Scheduler

| Property | Value |
|---------|-------|
| **Type** | Categorical |
| **Optuna options** | `cosine_warmup`, `step`, `reduce_on_plateau`, `cosine` |

**What it is:** A rule for changing the learning rate over the course of training. The initial learning rate (chosen by Optuna) is modified each epoch according to this schedule.

**What it controls:** How the learning rate evolves over time.

**How each option helps:**
- **cosine_warmup**: Start with a gradual increase (warmup), then smoothly decay. Often works well for transformers.
- **step**: Reduce the learning rate by a fixed factor at fixed epochs (e.g., every 30 epochs). Simple and predictable.
- **reduce_on_plateau**: Only reduce when validation performance stops improving. Adapts to your data.
- **cosine**: Smooth decay without warmup. Good when you want a simple schedule.

**Practical tip:** `cosine_warmup` is a strong default for transformer models. If training seems to plateau early, `reduce_on_plateau` can help by lowering the learning rate when progress stalls.

---

### Warmup ratio (warmup_ratio)

| Property | Value |
|---------|-------|
| **Type** | Float |
| **Optuna range** | 0.0 to 0.2 |

**What it is:** The fraction of training (by epoch) during which the learning rate increases linearly from 0 (or a small value) up to the chosen learning rate.

**What it controls:** How gently training "ramps up" at the start.

**How changing it helps:**
- **0.0**: No warmup; full learning rate from the first epoch. Can be unstable for large models.
- **0.05–0.1**: Short warmup; often enough to stabilize early training.
- **0.2**: Longer warmup; useful when the model is large or the dataset is small.

**Practical tip:** A small warmup (e.g., 0.05–0.1) usually helps. If you see high loss or NaN in the first few epochs, try increasing the warmup ratio.

---

## Data and Memory

### Batch size (batch_size)

| Property | Value |
|---------|-------|
| **Type** | Categorical |
| **Optuna options** | 16, 32, 64 |

**What it is:** The number of trials (samples) processed together in one forward and backward pass before updating the weights.

**What it controls:** Trade-off between speed, memory use, and gradient quality.

**How changing it helps:**
- **Smaller (16)**: Uses less GPU memory; gradients are noisier but can sometimes generalize better. Slower per epoch.
- **Larger (64)**: Faster per epoch; gradients are smoother. Requires more memory; may run out of memory on small GPUs.

**Practical tip:** Start with 32. If you get "CUDA out of memory," use 16. If you have plenty of memory and want faster runs, try 64.

---

## Model Size and Capacity

### Hidden dimension (hidden_dim)

| Property | Value |
|---------|-------|
| **Type** | Categorical |
| **Optuna options** | 64, 128, 256 |

**What it is:** The size of the vector that represents each "token" (e.g., each brain region) inside the transformer. Every token is projected into a hidden_dim-dimensional space.

**What it controls:** How much information each token can carry.

**How changing it helps:**
- **Smaller (64)**: Fewer parameters; faster and less prone to overfitting. May underfit complex patterns.
- **Larger (256)**: More capacity; can capture finer structure. Needs more data and compute; may overfit on small datasets.

**Practical tip:** For small datasets (e.g., &lt;500 trials), try 64 or 128. For larger datasets, 128 or 256 can help. Optuna will search; watch validation accuracy to see if a larger model helps or hurts.

---

### Number of heads (num_heads)

| Property | Value |
|---------|-------|
| **Type** | Categorical |
| **Optuna options** | 2, 4, 8 |

**What it is:** The number of "attention heads" in the transformer. Each head looks at different aspects of the relationships between brain regions (e.g., one head might focus on temporal patterns, another on spatial).

**What it controls:** How many different "views" of the data the model can attend to in parallel.

**How changing it helps:**
- **Fewer (2)**: Simpler; fewer parameters. May miss subtle interactions.
- **More (8)**: Richer attention patterns. Requires hidden_dim to be divisible by num_heads (e.g., hidden_dim=64 with num_heads=8 works; hidden_dim=64 with num_heads=5 does not).

**Constraint:** `hidden_dim` must be divisible by `num_heads`. Optuna automatically prunes invalid combinations.

**Practical tip:** 4 heads is a good default. With hidden_dim=256, 8 heads is natural; with hidden_dim=64, 2 or 4 heads work.

---

### Number of layers (num_layers)

| Property | Value |
|---------|-------|
| **Type** | Integer |
| **Optuna range** | 1 to 6 |

**What it is:** The number of transformer layers stacked on top of each other. Each layer applies self-attention and a feed-forward network.

**What it controls:** How deep the model is; deeper models can learn more abstract, hierarchical representations.

**How changing it helps:**
- **Shallow (1–2)**: Fast; good for simple tasks or small data. May underfit.
- **Deeper (4–6)**: More abstraction; can capture complex, multi-level structure. Slower and more prone to overfitting on small data.

**Practical tip:** For phase classification (early vs. late), 2–4 layers often suffice. For harder tasks or more data, 4–6 can help. Start shallow and add depth if you underfit.

---

### Feed-forward dimension (ff_dim)

| Property | Value |
|---------|-------|
| **Type** | Categorical |
| **Optuna options** | 128, 256, 512 |

**What it is:** The size of the hidden layer inside each transformer block's feed-forward network. Each block has: attention → feed-forward (hidden_dim → ff_dim → hidden_dim).

**What it controls:** How much computation happens "inside" each layer beyond attention.

**How changing it helps:**
- **Smaller (128)**: Fewer parameters; faster. May limit expressiveness.
- **Larger (512)**: More capacity per layer. Often paired with larger hidden_dim.

**Practical tip:** ff_dim is often 2–4× hidden_dim. For hidden_dim=64, 128 is typical; for hidden_dim=256, 256 or 512 works well.

---

## Regularization

### Dropout

| Property | Value |
|---------|-------|
| **Type** | Float |
| **Optuna range** | 0.1 to 0.5 |

**What it is:** During training, a random fraction of neurons is "dropped" (set to zero) each forward pass. This prevents the model from relying too heavily on any single pathway.

**What it controls:** Regularization strength; higher dropout reduces overfitting but can cause underfitting if too high.

**How changing it helps:**
- **Lower (0.1–0.2)**: Less regularization; model can fit the data more closely. Use when you have lots of data or the model is small.
- **Higher (0.3–0.5)**: Stronger regularization; reduces overfitting. Use when validation accuracy is much lower than training accuracy.

**Practical tip:** If training accuracy is high but validation accuracy is low, increase dropout. If both are low, try decreasing it.

---

### Weight decay (weight_decay)

| Property | Value |
|---------|-------|
| **Type** | Float (log scale) |
| **Optuna range** | 1e-8 to 1e-2 |

**What it is:** An L2 penalty on the weights: the optimizer is encouraged to keep weights small. This discourages the model from memorizing noise.

**What it controls:** Another form of regularization; complements dropout.

**How changing it helps:**
- **Very small (1e-8)**: Almost no penalty; behaves like no weight decay.
- **Moderate (1e-5 to 1e-3)**: Common range; helps generalization.
- **Large (1e-2)**: Strong penalty; can cause underfitting.

**Practical tip:** 1e-5 to 1e-4 is a good starting range. If you overfit, try increasing; if you underfit, try decreasing.

---

## Constraints and Tips

### Constraint: hidden_dim and num_heads

The transformer requires `hidden_dim % num_heads == 0`. For example:
- hidden_dim=64, num_heads=2, 4, or 8 ✓
- hidden_dim=128, num_heads=2, 4, or 8 ✓
- hidden_dim=64, num_heads=5 ✗ (Optuna will prune this automatically)

### Suggested reading order

1. **Start here**: lr, batch_size, dropout—these have the most intuitive effect on training.
2. **Model size**: hidden_dim, num_layers, num_heads, ff_dim—control capacity and compute.
3. **Advanced**: scheduler, warmup_ratio, weight_decay—fine-tune optimization dynamics.

### Quick reference table

| Hyperparameter | Increase when... | Decrease when... |
|---------------|------------------|------------------|
| lr | Learning is too slow | Loss explodes or oscillates |
| batch_size | You have GPU memory to spare | You get OOM errors |
| hidden_dim | Model underfits, data is large | Model overfits, data is small |
| num_layers | Task is complex, more abstraction needed | Task is simple, or overfitting |
| dropout | Validation << training accuracy | Both accuracies are low |
| weight_decay | Overfitting | Underfitting |

---

## See also

- [MATLAB GUI Tutorial](MATLAB-GUI-Tutorial) – Configure and run training from the GUI
- [scripts/run_optuna_cluster.sh](../../scripts/run_optuna_cluster.sh) – Minimal SLURM script for cluster HPO
