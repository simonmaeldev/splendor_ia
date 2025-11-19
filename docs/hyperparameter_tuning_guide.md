# Hyperparameter Tuning Guide for Splendor Imitation Learning

This guide provides recommendations for experimenting with model architectures and training hyperparameters to improve performance.

## Quick Start

### Using Pre-configured Architectures

Three configurations are provided:

1. **Small** (`configs/config_small.yaml`)
   - Trunk: [256, 128]
   - Heads: [64]
   - Use for: Quick testing, debugging, limited compute
   - Expected training time: ~30-45 minutes on RTX 4090

2. **Medium** (`configs/config_medium.yaml`) - **Default**
   - Trunk: [512, 256]
   - Heads: [128, 64]
   - Use for: Standard training, good balance of capacity and speed
   - Expected training time: ~2-3 hours on RTX 4090

3. **Large** (`configs/config_large.yaml`)
   - Trunk: [512, 512, 256]
   - Heads: [128, 128, 64]
   - Use for: Maximum capacity if underfitting
   - Expected training time: ~4-5 hours on RTX 4090

### Training with Different Configs

```bash
# Small architecture (quick test)
uv run python -m src.imitation_learning.train --config src/imitation_learning/configs/config_small.yaml

# Medium architecture (default)
uv run python -m src.imitation_learning.train --config src/imitation_learning/config.yaml

# Large architecture (max capacity)
uv run python -m src.imitation_learning.train --config src/imitation_learning/configs/config_large.yaml
```

## Hyperparameter Categories

### 1. Model Architecture

#### Trunk Dimensions (`trunk_dims`)

Controls the shared representation capacity.

**Recommendations:**
- Start with: `[512, 256]`
- If underfitting: `[512, 512, 256]` or `[768, 512, 256]`
- If overfitting: `[256, 128]` or `[384, 192]`

**Signs of underfitting:**
- Training loss plateaus at high value (>1.0)
- Both training and validation accuracy are low (<60%)
- Training and validation losses are similar

**Signs of overfitting:**
- Training loss much lower than validation loss
- Validation loss increases while training loss decreases
- Large gap between training and validation accuracy

#### Head Dimensions (`head_dims`)

Controls task-specific capacity.

**Recommendations:**
- Start with: `[128, 64]`
- If underfitting: `[256, 128]` or `[128, 128, 64]`
- If overfitting: `[64]` or `[96, 48]`

Generally, heads need less capacity than trunk since they solve simpler sub-problems.

#### Dropout (`dropout`)

Regularization to prevent overfitting.

**Recommendations:**
- Start with: `0.3`
- If overfitting: increase to `0.4` or `0.5`
- If underfitting: decrease to `0.2` or `0.1`

Higher dropout = more regularization = less overfitting.

### 2. Training Hyperparameters

#### Learning Rate (`learning_rate`)

Controls optimization step size.

**Recommendations:**
- Start with: `0.001` (1e-3)
- If training unstable (loss jumps): `0.0005` or `0.0001`
- If training too slow: `0.002` or `0.005`

**Signs of too high LR:**
- Loss explodes to NaN
- Loss oscillates wildly
- Model doesn't converge

**Signs of too low LR:**
- Training very slow
- Loss decreases linearly for many epochs
- May need 100+ epochs to converge

#### Batch Size (`batch_size`)

Number of samples per training step.

**Recommendations:**
- Start with: `256`
- Larger batch (512, 1024): faster training, more GPU memory, more stable gradients
- Smaller batch (128, 64): less GPU memory, noisier gradients, can help generalization

On RTX 4090 with 24GB VRAM, you can likely use 512 or even 1024.

#### Epochs (`epochs`)

Maximum number of training passes through data.

**Recommendations:**
- Start with: `50`
- Rely on early stopping to prevent overtraining
- If early stopping triggers before 20 epochs: reduce learning rate
- If training hasn't converged by 50 epochs: increase epochs or learning rate

### 3. Regularization

#### Gradient Clipping (`gradient_clip_norm`)

Prevents gradient explosion in deep networks.

**Recommendations:**
- Start with: `null` (disabled)
- If loss becomes NaN: enable with `1.0`
- For deeper networks (large config): use `1.0` or `0.5`

#### Class Weights (`use_class_weights`)

Handle class imbalance.

**Recommendations:**
- Start with: `false`
- Enable if model only predicts most common action type
- Check action_type distribution in preprocessing_stats.json
- If one action type >60% of data: consider enabling

### 4. Learning Rate Scheduling

The default configuration uses ReduceLROnPlateau to automatically reduce LR when validation loss plateaus.

**Configuration:**
```yaml
scheduler:
  enabled: true
  mode: "min"        # Minimize validation loss
  factor: 0.5        # Reduce LR by half
  patience: 5        # Wait 5 epochs before reducing
```

**Recommendations:**
- Keep enabled for most experiments
- Increase patience (7-10) if learning is noisy
- Decrease factor (0.3) for more aggressive reduction
- Disable if manually managing learning rate

### 5. Early Stopping

**Configuration:**
```yaml
patience: 10  # Stop if no improvement for 10 epochs
```

**Recommendations:**
- Start with: `10`
- Increase (15-20) if training is noisy
- Decrease (5-7) if training on limited time budget

## Interpreting Training Curves

### Healthy Training

```
Epoch   Train Loss   Val Loss   Action Acc
1       1.450        1.520      0.42
5       0.985        1.102      0.58
10      0.742        0.892      0.68
20      0.534        0.745      0.76
30      0.425        0.682      0.79
40      0.380        0.655      0.81
```

Signs:
- Losses steadily decrease
- Gap between train and val loss is small
- Accuracies improve consistently

### Overfitting

```
Epoch   Train Loss   Val Loss   Action Acc
1       1.450        1.520      0.42
10      0.742        0.912      0.68
20      0.334        0.998      0.72  ← Val loss increasing!
30      0.182        1.145      0.71  ← Accuracy decreasing!
```

**Solutions:**
- Increase dropout (0.3 → 0.4 or 0.5)
- Reduce model capacity
- Use early stopping (should trigger automatically)
- Add more regularization

### Underfitting

```
Epoch   Train Loss   Val Loss   Action Acc
1       1.450        1.520      0.42
10      1.142        1.185      0.52
20      0.982        1.032      0.59
40      0.875        0.952      0.64  ← Plateau
50      0.870        0.948      0.64  ← Not improving
```

**Solutions:**
- Increase model capacity (trunk_dims, head_dims)
- Decrease dropout (0.3 → 0.2 or 0.1)
- Increase learning rate (0.001 → 0.002)
- Train for more epochs

### Unstable Training

```
Epoch   Train Loss   Val Loss
1       1.450        1.520
2       2.340        2.687  ← Jumped up!
3       1.102        1.234
4       NaN          NaN    ← Exploded!
```

**Solutions:**
- Reduce learning rate (0.001 → 0.0005)
- Enable gradient clipping (1.0)
- Reduce batch size (256 → 128)
- Check data preprocessing for errors

## Recommended Experiment Sequence

### Phase 1: Establish Baseline (Medium Config)

```bash
uv run python -m src.imitation_learning.train \
  --config src/imitation_learning/config.yaml
```

Target metrics:
- Action type accuracy: 70-85%
- Overall action accuracy: 50-70%

### Phase 2: Architecture Search

If underfitting (accuracy < 65%):
```bash
# Try large architecture
uv run python -m src.imitation_learning.train \
  --config src/imitation_learning/configs/config_large.yaml
```

If overfitting (val loss increasing):
```bash
# Try small architecture with more dropout
# Edit config_small.yaml to set dropout: 0.4
uv run python -m src.imitation_learning.train \
  --config src/imitation_learning/configs/config_small.yaml
```

### Phase 3: Learning Rate Tuning

Create custom configs with different learning rates:

```yaml
# config_lr_high.yaml
training:
  learning_rate: 0.002

# config_lr_low.yaml
training:
  learning_rate: 0.0005
```

Run both and compare convergence speed.

### Phase 4: Advanced Tuning

- Enable class weights if needed
- Experiment with batch sizes (128, 256, 512, 1024)
- Try different dropout rates per layer (not currently supported, but easy to add)

## Comparing Experiments

Weights & Biases will track all experiments automatically. To compare:

1. Go to your wandb dashboard: https://wandb.ai/
2. Select project "splendor-ia"
3. Use "Runs" view to compare metrics across experiments
4. Key metrics to compare:
   - Final validation loss
   - Final action_type accuracy
   - Overall action accuracy
   - Training time

## Expected Performance Ranges

Based on similar imitation learning tasks:

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Action Type Acc | <60% | 60-70% | 70-80% | >80% |
| Overall Action Acc | <40% | 40-55% | 55-70% | >70% |
| Card Selection Acc | <50% | 50-65% | 65-80% | >80% |
| Gem Selection Acc | <60% | 60-75% | 75-85% | >85% |

If you're in the "Poor" range:
1. Check data preprocessing for bugs
2. Verify loss computation is correct
3. Try much larger architecture
4. Check if data quality is an issue (is MCTS playing well?)

## Creating Custom Configurations

To create a custom configuration:

1. Copy an existing config:
```bash
cp src/imitation_learning/config.yaml src/imitation_learning/configs/my_custom.yaml
```

2. Edit the values you want to change

3. Train with your config:
```bash
uv run python -m src.imitation_learning.train \
  --config src/imitation_learning/configs/my_custom.yaml
```

## Common Issues and Solutions

### Issue: Loss becomes NaN after few epochs

**Causes:**
- Learning rate too high
- Gradient explosion

**Solutions:**
- Reduce learning rate to 0.0001
- Enable gradient clipping: `gradient_clip_norm: 1.0`
- Use smaller batch size

### Issue: Model only predicts one action type

**Causes:**
- Severe class imbalance
- Model collapsed to always predicting most common class

**Solutions:**
- Enable class weights: `use_class_weights: true`
- Check action_type distribution in data
- Ensure sufficient examples of all action types

### Issue: Secondary heads (card selection, etc.) have very low accuracy

**Causes:**
- Not enough training samples for those action types
- Trunk not learning good representations
- Heads too small

**Solutions:**
- Increase head capacity: `head_dims: [256, 128]`
- Train longer (more epochs)
- Check that loss is being computed correctly with masking

### Issue: Validation loss much higher than training loss

**Cause:**
- Overfitting

**Solutions:**
- Increase dropout: `dropout: 0.4` or `0.5`
- Reduce model capacity
- Early stopping should handle this automatically
- Check if test set is significantly different from train

### Issue: Training very slow (>6 hours for 50 epochs)

**Causes:**
- Batch size too small
- Too many DataLoader workers causing overhead
- CPU bottleneck in data loading

**Solutions:**
- Increase batch size: `batch_size: 512` or `1024`
- Reduce num_workers: `num_workers: 2`
- Ensure data is on SSD not HDD
- Profile with nvidia-smi to check GPU utilization

## Grid Search (Advanced)

For systematic hyperparameter search, create a grid search script:

```python
import itertools
import subprocess

# Define parameter grid
trunk_dims = [[256, 128], [512, 256], [512, 512, 256]]
learning_rates = [0.0005, 0.001, 0.002]
dropouts = [0.2, 0.3, 0.4]

for trunk, lr, dropout in itertools.product(trunk_dims, learning_rates, dropouts):
    # Create custom config and run training
    # This is left as an exercise
    pass
```

## Further Reading

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [PyTorch Learning Rate Scheduling](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Practical Recommendations for Gradient-Based Training](https://arxiv.org/abs/1206.5533)
