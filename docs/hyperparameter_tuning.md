# Hyperparameter Tuning Notes

This page summarizes the hyperparameter ranges used during tuning and the practical behavior observed during experiments.

## PPO Hyperparameters

| Parameter | Default | Tuning range | What it controls | What can go wrong |
| --------- | ------- | ------------ | ---------------- | ----------------- |
| `learning_rate` | `3e-4` | Min: `1e-5`, Max: `1e-3` (log scale) | Step size for Adam updates | Too high: policy can diverge or oscillate. Too low: learning can stall. |
| `n_steps` | `2048` | `512, 1024, 2048, 4096` | Number of rollout steps collected before each update | Too large: slower wall-clock updates and higher memory use. Too small: noisier gradients and less stable training. |
| `batch_size` | `64` | `32, 64, 128, 256` | Mini-batch size used when replaying rollout data | Too large: lower sample diversity per update. Too small: very noisy gradients. Must satisfy `batch_size <= n_steps`. |
| `n_epochs` | `10` | `3, 5, 10, 20` | Number of passes over rollout data per update | Too many epochs can overfit stale data, break the KL trust region, and cause destructive updates. |
| `gamma` | `0.999` | Min: `0.95`, Max: `0.9999` (log scale) | Discount factor for long-term return | Too low: agent becomes short-sighted. Too high (near `1.0`): value targets can become unstable on long episodes. |
| `gae_lambda` | `0.98` | Min: `0.9`, Max: `1.0` | Bias-variance balance in GAE advantages | Too high: high-variance advantages and noisy updates. Too low: biased advantages and slower improvement. |
| `ent_coef` | `0.0` | Min: `0.0`, Max: `0.01` | Entropy regularization for exploration | Too high: policy stays too random. Too low: policy can collapse into premature determinism. |
| `clip_range` | `0.2` | Min: `0.1`, Max: `0.4` | Maximum policy update size per PPO step | Too large: unstable policy jumps. Too small: useful updates are clipped away and training slows. |
| `vf_coef` | `0.5` | Min: `0.25`, Max: `1.0` | Relative weight of the value loss | Too high: value head dominates and weakens policy learning. Too low: poorer value estimates and weaker advantages. |
| `max_grad_norm` | `0.5` | Min: `0.3`, Max: `1.0` | Gradient clipping threshold | Too low: suppresses useful updates. Too high: increases risk of exploding gradients, especially with larger `n_epochs`. |

## ICM Hyperparameters

| Parameter | Default | Tuning range | What it controls | What can go wrong |
| --------- | ------- | ------------ | ---------------- | ----------------- |
| `feature_dim` | `64` | `32, 64, 128, 256` | Size of the latent feature representation used by ICM | Too small: weak representation and noisy curiosity signal. Too large: slower updates and potential overfitting. |
| `lr` | `1e-3` | Min: `1e-4`, Max: `1e-2` (log scale) | Learning rate for encoder, inverse model, and forward model | Too high: unstable curiosity loss and exploding intrinsic reward. Too low: curiosity model learns too slowly. |
| `eta` | `0.01` | Min: `1e-3`, Max: `0.5` (log scale) | Scale factor for intrinsic reward | Too high: intrinsic reward dominates task reward. Too low: curiosity has little practical effect. |
| `beta` | `0.5` | Min: `0.1`, Max: `0.9` | Trade-off between inverse and forward model losses | Too high: overemphasizes forward loss. Too low: weak exploration pressure. |
| `update_freq` | `128` | `64, 128, 256, 512` | How often ICM optimization runs (in env steps) | Too frequent: expensive and noisy updates. Too sparse: stale curiosity model. |
| `buffer_capacity` | `1000` | `500, 1000, 5000` | Replay buffer size for ICM training transitions | Too small: poor diversity and overfitting risk. Too large: old/off-policy transitions can dominate. |
| `batch_size` | `64` | `32, 64, 128` | Mini-batch size for ICM updates | Too small: high-variance gradients. Too large: slower adaptation. |

## Practical Notes From Runs

- PPO trials are pruned when `batch_size > n_steps`.
- `n_epochs = 20` was too aggressive in this setup.
- `n_steps` values below `2048` underperformed; `1024` was consistently poor in these experiments.
- Learning rate was one of the most sensitive: some runs jumped from strong scores (for example, `+90`) to bad policies (for example,`-112`) within the same training process.