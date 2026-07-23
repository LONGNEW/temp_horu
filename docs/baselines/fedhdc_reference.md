# FedHDC reference and artifact decisions

The reference is Ergun, Chandrasekaran, and Rosing, *Federated
Hyperdimensional Computing* (arXiv:2312.15966, 2023). Its local model is a
class sample-hypervector bundle and the server aggregates local models.

This artifact implements `H=cos(XE)`, local class sums, full-model
sample-count-weighted server aggregation, and nonzero-row L2 normalization.
It does **not** claim complete paper reproduction.

## USER_SPECIFIED artifact differences

- Bootstrap is one-time before round 1: each client uploads its normalized
  bundled model, then the server aggregates and broadcasts a fresh global clone.
- Prediction uses dot product; zero prototype rows score negative infinity.
- Local training uses batch size 16, batch-start (stale) predictions, unit
  update targets, summed—not averaged—batch deltas, and normalization only of
  rows changed by a batch.
- The server uses client train-sample weights (not class-wise counts).
- Bootstrap copy/compute time and payload are reported separately from recurring
  round timing. The parallel estimate is a simulation estimate, not network latency.
- **Official performance metric:** after each aggregation, the global model is
  evaluated once over the concatenation of every participating client's test
  samples. This pooled global-test accuracy is the reported performance;
  per-client global-model accuracies are diagnostics only.
