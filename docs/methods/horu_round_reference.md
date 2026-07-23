# HoRU recurring round reference (T005)

| Relation | Implementation |
|---|---|
| Eq. (11)–(13) | Cached `z_c`, `z_g`, `z_p` form each query; the systems comparison uses the agreed direct coefficient dot product `q @ u_k`, without reconstruction or a Gram metric. |
| Eq. (14)–(17) | Predictions are fixed at batch start. Misclassified samples accumulate additive true/predicted-row deltas for `C`, `G`, `Δ`, and `P`; no post-update normalization is added. |
| Eq. (25)–(27) | Final train prediction produces local class count, error count, and `e=w/max(t,1)`. |
| Eq. (28)–(30) | Only `C,G` are class-count weighted on the server. Zero-denominator rows are zero. |
| Eq. (31)–(32) | RowGate computes `rollback=clamp(1-gate_alpha*e, gate_min, gate_max)` and `follow=1-rollback`; clients absorb the broadcast shared state using `eta_global * follow`. `Δ`, `P`, and `B_p` remain local and hash-checked. |

The recurring hot path stays entirely in coefficient space and contains only
the stated additive coefficient updates.
