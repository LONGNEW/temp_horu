# HoRU bootstrap reference (T004)

This implementation covers the one-time, pre-round initialization only. It does
not retrain coefficients, aggregate a shared state over rounds, or perform the
personalization update.

| Paper relation | T004 implementation |
|---|---|
| Eq. (5)–(6) | A shared Gaussian projection is created once and each query is encoded as `h = cos(xE)`. |
| Eq. (7)–(8) | Each client builds `M_i` from class-wise train means and L2-normalizes nonempty rows; `n_i,k` is retained. |
| Eq. (9)–(10) | Server forms `Σ_i M_iᵀM_i`, uses descending `eigh` eigenvectors as `[B_c, B_g]`, and canonicalizes signs. |
| Eq. (11)–(12) | `C_global` and `G_global` are class-count weighted from `M_i B_c` and `M_i B_g`; each client gets independent `C_i=C_global`, `G_i=G_global`, and zero `Δ_i`. |
| Eq. (13)–(14) | `R_i=M_i-(C_i,total B_cᵀ+G_global B_gᵀ)`; `B_p,i` is from its right singular vectors and `P_i=R_iB_p,i`. |
| Eq. (15) | The initial reconstruction is row-normalized `C_global B_cᵀ + G_global B_gᵀ + P_i B_p,iᵀ`. |
| Eq. (16) | Train/test queries are projected once to `z_c`, `z_g`, and `z_p`; T005 can consume these caches without projecting the full hypervector again. |

`full_svd` is a literal numerical completion for `r_p > K`: it uses the first
`r_p` vectors of the full right-singular matrix. Zero-singular directions are
not unique mathematically, so manifests label this behavior
`USER_SPECIFIED_NUMERICAL_COMPLETION`. The ordinary smoke configuration uses
`reduced_svd` with `r_p <= min(K,D)`.

Timing values are single-process compute/copy timings, not network latency.
Payload accounting communicates only the uploaded `M_i` and broadcast
`[B_c, B_g, C_global, G_global]`; local bases, coefficients, deltas, and query caches are
not communicated.
