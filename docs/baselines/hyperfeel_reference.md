# HyperFeel reference mapping

Reference: H. Li, F. Liu, Y. Chen, and L. Jiang, “HyperFeel: An Efficient
Federated Learning Framework Using Hyperdimensional Computing,” ASP-DAC 2024,
pp. 716–721, DOI 10.1109/ASP-DAC58780.2024.10473907. The supplied PDF digest is
`c298831aec5d1c929e67e6670ebcd5823be125ad6440b805a1cc9a7c2d4aa734`.

| Paper item | Artifact correspondence |
|---|---|
| Fig. 1 | One-time client AM upload, server central-AM sum, central-AM broadcast; then delta-only recurring communication. |
| Eq. (1) | `Q = cos(XE)` is provided by the shared T001 `NonlinearEncoder`; prediction uses dot product as specified by T003. |
| Eq. (2) | `retrain_samplewise` applies raw `lr × Q` to true/predicted AM rows and to the same rows of the client delta. |
| Eq. (3) | `personalization_weights` computes `error[k] / count[k]`, with a zero weight for missing classes; `apply_personalization` uses the preceding global delta. |
| Algorithm 1 | `sum_deltas` performs server sum; personalized AMs persist and are never uploaded or overwritten. |

Implementation scope is horizontal UCI-HAR only. It excludes the paper's vertical
FL path and does not claim paper-level accuracy reproduction. The smoke uses
T003's `D=256`, three clients, and two rounds rather than the paper's reported
`D=1000`, 30-client experiment.
