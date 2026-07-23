# Table III timing scope

Source: `docs/cases26-paper574.pdf`, Table III.

Every method emits the same per-round fields:

```text
local_round_ms
similarity_ms
update_ms
server_step_ms
uploaded_payload_bytes
```

Client timing fields are arithmetic means over participating clients.
`uploaded_payload_bytes` is the upload from one client, not the sum across
clients.  Encoding, shuffle/index construction, accuracy evaluation, hashes,
checkpoint I/O, and network latency are outside Table III.

## FedHDC

- `similarity_ms`: batch prototype scoring and argmax prediction.
- `update_ms`: misclassification mask, unit update-vector construction, true/predicted class
  push-pull, and changed-row normalization.
- `local_round_ms`: mean client `similarity_ms + update_ms`.
- `server_step_ms`: sample-count-weighted full-prototype aggregation and row
  normalization.
- `uploaded_payload_bytes`: `K * D * element_size`.

## HyperFeel

- `similarity_ms`: batch associative-memory scoring and argmax prediction.
- `update_ms`: prior-global-delta personalization, misclassification mask,
  class-count/error accounting, true/predicted push-pull, client-delta
  accumulation, and personalization-weight construction.
- `local_round_ms`: mean client `similarity_ms + update_ms`.
- `server_step_ms`: element-wise client-delta aggregation.
- `uploaded_payload_bytes`: `K * D * element_size`.

## HoRU

HoRU retains the Table II breakdown and additionally folds it into Table III:

- `similarity_ms`: Table II direct coefficient dot scoring and argmax prediction.
- `update_ms`: Table II misclassification mask and additive coefficient
  push-pull; no row normalization.
- `local_round_ms`: mean Table II client round total, including final train
  prediction and class-wise error statistics.
- `server_step_ms`: Table II synchronization total: common aggregation,
  global aggregation, and mean client shared-branch update.
- `uploaded_payload_bytes`: `K * (r_c + r_g) * element_size`.

Raw sums and maxima may be emitted with an `_aux_ms` suffix but are not used in
the Table III comparison.
