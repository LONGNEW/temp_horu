# HoRU timing scope from paper Tables I and II

Source: `docs/cases26-paper574.pdf`, Tables I and II.  Field names in the
artifact follow the table component names.  Dataset loading, encoding,
evaluation, diagnostics, persistence, and simulated communication copies are
not part of either table.

## Table I — Bootstrap cost

Table I reports server and client bootstrap separately.  It does not report a
combined server-plus-client wall time.

| Scope | Table I component | Artifact field |
|---|---|---|
| Server | Common/global basis computation | `server_common_global_basis_ms` |
| Server | Projection of client hypervectors | `server_client_hv_projection_ms` |
| Server | Common/global coefficients build | `server_common_global_coefficients_ms` |
| Server | Server bootstrap total | `server_bootstrap_total_ms` |
| Client | Residual class hypervectors build | `residual_construction_ms` |
| Client | Personal basis computation | `personal_basis_svd_ms` |
| Client | Projection of residual class hypervectors | `residual_coefficient_projection_ms` |
| Client | Cached common/global/personal query coefficient build | `query_coefficient_cache_ms` |
| Client | Client bootstrap wall time | `client_bootstrap_ms` |

The client class-hypervector construction that produces `M_i` precedes the
Table I boundary.  Reconstruction/orthogonality checks and hashes are
diagnostics.  Broadcast copies are system-simulation auxiliaries.  None of
these are included in the Table I server or client totals.

```text
server_bootstrap_total
  = common/global basis computation
  + projection of client hypervectors
  + common/global coefficients build
```

```text
client_bootstrap
  = residual class hypervectors build
  + personal basis computation
  + projection of residual class hypervectors
  + cached query coefficient build
```

## Table II — Recurring cost after bootstrap

| Scope | Table II component | Artifact field |
|---|---|---|
| Client | Local similarity | `coefficient_similarity_ms` |
| Client | Local coefficients update | `coefficient_update_ms` |
| Client | Final train dataset prediction | `final_train_prediction_ms` |
| Client | Class-wise error statistics | `class_error_statistics_ms` |
| Client | Client round total | `client_round_table_ii_ms` |
| Server | Aggregation of common coefficients | `common_aggregation_ms` |
| Server | Aggregation of global coefficients | `global_aggregation_ms` |
| Client | Client-side shared branch update with class-wise update weights | `client_shared_branch_update_ms` |
| Mix | Synchronization step total | `synchronization_table_ii_ms` |

```text
client_round_table_ii
  = local similarity
  + local coefficients update
  + final train dataset prediction
  + class-wise error statistics
```

```text
synchronization_table_ii
  = common coefficient aggregation
  + global coefficient aggregation
  + mean client shared-branch update
```

Client values reported in Tables II and III are means over clients.  Maximum
and sum values remain available only as auxiliary simulation metrics.
