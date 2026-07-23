# Checkpoint Comparison

- device: `cuda`
- seeds: `[42]`
- datasets: `['ninapro_db1']`
- methods: `['horu_hd', 'fedhdc', 'hyperfeel']`
- inference only: `False`
- inference modes: `['fused', 'shared', 'personal', 'routed']`
- checkpoint states: `results/hd_checkpoint_states`
- round checkpoints: `[1, 25]`
- local epochs: `3`
- batch size: `32`
- client participation: `1.0`
- hd dim: `2000`
- hd lr: `0.035`
- hd cosine random phase: `False`
- nn lr: `0.001`
- measure energy: `False`
- large-dataset train cap: threshold `100000`, cap `50000`

## ninapro_db1

- classes: `52`, chance accuracy: `0.0192`
- train sampling: `135000 -> 50000` across `27` clients

| method | primary metric | R1 | R25 | runtime mean (s) |
| --- | --- | ---: | ---: | ---: |
| horu_hd | mean_personalized_accuracy | 0.7536 | 0.7568 | 76.74 |
| fedhdc | global_test_accuracy | 0.1016 | 0.3086 | 45.82 |
| hyperfeel | mean_personalized_accuracy | 0.6109 | 0.6148 | 24.54 |

