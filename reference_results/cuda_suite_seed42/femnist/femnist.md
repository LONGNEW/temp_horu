# Checkpoint Comparison

- device: `cuda`
- seeds: `[42]`
- datasets: `['femnist']`
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

## femnist

- classes: `62`, chance accuracy: `0.0161`

| method | primary metric | R1 | R25 | runtime mean (s) |
| --- | --- | ---: | ---: | ---: |
| horu_hd | mean_personalized_accuracy | 0.6828 | 0.6854 | 94.54 |
| fedhdc | global_test_accuracy | 0.2623 | 0.5726 | 52.73 |
| hyperfeel | mean_personalized_accuracy | 0.6428 | 0.6778 | 52.78 |

