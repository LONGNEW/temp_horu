# Checkpoint Comparison

- device: `cuda`
- seeds: `[42]`
- datasets: `['uci_har']`
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

## uci_har

- classes: `6`, chance accuracy: `0.1667`

| method | primary metric | R1 | R25 | runtime mean (s) |
| --- | --- | ---: | ---: | ---: |
| horu_hd | mean_personalized_accuracy | 0.9712 | 0.9766 | 16.59 |
| fedhdc | global_test_accuracy | 0.8551 | 0.9525 | 4.05 |
| hyperfeel | mean_personalized_accuracy | 0.9739 | 0.9808 | 3.10 |

