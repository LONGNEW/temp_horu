# Checkpoint Comparison

- device: `cuda`
- seeds: `[42]`
- datasets: `['isolet_raw']`
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

## isolet_raw

- classes: `26`, chance accuracy: `0.0385`

| method | primary metric | R1 | R25 | runtime mean (s) |
| --- | --- | ---: | ---: | ---: |
| horu_hd | mean_personalized_accuracy | 0.8792 | 0.8814 | 7.04 |
| fedhdc | global_test_accuracy | 0.9073 | 0.9385 | 2.27 |
| hyperfeel | mean_personalized_accuracy | 0.9057 | 0.9059 | 2.15 |

