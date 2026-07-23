# Checkpoint Comparison

- device: `cuda`
- seeds: `[42]`
- datasets: `['wisdm']`
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

## wisdm

- classes: `18`, chance accuracy: `0.0556`
- train sampling: `255000 -> 50000` across `51` clients

| method | primary metric | R1 | R25 | runtime mean (s) |
| --- | --- | ---: | ---: | ---: |
| horu_hd | mean_personalized_accuracy | 0.4805 | 0.5669 | 106.35 |
| fedhdc | global_test_accuracy | 0.0949 | 0.0695 | 52.30 |
| hyperfeel | mean_personalized_accuracy | 0.4526 | 0.5111 | 84.59 |

