# NinaPro DB1

Subjects 1–27 are clients.  The loader combines 10 EMG and 22 glove channels,
uses `restimulus` labels with rest removed, makes non-overlapping 20-sample
(200 ms at 100 Hz) windows without crossing gesture or repetition boundaries,
and holds out repetitions 2, 5, and 7.  Train-only pooled standardization and
per-client total class-stratified caps of 5,000 train/1,000 test samples are
applied.
