# WISDM

The official transformed set omits subject 1614 (a documented upstream archive
omission), while the official raw phone accelerometer file for 1614 is present.
At the user's request, the loader restores the original 1600–1650, 51-client
set by recovering only the selected 43 basic features from that raw file using
the archive's `arffmagic` feature definition.  The recovery has its own raw and
algorithm provenance and must not be described as a byte-identical official
ARFF file.  Existing client 1600 raw recovery matches its supplied transformed
43-feature rows within float rounding (`max_abs < 5e-5`). The loader removes
non-finite rows (at most 1%), applies deterministic splits/caps, then computes
train-only pooled standardization.
