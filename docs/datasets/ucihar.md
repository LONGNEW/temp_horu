# UCI-HAR

Official UCI archive; 30 subjects are ordered clients.  The original train and
test files are combined, then each subject receives a deterministic,
class-stratified 70/30 split (singletons stay in train).  Features use
sample-wise L2 normalization.  See `data/ucihar/manifest.json` for raw and
split hashes.
