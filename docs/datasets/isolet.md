# ISOLET

The two official ISOLET data files are combined and sample-wise L2 normalized.
The cache uses eight class-wise Dirichlet clients (`alpha=0.05`, seed 0) with
minimum 50 samples and two classes, followed by deterministic client-internal
70/30 class-stratified splits.  The partition retry count and label histograms
are immutable manifest fields.
