# FEMNIST

This loader consumes LEAF writer JSON directly.  It intersects train and test
writer IDs, orders them lexically, and takes the first 200.  No supplied LEAF
split is re-partitioned.  Images are flattened to 784 values, divided by 255,
then L2 normalized per sample.
