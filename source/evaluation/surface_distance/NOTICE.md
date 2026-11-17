# Surface-distance attribution

The Python sources in this directory are adapted from DeepMind's
[`surface-distance`](https://github.com/deepmind/surface-distance) library,
which is Copyright 2018 Google Inc. and licensed under Apache License 2.0.
The complete license text is included in [`LICENSE`](LICENSE).

BraTS parity was established against the vendored copy in the official
[`BraTS-2023-Metrics`](https://github.com/rachitsaluja/BraTS-2023-Metrics)
repository at commit
`43c905242b2eecf421d4ab2da7af8ece9777d322` (12 October 2023). That code calls
these primitives for its robust HD95 and preserves the marching-cubes surfel
area weighting. The adaptation keeps the numerical algorithm and lookup table,
while replacing APIs deprecated by current NumPy/SciPy releases and using the
framework's package layout.

