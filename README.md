# Pose2vec
This repository contains the following:
  - Utilities for various human skeleton preprocessing steps in numpy and tensorflow.
  - Tensorflow model to learn a continuous pose embedding space.

This code has been used to train the Pose _EnGAN_ Model in the paper: "**Maharshi Gor***, Jogendra Nath Kundu*, R Venkatesh Babu, [Unsupervised Feature Learning of Human Actions as Trajectories in Pose Embedding Manifold](https://arxiv.org/abs/1812.02592), _IEEE Winter Conference on Applications of Computer Vision (WACV)_".

#### Qualitative Results:
- Grid Interpolations <br>
<img src="./docs/interpolation.png" width="400">
<img src="./docs/interpolation_2.png" width="400">
<img src="./docs/interpolation_3.png" width="400">

- Reconstructions (left: Ground Truth, right: Reconstruction)
<img src="./docs/recon.png" width="600">
