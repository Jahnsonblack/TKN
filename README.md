# TKN
This repository contains the supplementary materials for TKN: Transformer-based Keypoint Prediction Network for Real-time Video Prediction
## Related Works
Unsupervised methods can reduce the cost of manual annotation which is a common requirement for video datasets.
### Unsupervised keypoint learning
Due to the similarity of pixels in consecutive video frames, the keypoints in each frame can be learned via unsupervised reconstruction of the other frames. Jakab et al.[<sup>1</sup>](#jakab2018conditional) propose to learn the object landmarks via conditional image generation and representation space shaping. Minderer et al.[<sup>2</sup>](#minderer2019unsupervised) introduce keypoints to video prediction using stochastic dynamics learning for the first time, which drastically reduces computational complexity. Gao et al.[<sup>3</sup>](#gao2021accurate) applied grids on top of [<sup>2</sup>](#minderer2019unsupervised) for a clearer expression of the keypoint distribution.



## 参考

<div id="jakab2018conditional"></div>
- [1] [Conditional image generation for learning the structure of visual objects]
<div id="minderer2019unsupervised"></div>
- [2] [Unsupervised learning of object structure and dynamics from videos]
<div id="gao2021accurate"></div>
- [2] [Accurate Grid Keypoint Learning for Efficient Video Prediction]
