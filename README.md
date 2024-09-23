# TKN
This repository contains the supplementary materials for TKN: Transformer-based Keypoint Prediction Network for Real-time Video Prediction
## Related Works
Unsupervised methods can reduce the cost of manual annotation which is a common requirement for video datasets.
### Unsupervised keypoint learning
Due to the similarity of pixels in consecutive video frames, the keypoints in each frame can be learned via unsupervised reconstruction of the other frames. Jakab et al.[<sup>1</sup>](#jakab2018conditional) propose to learn the object landmarks via conditional image generation and representation space shaping. Minderer et al.[<sup>2</sup>](#minderer2019unsupervised) introduce keypoints to video prediction using stochastic dynamics learning for the first time, which drastically reduces computational complexity. Gao et al.[<sup>3</sup>](#gao2021accurate) applied grids on top of [<sup>2</sup>](#minderer2019unsupervised) for a clearer expression of the keypoint distribution.
### Unsupervised video prediction
uses the pixel values of the video frames as the labels for unsupervised prediction. Existing studies can be classified into two categories, as shown by Fig.~ref{fig:otherstrutures}. 
The first category of works focuses on improving the performance of the well-known RNN by adapting the intermediate recurrent structure [<sup>4</sup>](#wang2018eidetic) [<sup>5</sup>](#wang2021predrnn) [<sup>6</sup>](#oliu2018folded)[<sup>7</sup>](#wang2018predrnn++) [<sup>8</sup>](#castrejon2019improved) [<sup>9</sup>](#chen2017learning) [<sup>10</sup>](#yang2019structured)
. For example, E3D-LSTM[<sup>4</sup>](#wang2018eidetic) integrates 3DCNN with LSTM to extract short-term dependent representations and motion features. PredRNN[<sup>5</sup>](#wang2021predrnn) enables the cross-level communication for the learned visual dynamics by propagating the memory flow in both bottom-up and top-down orientations. 
The second category focuses on disentangling the dynamic objects and the static background in the video frames, mostly by adapting the Convolutional Neural Network(CNN) structure [<sup>11</sup>](#ying2018better) [<sup>12</sup>](#guen2020disentangling ) [<sup>13</sup>](#denton2017unsupervised) [<sup>14</sup>](#blattmann2021understanding) [<sup>15</sup>](#xu2019unsupervised) [<sup>16</sup>](#xu2022active)
For instance, DGGAN[<sup>10</sup>](#ying2018better) trains a multi-stage generative network for prediction guided by synthetic inter-frame difference. PhyDNet[<sup>12</sup>](#guen2020disentangling)uses a latent space to untangle physical dynamics from residual information. 
The methods in both categories use so-called ``sequential prediction'', that is, using the previous prediction frame as the input frame for the next round of prediction. The prediction speed is proportional to the number of frames to be predicted and thus leads to an intolerably long delay for long-term prediction. 
Therefore, we propose a parallel prediction scheme, as shown in Fig.~ref{fig:mystructure}, to extract the features of multiple frames and output multiple predicted frames in parallel, which greatly accelerates the prediction process.
### Transformer
Transformer has been utilized extensively in NLP due to its benefits over RNN in feature extraction and long-range feature capture. It monitors global attention to prevent the loss of prior knowledge which often occurs with RNN. Its parallel processing capacity can significantly accelerate the process. Recently, the field of computer vision has begun to explore its potential and produced positive results~\cite{dosovitskiy2020image,liu2021swin,liu2021video,arnab2021vivit,liang2021swinir,xu2023lgvit,huang2023semi,wang2023deep,khan2023artrivit}. Most related works input segmented patches of images to the transformer to calculate inter-patch attention and obtain the features. There are also a number of vision transformer (VIT) approaches applied to video analysis[<sup>18</sup>](#arnab2021vivit) [<sup>19</sup>](#liu2022video) [<sup>20</sup>](#man2022scenario)[<sup>21</sup>](#feng2023efficient) [<sup>22</sup>](#xiang2023emhiformer) [<sup>23</sup>](#li2021video). For example, VIVIT[<sup>18</sup>](#arnab2021vivit) proposes four different video transformer structures to solve video classification problems using the spatio-temporal attention mechanism. [<sup>19</sup>](#liu2022video) applies the swin transformer structure to video and uses an inductive bias of locality. In this paper we select CNN as the feature extractor instead of the VIT structure because of the huge computational cost of VIT compared to CNN. We select the transformer structure as the predictor because it outperformed RNN, mix-mlp, and other structures, in terms of predicting spatio-temporal features in our empirical experiments.

Most of the aforementioned video prediction methods extract from each frame complex features, typically of tens of thousands of bytes[<sup>24</sup>](#shi2015convolutional) [<sup>4</sup>](#wang2018eidetic)[<sup>12</sup>](#guen2020disentangling) [<sup>25</sup>](#akan2021slamp), resulting in excessive numbers of floating point operations in both the feature extraction module and the prediction module. Moreover, they employ sequential (frame-by-frame) prediction process. Hence, both training and testing consume a great deal of time and memory. In the meanwhile, many videos, particularly human activity records, have a significant amount of background redundancy[<sup>26</sup>](#schuldt2004recognizing) [<sup>27</sup>](#h36m_pami) that can be removed by extracting information only from the key motions. Therefore, in this work, we try to couple the unique advantages of the transformer and the keypoint-based prediction methods to maximize their benefits.
## Models

### Coordinate Generation (CG)
module converts the heatmap generated by the encoder's last layer to the keypoints.  We use a similar CG structure as in [<sup>1</sup>](#jakab2018conditional) which first uses a fully connected layer to convert the encoder heatmap $h_n$ from ${\Bbb R}^{H_n \times W_n\times C_n}$ into $ {\Bbb R}^{H_n \times W_n\times K}$, where $K$ refers to the number of keypoints. We do this in the hope of compressing $H_n \times W_n$ into the form of point coordinates in the dimension of $K$. The converted heatmap $h^{'}_n$ can be rewritten as $h^'(x;y;i)$, where $x=1,2,...,W_n,y=1,2...,H_n, i=1,2,...,K$, represent the three dimensions of of $ h^'_n$, respectively. Then we can calculate the coordinates of the $k$-th keypiont in width $p_ix$ as follows:

$$
h_n^{'}(x;i)= \frac{\sum_{y} h_n^{'}(x;y;i)}{\sum_{x,y} h_n^{'}(x;y;i)}  , \label{4}
$$


## 参考

<div id="jakab2018conditional"></div>
- [1] [Conditional image generation for learning the structure of visual objects]
<div id="minderer2019unsupervised"></div>
- [2] [Unsupervised learning of object structure and dynamics from videos]
<div id="gao2021accurate"></div>
- [3] [Accurate Grid Keypoint Learning for Efficient Video Prediction]
<div id="wang2018eidetic"></div>
- [4] [Eidetic 3d lstm: A model for video prediction and beyond]
<div id="wang2021predrnn"></div>
- [5] [PredRNN: A recurrent neural network for spatiotemporal predictive learning]
<div id="oliu2018folded"></div>
- [6] [Folded recurrent neural networks for future video prediction]
<div id="wang2018predrnn++"></div>
- [7] [Predrnn++: Towards a resolution of the deep-in-time dilemma in spatiotemporal predictive learning]
<div id="castrejon2019improved"></div>
- [8] [Improved conditional vrnns for video prediction]
<div id="chen2017learning"></div>
- [9] [Learning object-centric transformation for video prediction]
<div id="yang2019structured"></div>
- [10] [Structured stochastic recurrent network for linguistic video prediction]
<div id="ying2018better"></div>
- [11] [Better guider predicts future better: Difference guided generative adversarial networks]
<div id="guen2020disentangling"></div>
- [12] [Disentangling physical dynamics from unknown factors for unsupervised video prediction]
<div id="denton2017unsupervised"></div>
- [13] [Unsupervised learning of disentangled representations from video]
<div id="blattmann2021understanding"></div>
- [14] [Understanding object dynamics for interactive image-to-video synthesis]
<div id="xu2019unsupervised"></div>
- [15] [Unsupervised discovery of parts, structure, and dynamics]
<div id="xu2022active"></div>
- [16] [Active Patterns Perceived for Stochastic Video Prediction]
<div id="h36m_pami"></div>
- [17] [Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian]
<div id="arnab2021vivit"></div>
- [18] [Vivit: A video vision transformer]
<div id="liu2022video"></div>
- [19] [Video swin transformer]
<div id="man2022scenario"></div>
- [20] [Scenario-aware recurrent transformer for goal-directed video captioning]
<div id="feng2023efficient"></div>
- [21] [Efficient video transformers via spatial-temporal token merging for action recognition]
<div id="xiang2023emhiformer"></div>
- [22] [EMHIFormer: An Enhanced Multi-Hypothesis Interaction Transformer for 3D human pose estimation in video]
<div id="li2021video"></div>
- [23] [Video semantic segmentation via sparse temporal transformer]
<div id="shi2015convolutional"></div>
- [24] [Convolutional LSTM network: A machine learning approach for precipitation nowcasting]
<div id="akan2021slamp"></div>
- [25] [SLAMP: Stochastic Latent Appearance and Motion Prediction]
<div id="schuldt2004recognizing"></div>
- [26] [Recognizing human actions: a local SVM approach]
