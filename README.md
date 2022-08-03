# Evaluation Metrics for Visual Tracking

## Average Overlap (AO)

Average Overlap (AO) is the average overlap of the sequence of IoU (Intersection over Union) of the pixels between tracking result and groundtruth bounding box.

$$ AO = 100 \times \frac {1} {N} \sum_{i=1}^N \mathrm{IoU} \left( S_{1,i}, S_{2,i} \right) $$

$$ \mathrm{IoU} \left( S_{1,i}, S_{2,i} \right) = \frac {| S_{1,i} \cap S_{2,i} |} {| S_{1,i} \cup S_{2,i} |} $$

where $N$ is the number of frames, $S_{1,i}$ represents the area of the predicted object of frame $i$, $S_{2,i}$ represents the area of the ground truth of frame $i$, and $\mathrm{IoU} \left( S_{1,i}, S_{2,i} \right)$ is the IoU value of $S_{1,i}$ and $S_{2,i}$.

* K-Means vs. Fuzzy C-Means for segmentation of orchid flowers, III. RESULT AND EVALUATION
[[paper](https://www.researchgate.net/publication/311409493_K-Means_vs_Fuzzy_C-Means_for_Segmentation_of_Orchid_Flowers)]
* https://github.com/got-10k/toolkit/blob/v0.1.3/got10k/experiments/got10k.py#L265

## Area Under Curve (AUC)

Area Under Curve (AUC) is the average of the success rates corresponding to the sampled overlap thresholds. The AO is recently proved to be equivalent to the AUC. AUC value is usually used to ranking the trackers in success plot $(S)$. The success rate $SR$ is measured as the $\mathrm{IoU}$, by testing whether $\mathrm{IoU}$ is larger than a certain threshold $t$ (e.g., $t=0.5$).

$$ SR_{0.5} = \frac {1} {N} \sum_{i=1}^N 1\left(\mathrm{IoU} \left( S_{1,i}, S_{2,i} \right) > 0.5 \right) $$ 

$$ AUC = \int_{0}^1 SR_t dt = \int_{0}^1 \frac {1} {N} \sum_{i=1}^N 1\left(\mathrm{IoU} \left( S_{1,i}, S_{2,i} \right) > t \right)dt$$

* GOT-10k: A large high-diversity benchmark for generic object tracking in the wild, 4.2 Evaluation methodology section
[[paper](https://arxiv.org/pdf/1810.11981.pdf)]
* [https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/experiments/lasot.py#L144](https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/experiments/lasot.py#L144)

## Precision (P)
P stands for precision score. Usually measured as the distance in pixels between the centers $C^{gt}$ and $C^{tr}$ of the ground truth and the tracker bounding box, respectively. Then, The trackers are ranked using this metric with a conventional threshold of 20 pixels.

$$ P = \frac {1} {N} \sum_{i=1}^N 1\left(\left\Vert C^{tr}_i - C^{gt}_i \right\Vert _2 \le 20 \right) $$

$$ \left\Vert C_{i}^{tr} - C_{i}^{gt} \right\Vert _2 = \left\Vert \begin{pmatrix} \ C^{tr}_{x,i} \\ 
C_{y,i}^{tr} 
\end{pmatrix} - 
\begin{pmatrix} 
C_{x,i}^{gt} \\ 
C_{y,i}^{gt} 
\end{pmatrix} \right\Vert _2 = \left\Vert \begin{pmatrix} C_{x,i}^{tr} - C_{x,i}^{gt} \\ 
C_{y,i}^{tr} - C_{y,i}^{gt} 
\end{pmatrix} \right\Vert _2 $$

where $\Vert \cdot \Vert _2$ is Euclidean distance
* TrackingNet: A large-scale dataset and benchmark for object tracking in the wild,3.4 Evaluation section [[paper](https://arxiv.org/pdf/1803.10794.pdf)]
* [https://github.com/got-10k/toolkit/blob/v0.1.3/got10k/utils/metrics.py#L7](https://github.com/got-10k/toolkit/blob/v0.1.3/got10k/utils/metrics.py#L7)

## Normalize Precision (Pnorm)
$P_{norm}$ normalize the precision over the ground truth bounding box. The trackers are then ranked using the average for normalized precision between 0 and 0.5.

$$ P_{norm} = \frac {1} {|T|} \sum_{t\in T} \frac {1} {N} \sum_{i=1}^N 1\left(\left\Vert W_i^{-1} \left( C^{tr}_i - C^{gt}_i \right\Vert _2 \right) \le t \right) $$

where,

$$ T = linspace(0.0, 0.5, 51)$$

$$ T = \lbrace x \in \mathbb{R} | x = 0.1 \times n \land n \in \lbrace a \in \mathbb{N} | 0 \le a \le 50 \rbrace \rbrace$$

$$ W_i = diag \left( BB_{x,i}^{gt}, BB_{y,i}^{gt}\right) = \begin{pmatrix} BB_{x,i}^{gt} & 0 \\ 
0 & BB_{y,i}^{gt} 
\end{pmatrix}$$

then, 

$$\Vert W_i^{-1} \left( C_i^{tr} - C_i^{gt} \right) \Vert_2 = \left\Vert \begin{pmatrix} \frac {1} {BB_{x,i}^{gt}} & 0 \\ 
0 & \frac {1} {BB_{y,i}^{gt}} 
\end{pmatrix} \begin{pmatrix} C_{x,i}^{tr} - C_{x,i}^{gt} \\ 
C_{y,i}^{tr} - C_{y,i}^{gt} 
\end{pmatrix} \right\Vert _2$$

$W$ is a diagonal covariance matrix whose elements are the standard deviations of the target state parameters. $BB^{gt}$ is ground truth bounding boxes and $BB^{tr}$ is the ones generated by tracker. $N$ is the number of frames.

* TrackingNet: A large-scale dataset and benchmark for object tracking in the wild,3.4 Evaluation section [[paper](https://arxiv.org/pdf/1803.10794.pdf)]
* [https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/utils/metrics.py#L22](https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/utils/metrics.py#L22)

# Execute
```
# This is an example for GOT-10k, TrackingNet, and LaSOT dataset
pip install numpy pandas opencv-python tqdm
python metric.py
```

# Result
```
--- Evaluate for GOT-10k ---
Average Overlap (AO): 28.40 %
Success 0.5 (SR0.5): 22.12 %
Success 0.75 (SR0.75): 16.81 %
--- Evaluate for TrackingNet & LaSOT ---
Success score (AUC): 28.78 %
Precision score (P): 23.89 %
NPrecision score (P_norm): 21.90 %
```

## Reference

* [1] GOT-10k: A large high-diversity benchmark for generic object tracking in the wild
[[paper](https://arxiv.org/pdf/1810.11981.pdf)]
* [2] TrackingNet: A large-scale dataset and benchmark for object tracking in the wild
[[paper](https://arxiv.org/pdf/1803.10794.pdf)]
* [3] LaSOT: A high-quality benchmark for large-scale single object tracking
[[paper](https://arxiv.org/pdf/1809.07845v2.pdf)]
* [4] Object tracking benchmark
[[paper](https://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf)]
* [5] Visual object tracking performance measures revisited
[[paper](https://arxiv.org/pdf/1502.05803.pdf)]
* [6] K-Means vs. Fuzzy C-Means for segmentation of orchid flowers
[[paper](https://www.researchgate.net/publication/311409493_K-Means_vs_Fuzzy_C-Means_for_Segmentation_of_Orchid_Flowers)]
* [7] Robust visual tracking with reliable object information and Kalman filter
[[paper](https://www.researchgate.net/publication/348859011_Robust_Visual_Tracking_with_Reliable_Object_Information_and_Kalman_Filter)]
* [8] Online Object Tracking: A Benchmark
[[paper](https://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf)]
* [9] Intersection over Union (IoU) for object detection
[[link](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)]
* [10] Understanding AUC-ROC Curve
[[link](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)]
