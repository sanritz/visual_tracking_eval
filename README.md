# Evaluation Metrics for Visual Tracking

## Average Overlap (AO)

Average Overlap (AO) is the average of the sequence of IoU (Intersection over Union).

$$ AO = 100 \times \frac {1} {N} \sum_{i=1}^N \mathrm{IoU} \left( S_{1,i}, S_{2,i} \right) $$

$$ \mathrm{IoU} \left( S_{1,i}, S_{2,i} \right) = \frac {| S_{1,i} \cap S_{2,i} |} {| S_{1,i} \cup S_{2,i} |} $$

where $N$ is the number of frames, $S_{1,i}$ represents the area of the predicted object of frame $i$, $S_{2,i}$ represents the area of the ground truth of frame $i$, and $\mathrm{IoU} \left( S_{1,i}, S_{2,i} \right)$ is an IoU value of $S_{1,i}$ and $S_{2,i}$.

* [6] K-Means vs. Fuzzy C-Means for segmentation of orchid flowers, III. RESULT AND EVALUATION
[[paper](https://www.researchgate.net/publication/311409493_K-Means_vs_Fuzzy_C-Means_for_Segmentation_of_Orchid_Flowers)]
* https://github.com/got-10k/toolkit/blob/v0.1.3/got10k/experiments/got10k.py#L265

## Area Under Curve (AUC)

Area Under Curve (AUC) is the average of the success rates corresponding to the sampled overlap thresholds.The AO is recently proved to be equivalent to the AUC. AUC value is usually used to ranking the trackers in success plot $(S)$. \
*[1] - 4.2 Evaluation methodology section\
[https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/experiments/lasot.py#L144](https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/experiments/lasot.py#L144)

## Precision (P)
P is stand for precision score. Usually measured as the distance in pixels between the centers $C^{gt}$ and $C^{tr}$ of the ground truth and the tracker bounding box, respectively.

$$ P =\left\Vert C^{tr} - C^{gt} \right\Vert _2 = \left\Vert \begin{pmatrix} C^{tr}_x \\ 
C^{tr}_y 
\end{pmatrix} - 
\begin{pmatrix} C^{gt}_x \\ 
C^{gt}_y 
\end{pmatrix} \right\Vert _2 = \left\Vert \begin{pmatrix} C^{tr}_x - C^{gt}_x \\ 
C^{tr}_y - C^{gt}_y 
\end{pmatrix} \right\Vert _2 $$

where $\Vert \cdot \Vert _2$ is Euclidean distance\
[https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/utils/metrics.py#L7](https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/utils/metrics.py#L7)

## Normalize Precision (Pnorm)
Normalize the precision over the ground truth bounding box.

$$P_{norm} = \Vert W \left( C^{tr} - C^{gt} \right) \Vert_2$$

$$ W_i = diag \left( BB_{x,i}^{gt}, BB_{y,i}^{gt}\right) = \begin{pmatrix} BB_{x,i}^{gt} & 0 \\ 
0 & BB_{y,i}^{gt} 
\end{pmatrix}$$

then,

$$ P_{norm} = \left\Vert \begin{pmatrix} BB_x^{gt} & 0 \\ 
0 & BB_y^{gt} 
\end{pmatrix} \begin{pmatrix} C^{tr}_x - C^{gt}_x \\ 
C^{tr}_y - C^{gt}_y 
\end{pmatrix} \right\Vert _2$$

where $W$ is a diagonal covariance matrix whose elements are the standard deviations of the target state parameters. $BB^{gt}$ is ground truth bounding boxes and $BB^{tr}$ is the ones generated by tracker.

The success $S$ is measured as the IoU, by testing whether $S$ is larger than a certain threshold $t_o$ (e.g., $t_o=0.5$).

$$ S = \frac {| BB^{tr} \cap BB^{gt} |} {| BB^{tr} \cup BB^{gt}|} $$ 

*[2] - 3.4 Evaluation section\
[https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/utils/metrics.py#L22](https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/utils/metrics.py#L22)

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
Precision score (P): 30.57 %
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
