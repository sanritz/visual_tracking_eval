# Evaluation Metrics for Visual Tracking

## Average Overlap (AO)
Average overlap is percentage of IoU (Intersection over Union)

$$ AO = 100 \times  \frac {| S_1 \cap S_2 |} {| S_1 \cup S_2 |} $$

where $S_1$ represents the area of predict object and $S_2$ represents the area of ground truth object \
*[6] - 3. Result and evaluation section\
[https://github.com/got-10k/toolkit/blob/master/got10k/experiments/got10k.py#L198](https://github.com/got-10k/toolkit/blob/master/got10k/experiments/got10k.py#L198)
## AUC
Stand for area under curve, which is the average of the success rates corresponding to the sampled overlap thresholds.The AO is recently proved to be equivalent to the AUC. AUC value is usually used to ranking the trackers in success plot $(S)$. \
*[1] - 4.2 Evaluation methodology section

## P
P is stand for precision score. Usually measured as the distance in pixels between the centers $C^{gt}$ and $C^{tr}$ of the ground truth and the tracker bounding box, respectively.

$$ P =\left\Vert C^{tr} - C^{gt} \right\Vert _2 = \left\Vert \begin{pmatrix} C^{tr}_x \\ 
C^{tr}_y 
\end{pmatrix} - 
\begin{pmatrix} C^{gt}_x \\ 
C^{gt}_y 
\end{pmatrix} \right\Vert _2 = \left\Vert \begin{pmatrix} C^{tr}_x - C^{gt}_x \\ 
C^{tr}_y - C^{gt}_y 
\end{pmatrix} \right\Vert _2 $$

where $\Vert \cdot \Vert _2$ is Euclidean distance

## Pnorm
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

The success S is measured as the IoU, by testing whether $S$ is larger than a certain threshold $t_o$ (e.g., $t_o=0.5$).

$$ S = \frac {| BB^{tr} \cap BB^{gt} |} {| BB^{tr} \cup BB^{gt}|} $$ 

*[2] - 3.4 Evaluation section\
[https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/utils/metrics.py#L22](https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/utils/metrics.py#L22)

# Execute
```
# This is an example for GOT-10k, TrackingNet, and LaSOT dataset
pip install numpy pandas opencv-python tqdm
python metric.py
```

## Reference
[1]: GOT-10k: A large high-diversity benchmark for generic object tracking in the wild [[paper](https://arxiv.org/pdf/1810.11981.pdf)] \
[2]: TrackingNet: A large-scale dataset and benchmark for object tracking in the wild [[paper](https://arxiv.org/pdf/1803.10794.pdf)] \
[3]: LaSOT: A high-quality benchmark for large-scale single object tracking [[paper](https://arxiv.org/pdf/1809.07845v2.pdf)] \
[4]: Object tracking benchmark [[paper](https://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf)] \
[5]: Visual object tracking performance measures revisited [[paper](https://arxiv.org/pdf/1502.05803.pdf)] \
[6]: K-Means vs. Fuzzy C-Means for segmentation of orchid flowers [[paper](https://www.researchgate.net/publication/311409493_K-Means_vs_Fuzzy_C-Means_for_Segmentation_of_Orchid_Flowers)] \
[7]: Robust visual tracking with reliable object information and Kalman filter [[paper](https://www.researchgate.net/publication/348859011_Robust_Visual_Tracking_with_Reliable_Object_Information_and_Kalman_Filter)] \
[8]: Online Object Tracking: A Benchmark [[paper](https://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf)] \
[9]: Intersection over Union (IoU) for object detection [[link](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)] \
[10]: Understanding AUC-ROC Curve [[link](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)] \
[11]: Visual Object Tracking on GOT-10k dataset [[link](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k)] 
