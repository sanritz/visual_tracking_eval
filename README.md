# Evaluation guidance
This is evaluation metrics for visual tracking benchmark.
## AO
Average overlap is percentage of IoU (Intersection over Union)
$$ AO = 100 \times  \frac {| S_1 \cap S_2 |} {| S_1 \cup S_2 |} $$
where $S_1$ represents the area of predict object and $S_2$ represents the area of ground truth object 

## AUC
Stand for area under curve, which is the average of the success rates corresponding to the sampled overlap thresholds.The AO is recently proved to be equivalent to the AUC. AUC value is usually used to ranking the trackers in success plot $(S)$.


## P
P is stand for precision score. Usually measured as the distance in pixels between the centers $C^{gt}$ and $C^{tr}$ of the ground truth and the tracker bounding box, respectively.
$$ P =\| C^{tr} - C^{gt} \| _2$$

## Pnorm
Normalize the precision over the ground truth bounding box.
$$ P_{norm} = \| W \left( C^{tr} - C^{gt} \right) \| _ 2$$
$$ W = diag \left( BB_x^{gt}, BB_y^{gt}\right) $$
where $BB^{gt}$ is ground truth bounding boxes and $BB^{tr} $ is the ones generated by tracker.

The success S is measured as the IoU, by testing whether $S$ is larger than a certain threshold $t_o$ (e.g., $t_o=0.5$).
$$ S = \frac {| BB^{tr} \cap BB^{gt} |} {| BB^{tr} \cup BB^{gt}|} $$
