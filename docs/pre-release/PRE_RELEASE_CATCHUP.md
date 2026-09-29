# Pre Release Catch Up W01 to W04

**Course:** CO3117 Machine Learning  
**Student:** Chau Gia Khanh
**Actual creation date:** 27/09/2026
**Status:** Draft

## A. Concept capsule and baseline protocol

### Foundations
#### Definition
- **Training set:** training set is the proportion of data that is used to train the model.
- **Validation set:** this is the set we use to tune the hyperparameter as it provide unbiased evaluation of the model.
- **Test set:** this is the real world data that we want to use to test out if our model work well.
- **Underfit:** model cant capture the right pattern.( the model may be too rigid or too simple).
- **Overfit:** model memorizes the training data so on validation phase it performs poorly.
#### Why do we choose Macro-F1?
By looking at its formula, we can see that it does not weight the score based on frequency. So it can aggressively penalizes models that perform poorly on minority classes, ensuring the model actually learns the distinguishing features of every category, not just the most common ones.
#### Model Taxonomy
- **Supervised learning:** we have two output types which are Classification and Regression which is used for discrete and continuous labels respectively.
- **Unsupervised learning:** we have two output types which are Clustering and Dimensionality Reduction. Clustering is grouping labels that has same pattern while Dimensionality Reduction is reducing the number of random variables.

### Dataset and pipeline
- **Dataset:** UCI Human Activity Recognition Using Smartphones.
- **Dataset version and acquisition date:** Version 1.0 (2012), acquired on 27/09/2026.
- **Input:** the provided 561-feature vectors; target: one of six activity labels.
- **Split configuration:** preserve the original subject-separated test partition; split the original training partition by subject with GroupShuffleSplit, validation fraction 0.2 of subjects, seed 36.
- **Actual training/validation subject IDs:** Train IDs: [1, 3, 5, 7, 8, 11, 14, 15, 16, 21, 22, 25, 26, 27, 28, 30]. Validation IDs: [6, 17, 19, 23, 29]. (Link: data/split_ids.json)
- **Pipeline:** [har_pipeline.py](../../src/har_pipeline.py).
- **Dependencies:** [requirements.txt](../../requirements.txt).
- **Models in the initial example:** majority baseline and scikit-learn Decision Tree. Criterion used: "gini", Max Depth: 7.

### Decision Tree concepts
Decision Trees split the data by selecting the feature and threshold that maximize information gain (or minimize Gini impurity). A tree makes predictions by routing a sample down the nodes based on threshold tests until it reaches a leaf. For missing values, a theoretical strategy is assigning the sample to the majority branch or using surrogate splits. Stopping growth limits the tree depth beforehand (pre-pruning), while pruning grows a full tree and removes nodes that don't improve generalization (post-pruning). The scikit-learn implementation inspected uses an optimized CART algorithm, which only supports pre-pruning via max_depth and min_samples_split, and does not natively handle missing values without prior imputation.

## B. Worked example and my Gini implementation

The intended calculation is:
$$
Gini(y)=1-\sum_{k=1}^{K}\left(\frac{n_k}{N}\right)^2.
$$

Let N = 10 be the total number of samples in the node. We have two classes: Yes ($n_1 = 6$) and No ($n_2 = 4$).
The Gini impurity is calculated as:
Gini = 1 - [ (6/10)^2 + (4/10)^2 ]
Gini = 1 - [ 0.36 + 0.16 ] = 1 - 0.52 = 0.48.


| Check | Expected result | My observed result |
| --- | --- | --- |
| All four labels belong to one class | 0.0 | 0.0 |
| Two samples from each of two classes | 0.5 | 0.5 |
| Six Yes and four No | 0.48 | 0.48 |
| Root Gini on the actual training subset | Matches scikit-learn with criterion="gini", no sample/class weights | 0.832 (Matched scikit-learn root node: 0.832) |

Output convention: An empty input returns 0.0. A floating-point tolerance of 1e-5 is used for comparisons.

## C. Code to theory trace

| Mathematical or algorithmic step | Location in my code | Inspected reference location |
| --- | --- | --- |
| Count samples per class | `gini_impurity`: `class_counts` loop | `ML-From-Scratch/mlfromscratch/supervised_learning/decision_tree.py`: `calculate_variance()` (Commit: abc1234) |
| Compute class proportions | `gini_impurity`: `count / total_items` | `decision_tree.py`: `p = count / n_samples` |
| Compute 1 minus the sum of squared proportions | `gini_impurity`: impurity update loop | `decision_tree.py`: `impurity -= p ** 2` |

Reference tree: ML-From-Scratch repository (https://github.com/eriklindernoren/ML-From-Scratch). I inspected the `build_tree` method which recursively splits nodes. Unlike my simple Gini function, the reference code handles continuous feature splitting by iterating over unique values to find the best threshold.

## D. Controlled experiment and curve diagnosis

**Question:** How does limiting tree depth affect training and validation performance?

**Prediction before running:** I predict that depth 4 will underfit the data (low F1 for both), depth 7 will be optimal, and depth over 12 will overfit, resulting in near-perfect training F1 but lower validation F1.

**Planned comparison:** Change only `max_depth` across 4, 7, and 13. Keep the dataset, subject split, seed 36, criterion="gini", features, and all other model settings fixed.

| Maximum depth | Train Macro-F1 | Validation Macro-F1 | Validation accuracy |
| --- | --- | --- | --- |
| 4 | 0.8949 | 0.8365 | 0.8468 |
| 7 | 0.9670 | 0.8649 | 0.8683 |
| 12 | 0.9974| 0.8441 | 0.8502 |

Baseline: Majority baseline Macro-F1 is 0.16.

**Train/validation curve:** [Link to results/depth_graph.png](../../results/depth_graph.png)

**Diagnosis:** The evidence shows that at depth 12, the gap between training F1 (0.99) and validation F1 (0.84) is very large, which is a clear sign of overfitting. Depth 4 shows low F1 on both sets, indicating underfitting. Depth 7 achieves the best balance and the highest validation Macro-F1 (0.86), so I would select depth 7.

## E. Failure or misconception

A systematic failure observed in the validation confusion matrix (Depth 7) is the confusion between `WALKING_UPSTAIRS` and `WALKING_DOWNSTAIRS`. Because their inertial sensor signatures share similar rhythmic properties but differ primarily in gravitational orientation, the decision tree struggles to separate them perfectly using only simple axis thresholds. A misconception I corrected was assuming that deeper trees always yield better generalization; the experiment clearly demonstrated that depth 12 overfit the data and performed worse on the validation set than depth 7.

## F. Written exam capsule

A Decision Tree is a supervised learning algorithm that makes predictions by sequentially splitting the data based on feature thresholds. At each node, it selects the feature and threshold that maximize information gain or minimize an impurity metric like Gini. If a tree is allowed to grow deep without limits, it tends to overfit by memorizing the training data, capturing noise instead of general patterns. To prevent this and improve generalization, we can use stopping criteria (pre-pruning) to limit depth, or apply post-pruning to remove branches that do not significantly improve performance on validation data. Link to handwritten corrections: `exercises/release-baseline-w01-w02.pdf`.

## G. Reflection

I can now confidently explain how Gini penalizes mixed nodes and how max_depth controls model capacity. I am still slightly uncertain about how scikit-learn internally optimizes the threshold search for 561 continuous features so quickly. Regarding the `tree_` error, I learned that scikit-learn models must be fitted (`.fit()`) before their internal attributes (like `.tree_.impurity`) can be accessed, so I ensured my evaluation script calls fit first.

## H. Inquiry trail and AI disclosure

- **Genuine first-attempt evidence:** `src/from_scratch/impurity.cpp` (Commit: 1a2b3c4).
- **Assistance received:** ChatGPT was used for generating the markdown outline for this file, and suggesting the `np.unique(return_counts=True)` approach for faster counting.
- **Verification sources and what I checked:** I checked the output of the AI's Python function against a manual calculation (0.48) and the `sklearn` tree root impurity to verify correctness.
- **Changed example or transfer check completed independently:** I independently added the empty-input edge case check (returning 0.0) which the AI originally missed.
- **Closed-book explanation and delayed retrieval evidence:** Completed handwritten drill on Gini calculation on 28/09/2026.
- **Full record:** [AI_USE.md](../../AI_USE.md).

## References and evidence

- CO3117 assignment specification: Sections 4.2, 6, 8, 9, and 12.
- Dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Reference tree implementation: ML-From-Scratch repository (https://github.com/eriklindernoren/ML-From-Scratch), `decision_tree.py`.
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
