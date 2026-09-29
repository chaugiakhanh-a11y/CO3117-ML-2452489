# Dataset and Data Protocol

## 1. Dataset Source

- Dataset: Human Activity Recognition Using Smartphones
- Provider: UCI Machine Learning Repository
- Dataset ID: 240
- Source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- DOI: https://doi.org/10.24432/C54S4K
- Version: 1.0
- Download date: 29-09-2026.

## 2. Purpose
The purpose of this mini project is to predict a person’s current physical activity from smartphone inertial measurements.

We will use different ML models to compare with each other on the same benchmark.

## 3. input features and prediction output

For the initial experiments, each input sample is a vector of
561 features extracted from accelerometer and gyroscope signals.

The prediction target is one of six activity labels:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

The subject identifier will be used to control data splitting.
It will not be used as an input feature.


## 4. Planned Data Splitting and Preprocessing
## Data Splitting

- The original UCI HAR training/test partition is preserved.
- The validation set is created from the original training partition.
- Split method: GroupShuffleSplit.
- Number of splits: 1.
- Validation fraction: 0.2 of the subjects in the original training set,
  rounded up. The fraction of samples may differ from 20%.
- Random seed: 36.
- Training and validation sets contain different subjects.
- The original test set is reserved for the final evaluation.

Training subject IDs: TODO
Validation subject IDs: TODO

The exact subject IDs and row indices are saved in:
results/har_pipeline_seed36/split.json

## Preprocessing

The initial pipeline uses the 561 features provided by UCI HAR.
Subject identifiers are used only for splitting, not as input features.

No additional scaling is applied in this initial pipeline because
the majority baseline and Decision Tree do not require it.

## Initial Pipeline

1. Load the original training features, labels, and subject IDs.
2. Check data dimensions, valid labels, and subject separation.
3. Split the original training data into training and validation sets.
4. Fit the majority baseline and a scikit-learn Decision Tree
   using entropy and a maximum depth of 5.
5. Evaluate both models on training and validation data.
6. Save the split information, metrics, confusion matrices,
   and environment details.

The original test features and labels are not loaded or evaluated.



## 5. Evaluation Metrics

- Primary metric: Macro-F1.
- Secondary evidence: Accuracy and confusion matrix.
