# AI Use Log
## Pipeline Setup — 2026-09-29

### Learning Question
How can I load the UCI HAR dataset, create a subject-based
training/validation split, and evaluate baseline models?

### AI Tool
ChatGPT.

### AI Assistance Received
ChatGPT generated an example pipeline for:
- Loading the dataset and checking its structure.
- Splitting the original training data by subject.
- Training a majority baseline and a scikit-learn Decision Tree.
- Saving metrics, confusion matrices, and split information.

At my request, ChatGPT changed the default random seed
from 42 to 36 and provided instructions for running locally.

### Affected Files
- [Pipeline code](src/har_pipeline.py)
- [Dependencies](requirements.txt)
- [Dataset documentation](data/README.md)

### First-Attempt Evidence
This pipeline started from an AI-generated example.
It is not presented as my unaided first attempt or as
a from-scratch implementation of the learning algorithms.

Any genuine earlier attempt will be linked separately.

### Verification Sources
- CO3117 assignment specification, Sections 9 and 12.
- UCI HAR documentation included with the downloaded dataset.
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

### Independent Verification
The following checks are pending documentation:

- [x] Explain the roles of X, y, and subject IDs.
- [x] Confirm that training and validation subjects do not overlap.
- [x] Confirm that the random seed is 36.
- [x] Confirm that models are fitted only on training data.
- [] Confirm that test features and labels are not evaluated.
- [ ] Compare the printed results with the saved output files.

### Changes Made Independently
TODO: Describe the changes I actually make and explain why.

### Closed-Book Reproduction
Not yet documented. I will explain or reconstruct the
splitting and evaluation process without AI or notes.
