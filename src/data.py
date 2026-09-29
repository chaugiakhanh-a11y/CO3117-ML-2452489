# TODO: implement data loading, splitting, and preprocessing
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Constants
ROOT = Path(__file__).resolve().parents[1]
LABELS = [1, 2, 3, 4, 5, 6]  
ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]

@dataclass
class HARData:
    """Dataclass to hold the parsed Human Activity Recognition dataset."""
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    test_subjects: np.ndarray
    hashes: Dict[str, str]


def load_data(data_dir: Path) -> HARData:
    """
    Loads dataset from the specified directory.
    X: features; y: activity labels; subjects: subject IDs (not used as features).
    """
    files = {
        "X_train": data_dir / "train/X_train.txt",
        "y_train": data_dir / "train/y_train.txt",
        "subjects_train": data_dir / "train/subject_train.txt",
        "subjects_test": data_dir / "test/subject_test.txt",
    }

    # 1. Check for missing files
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files:\n" + "\n".join(missing))

    # 2. Load data
    X = np.loadtxt(files["X_train"], ndmin=2)
    y = np.loadtxt(files["y_train"], dtype=int, ndmin=1)
    subjects = np.loadtxt(files["subjects_train"], dtype=int, ndmin=1)
    test_subjects = np.unique(np.loadtxt(files["subjects_test"], dtype=int, ndmin=1))

    # 3. Validate data dimensions and integrity
    if X.shape[0] != len(y) or len(y) != len(subjects):
        raise ValueError("X_train, y_train, and subject_train must have an equal number of rows.")
    
    if X.shape[1] != 561 or not np.isfinite(X).all():
        raise ValueError("Expected 561 finite features per sample; inspect the data.")
    
    if set(np.unique(y)) != set(LABELS):
        raise ValueError("Expected all six UCI HAR activity labels, numbered 1 to 6.")
    
    if np.intersect1d(subjects, test_subjects).size > 0:
        raise ValueError("Subject leakage detected between original training and test sets.")

    # 4. Generate SHA256 hashes for tracking/reproducibility
    file_hashes = {
        str(path.relative_to(data_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files.values()
    }

    return HARData(X=X, y=y, subjects=subjects, test_subjects=test_subjects, hashes=file_hashes)


def split_by_subject(
    X: np.ndarray, 
    y: np.ndarray, 
    subjects: np.ndarray, 
    seed: int, 
    valid_fraction: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the training data into train and validation sets based on subjects.
    Ensures that a single subject's data belongs exclusively to one set.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_fraction, random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y, groups=subjects))

    # Sanity checks
    train_ids = np.unique(subjects[train_idx])
    valid_ids = np.unique(subjects[valid_idx])
    
    if np.intersect1d(train_ids, valid_ids).size > 0:
        raise ValueError("Data leakage: A subject appears in both train and validation sets.")

    # Ensure all classes are represented in both splits
    for name, idx in [("train", train_idx), ("validation", valid_idx)]:
        if set(np.unique(y[idx])) != set(LABELS):
            raise ValueError(f"The '{name}' split is missing activity classes; check your split fraction/seed.")

    return train_idx, valid_idx
