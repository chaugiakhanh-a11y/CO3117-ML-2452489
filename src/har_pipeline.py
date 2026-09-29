"""UCI HAR: đọc dữ liệu -> chia theo người -> train -> validation.

Mẫu hỗ trợ học tập do ChatGPT tạo; ghi việc sử dụng vào AI_USE.md.
Các mô hình bên dưới là implementation có sẵn của scikit-learn.
Chạy từ thư mục làm việc trên máy: python src/har_pipeline.py
"""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
LABELS = [1, 2, 3, 4, 5, 6]
ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


def load_data(data_dir):
    """X: đặc trưng; y: nhãn hoạt động; subjects: mã người, không phải feature."""
    files = {
        "X_train": data_dir / "train/X_train.txt",
        "y_train": data_dir / "train/y_train.txt",
        "subjects_train": data_dir / "train/subject_train.txt",
        "subjects_test": data_dir / "test/subject_test.txt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))

    # Chỉ đọc features/labels của train gốc. Test chỉ đọc mã người để kiểm tra split.
    X = np.loadtxt(files["X_train"], ndmin=2)
    y = np.loadtxt(files["y_train"], dtype=int, ndmin=1)
    subjects = np.loadtxt(files["subjects_train"], dtype=int, ndmin=1)
    test_subjects = np.unique(
        np.loadtxt(files["subjects_test"], dtype=int, ndmin=1)
    )
    if X.shape[0] != len(y) or len(y) != len(subjects):
        raise ValueError("X_train, y_train and subject_train must have equal rows.")
    if X.shape[1] != 561 or not np.isfinite(X).all():
        raise ValueError("Expected 561 finite features per sample; inspect the data.")
    if set(np.unique(y)) != set(LABELS):
        raise ValueError("Expected all six UCI HAR activity labels, numbered 1 to 6.")
    if np.intersect1d(subjects, test_subjects).size:
        raise ValueError("Subject leakage between original training and test sets.")

    # Fingerprint để phát hiện dữ liệu đã đổi khi chạy lại.
    hashes = {
        str(path.relative_to(data_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files.values()
    }
    return X, y, subjects, test_subjects, hashes


def split_by_subject(X, y, subjects, seed, valid_fraction):
    """Tách validation từ train gốc; một người chỉ thuộc một tập."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=valid_fraction, random_state=seed
    )
    # Trong API, test_size ở đây chính là phần VALIDATION, tính theo số người.
    train_idx, valid_idx = next(splitter.split(X, y, groups=subjects))
    train_ids = np.unique(subjects[train_idx])
    valid_ids = np.unique(subjects[valid_idx])
    if np.intersect1d(train_ids, valid_ids).size:
        raise ValueError("A subject appears in both train and validation.")
    for name, idx in [("train", train_idx), ("validation", valid_idx)]:
        if set(np.unique(y[idx])) != set(LABELS):
            raise ValueError(f"{name} is missing activity classes; inspect the split.")
    return train_idx, valid_idx


def run(data_dir, output_dir, seed=36, valid_fraction=0.2):
    # 1. Đọc và kiểm tra dữ liệu gốc.
    X, y, subjects, test_subjects, hashes = load_data(data_dir)

    # 2. Chia theo mã NGƯỜI, sau đó lấy các dòng tương ứng.
    train_idx, valid_idx = split_by_subject(X, y, subjects, seed, valid_fraction)
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    train_ids = np.unique(subjects[train_idx]).tolist()
    valid_ids = np.unique(subjects[valid_idx]).tolist()

    # 3. Lưu đúng split để các mô hình sau dùng chung.
    manifest = {
        "seed": seed,
        "validation_fraction_of_subjects": valid_fraction,
        "train_subject_ids": train_ids,
        "validation_subject_ids": valid_ids,
        "test_subject_ids": test_subjects.tolist(),
        "train_row_indices_zero_based": train_idx.tolist(),
        "validation_row_indices_zero_based": valid_idx.tolist(),
        "source_sha256": hashes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "split.json"
    if split_path.exists():
        if json.loads(split_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                "Existing split.json differs from this run. Keep the frozen split; "
                "use another --output-dir only for a deliberate new protocol."
            )
    else:
        split_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Training subject IDs:", train_ids)
    print("Validation subject IDs:", valid_ids)
    print("Random seed:", seed)
    print("X_train shape:", X_train.shape)
    print("X_valid shape:", X_valid.shape)
    print("Original test features/labels: not loaded or evaluated.")

    # 4. Hai benchmark có sẵn; cây dùng entropy để nối với ví dụ học trên giấy.
    # Các đặc trưng UCI đã được trích xuất. Cây và Dummy không cần StandardScaler.
    # max_depth=5 là giá trị khởi đầu minh họa, chưa được tối ưu.
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "sklearn_tree_depth5": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, random_state=seed
        ),
    }
    rows = []
    matrices = {"label_order": LABELS, "activity_order": ACTIVITIES,
                "rows": "true label", "columns": "predicted label", "models": {}}
    for model_name, model in models.items():
        # fit CHỈ nhận tập train. Không fit lại trên validation.
        start = perf_counter()
        model.fit(X_train, y_train)
        train_seconds = perf_counter() - start
        matrices["models"][model_name] = {}

        # 5. Đánh giá train để chẩn đoán, validation để chọn mô hình sau này.
        for split_name, X_part, y_part in [
            ("train", X_train, y_train), ("validation", X_valid, y_valid)
        ]:
            start = perf_counter()
            prediction = model.predict(X_part)
            inference_seconds = perf_counter() - start
            macro_f1 = f1_score(y_part, prediction, labels=LABELS,
                               average="macro", zero_division=0)
            accuracy = accuracy_score(y_part, prediction)
            rows.append({
                "model": model_name, "split": split_name,
                "macro_f1": float(macro_f1), "accuracy": float(accuracy),
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds, "seed": seed,
            })
            matrices["models"][model_name][split_name] = confusion_matrix(
                y_part, prediction, labels=LABELS
            ).tolist()
            print(f"{model_name:24} {split_name:10} "
                  f"Macro-F1={macro_f1:.4f}  Accuracy={accuracy:.4f}")

    # 6. Lưu kết quả thật của lần chạy và thông tin dùng để cập nhật README.
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "confusion_matrices.json").write_text(
        json.dumps(matrices, indent=2), encoding="utf-8"
    )
    (output_dir / "readme_split.txt").write_text(
        f"Random seed: {seed}\n"
        f"Training subject IDs: {', '.join(map(str, train_ids))}\n"
        f"Validation subject IDs: {', '.join(map(str, valid_ids))}\n"
        f"Training samples: {len(train_idx)}\nValidation samples: {len(valid_idx)}\n"
        f"Validation policy: GroupShuffleSplit, fraction={valid_fraction} of subjects.\n",
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "preprocessing": "Provided UCI feature vectors; no extra scaling.",
        "tree_parameters": models["sklearn_tree_depth5"].get_params(),
    }, indent=2), encoding="utf-8")
    print("Saved results to:", output_dir.resolve())
    return manifest, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/raw/UCI HAR Dataset")
    parser.add_argument("--output-dir", type=Path,
                        default=None)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.valid_fraction < 1:
        parser.error("--valid-fraction must be between 0 and 1.")
    output_dir = args.output_dir or ROOT / f"results/har_pipeline_seed{args.seed}"
    try:
        run(args.data_dir, output_dir, args.seed, args.valid_fraction)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
