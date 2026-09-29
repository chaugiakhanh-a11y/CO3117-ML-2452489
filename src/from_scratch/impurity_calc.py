


def gini_impurity(labels):

    if len(labels) == 0:
        return 0.0

    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    impurity = 1.0
    total_items = len(labels)

    for count in class_counts.values():
        probability = count / total_items
        impurity -= probability * probability

    return impurity


if __name__ == "__main__":
    examples = [
        [],
        [1, 1, 1, 1],
        [1, 1, 2, 2],
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        [1, 2, 3, 4, 5, 6],
    ]
    for labels in examples:
        print(f"{labels} -> Gini = {gini_impurity(labels):.6f}")