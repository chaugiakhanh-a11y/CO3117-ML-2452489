# TODO: implement evaluation helpers

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def evaluate_classification(y_true, y_pred, labels):
    """Tính toán macro-F1, accuracy và confusion matrix"""
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "confusion_matrix": conf_matrix
    }
