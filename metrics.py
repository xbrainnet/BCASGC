from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_pred = probabilities.argmax(1)
    classes = np.unique(y_true)
    acc = accuracy_score(y_true, y_pred)
    if len(classes) == 2:
        auc = roc_auc_score(y_true, probabilities[:, 1])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spe = tn / max(tn + fp, 1)
        sen = tp / max(tp + fn, 1)
    else:
        auc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        recalls = np.diag(cm) / np.maximum(cm.sum(1), 1)
        sen = float(recalls.mean())
        specificities = []
        for i in range(len(classes)):
            tp = cm[i, i]
            fn = cm[i].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            specificities.append(tn / max(tn + fp, 1))
        spe = float(np.mean(specificities))
    return {"ACC": float(acc), "AUC": float(auc), "SPE": float(spe), "SEN": float(sen)}
