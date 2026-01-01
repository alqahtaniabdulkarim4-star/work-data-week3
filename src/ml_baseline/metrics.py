from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except ValueError:
            metrics["log_loss"] = None

    return metrics

