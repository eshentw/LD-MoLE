import numpy as np
from evaluate import load

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def compute_metrics(dataset, task, preds, labels, dtype=None, metric=None, is_regression=False):
    metric = load(dataset, task)
    preds = preds[0] if isinstance(preds, tuple) else preds
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    if task is not None:
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    elif is_regression:
        return {"mse": ((preds - labels) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == labels).astype(np.float32).mean().item()}