import numpy as np


def evaluate(B_true, B_est):
    """
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    @:parameter
    ----------
    B_est: np.ndarray
        [d, d, t] estimate, {0, 1}.
    B_true: np.ndarray
        [d, d, t] ground truth graph, {0, 1}.
    """
    assert B_true.shape == B_est.shape
    B_true = np.where(B_true == 0, 0, 1)
    B_est = np.where(B_est == 0, 0, 1)
    eunit = np.ones(np.shape(B_est))
    TP = np.sum(B_true*B_est)
    TN = np.sum((eunit-B_true)*(eunit-B_est))
    FN = np.sum((B_true)*(eunit-B_est))
    FP = np.sum((eunit-B_true)*(B_est))
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    TPR = TP / (TP + FN)
    NNZ = TP + FP
    gscore = max(0, (TP - FP)) / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    result = {'Accuracy': Accuracy, 'Precision': Precision,
              'Recall': Recall, 'TPR': TPR, 'NNZ': NNZ, 'F1': F1, 'gscore': gscore}
    return result
