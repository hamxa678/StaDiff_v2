import numpy as np

from stadiff.evaluation.metrics import auroc, average_precision, best_f1, binary_metrics


def test_auroc():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))


def test_average_precision():
    assert average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    ap = average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert abs(ap - (0.5 * 1.0 + 0.5 * 2 / 3)) < 1e-9


def test_f1():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["tp"] == 1 and m["fp"] == 1 and abs(m["f1"] - 0.5) < 1e-9
    assert best_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])["f1"] == 1.0
