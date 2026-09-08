import torch

from eoflood.metrics import SegMetrics


def test_perfect_prediction():
    m = SegMetrics()
    y = torch.tensor([[[0, 1], [1, 0]]])
    m.update(y, y)
    r = m.compute()
    assert r["water_iou"] == 1.0 and r["miou"] == 1.0 and r["acc"] == 1.0


def test_ignore_index_is_skipped():
    m = SegMetrics()
    pred = torch.tensor([[[1, 1], [0, 0]]])
    target = torch.tensor([[[1, -1], [0, -1]]])
    m.update(pred, target)
    r = m.compute()
    assert r["acc"] == 1.0
    assert m.cm.sum().item() == 2


def test_known_confusion_matrix():
    # target: 4 water, 4 non-water; pred gets 3/4 water right and 2/4 non-water right
    target = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    pred = torch.tensor([[1, 1, 1, 0, 0, 0, 1, 1]])
    m = SegMetrics()
    m.update(pred, target)
    r = m.compute()
    tp, fp, fn = 3, 2, 1
    assert abs(r["water_iou"] - tp / (tp + fp + fn)) < 1e-6
    assert abs(r["water_precision"] - tp / (tp + fp)) < 1e-6
    assert abs(r["water_recall"] - tp / (tp + fn)) < 1e-6


def test_accumulates_over_batches():
    a, b = SegMetrics(), SegMetrics()
    t1, p1 = torch.tensor([[1, 0, 1]]), torch.tensor([[1, 1, 1]])
    t2, p2 = torch.tensor([[0, 0, 1]]), torch.tensor([[0, 0, 0]])
    a.update(p1, t1)
    a.update(p2, t2)
    b.update(torch.cat([p1, p2], 1), torch.cat([t1, t2], 1))
    assert torch.equal(a.cm, b.cm)
