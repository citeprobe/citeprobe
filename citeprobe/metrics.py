"""The two ranking metrics the paper reports: mAP and Top-1."""

import numpy


def average_precision(scores: numpy.ndarray, gold_mask: numpy.ndarray) -> float:
    """AP of one ranking against its gold set (threshold-free, multi-gold)."""
    order = numpy.argsort(scores)[::-1]
    hits = 0
    precision_sum = 0.0
    for rank, unit in enumerate(order, start=1):
        if gold_mask[unit]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / max(int(gold_mask.sum()), 1)


def mean_average_precision(
    score_rows: list[numpy.ndarray], gold_masks: list[numpy.ndarray]
) -> float:
    """Mean AP over examples, each row scored against its own gold mask."""
    values = [
        average_precision(row, mask) for row, mask in zip(score_rows, gold_masks, strict=True)
    ]
    return float(numpy.mean(values))


def top1_gold_accuracy(
    score_rows: list[numpy.ndarray], gold_masks: list[numpy.ndarray]
) -> float:
    """Fraction of examples whose top-scored sentence is gold."""
    hits = [
        bool(mask[int(numpy.argmax(row))])
        for row, mask in zip(score_rows, gold_masks, strict=True)
    ]
    return float(numpy.mean(hits))
