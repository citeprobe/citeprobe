"""The linear readout of Equation 3, its training data, training and scoring.

One training item is one (statement, sentence) pair: the sentence's
standardized [L, H] attention grid, labelled 1 if the sentence is gold.
Statement ids come along so the ranking metrics group pairs correctly.
"""

import pathlib
from dataclasses import dataclass

import numpy
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from citeprobe.capture import load_capture
from citeprobe.data import CiteDataset
from citeprobe.metrics import mean_average_precision, top1_gold_accuracy

SCORING_BATCH = 4096


@dataclass
class Pairs:
    """All (statement, sentence) pairs of one or more splits."""

    grids: torch.Tensor           # [num_pairs, L, H] standardized attention
    labels: torch.Tensor          # [num_pairs] float 0/1
    statement_ids: numpy.ndarray  # [num_pairs] int, groups pairs by statement

    def to(self, device: str) -> "Pairs":
        return Pairs(self.grids.to(device), self.labels.to(device), self.statement_ids)


@dataclass
class TrainingConfig:
    """The paper's recipe (Appendix, Table "Readout and training configuration")."""

    learning_rate: float
    batch_size: int = 128
    epochs: int = 300
    patience: int = 30


@dataclass
class FitReport:
    """What one fit achieved on validation, at the kept epoch."""

    best_epoch: int
    val_map: float
    val_top1: float
    parameters: int


@dataclass
class Scores:
    """Ranking quality over the statements that have at least one gold sentence."""

    map: float
    top1: float
    statements: int


# ----------------------------------------------------------------------------
# Training data
# ----------------------------------------------------------------------------


def standardize(attention_sum: numpy.ndarray) -> numpy.ndarray:
    """Equation 2: z-score each head's sentence scores within the example.

    attention_sum is [L, H, m, d]; the statistics run over the sentence axis.
    """
    means = attention_sum.mean(axis=-1, keepdims=True)
    deviations = attention_sum.std(axis=-1, keepdims=True)
    return (attention_sum - means) / numpy.maximum(deviations, 1e-9)


def build_pairs(
    capture_directory: pathlib.Path, dataset: CiteDataset, keep: frozenset[int] | None = None
) -> Pairs:
    """Pairs from every capture in a directory, labelled with the split's gold.

    keep restricts the pairs to those dataset_index values (the examples the
    model answered correctly, when scoring a test split).
    """
    npz_paths = sorted(capture_directory.glob("example_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No captures in {capture_directory}")
    parts = [
        _example_pairs(path, dataset)
        for path in npz_paths
        if keep is None or int(path.stem.split("_")[1]) in keep
    ]
    if not parts:
        raise ValueError(f"{capture_directory}: no examples kept")
    return concatenate_pairs(parts)


def _example_pairs(npz_path: pathlib.Path, dataset: CiteDataset) -> Pairs:
    """One example's pairs: every (statement, sentence) grid with its gold label."""
    example = dataset.by_dataset_index(int(npz_path.stem.split("_")[1]))
    gold_units = set(example.gold_unit_positions(example.partition()))
    standardized = standardize(load_capture(npz_path).attention_sum)  # [L, H, m, d]
    num_statements, num_sentences = standardized.shape[2:]

    # [L, H, m, d] -> [m, d, L, H] -> [m * d, L, H]: statement-major, one grid per sentence.
    grids = numpy.moveaxis(standardized, (2, 3), (0, 1)).reshape(-1, *standardized.shape[:2])
    gold_mask = numpy.array([unit in gold_units for unit in range(num_sentences)], dtype=numpy.float32)
    statement_ids = numpy.repeat(numpy.arange(num_statements), num_sentences)
    return Pairs(
        grids=torch.from_numpy(numpy.ascontiguousarray(grids, dtype=numpy.float32)),
        labels=torch.from_numpy(numpy.tile(gold_mask, num_statements)),
        statement_ids=statement_ids,
    )


def concatenate_pairs(parts: list[Pairs]) -> Pairs:
    """Join pair sets, keeping every statement group distinct."""
    grids, labels, statement_ids = [], [], []
    offset = 0
    for part in parts:
        grids.append(part.grids)
        labels.append(part.labels)
        statement_ids.append(part.statement_ids + offset)
        offset += int(part.statement_ids.max()) + 1
    return Pairs(torch.cat(grids), torch.cat(labels), numpy.concatenate(statement_ids))


# ----------------------------------------------------------------------------
# The probe
# ----------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Equation 3: one weight per head over the standardized grid, plus a bias."""

    def __init__(self, layers: int, heads: int) -> None:
        super().__init__()
        self.layers = layers
        self.heads = heads
        self.linear = nn.Linear(layers * heads, 1)

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        """grids: [batch, L, H] -> logits: [batch]"""
        return self.linear(grids.flatten(start_dim=1)).squeeze(-1)

    def head_weights(self) -> torch.Tensor:
        """The weight vector as an [L, H] grid."""
        return self.linear.weight.detach().reshape(self.layers, self.heads)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"weight": self.head_weights().cpu().contiguous(),
             "bias": self.linear.bias.detach().cpu().contiguous()},
            str(path),
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "LinearProbe":
        tensors = load_file(str(path))
        layers, heads = tensors["weight"].shape
        probe = cls(layers, heads)
        with torch.no_grad():
            probe.linear.weight.copy_(tensors["weight"].reshape(1, layers * heads))
            probe.linear.bias.copy_(tensors["bias"])
        return probe.eval()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def train_probe(
    train: Pairs, validation: Pairs, config: TrainingConfig, seed: int, device: str
) -> tuple[LinearProbe, FitReport]:
    """Fit the probe with AdamW and positive-weighted BCE, keeping the best epoch.

    The kept epoch is the one with the best harmonic mean of validation mAP
    and Top-1. Training stops after `patience` epochs without improving it,
    and the best epoch's weights are returned.
    """
    torch.manual_seed(seed)
    layers, heads = train.grids.shape[1:]
    probe = LinearProbe(int(layers), int(heads)).to(device)
    train = train.to(device)
    loss_function = _class_balanced_loss(train.labels)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.learning_rate)

    best_score = -1.0
    best = FitReport(best_epoch=-1, val_map=-1.0, val_top1=-1.0, parameters=0)
    best_state: dict = {}
    stagnant = 0
    for epoch in range(config.epochs):
        _train_one_epoch(probe, train, loss_function, optimizer, config.batch_size)
        scores = evaluate(probe, validation, device)
        score = harmonic_mean(scores.map, scores.top1)
        if score > best_score:
            best_score = score
            best = FitReport(epoch, scores.map, scores.top1, probe.parameter_count())
            best_state = {key: value.detach().cpu().clone()
                          for key, value in probe.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= config.patience:
                break
    probe.load_state_dict(best_state)
    return probe.eval(), best


def _class_balanced_loss(labels: torch.Tensor) -> nn.Module:
    """BCE with the loss on positives scaled by (N - P) / P.

    Positives are 4 to 5% of pairs, so without this the probe would learn to
    say no to everything.
    """
    positives = float(labels.sum())
    pos_weight = torch.tensor((len(labels) - positives) / positives, device=labels.device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _train_one_epoch(probe: LinearProbe, train: Pairs, loss_function: nn.Module,
                     optimizer: torch.optim.Optimizer, batch_size: int) -> None:
    """One pass over the training pairs in a fresh random order."""
    probe.train()
    permutation = torch.randperm(len(train.labels), device=train.labels.device)
    for start in range(0, len(train.labels), batch_size):
        batch = permutation[start:start + batch_size]
        optimizer.zero_grad()
        loss = loss_function(probe(train.grids[batch]), train.labels[batch])
        loss.backward()
        optimizer.step()


def harmonic_mean(precision: float, hits: float) -> float:
    """Harmonic mean of mAP and Top-1, zero if either is zero.

    This is the validation quantity that picks both the kept epoch of a fit
    and the learning rate of a model.
    """
    if precision <= 0.0 or hits <= 0.0:
        return 0.0
    return 2.0 * precision * hits / (precision + hits)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def evaluate(probe: LinearProbe, pairs: Pairs, device: str) -> Scores:
    """mAP and Top-1 over statements; statements with no gold are unrankable and skipped."""
    scores = _score_pairs(probe, pairs, device)
    labels = pairs.labels.cpu().numpy().astype(bool)
    rows: list[numpy.ndarray] = []
    masks: list[numpy.ndarray] = []
    for statement_id in numpy.unique(pairs.statement_ids):
        selector = pairs.statement_ids == statement_id
        if labels[selector].any():
            rows.append(scores[selector])
            masks.append(labels[selector])
    return Scores(mean_average_precision(rows, masks), top1_gold_accuracy(rows, masks), len(rows))


def _score_pairs(probe: LinearProbe, pairs: Pairs, device: str) -> numpy.ndarray:
    """The probe's logit for every pair, computed in batches."""
    probe.eval()
    with torch.no_grad():
        logits = [
            probe(pairs.grids[start:start + SCORING_BATCH].to(device)).cpu()
            for start in range(0, len(pairs.labels), SCORING_BATCH)
        ]
    return torch.cat(logits).numpy()
