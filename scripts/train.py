"""Step 2: fit the probes under the four-corpus leave-one-out protocol.

    uv run python -m scripts.train --model phi4-mini

For every held-out corpus, the probe is fitted on the train splits of the
other three, its epoch chosen on their pooled val splits, and scored on the
held-out test split gated to correctly answered examples. Each cell is run
for every learning rate in the grid and three seeds. The learning rate is
then chosen ONCE per model, on validation pooled over all holdouts and
seeds, so no held-out corpus picks its own hyperparameters. Both choices
maximize the harmonic mean of validation mAP and Top-1.

Writes runs/<model>/grid.jsonl (one row per fit, resumable), every fitted
probe under runs/<model>/grid/, and the chosen recipe's probes in the same
layout weights/ uses, so `scripts.evaluate --weights runs` reads them.
"""

import json
import pathlib
import shutil
import statistics
from dataclasses import asdict, dataclass

import torch

from citeprobe.data import CORPORA, CiteDataset
from citeprobe.gate import answered_correctly
from citeprobe.probe import (
    LinearProbe,
    Pairs,
    TrainingConfig,
    build_pairs,
    concatenate_pairs,
    evaluate,
    harmonic_mean,
    train_probe,
)

LEARNING_RATES = "1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2"


@dataclass
class Cell:
    """One fit, scored on its held-out corpus."""

    model: str
    holdout: str
    seed: int
    learning_rate: float
    batch_size: int
    best_epoch: int
    val_map: float
    val_top1: float
    test_map: float
    test_top1: float
    parameters: int


@dataclass
class Splits:
    """The training pool, its validation pool and the gated held-out test set."""

    train: Pairs
    validation: Pairs
    test: Pairs


def main(
    model: str,
    device: str = "",
    captures: str = "captures",
    output: str = "runs",
    seeds: int = 3,
    learning_rates: str = LEARNING_RATES,
    batch_size: int = 128,
) -> None:
    """Run the grid for one model, then choose and export its recipe."""
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    rates = [float(rate) for rate in learning_rates.split(",")]
    model_directory = pathlib.Path(output) / model
    done = _completed(model_directory / "grid.jsonl")

    for holdout in CORPORA:
        pending = [(rate, seed) for rate in rates for seed in range(seeds)
                   if (holdout, seed, rate, batch_size) not in done]
        if not pending:
            print(f"{model} {holdout}: all cells done", flush=True)
            continue
        splits = assemble(pathlib.Path(captures) / model, holdout)
        print(f"{model} holdout {holdout}: {len(splits.train.labels)} train pairs, "
              f"{len(splits.validation.labels)} val pairs, "
              f"{len(splits.test.labels)} gated test pairs", flush=True)
        for rate, seed in pending:
            config = TrainingConfig(learning_rate=rate, batch_size=batch_size)
            cell = run_cell(model, holdout, seed, config, splits, device, model_directory)
            print(f"  lr {rate:g} seed {seed}: val mAP {cell.val_map:.3f}  "
                  f"test mAP {cell.test_map:.3f} Top-1 {cell.test_top1:.3f}  epoch {cell.best_epoch}",
                  flush=True)
    export_recipe(model_directory, seeds, batch_size)


def assemble(model_captures: pathlib.Path, holdout: str) -> Splits:
    """Pool the other corpora's train and val splits; gate the holdout's test split."""
    train_parts: list[Pairs] = []
    validation_parts: list[Pairs] = []
    for corpus in CORPORA:
        if corpus == holdout:
            continue
        for split, parts in (("train", train_parts), ("val", validation_parts)):
            parts.append(build_pairs(model_captures / corpus / split, CiteDataset.load(corpus, split)))
    test_dataset = CiteDataset.load(holdout, "test")
    test_directory = model_captures / holdout / "test"
    keep = answered_correctly(test_dataset, model_captures.name, test_directory)
    return Splits(
        train=concatenate_pairs(train_parts),
        validation=concatenate_pairs(validation_parts),
        test=build_pairs(test_directory, test_dataset, keep),
    )


def run_cell(model: str, holdout: str, seed: int, config: TrainingConfig, splits: Splits,
             device: str, model_directory: pathlib.Path) -> Cell:
    """Fit one probe, score it on the gated holdout, save it and log the row."""
    probe, report = train_probe(splits.train, splits.validation, config, seed, device)
    test = evaluate(probe, splits.test, device)
    probe.save(_grid_probe_path(model_directory, config.learning_rate, holdout, seed))
    cell = Cell(model, holdout, seed, config.learning_rate, config.batch_size,
                report.best_epoch, report.val_map, report.val_top1, test.map, test.top1,
                report.parameters)
    grid_path = model_directory / "grid.jsonl"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    with grid_path.open("a") as handle:
        handle.write(json.dumps(asdict(cell)) + "\n")
    return cell


def export_recipe(model_directory: pathlib.Path, seeds: int, batch_size: int) -> None:
    """Choose the learning rate on pooled validation and copy its probes out.

    Only rates with every (holdout, seed) cell finished are eligible, so a
    rate cannot win on a lucky subset. The chosen probes land beside
    recipe.json as holdout_<corpus>/seed<n>.safetensors.
    """
    complete = _complete_rates(model_directory / "grid.jsonl", seeds, batch_size)
    if not complete:
        print("grid incomplete: no learning rate has every cell yet", flush=True)
        return
    rate = max(complete, key=lambda candidate: _pooled_validation(complete[candidate]))

    for holdout in CORPORA:
        for seed in range(seeds):
            source = _grid_probe_path(model_directory, rate, holdout, seed)
            target = model_directory / f"holdout_{holdout}" / f"seed{seed}.safetensors"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            LinearProbe.load(target)  # fail loudly if the copy is unreadable
    (model_directory / "recipe.json").write_text(json.dumps(
        {"learning_rate": rate, "batch_size": batch_size, "seeds": seeds}, indent=2) + "\n")

    print(f"\n{model_directory.name}: learning rate {rate:g} chosen on pooled validation")
    for holdout in CORPORA:
        cells = [cell for cell in complete[rate] if cell["holdout"] == holdout]
        print(f"  {holdout:10} mAP {statistics.mean(c['test_map'] for c in cells):.3f}  "
              f"Top-1 {statistics.mean(c['test_top1'] for c in cells):.3f}")


def _complete_rates(grid_path: pathlib.Path, seeds: int,
                    batch_size: int) -> dict[float, list[dict]]:
    """Grid rows grouped by learning rate, keeping only rates with every cell done."""
    by_rate: dict[float, list[dict]] = {}
    for row in map(json.loads, filter(str.strip, grid_path.read_text().splitlines())):
        if row["batch_size"] == batch_size:
            by_rate.setdefault(row["learning_rate"], []).append(row)
    return {rate: cells for rate, cells in by_rate.items() if len(cells) == len(CORPORA) * seeds}


def _pooled_validation(cells: list[dict]) -> float:
    """Harmonic mean of validation mAP and Top-1, averaged over every holdout and seed."""
    return statistics.mean(harmonic_mean(cell["val_map"], cell["val_top1"]) for cell in cells)


def _grid_probe_path(model_directory: pathlib.Path, rate: float, holdout: str,
                     seed: int) -> pathlib.Path:
    return (model_directory / "grid" / f"lr{rate:g}" / f"holdout_{holdout}"
            / f"seed{seed}.safetensors")


def _completed(grid_path: pathlib.Path) -> set:
    """Cells already on disk, so a restart skips them."""
    if not grid_path.exists():
        return set()
    return {(row["holdout"], row["seed"], row["learning_rate"], row["batch_size"])
            for row in map(json.loads, filter(str.strip, grid_path.read_text().splitlines()))}


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
