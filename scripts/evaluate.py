"""Step 3: score the probes on the held-out test splits and print the main results rows.

    uv run python -m scripts.evaluate                     # released weights
    uv run python -m scripts.evaluate --weights runs      # probes you trained

Each corpus is scored by the probes that held it out, one per seed, on the
test examples the model answered correctly; the three seeds are averaged.
"""

import pathlib
import statistics
from dataclasses import dataclass

import torch

from citeprobe.data import CORPORA, CiteDataset
from citeprobe.gate import answered_correctly
from citeprobe.probe import LinearProbe, build_pairs, evaluate

LABELS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "tydiqa": "TyDiQA", "wice": "WiCE"}


@dataclass
class Cell:
    """Seed-averaged scores of one model on one held-out corpus."""

    map: float
    top1: float
    examples: int


def main(
    captures: str = "captures",
    weights: str = "weights",
    models: str = "",
    device: str = "",
) -> None:
    """models narrows the run to a comma-separated subset of the weight folders."""
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_root = pathlib.Path(weights)
    chosen = models.split(",") if models else sorted(
        path.name for path in weights_root.iterdir() if (path / "recipe.json").exists())
    header = "| Model | " + " | ".join(f"{LABELS[c]} mAP | {LABELS[c]} Top-1" for c in CORPORA) \
        + " | Avg mAP | Avg Top-1 |"
    print(header)
    print("|" + "---|" * (2 * len(CORPORA) + 3))
    for model in chosen:
        cells = [score_cell(model, corpus, pathlib.Path(captures), weights_root, device)
                 for corpus in CORPORA]
        numbers = "".join(f" {cell.map:.3f} | {cell.top1:.3f} |" for cell in cells)
        print(f"| {model} |{numbers} {statistics.mean(c.map for c in cells):.3f} | "
              f"{statistics.mean(c.top1 for c in cells):.3f} |", flush=True)


def score_cell(model: str, corpus: str, captures: pathlib.Path, weights: pathlib.Path,
               device: str) -> Cell:
    """Average the held-out probes of every seed on the gated test split."""
    dataset = CiteDataset.load(corpus, "test")
    test_directory = captures / model / corpus / "test"
    keep = answered_correctly(dataset, model, test_directory)
    pairs = build_pairs(test_directory, dataset, keep)
    probe_paths = sorted((weights / model / f"holdout_{corpus}").glob("seed*.safetensors"))
    if not probe_paths:
        raise FileNotFoundError(f"no probes under {weights / model / f'holdout_{corpus}'}")
    scores = [evaluate(LinearProbe.load(path).to(device), pairs, device) for path in probe_paths]
    return Cell(
        map=statistics.mean(score.map for score in scores),
        top1=statistics.mean(score.top1 for score in scores),
        examples=scores[0].statements,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
