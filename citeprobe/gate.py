"""Which test examples the model answered correctly.

QA corpora use answer containment under the official HotpotQA normalization.
WiCE uses the LLM judge's verdicts in data/wice/judgements/.
"""

import json
import pathlib
import re
import string

from citeprobe.capture import CaptureRecord, load_record
from citeprobe.data import DATA_ROOT, CiteDataset, DatasetExample

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def answered_correctly(
    dataset: CiteDataset, model_key: str, capture_directory: pathlib.Path
) -> frozenset[int]:
    """dataset_index values of the captured examples answered correctly.

    Examples with no gold annotation are left out: there is nothing to score
    them against either way.
    """
    judged = _wice_verdicts(model_key) if dataset.corpus == "wice" else None
    correct: set[int] = set()
    for path in sorted(capture_directory.glob("example_*.json")):
        record = load_record(path)
        example = dataset.by_dataset_index(record.dataset_index)
        if not example.gold_unit_positions(example.partition()):
            continue
        if _is_correct(record, example, judged):
            correct.add(record.dataset_index)
    return frozenset(correct)


def _is_correct(record: CaptureRecord, example: DatasetExample, judged: dict[int, bool] | None) -> bool:
    """The judge's verdict when there is one, else containment of the reference answer."""
    if judged is not None:
        return judged.get(record.dataset_index, False)
    gold_answer = _normalize(example.answer)
    return bool(gold_answer) and gold_answer in _normalize(_cited_text(record))


def _cited_text(record: CaptureRecord) -> str:
    """The part of the response the probe scored."""
    return " ".join(record.response[start:end] for start, end in record.statement_char_ranges)


def _wice_verdicts(model_key: str) -> dict[int, bool]:
    """The judge's verdict per WiCE test example, for one model."""
    path = DATA_ROOT / "wice" / "judgements" / f"{model_key}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"WiCE gating needs {path}")
    verdicts: dict[int, bool] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            verdicts[row["dataset_index"]] = bool(row["correct"])
    return verdicts


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    lowered = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", lowered).split())
