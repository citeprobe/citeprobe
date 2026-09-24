"""Typed access to the four citation corpora (HotpotQA, MuSiQue, TyDi QA, WiCE).

Every split has one shape: each source paragraph is a list of sentences and
the gold annotations are (source_index, sentence_index) pairs into those
lists. The sentences are also the citation units, so gold indices and
score-vector indices line up by construction.
"""

import json
import pathlib
from dataclasses import dataclass

DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
CORPORA = ["hotpotqa", "musique", "tydiqa", "wice"]


@dataclass
class Source:
    """One context paragraph, kept as its original sentence list."""

    title: str
    text: list[str]
    is_gold: bool
    gold_sentence_indices: list[int]


@dataclass
class ContextPartition:
    """The flattened sentence units of one example's context.

    parts[k] is the k-th citation unit; unit_indices[k] is its
    (source_index, sentence_index) address; joining separators[k] + parts[k]
    in order reproduces the context string exactly.
    """

    context: str
    parts: list[str]
    separators: list[str]
    unit_indices: list[list[int]]

    def source_char_spans(self) -> list[list[int]]:
        """Char range of every unit inside the context string."""
        spans: list[list[int]] = []
        position = 0
        for separator, part in zip(self.separators, self.parts, strict=True):
            position += len(separator)
            spans.append([position, position + len(part)])
            position += len(part)
        return spans

    def unit_position(self, source_index: int, sentence_index: int) -> int:
        """Flat unit index of a (source, sentence) address."""
        return self.unit_indices.index([source_index, sentence_index])


@dataclass
class DatasetExample:
    """One question, its context paragraphs, the reference answer and gold."""

    dataset_index: int
    question: str
    answer: str
    gold_source_indices: list[list[int]]
    sources: list[Source]

    def partition(self) -> ContextPartition:
        """Flatten the sources into citation units.

        Sentences are stored stripped, so they join with a single space inside
        a paragraph; paragraphs join with a blank line.
        """
        parts: list[str] = []
        separators: list[str] = []
        unit_indices: list[list[int]] = []
        for source_index, source in enumerate(self.sources):
            for sentence_index, sentence in enumerate(source.text):
                if not parts:
                    separators.append("")
                elif sentence_index == 0:
                    separators.append("\n\n")
                else:
                    separators.append(" ")
                parts.append(sentence)
                unit_indices.append([source_index, sentence_index])
        context = "".join(
            separator + part for separator, part in zip(separators, parts, strict=True)
        )
        return ContextPartition(context, parts, separators, unit_indices)

    def gold_unit_positions(self, partition: ContextPartition) -> list[int]:
        """Flat unit indices of the gold sentences."""
        return [
            partition.unit_position(source_index, sentence_index)
            for source_index, sentence_index in self.gold_source_indices
        ]


class CiteDataset:
    """One split (train, val or test) of one corpus."""

    def __init__(self, corpus: str, split: str, examples: list[DatasetExample]) -> None:
        self.corpus = corpus
        self.split = split
        self.examples = examples
        self._by_dataset_index = {example.dataset_index: example for example in examples}

    @classmethod
    def load(cls, corpus: str, split: str) -> "CiteDataset":
        """Read data/<corpus>/<split>.json."""
        path = DATA_ROOT / corpus / f"{split}.json"
        payload = json.loads(path.read_text())
        examples = [
            DatasetExample(
                dataset_index=item["dataset_index"],
                question=item["question"],
                answer=item["answer"],
                gold_source_indices=item["gold_source_indices"],
                sources=[
                    Source(
                        title=source["title"],
                        text=source["text"],
                        is_gold=source["is_gold"],
                        gold_sentence_indices=source["gold_sentence_indices"],
                    )
                    for source in item["sources"]
                ],
            )
            for item in payload["examples"]
        ]
        return cls(corpus, split, examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def by_dataset_index(self, dataset_index: int) -> DatasetExample:
        return self._by_dataset_index[dataset_index]
