"""Cite an LLM's own answer with CiteProbe.

    uv run python cite.py --model phi4-mini                       # smoke test on a built-in passage
    uv run python cite.py --model gemma4-12b --question "Who built it?" --context_file examples/sample-passage.txt
    uv run python cite.py --model phi4-mini --question "Who built it?" --context_file examples/sample-passage.txt \\
        --answer "Gustave Eiffel's company built it."

The model answers the question over the context, or is handed an answer
written elsewhere (surrogacy: a closed model answers, an open model
cites). One fused attention pass over the sequence gives every
context sentence its layer-by-head attention profile, and the probe turns
each profile into the probability that the sentence supports the answer.
"""

import pathlib
from dataclasses import dataclass

import numpy
import torch

from citeprobe import capture
from citeprobe.models import ModelSpec, get_model_spec
from citeprobe.probe import LinearProbe, standardize

# Used when no question and context are given: a one-command check that the
# install, the model download and the probe all work.
SMOKE_TEST_QUESTION = "Which structure overtook the Eiffel Tower as the tallest in the world, and when?"
SMOKE_TEST_PASSAGE = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower "
    "from 1887 to 1889.",
    "Locally nicknamed La dame de fer, it was constructed as the centerpiece of the 1889 "
    "World's Fair.",
    "The tower is 330 metres tall, about the same height as an 81-storey building.",
    "It was the tallest man-made structure in the world until the Chrysler Building in New York "
    "was finished in 1930.",
    "The tower has three levels for visitors, with restaurants on the first and second levels.",
    "Paris is also home to the Louvre, the world's most-visited museum.",
]


@dataclass
class LoadedModel:
    """An LLM on device, its tokenizer and the registry entry describing it."""

    spec: ModelSpec
    model: object
    tokenizer: object


@dataclass
class Citation:
    """One context sentence and the probe's probability that it supports the answer."""

    sentence: str
    probability: float


@dataclass
class CitedAnswer:
    """The model's answer and every context sentence, most likely citation first."""

    answer: str
    citations: list[Citation]


def main(
    model: str = "phi4-mini",
    question: str = "",
    context_file: str = "",
    answer: str = "",
    probe: str = "",
    device: str = "cuda:0",
    max_gpu_memory: str = "",
) -> None:
    """Answer a question over a context and print the sentences ranked as citations.

    context_file holds one sentence per line and needs its own question;
    without it the built-in smoke test question and passage are used.
    answer, when given, is cited in place of the model's own generation.
    probe defaults to the released probe for this model fitted with WiCE
    held out, so it was trained on the three QA corpora. max_gpu_memory
    overrides the automatic GPU budget (e.g. "18GiB").
    """
    probe_path = pathlib.Path(probe or f"weights/{model}/holdout_wice/seed0.safetensors")
    if context_file and not question:
        raise ValueError("--context_file needs a --question about it")
    question = question or SMOKE_TEST_QUESTION
    sentences = read_sentences(context_file) if context_file else SMOKE_TEST_PASSAGE

    loaded = load_model(model, device, max_gpu_memory)
    readout = LinearProbe.load(probe_path).to(device)
    result = cite(loaded, readout, question, sentences, answer or None)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result.answer}")
    print("\nContext sentences, most likely citation first:")
    for rank, citation in enumerate(result.citations, start=1):
        print(f"  {rank}. p={citation.probability:.3f}  {citation.sentence}")


def load_model(model_key: str, device: str, max_gpu_memory: str = "") -> LoadedModel:
    """Load a registry model from Hugging Face onto device."""
    spec = get_model_spec(model_key)
    model, tokenizer = capture.load_model_and_tokenizer(spec, device, max_gpu_memory)
    return LoadedModel(spec, model, tokenizer)


def cite(
    loaded: LoadedModel, probe: LinearProbe, question: str, sentences: list[str],
    answer: str | None = None,
) -> CitedAnswer:
    """Answer the question (or take the given answer) and score every sentence as its citation."""
    context = " ".join(sentences)
    prompt = capture.build_prompt(loaded.tokenizer, context, question)
    if answer is None:
        generation = capture.generate(loaded.model, loaded.tokenizer, loaded.spec, prompt)
    else:
        generation = capture.teacher_force(loaded.spec, prompt, answer)

    # The citable answer: the response, minus any reasoning segment and trailing special tokens.
    response = capture.response_text(loaded.tokenizer, generation)
    answer_range = capture.citable_char_range(response, loaded.tokenizer, loaded.spec)
    if answer_range is None:
        raise RuntimeError("the model produced nothing citable")
    cited = response[answer_range[0]:answer_range[1]]

    # Where the answer and each sentence sit in the token sequence, then one fused pass.
    spans = capture.compute_token_spans(
        loaded.tokenizer, generation, context, [answer_range], sentence_char_spans(sentences)
    )
    token_ids = loaded.tokenizer(generation.output_text, add_special_tokens=False)["input_ids"]
    grid = capture.capture_fused(loaded.model, token_ids, spans)

    probabilities = score_sentences(probe, grid.attention_sum)
    order = numpy.argsort(-probabilities)
    return CitedAnswer(
        answer=cited.strip(),
        citations=[Citation(sentences[index], float(probabilities[index])) for index in order],
    )


def score_sentences(probe: LinearProbe, attention_sum: numpy.ndarray) -> numpy.ndarray:
    """Equation 2 then Equation 3: standardize within the example, score each sentence.

    attention_sum is [layers, heads, 1 statement, sentences]; the result is
    one probability per sentence.
    """
    standardized = standardize(attention_sum)[:, :, 0, :]             # [L, H, sentences]
    grids = torch.from_numpy(numpy.moveaxis(standardized, -1, 0))     # [sentences, L, H]
    device = probe.linear.weight.device
    with torch.no_grad():
        logits = probe(grids.to(device))
    return torch.sigmoid(logits).cpu().numpy()


def sentence_char_spans(sentences: list[str]) -> list[list[int]]:
    """Char range of each sentence inside the context, which joins them with one space."""
    spans: list[list[int]] = []
    position = 0
    for sentence in sentences:
        spans.append([position, position + len(sentence)])
        position += len(sentence) + 1
    return spans


def read_sentences(path: str) -> list[str]:
    """One sentence per non-empty line."""
    return [line.strip() for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
