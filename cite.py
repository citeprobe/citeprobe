"""Cite an LLM's own answer with CiteProbe.

    uv run python cite.py --model phi4-mini                       # smoke test on a built-in passage
    uv run python cite.py --model gemma4-12b --question "Who built it?" --context_file examples/sample-passage.txt
    uv run python cite.py --model phi4-mini --question "Who built it?" --context_file examples/sample-passage.txt \\
        --answer "Gustave Eiffel's company built it."
    uv run python cite.py --model phi4-mini --api                 # HTTP server: POST /cite

The model answers the question over the context, or is handed an answer
written elsewhere (surrogacy: a closed model answers, an open model
cites). The answer is split into sentences. One fused attention pass over
the sequence gives every context sentence its layer-by-head attention
profile from each answer sentence, and the probe turns each profile into
the confidence that the context sentence supports that answer sentence.
"""

import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

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
    """One context sentence and the probe's confidence that it supports a statement."""

    context: int  # position of the sentence in the context list, from 0
    confidence: float


@dataclass
class CitedStatement:
    """One answer sentence and every context sentence, most likely citation first."""

    statement: str
    citations: list[Citation]


@dataclass
class CitedAnswer:
    """The model's answer, cited sentence by sentence in answer order."""

    answer: str
    statements: list[CitedStatement]


def main(
    model: str = "phi4-mini",
    question: str = "",
    context_file: str = "",
    answer: str = "",
    probe: str = "",
    device: str = "cuda:0",
    max_gpu_memory: str = "",
    threshold: float = 0.7,
    top_k: int | None = None,
    api: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Answer a question over a context and print the cited answer as JSON.

    context_file holds one sentence per line and needs its own question;
    without it the built-in smoke test question and passage are used.
    answer, when given, is cited in place of the model's own generation.
    probe defaults to the released probe for this model fitted with WiCE
    held out, so it was trained on the three QA corpora. max_gpu_memory
    overrides the automatic GPU budget (e.g. "18GiB"). Each answer sentence
    keeps the citations with confidence at least threshold, or, when top_k
    is given, its top_k citations whatever their confidence. api serves POST /cite on
    host:port instead of answering once.
    """
    probe_path = pathlib.Path(probe or f"weights/{model}/holdout_wice/seed0.safetensors")
    if api:
        loaded = load_model(model, device, max_gpu_memory)
        readout = LinearProbe.load(probe_path).to(device)
        serve(loaded, readout, host, port, threshold, top_k)
        return
    if context_file and not question:
        raise ValueError("--context_file needs a --question about it")
    question = question or SMOKE_TEST_QUESTION
    sentences = read_sentences(context_file) if context_file else SMOKE_TEST_PASSAGE

    loaded = load_model(model, device, max_gpu_memory)
    readout = LinearProbe.load(probe_path).to(device)
    result = cite(loaded, readout, question, sentences, answer or None)
    output = to_json(sentences, question, result, threshold, top_k)
    print(json.dumps(output, indent=2, ensure_ascii=False))


def to_json(
    sentences: list[str], question: str, result: "CitedAnswer", threshold: float,
    top_k: int | None,
) -> dict:
    """The response: context, question, answer, and each statement's kept citations."""
    return {
        "context": sentences,
        "question": question,
        "answer": result.answer,
        "statements": [
            {"statement": statement.statement,
             "citations": [asdict(citation)
                           for citation in keep_citations(statement.citations, threshold, top_k)]}
            for statement in result.statements
        ],
    }


def keep_citations(
    citations: list["Citation"], threshold: float, top_k: int | None
) -> list["Citation"]:
    """The top_k citations when top_k is given, else those with confidence >= threshold."""
    if top_k is not None:
        return citations[:top_k]
    return [citation for citation in citations if citation.confidence >= threshold]


def serve(
    loaded: "LoadedModel", probe: LinearProbe, host: str, port: int, threshold: float,
    top_k: int | None,
) -> None:
    """POST /cite with {"context", "question", and optionally "answer", "top_k", "threshold"}.

    Without "answer" the model answers; with it, that answer is cited
    (surrogate mode). One request at a time: there is one model on one GPU.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/cite":
                self._send(404, {"error": "not found; use POST /cite"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                sentences, question = body["context"], body["question"]
                answer = body.get("answer") or None
                if (not isinstance(sentences, list) or not sentences
                        or not all(isinstance(sentence, str) and sentence.strip() for sentence in sentences)):
                    raise ValueError('"context" must be a non-empty list of non-empty sentences')
                if not isinstance(question, str) or not question.strip():
                    raise ValueError('"question" must be a non-empty string')
                # A request's own top_k or threshold wins over the server's defaults.
                if body.get("top_k") is not None:
                    limit = int(body["top_k"])
                elif "threshold" in body:
                    limit = None
                else:
                    limit = top_k
                minimum = float(body.get("threshold", threshold))
            except KeyError as error:
                self._send(400, {"error": f"missing field {error}"})
                return
            except (ValueError, TypeError) as error:
                self._send(400, {"error": f"bad request: {error}"})
                return
            result = cite(loaded, probe, question, sentences, answer)
            self._send(200, to_json(sentences, question, result, minimum, limit))

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:
            print(f"{self.address_string()} {format % args}", file=sys.stderr)

    server = HTTPServer((host, port), Handler)
    print(f"Serving POST http://{host}:{port}/cite", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def load_model(model_key: str, device: str, max_gpu_memory: str = "") -> LoadedModel:
    """Load a registry model from Hugging Face onto device."""
    spec = get_model_spec(model_key)
    model, tokenizer = capture.load_model_and_tokenizer(spec, device, max_gpu_memory)
    return LoadedModel(spec, model, tokenizer)


def cite(
    loaded: LoadedModel, probe: LinearProbe, question: str, sentences: list[str],
    answer: str | None = None,
) -> CitedAnswer:
    """Answer the question (or take the given answer) and score every sentence as its citation.

    An answer with nothing citable comes back with no statements.
    """
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
        # Nothing citable: the reasoning never closed, or the answer is empty.
        return CitedAnswer(answer="", statements=[])
    cited = response[answer_range[0]:answer_range[1]]

    # Each answer sentence is one statement; all of them share one fused pass.
    statement_ranges = capture.split_statements(response, answer_range)
    spans = capture.compute_token_spans(
        loaded.tokenizer, generation, context, statement_ranges, sentence_char_spans(sentences)
    )
    token_ids = loaded.tokenizer(generation.output_text, add_special_tokens=False)["input_ids"]
    grid = capture.capture_fused(loaded.model, token_ids, spans)

    probabilities = score_sentences(probe, grid.attention_sum)    # [answer sentences, sentences]
    return CitedAnswer(
        answer=cited.strip(),
        statements=[
            CitedStatement(response[start:end], rank_citations(row))
            for (start, end), row in zip(statement_ranges, probabilities, strict=True)
        ],
    )


def rank_citations(probabilities: numpy.ndarray) -> list[Citation]:
    """Every context sentence with its confidence, most likely citation first."""
    return [
        Citation(int(index), round(float(probabilities[index]), 4))
        for index in numpy.argsort(-probabilities)
    ]


def score_sentences(probe: LinearProbe, attention_sum: numpy.ndarray) -> numpy.ndarray:
    """Equation 2 then Equation 3: standardize within the example, score each sentence.

    attention_sum is [layers, heads, statements, sentences]; the result is
    [statements, sentences] probabilities. Standardization runs over the
    sentence axis, so each statement is standardized on its own.
    """
    standardized = standardize(attention_sum)                         # [L, H, m, d]
    grids = torch.from_numpy(numpy.moveaxis(standardized, (2, 3), (0, 1)))  # [m, d, L, H]
    device = probe.linear.weight.device
    with torch.no_grad():
        logits = probe(grids.reshape(-1, *grids.shape[2:]).to(device))
    return torch.sigmoid(logits).reshape(grids.shape[:2]).cpu().numpy()


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


def normalize_arguments(arguments: list[str]) -> list[str]:
    """Accept --top-k for --top_k, and a bare --api for --api true (jsonargparse wants both spelled out)."""
    normalized: list[str] = []
    for index, argument in enumerate(arguments):
        if argument.startswith("--"):
            name, equals, value = argument.partition("=")
            argument = name.replace("-", "_").replace("__", "--", 1) + equals + value
        if argument == "--api" and arguments[index + 1:index + 2] not in (["true"], ["false"]):
            argument = "--api=true"
        normalized.append(argument)
    return normalized


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False, args=normalize_arguments(sys.argv[1:]))
