# CiteProbe

Sentence-level citations for RAG answers, decoded directly from a frozen LLM's attention in the same inference pass. For each source sentence, CiteProbe aggregates attention from answer tokens across all layers and heads into an `L × H` grid, standardizes it within the example, and scores it with a linear probe with one weight per head.

**No extra LLM passes.**

## Install

```bash
uv sync
```

Requires Python 3.12, PyTorch, and Transformers 5.14. Capturing attention requires a GPU.

For Qwen3.5, installing `flash-linear-attention` speeds up its linear-attention layers.

## Use CiteProbe

```bash
# Your question and context, one source sentence per line.
uv run python cite.py --model gemma4-12b --threshold 0.6 \
    --question "Who built the tower, and how tall is it? Answer in two sentences." \
    --context_file examples/sample-passage.txt
```

Outputs JSON:

```json
{
    "context": [
        "The Eiffel Tower is...",
        "It is named after...",
        "...more sentences..."
    ],
    "question": "Who built the tower, and how tall is it? Answer in two sentences.",
    "answer": "The tower built by Gustave Eiffel. It stands 330 metres tall.",
    "statements": [
        {
            "statement": "The tower was built by engineer Gustave Eiffel.",
            "citations": [{
                "context": 1, // refers to context[1]
                "confidence": 0.9981
            }, {
                "context": 0, 
                "confidence": 0.6088
            }]
        },
        {
            "statement": "It stands 330 metres tall.",
            // same as above for second statement in answer
        }
    ]
}
```

## Surrogate mode
```bash
# Surrogacy: cite an answer produced by any other model.
uv run python cite.py --model phi4-mini \
    --question "Who built it?" \
    --context_file examples/sample-passage.txt \
    --answer "Gustave Eiffel's company built it."
```

Models: `phi4-mini`, `ministral-8b`, `qwen3.5-9b`, `gemma4-12b`.

**Note:** Weights are automatically offloaded to CPU when GPU memory is insufficient.

## API

```bash
uv run python cite.py --model phi4-mini --api    # --host 127.0.0.1 --port 8000 by default
```

Call the API (`answer` optionally turns API into surrogate mode):

```bash
curl -X POST localhost:8000/cite -d '{
    "context": ["The Eiffel Tower is...", "It is named after...", "..."],
    "question": "Who built it?",
    "answer": "The company of Gustave Eiffel built it."
}'
```

## Additional options

- `--threshold` (default 0.7): lowest confidence a kept citation may have.
- `--top-k`: gives top-k citations. overrides --threshold.


Both work on the command line and as the server's defaults; a request can override them with `top_k` and `threshold`.

## Reproduce the probes

```bash
# 1. Download the paper's attention captures (2.4 GB).
uv run python -m scripts.download_captures
#    Or recapture them yourself (GPU required):
#    uv run python -m scripts.capture --model phi4-mini

# 2. Train probes into runs/. Skip this to use the released probes in weights/.
uv run python -m scripts.train --model phi4-mini

# 3. Evaluate.
uv run python -m scripts.evaluate --weights runs    # or --weights weights
```

This prints mAP and Top-1 per model and held-out corpus, over the test examples the model answered correctly: answer containment for the QA corpora, and the paper's LLM-judge verdicts in `data/wice/judgements/` for WiCE, so recaptured WiCE answers may differ on a few examples.

## Layout

```text
cite.py              load a model, attach a probe, cite
citeprobe/
    models.py        model definitions
    data.py          corpus loading
    capture.py       prompt, generate, reduce attention to per-head grids
    fused_kernel.py  compute the same grids without the full attention matrix
    probe.py         standardization, linear probe, training, scoring
    gate.py          determine which test examples were answered correctly
    metrics.py       mAP and Top-1
scripts/             capture.py, train.py, evaluate.py, download_captures.py
data/                train/val/test splits; model answers; WiCE answer-correctness verdicts
weights/             released probes by model, held-out corpus, and seed
examples/            sample passage for cite.py, one sentence per line
```

## Citation

```bibtex
@article{citeprobe2026,
  title  = {CiteProbe: From LLM Attention to Citations},
  author = {Anonymous},
  year   = {2026},
}
```
