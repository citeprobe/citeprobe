"""Step 1: generate answers and capture attention grids.

    uv run python -m scripts.capture --model phi4-mini                 # every corpus and split
    uv run python -m scripts.capture --model phi4-mini --corpus wice --split test

Writes captures/<model>/<corpus>/<split>/example_<index>.{npz,json}. Examples
already on disk are skipped, so an interrupted run resumes where it stopped,
and --start/--count shard a split across GPUs.
"""

import pathlib
import time
import traceback
from dataclasses import dataclass

import torch

from citeprobe.capture import BACKENDS, capture_example, load_model_and_tokenizer, save_capture
from citeprobe.data import CORPORA, CiteDataset, DatasetExample
from citeprobe.models import ModelSpec, get_model_spec

SPLITS = ["train", "val", "test"]

# Training examples per corpus. Three splits hold exactly this many; WiCE's
# holds 1000, of which the paper used the first 500.
TRAIN_EXAMPLES = 500

# The paper's prompt budget. Longer prompts are skipped, so this decides which
# examples a split contains.
MAX_PROMPT_TOKENS = 3200


@dataclass
class Job:
    """One split of one corpus to capture, and where its files go."""

    corpus: str
    split: str
    examples: list[DatasetExample]
    output_directory: pathlib.Path


@dataclass
class Settings:
    """What every job in a run shares."""

    backend: str
    max_prompt_tokens: int


def main(
    model: str,
    corpus: str = "",
    split: str = "",
    device: str = "cuda:0",
    backend: str = "fused",
    start: int = 0,
    count: int | None = None,
    captures: str = "captures",
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    seed: int = 42,
    max_gpu_memory: str = "",
) -> None:
    """Capture one model over the chosen corpora and splits (all of them by default).

    corpus and split take comma-separated names. backend is "fused" (the
    tiled pass of citeprobe.fused_kernel, never holding a T x T matrix) or
    "eager" (materializes attention, what the paper's captures used).
    Weights are automatically offloaded to CPU when GPU memory is insufficient.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}'; known: {sorted(BACKENDS)}")
    torch.manual_seed(seed)
    spec = get_model_spec(model)
    jobs = plan_jobs(spec, corpus, split, start, count, pathlib.Path(captures))
    settings = Settings(backend, max_prompt_tokens)

    model_instance, tokenizer = load_model_and_tokenizer(spec, device, max_gpu_memory)
    for job in jobs:
        run_job(model_instance, tokenizer, spec, job, settings)


def plan_jobs(
    spec: ModelSpec, corpus: str, split: str, start: int, count: int | None,
    captures: pathlib.Path,
) -> list[Job]:
    """Expand the corpus and split arguments into the list of jobs to run."""
    corpora = corpus.split(",") if corpus else CORPORA
    splits = split.split(",") if split else SPLITS
    jobs: list[Job] = []
    for name in corpora:
        for part in splits:
            examples = CiteDataset.load(name, part).examples
            if part == "train":
                examples = examples[:TRAIN_EXAMPLES]
            examples = examples[start:] if count is None else examples[start:start + count]
            jobs.append(Job(name, part, examples, captures / spec.key / name / part))
    return jobs


def run_job(model, tokenizer, spec: ModelSpec, job: Job, settings: Settings) -> None:
    """Capture every example of one job, skipping those already on disk."""
    print(f"\n{spec.key} {job.corpus}/{job.split}: {len(job.examples)} examples", flush=True)
    done = 0
    for example in job.examples:
        stem = job.output_directory / f"example_{example.dataset_index:05d}"
        if stem.with_suffix(".npz").exists():
            done += 1
            continue
        started = time.time()
        try:
            captured = capture_example(
                model, tokenizer, spec, example, settings.backend, settings.max_prompt_tokens
            )
        except Exception as error:
            print(f"[{example.dataset_index}] FAILED: {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            continue
        if captured is None:
            print(f"[{example.dataset_index}] skipped: prompt too long or nothing citable",
                  flush=True)
            continue
        record, capture = captured
        save_capture(job.output_directory, record, capture)
        done += 1
        print(f"[{example.dataset_index}] captured ({done}/{len(job.examples)}) "
              f"in {time.time() - started:.1f}s", flush=True)
    print(f"{job.corpus}/{job.split}: {done}/{len(job.examples)} examples in "
          f"{job.output_directory}", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
