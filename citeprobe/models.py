"""The four LLMs the paper reports, and how each one is prompted.

Thinking models write a reasoning segment before the answer; the markers
tell the pipeline where the citable answer starts. The reasoning stays in
the teacher-forced sequence but is not cited.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One model.

    max_new_tokens is the answer budget. For thinking models thinking_budget
    caps the reasoning segment: if think_end_marker has not appeared by then
    it is injected and the model answers from the reasoning it has.

    template_thinking marks a chat template that defaults to thinking OFF and
    takes an enable_thinking flag (Gemma 4). Qwen's template thinks by default
    and ends the generation prompt with think_start_marker itself.
    """

    key: str
    huggingface_id: str
    max_new_tokens: int
    thinking: bool = False
    think_start_marker: str | None = None
    think_end_marker: str | None = None
    thinking_budget: int | None = None
    template_thinking: bool = False


MODELS = {
    "phi4-mini": ModelSpec(
        key="phi4-mini",
        huggingface_id="microsoft/Phi-4-mini-instruct",
        max_new_tokens=256,
    ),
    "ministral-8b": ModelSpec(
        key="ministral-8b",
        huggingface_id="mistralai/Ministral-3-8B-Instruct-2512-BF16",
        max_new_tokens=256,
    ),
    "qwen3.5-9b": ModelSpec(
        key="qwen3.5-9b",
        huggingface_id="Qwen/Qwen3.5-9B",
        max_new_tokens=512,
        thinking=True,
        think_start_marker="<think>",
        think_end_marker="</think>",
        thinking_budget=1024,
    ),
    "gemma4-12b": ModelSpec(
        key="gemma4-12b",
        huggingface_id="google/gemma-4-12B-it",
        max_new_tokens=256,
        thinking=True,
        think_start_marker="<|channel>thought",
        think_end_marker="<channel|>",
        thinking_budget=1024,
        template_thinking=True,
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    """The registry entry for a model key, with a helpful error otherwise."""
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Known: {sorted(MODELS)}")
    return MODELS[key]
