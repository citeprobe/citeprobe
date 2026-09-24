"""One generation pass, reduced to the attention grid of Equation 1.

The pipeline for one example is: build the chat prompt, let the model
answer, locate the citable answer and every source sentence in the token
sequence, then run one teacher-forced forward pass and reduce each head's
attention to a [statements, sentences] matrix. Two backends do that last
step. "eager" materializes every layer's [H, T, T] attention and reduces
it. "fused" accumulates the sentence sums inside a tiled online softmax
(citeprobe.fused_kernel) and never holds the T x T matrix, so long contexts
fit. Both give the same attention_sum, up to floating point.
"""

import functools
import json
import pathlib
from dataclasses import asdict, dataclass

import numpy
import torch
from accelerate.utils import get_max_memory
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

from citeprobe.data import DatasetExample
from citeprobe.fused_kernel import expand_key_values, fused_sentence_attention
from citeprobe.models import ModelSpec

PROMPT_TEMPLATE = "Context: {context}\n\nQuery: {query}"
EAGER_IMPLEMENTATION = "citeprobe_eager"


@dataclass
class Prompt:
    """The chat-templated prompt as text and as token ids."""

    text: str
    token_ids: list[int]


@dataclass
class Generation:
    """The model's full output: prompt text followed by its response.

    response_start is the number of prompt tokens, so the response occupies
    token positions [response_start, len(sequence)) of the tokenized output.
    """

    output_text: str
    response_start: int


@dataclass
class TokenSpans:
    """Token index sets inside the full teacher-forced sequence."""

    statement_token_sets: list[list[int]]
    source_token_sets: list[list[int]]


@dataclass
class AttentionGeometry:
    """The shape facts read off a model config.

    Hybrid models (Qwen3.5) interleave linear-attention layers that produce
    no attention matrix; full_attention_indices names the layers that do,
    and the rest stay zero in the [L, H, m, d] grid.
    """

    num_layers: int
    num_heads: int
    full_attention_indices: list[int]


@dataclass
class Capture:
    """Equation 1 for one example, every layer and head at once.

    attention_sum[l, h, i, j] is the mean over statement-i tokens of the
    summed attention paid to sentence-j tokens at head (l, h).
    """

    attention_sum: numpy.ndarray


@dataclass
class CaptureRecord:
    """The text side of one capture, saved as JSON beside the grid.

    statement_char_ranges are in response coordinates; the text they cover
    is what the probe scores, so it is also what answer correctness is
    judged on.
    """

    dataset_index: int
    model_key: str
    question: str
    context: str
    response: str
    sources: list[str]
    statements: list[str]
    statement_char_ranges: list[list[int]]
    prompt_token_count: int
    response_token_count: int


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


# GPU memory kept free for activations (peak measured: 3.5 GiB, Gemma 4 12B).
ACTIVATION_RESERVE_BYTES = 4 * 2**30


def load_model_and_tokenizer(spec: ModelSpec, device: str, max_gpu_memory: str = "") -> tuple:
    """Load a registry model in fp16 with SDPA attention, and its tokenizer.

    Generation runs on SDPA. The capture backends switch the attention
    implementation around their single forward pass and switch back.
    Weights that do not fit on the GPU are offloaded to CPU; max_gpu_memory
    (e.g. "18GiB") overrides the automatic budget.
    """
    tokenizer = load_tokenizer(spec)
    device_map, max_memory = _placement(device, max_gpu_memory)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec.huggingface_id, dtype=torch.float16, attn_implementation="sdpa",
            device_map=device_map, max_memory=max_memory,
        )
    except ValueError:
        # Vision-composite checkpoints (Ministral 3) load through the
        # image-text class; text-only use is unchanged.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            spec.huggingface_id, dtype=torch.float16, attn_implementation="sdpa",
            device_map=device_map, max_memory=max_memory,
        )
    model.eval()
    _report_offload(model)
    return model, tokenizer


def _placement(device: str, max_gpu_memory: str) -> tuple[str, dict[int | str, str | int] | None]:
    """device_map and max_memory: one GPU up to a budget, the rest on CPU."""
    target = torch.device(device)
    if target.type != "cuda":
        if max_gpu_memory:
            raise ValueError(f"max_gpu_memory offloads from a CUDA device, not '{device}'")
        return device, None
    gpu_index = target.index or 0
    if max_gpu_memory:
        budget: str | int = max_gpu_memory
    else:
        free_bytes, _ = torch.cuda.mem_get_info(gpu_index)
        budget = max(free_bytes - ACTIVATION_RESERVE_BYTES, 0)
    return "auto", {gpu_index: budget, "cpu": get_max_memory()["cpu"]}


def _report_offload(model) -> None:
    device_map = getattr(model, "hf_device_map", None) or {}
    offloaded = [name for name, placed in device_map.items() if placed == "cpu"]
    if offloaded:
        print(f"Offloaded {len(offloaded)} of {len(device_map)} modules to CPU (slower).")


def load_tokenizer(spec: ModelSpec):
    """The tokenizer with the model's chat-template quirks baked in."""
    if "Ministral" in spec.huggingface_id:
        # Mistral ships a tokenizer with a known regex bug; the flag fixes it.
        tokenizer = AutoTokenizer.from_pretrained(spec.huggingface_id, fix_mistral_regex=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.huggingface_id)
    if spec.template_thinking:
        tokenizer.apply_chat_template = functools.partial(
            tokenizer.apply_chat_template, enable_thinking=True
        )
    return tokenizer


# ----------------------------------------------------------------------------
# Prompting and generation
# ----------------------------------------------------------------------------


def build_prompt(tokenizer, context: str, question: str) -> Prompt:
    """One user turn holding the context and the question, chat templated."""
    content = PROMPT_TEMPLATE.format(context=context, query=question)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    return Prompt(text=text, token_ids=tokenizer.encode(text, add_special_tokens=False))


def generate(model, tokenizer, spec: ModelSpec, prompt: Prompt) -> Generation:
    """Greedy generation, with budget forcing for thinking models."""
    if spec.thinking:
        return _generate_with_thinking_budget(model, tokenizer, spec, prompt)
    return _generate_plain(model, tokenizer, spec, prompt)


def _generate_plain(model, tokenizer, spec: ModelSpec, prompt: Prompt) -> Generation:
    """One greedy pass; the prompt text is kept exactly as templated."""
    input_ids = torch.tensor([prompt.token_ids], device=model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=spec.max_new_tokens, do_sample=False)[0]
    # Decode the whole sequence and cut the prompt off by decoded length.
    raw_output = tokenizer.decode(output_ids)
    prompt_length = len(tokenizer.decode(prompt.token_ids))
    return Generation(prompt.text + raw_output[prompt_length:], len(prompt.token_ids))


def _generate_with_thinking_budget(model, tokenizer, spec: ModelSpec, prompt: Prompt) -> Generation:
    """Reason within thinking_budget tokens, then answer.

    If the model has not closed its reasoning with think_end_marker by the
    budget, the marker is injected and the answer is generated from the
    reasoning so far. The forced tokens become part of the response like any
    generated token.
    """
    if spec.thinking_budget is None or spec.think_end_marker is None:
        raise ValueError(f"{spec.key} is a thinking model without a budget or end marker")
    prompt_length = len(prompt.token_ids)
    input_ids = torch.tensor([prompt.token_ids], device=model.device)

    with torch.no_grad():
        reasoning = model.generate(input_ids, max_new_tokens=spec.thinking_budget, do_sample=False)
    reasoning_text = tokenizer.decode(reasoning[0, prompt_length:])
    finished = (
        reasoning[0, -1].item() == tokenizer.eos_token_id
        or reasoning.shape[1] - prompt_length < spec.thinking_budget
    )
    if spec.think_end_marker not in reasoning_text:
        forced_ids = tokenizer.encode(f"\n{spec.think_end_marker}\n\n", add_special_tokens=False)
        reasoning = torch.cat([reasoning, torch.tensor([forced_ids], device=model.device)], dim=1)
        finished = False

    sequence = reasoning
    if not finished:
        with torch.no_grad():
            sequence = model.generate(reasoning, max_new_tokens=spec.max_new_tokens, do_sample=False)
    return Generation(prompt.text + tokenizer.decode(sequence[0, prompt_length:]), prompt_length)


def teacher_force(spec: ModelSpec, prompt: Prompt, answer: str) -> Generation:
    """Place an answer the model did not write where its own answer would be.

    This is surrogacy: a closed model answers, an open model cites. A
    thinking model's citable text is whatever follows think_end_marker, so
    the foreign answer is given an empty reasoning segment. Qwen's chat
    template already ends the prompt with the opener, so only the closer is
    injected; Gemma writes its own opener, so both are. The prompt decides.
    """
    if not spec.thinking:
        return Generation(prompt.text + answer, len(prompt.token_ids))
    if spec.think_start_marker is None or spec.think_end_marker is None:
        raise ValueError(f"Thinking model {spec.key} is missing a think marker")
    already_open = prompt.text.rstrip().endswith(spec.think_start_marker)
    opener = "" if already_open else spec.think_start_marker
    response = f"{opener}{spec.think_end_marker}\n\n{answer}"
    return Generation(prompt.text + response, len(prompt.token_ids))


# ----------------------------------------------------------------------------
# Locating the answer and the sentences
# ----------------------------------------------------------------------------


def response_text(tokenizer, generation: Generation) -> str:
    """The response as text, cut at the first response token's char offset."""
    tokens = tokenizer(generation.output_text, add_special_tokens=False)
    start = tokens.token_to_chars(generation.response_start).start
    return generation.output_text[start:]


def citable_char_range(response: str, tokenizer, spec: ModelSpec) -> list[int] | None:
    """The answer's char range inside the response, as ONE citable statement.

    For thinking models only the text after think_end_marker is citable; the
    reasoning stays in the sequence but is not scored. The answer is cut at
    the first special token after that point and stripped. None means the
    example yields nothing to cite (reasoning never closed, or an empty
    answer).
    """
    answer_start = _answer_start(response, spec)
    if answer_start is None:
        return None
    answer_end = _first_special_token(response, answer_start, tokenizer, spec)
    citable = response[answer_start:answer_end]
    stripped = citable.strip()
    if len(stripped) < 2:
        return None
    offset = answer_start + citable.find(stripped)
    return [offset, offset + len(stripped)]


def _answer_start(response: str, spec: ModelSpec) -> int | None:
    """Where the answer begins: after the reasoning for thinking models, else 0."""
    if not spec.thinking:
        return 0
    if spec.think_end_marker is None:
        raise ValueError(f"Thinking model {spec.key} has no think_end_marker")
    marker_position = response.find(spec.think_end_marker)
    if marker_position == -1:
        return None
    return marker_position + len(spec.think_end_marker)


def _first_special_token(response: str, start: int, tokenizer, spec: ModelSpec) -> int:
    """Position of the first special token at or after start, else the response length."""
    special_strings = set(tokenizer.all_special_tokens)
    special_strings.update(str(token) for token in tokenizer.added_tokens_decoder.values())
    end = len(response)
    for special_token in special_strings:
        if spec.think_end_marker and special_token in spec.think_end_marker:
            continue
        position = response.find(special_token, start)
        if position != -1:
            end = min(end, position)
    return end


def compute_token_spans(
    tokenizer,
    generation: Generation,
    context: str,
    statement_char_ranges: list[list[int]],
    source_char_spans: list[list[int]],
) -> TokenSpans:
    """Map statement and source char spans to token index sets.

    Statement ranges are in response coordinates, source spans in context
    coordinates; both end up as positions in the tokenized output.
    """
    encoded = tokenizer(generation.output_text, add_special_tokens=False, return_offsets_mapping=True)
    return TokenSpans(
        statement_token_sets=_statement_token_sets(
            encoded, generation.response_start, statement_char_ranges),
        source_token_sets=_source_token_sets(
            encoded, generation.output_text, context, source_char_spans),
    )


def _statement_token_sets(encoded, response_start: int, char_ranges: list[list[int]]) -> list[list[int]]:
    """Response char ranges -> token positions, by walking token boundaries."""
    response_char_start = encoded.token_to_chars(response_start).start
    token_sets: list[list[int]] = []
    for start_char, end_char in char_ranges:
        first = _char_to_token(encoded, start_char + response_char_start)
        last = _char_to_token(encoded, end_char + response_char_start - 1) + 1
        token_sets.append(list(range(first, last)))
    return token_sets


def _char_to_token(encoded, char_index: int) -> int:
    """Index of the token whose chars contain char_index."""
    position = 0
    for position in range(len(encoded["input_ids"]) - 1):
        span = encoded.token_to_chars(position + 1)
        if span is not None and char_index < span.start:
            return position
    return position + 1


def _source_token_sets(encoded, output_text: str, context: str,
                       char_spans: list[list[int]]) -> list[list[int]]:
    """Context char spans -> the tokens overlapping them, via the offset mapping."""
    context_char_start = output_text.find(context)
    if context_char_start == -1:
        raise ValueError("Context string not found inside the chat prompt")
    offsets = encoded["offset_mapping"]
    token_sets: list[list[int]] = []
    for span_start, span_end in char_spans:
        start = context_char_start + span_start
        end = context_char_start + span_end
        token_sets.append([
            index for index, (offset_start, offset_end) in enumerate(offsets)
            if offset_start < end and offset_end > start
        ])
    return token_sets


# ----------------------------------------------------------------------------
# Reducing attention to the grid
# ----------------------------------------------------------------------------


def attention_geometry(model) -> AttentionGeometry:
    """Layer count, heads per layer and the full-attention layer set."""
    configuration = getattr(model.config, "text_config", model.config)
    num_layers = configuration.num_hidden_layers
    layer_types = getattr(configuration, "layer_types", None)
    if layer_types is None:
        full_attention_indices = list(range(num_layers))
    else:
        full_attention_indices = [
            index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
        ]
    return AttentionGeometry(
        num_layers=num_layers,
        num_heads=configuration.num_attention_heads,
        full_attention_indices=full_attention_indices,
    )


def _eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Plain eager attention that returns its softmax weights.

    Registered under one name for every model so the captured weights are
    the same softmax(QK^T / sqrt(d)) everywhere. The **kwargs is mandated by
    the transformers AttentionInterface signature.
    """
    if attention_mask is None and query.shape[2] > 1:
        raise ValueError("citeprobe_eager received no causal mask")
    key_states = expand_key_values(key, module.num_key_value_groups)
    value_states = expand_key_values(value, module.num_key_value_groups)
    weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    return output, weights


AttentionInterface.register(EAGER_IMPLEMENTATION, _eager_attention)
AttentionMaskInterface.register(EAGER_IMPLEMENTATION, ALL_MASK_ATTENTION_FUNCTIONS["eager"])


def capture_eager(model, token_ids: list[int], spans: TokenSpans) -> Capture:
    """Materialize every layer's attention and reduce it to the grid."""
    geometry = attention_geometry(model)
    statement_average, source_membership = _reduction_operators(spans, len(token_ids), model.device)

    attentions = _forward_with_attention(model, token_ids)
    layer_indices = _match_layer_indices(len(attentions), geometry)
    attention_sum = _empty_grid(geometry, spans)
    with torch.no_grad():
        for layer_index, attention in zip(layer_indices, attentions, strict=True):
            # [H, T, T] -> [H, m, T] (average over statement tokens) -> [H, m, d] (sum over sentence tokens)
            rows = torch.einsum("mt,htq->hmq", statement_average, attention[0].float())
            attention_sum[layer_index] = torch.einsum("hmq,qd->hmd", rows, source_membership).cpu().numpy()
    del attentions
    torch.cuda.empty_cache()
    return Capture(attention_sum)


def _reduction_operators(spans: TokenSpans, sequence_length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation 1 as two matrices: statement rows average, source columns sum."""
    statement_average = torch.zeros(len(spans.statement_token_sets), sequence_length, device=device)
    for statement_index, token_set in enumerate(spans.statement_token_sets):
        statement_average[statement_index, token_set] = 1.0 / len(token_set)
    source_membership = torch.zeros(sequence_length, len(spans.source_token_sets), device=device)
    for source_index, token_set in enumerate(spans.source_token_sets):
        source_membership[token_set, source_index] = 1.0
    return statement_average, source_membership


def _forward_with_attention(model, token_ids: list[int]) -> list[torch.Tensor]:
    """One teacher-forced pass under the eager implementation; one [1, H, T, T] map per layer."""
    input_ids = torch.tensor([token_ids], device=model.device)
    model.set_attn_implementation(EAGER_IMPLEMENTATION)
    try:
        with torch.no_grad():
            output = model(input_ids=input_ids, output_attentions=True, logits_to_keep=1)
    finally:
        model.set_attn_implementation("sdpa")
    return [entry for entry in output.attentions if entry is not None]


def capture_fused(model, token_ids: list[int], spans: TokenSpans, tile: int = 256) -> Capture:
    """The same grid through the fused pass, never holding a T x T matrix."""
    geometry = attention_geometry(model)
    grids = fused_sentence_attention(
        model, token_ids, spans.source_token_sets, spans.statement_token_sets, tile
    )
    layer_indices = _match_layer_indices(len(grids), geometry)
    attention_sum = _empty_grid(geometry, spans)
    for layer_index, grid in zip(layer_indices, grids, strict=True):
        attention_sum[layer_index] = grid.numpy()
    return Capture(attention_sum)


BACKENDS = {"eager": capture_eager, "fused": capture_fused}


# ----------------------------------------------------------------------------
# Putting it together, and files on disk
# ----------------------------------------------------------------------------


def capture_example(
    model, tokenizer, spec: ModelSpec, example: DatasetExample, backend: str,
    max_prompt_tokens: int,
) -> tuple[CaptureRecord, Capture] | None:
    """Prompt, generate and capture one dataset example.

    None when the example is skipped: the prompt exceeds the token budget,
    or the answer holds nothing citable.
    """
    partition = example.partition()
    prompt = build_prompt(tokenizer, partition.context, example.question)
    if len(prompt.token_ids) > max_prompt_tokens:
        return None
    generation = generate(model, tokenizer, spec, prompt)
    response = response_text(tokenizer, generation)
    char_range = citable_char_range(response, tokenizer, spec)
    if char_range is None:
        return None
    spans = compute_token_spans(
        tokenizer, generation, partition.context, [char_range], partition.source_char_spans()
    )
    token_ids = tokenizer(generation.output_text, add_special_tokens=False)["input_ids"]
    capture = BACKENDS[backend](model, token_ids, spans)
    record = CaptureRecord(
        dataset_index=example.dataset_index,
        model_key=spec.key,
        question=example.question,
        context=partition.context,
        response=response,
        sources=partition.parts,
        statements=[response[char_range[0]:char_range[1]]],
        statement_char_ranges=[char_range],
        prompt_token_count=len(prompt.token_ids),
        response_token_count=len(token_ids) - generation.response_start,
    )
    return record, capture


def save_capture(directory: pathlib.Path, record: CaptureRecord, capture: Capture) -> None:
    """Write example_<index>.npz (the grid) and .json (the text) to directory."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"example_{record.dataset_index:05d}"
    numpy.savez_compressed(stem.with_suffix(".npz"), attention_sum=capture.attention_sum)
    stem.with_suffix(".json").write_text(json.dumps(asdict(record), indent=2))


def load_capture(npz_path: pathlib.Path) -> Capture:
    """Read one saved grid."""
    with numpy.load(npz_path) as archive:
        return Capture(attention_sum=archive["attention_sum"])


def load_record(json_path: pathlib.Path) -> CaptureRecord:
    """Read one saved record, keeping only the CaptureRecord fields."""
    payload = json.loads(json_path.read_text())
    names = CaptureRecord.__dataclass_fields__
    return CaptureRecord(**{name: payload[name] for name in names})


def _match_layer_indices(count: int, geometry: AttentionGeometry) -> list[int]:
    """Which stack positions a per-attention-call sequence belongs to.

    Models emit one map per layer, or one per full-attention layer for
    hybrids whose other layers are attention-free.
    """
    if count == geometry.num_layers:
        return list(range(geometry.num_layers))
    if count == len(geometry.full_attention_indices):
        return list(geometry.full_attention_indices)
    raise ValueError(
        f"{count} attention tensors match neither {geometry.num_layers} layers nor "
        f"{len(geometry.full_attention_indices)} full-attention layers"
    )


def _empty_grid(geometry: AttentionGeometry, spans: TokenSpans) -> numpy.ndarray:
    return numpy.zeros(
        (
            geometry.num_layers,
            geometry.num_heads,
            len(spans.statement_token_sets),
            len(spans.source_token_sets),
        ),
        dtype=numpy.float32,
    )
