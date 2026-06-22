# CS336 Assignment 1 — Implementation Plan

Branch: `assignment1-impl`

## How this repo works

- Implementation code lives in `cs336_basics/` (currently only `__init__.py` and `pretokenization_example.py`).
- `tests/adapters.py` is the bridge: every test calls an adapter function, which currently `raise NotImplementedError`.
- Workflow: write implementations in `cs336_basics/`, then wire each adapter to call them.
- Run a single test file: `uv run pytest tests/test_xxx.py -v`
- Run everything: `uv run pytest`

## Implementation steps (suggested order)

Each step lists: adapter function → test file → what to figure out → notes.

### Stage 1 — Basic ops

1. **`run_softmax`** → `test_nn_utils.py::test_softmax_matches_pytorch`
   - Numerically stable softmax (subtract row max).
   - Arbitrary input dims.

2. **`run_cross_entropy`** → `test_nn_utils.py::test_cross_entropy`
   - log-softmax → gather targets → negative → mean.
   - Numerical stability required.

3. **`run_silu`** → `test_model.py::test_silu_matches_pytorch`
   - `x * sigmoid(x)`.

### Stage 2 — Simple model components

4. **`run_linear`** → `test_model.py::test_linear`
   - `in_features @ weights.T`, weights are `(d_out, d_in)`.

5. **`run_embedding`** → `test_model.py::test_embedding`
   - Index rows of `weights` by `token_ids`.

6. **`run_rmsnorm`** → `test_model.py::test_rmsnorm`
   - `x * weight / sqrt(mean(x^2) + eps)` over last dim.

### Stage 3 — Transformer sub-components

7. **`run_scaled_dot_product_attention`** → `test_model.py::test_scaled_dot_product_attention`
   - `softmax(QK^T / sqrt(d_k) + mask) @ V`.
   - Bool mask: True positions are masked (set to -inf).
   - Must support 3D and 4D inputs.

8. **`run_rope`** → `test_model.py::test_rope`
   - Rotary position embedding on last dim, pairwise rotation.
   - Frequencies `1 / theta^(2i/d_k)`.

9. **`run_swiglu`** → `test_model.py::test_swiglu`
   - `SiLU(x @ W1.T) * (x @ W3.T) @ W2.T`.

10. **`run_multihead_self_attention`** → `test_model.py::test_multihead_self_attention`
    - One matmul for all Q/K/V projections.
    - Reshape `(batch, seq, d_model)` → `(batch, heads, seq, d_k)`.
    - No RoPE, no causal mask (test uses random mask).

11. **`run_multihead_self_attention_with_rope`** → `test_model.py::test_multihead_self_attention_with_rope`
    - Same as 10, apply RoPE to Q and K (not V) after projection.

### Stage 4 — Full model

12. **`run_transformer_block`** → `test_model.py::test_transformer_block`
    - Pre-norm: `x = x + attn(rmsnorm1(x))`; `x = x + ffn(rmsnorm2(x))`.
    - Weight dict key mapping (see adapter docstring).

13. **`run_transformer_lm`** → `test_model.py::test_transformer_lm` + `test_transformer_lm_truncated_input`
    - embedding → N blocks → rmsnorm → lm_head.
    - Handle truncated (shorter) input.

### Stage 5 — Training infrastructure

14. **`run_get_lr_cosine_schedule`** → `test_optimizer.py::test_get_lr_cosine_schedule`
    - Linear warmup 0→max_lr; then cosine max_lr→min_lr.
    - `cosine_cycle_iters` is total period.

15. **`run_gradient_clipping`** → `test_nn_utils.py::test_gradient_clipping`
    - Global L2 norm over all grads; scale if exceeds max.
    - Skip params with `grad is None`.

16. **`get_adamw_cls`** → `test_optimizer.py::test_adamw`
    - Subclass `torch.optim.Optimizer`.
    - Weight decay applied before gradient update (per handout).
    - Test accepts match with PyTorch AdamW OR course snapshot.

17. **`run_get_batch`** → `test_data.py::test_get_batch`
    - Sample `batch_size` start indices in `[0, len - context_length)`.
    - `x = dataset[start : start+ctx]`, `y = dataset[start+1 : start+ctx+1]`.
    - Honor `device` string (invalid device should error).

### Stage 6 — Serialization

18. **`run_save_checkpoint` / `run_load_checkpoint`** → `test_serialization.py::test_checkpointing`
    - Save `{model, optimizer, iteration}` via `torch.save`.
    - Load with `map_location`, restore state dicts, return iteration.

### Stage 7 — BPE tokenizer (hardest)

19. **`run_train_bpe`** → `test_train_bpe.py`
    - Read corpus → pre-tokenize with GPT-2 regex (use `regex` module) → count pairs → iteratively merge.
    - Handle special tokens (never merged/split).
    - Speed: must finish small dataset in <1.5s (needs heap/index optimization).
    - `find_chunk_boundaries` in `pretokenization_example.py` supports parallelization.

20. **`get_tokenizer`** → `test_tokenizer.py` (~20 tests)
    - `encode(text) -> list[int]`, `decode(ids) -> str`, `encode_iterable(file) -> Iterator[int]`.
    - Split on special tokens first, then BPE the rest.
    - Match tiktoken GPT-2 encoding.
    - `encode_iterable` must be memory-bounded (<1MB extra).

## Order summary table

| # | Function | Difficulty | Depends on |
|---|----------|-----------|------------|
| 1 | run_softmax | ⭐ | — |
| 2 | run_cross_entropy | ⭐ | 1 |
| 3 | run_silu | ⭐ | — |
| 4 | run_linear | ⭐ | — |
| 5 | run_embedding | ⭐ | — |
| 6 | run_rmsnorm | ⭐ | — |
| 7 | run_scaled_dot_product_attention | ⭐⭐ | 1 |
| 8 | run_rope | ⭐⭐ | — |
| 9 | run_swiglu | ⭐⭐ | 4, 3 |
| 10 | run_multihead_self_attention | ⭐⭐⭐ | 7 |
| 11 | run_multihead_self_attention_with_rope | ⭐⭐⭐ | 10, 8 |
| 12 | run_transformer_block | ⭐⭐⭐ | 11, 9, 6 |
| 13 | run_transformer_lm | ⭐⭐ | 12, 5, 6, 4 |
| 14 | run_get_lr_cosine_schedule | ⭐ | — |
| 15 | run_gradient_clipping | ⭐ | — |
| 16 | get_adamw_cls | ⭐⭐⭐ | — |
| 17 | run_get_batch | ⭐ | — |
| 18 | checkpoint save/load | ⭐⭐ | 16 |
| 19 | run_train_bpe | ⭐⭐⭐⭐ | — |
| 20 | get_tokenizer | ⭐⭐⭐⭐⭐ | 19 |

## Progress tracking

Tick these off as tests pass:

- [ ] 1. run_softmax
- [ ] 2. run_cross_entropy
- [ ] 3. run_silu
- [ ] 4. run_linear
- [ ] 5. run_embedding
- [ ] 6. run_rmsnorm
- [ ] 7. run_scaled_dot_product_attention
- [ ] 8. run_rope
- [ ] 9. run_swiglu
- [ ] 10. run_multihead_self_attention
- [ ] 11. run_multihead_self_attention_with_rope
- [ ] 12. run_transformer_block
- [ ] 13. run_transformer_lm
- [ ] 14. run_get_lr_cosine_schedule
- [ ] 15. run_gradient_clipping
- [ ] 16. get_adamw_cls
- [ ] 17. run_get_batch
- [ ] 18. checkpoint save/load
- [ ] 19. run_train_bpe
- [ ] 20. get_tokenizer
