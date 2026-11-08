"""Optional: exercise real Qwen generation + cache on both GPUs without model downloads."""

import pytest


def test_tiny_qwen_chunked_prefill_matches_single_gpu():
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    transformers = pytest.importorskip("transformers")
    from accelerate import dispatch_model

    from zaic.model import make_split_map

    torch.manual_seed(42)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [4, 6, 6],
        },
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3_5ForCausalLM(config).half().eval().to("cuda:0")
    inputs = torch.arange(3, 32, device="cuda:0").unsqueeze(0)
    options = {
        "max_new_tokens": 4,
        "do_sample": False,
        "logits_to_keep": 1,
        "use_cache": True,
        "cache_implementation": "dynamic",
    }
    try:
        with torch.inference_mode():
            reference = model.generate(inputs, **options)
        mapping = make_split_map(set(dict(model.named_modules())), 4, 2)
        model = dispatch_model(model, device_map=mapping)
        with torch.inference_mode():
            actual = model.generate(inputs, prefill_chunk_size=8, **options)
            # Repeat with a fresh cache to detect state leaked across calls.
            repeated = model.generate(inputs, prefill_chunk_size=8, **options)
        assert torch.equal(actual.to("cpu"), reference.to("cpu"))
        assert torch.equal(actual.to("cpu"), repeated.to("cpu"))
    finally:
        del model
        torch.cuda.empty_cache()


class CharTokenizer:
    """Char-level stand-in tokenizer with a simple chat template (no download)."""

    pad_token_id, eos_token_id = 0, 2
    all_special_tokens = ()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        from transformers import BatchEncoding

        ids = [3 + ord(c) % 125 for c in text]
        if return_tensors == "pt":
            import torch

            ids = torch.tensor([ids])
            return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return "".join(chr(97 + i % 26) for i in ids)


def tiny_qwen(transformers, torch, device, dtype):
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention"] * 2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return transformers.Qwen3_5ForCausalLM(config).to(dtype).eval().to(device)


def test_prefix_cache_generation_matches_full_prefill():
    """Reusing the system/turn prefix state must not change greedy output."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from zaic.model import HFBackend

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = tiny_qwen(transformers, torch, device, torch.float32)
    options = {"max_new_tokens": 5, "prefill_chunk_size": 16, "context_tokens": 4096}
    cached = HFBackend.from_loaded(model, CharTokenizer(), **options)
    plain = HFBackend.from_loaded(model, CharTokenizer(), prefix_cache=False, **options)
    system = {"role": "system", "content": "RULES " * 30}
    q1 = [system, {"role": "user", "content": "question one " * 5}]
    q1_step2 = q1 + [
        {"role": "assistant", "content": '{"action":"tool"}'},
        {"role": "user", "content": "Tool result: 42 " * 4},
    ]
    q2 = [system, {"role": "user", "content": "another question " * 4}]
    for messages, expect_cached in [
        (q1, len(system["content"]) + len("<system>")),  # first call builds the cache
        (q1_step2, None),
        (q2, None),
    ]:
        reply, usage = cached.generate(messages)
        reference, ref_usage = plain.generate(messages)
        assert reply == reference
        assert ref_usage["cached_tokens"] == 0
        assert usage["cached_tokens"] > 0
        assert cached.prefix_cache  # no fallback happened
    # The turn snapshot covers everything up to the last user message.
    assert usage["cached_tokens"] >= len("<system>" + system["content"])
    assert set(cached.prefix.entries) == {"system", "turn"}
