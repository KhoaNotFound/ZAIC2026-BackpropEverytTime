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
