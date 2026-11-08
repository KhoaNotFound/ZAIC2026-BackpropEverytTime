"""Check whether optional fast kernels are correct and faster on this GPU.

Builds a small random Qwen3.5 decoder (same layer types as the real model, no
download), times a long-prompt forward pass, and saves its last logits. Run it
once without flash-linear-attention installed (--save ref.pt) and once with it
(--compare ref.pt); the comparison prints one JSON line with "ok": true only if
the logits agree and the forward pass is faster.
"""

import argparse
import json
import time

SEQ_LEN = 4096


def run():
    import torch
    import transformers
    from transformers.utils import import_utils

    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 3, 3],
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3_5ForCausalLM(config).half().eval().to("cuda:0")
    ids = torch.randint(3, 1024, (1, SEQ_LEN), generator=torch.Generator().manual_seed(1)).to("cuda:0")
    with torch.inference_mode():
        model(ids, use_cache=True, logits_to_keep=8)  # warm-up / kernel compilation
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(3):
            logits = model(ids, use_cache=True, logits_to_keep=8).logits
        torch.cuda.synchronize()
    fla = getattr(import_utils, "is_flash_linear_attention_available", lambda: False)()
    return {
        "logits": logits.float().cpu(),
        "seconds": (time.perf_counter() - started) / 3,
        "fla": bool(fla),
    }


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", help="write reference result here")
    group.add_argument("--compare", help="compare against this reference result")
    args = p.parse_args()
    try:
        result = run()
    except Exception as exc:  # noqa: BLE001 -- report kernel failures as data
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}))
        return
    if args.save:
        torch.save(result, args.save)
        print(json.dumps({"ok": True, "seconds": result["seconds"], "fla": result["fla"]}))
        return
    ref = torch.load(args.compare)
    diff = (result["logits"] - ref["logits"]).abs().max().item()
    scale = ref["logits"].abs().max().item() or 1.0
    speedup = ref["seconds"] / result["seconds"]
    print(
        json.dumps(
            {
                "ok": result["fla"] and diff / scale < 2e-2 and speedup > 1.05,
                "fla": result["fla"],
                "relative_diff": diff / scale,
                "reference_seconds": ref["seconds"],
                "seconds": result["seconds"],
                "speedup": speedup,
            }
        )
    )


if __name__ == "__main__":
    main()
