"""Save a ready-to-load model directory for offline runs (run once, with internet).

The default saves the text-only Qwen3.5-9B decoder already quantized to NF4 with
exactly the settings inference uses online (zaic.model.quantization_config), so
later runs read ~7 GB from disk instead of downloading ~19 GB and quantizing.
Pass --quantization none to save a text-only FP16 copy (~18 GB) instead.

    python scripts/prepare_model.py --output /kaggle/working/qwen3.5-9b-nf4 --verify
"""

import argparse
import datetime as dt
import importlib.metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MARKER = "zaic_model.json"


def versions():
    out = {}
    for pkg in ["torch", "transformers", "accelerate", "bitsandbytes"]:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def save(args):
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from zaic.model import model_class, quantization_config

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    cls, load_config = model_class(config)
    kwargs = {"dtype": torch.float16, "low_cpu_mem_usage": True}
    if args.quantization == "4bit":
        if not torch.cuda.is_available():
            raise SystemExit("4-bit quantization needs a CUDA GPU (bitsandbytes quantizes on GPU)")
        kwargs["quantization_config"] = quantization_config("4bit")
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = "cpu"
    started = time.perf_counter()
    model = cls.from_pretrained(
        args.model, config=load_config, revision=args.revision, **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    print(f"Loaded {args.model} as {cls.__name__} in {time.perf_counter() - started:.0f}s", flush=True)
    model.save_pretrained(args.output, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output)
    meta = {
        "source_model": args.model,
        "source_revision": args.revision or getattr(config, "_commit_hash", None),
        "quantization": args.quantization,
        "model_class": cls.__name__,
        "versions": versions(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (args.output / MARKER).write_text(json.dumps(meta, indent=2) + "\n")
    size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    print(f"Saved {args.output} ({size / 2**30:.2f} GiB)", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify(args):
    """Reload the saved directory exactly as inference does and generate once."""
    import torch

    from zaic.model import HFBackend

    started = time.perf_counter()
    backend = HFBackend(
        str(args.output),
        device_map="split" if torch.cuda.device_count() >= 2 else "cuda:0",
        context_tokens=4096,
        max_new_tokens=64,
        temperature=0,
    )
    print(f"Reloaded in {time.perf_counter() - started:.0f}s", flush=True)
    reply, usage = backend.generate(
        [
            {"role": "system", "content": "Return exactly one JSON object."},
            {"role": "user", "content": 'Reply with {"ok": true}.'},
        ]
    )
    print("Verification reply:", repr(reply), usage, flush=True)
    if not reply.strip():
        raise SystemExit("Saved model produced an empty reply")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--revision", help="Hub revision to pin (default: latest)")
    p.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify", action="store_true", help="Reload and generate once after saving")
    args = p.parse_args()
    if (args.output / MARKER).exists():
        print(f"{args.output} already prepared; skipping save", flush=True)
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise SystemExit(f"{args.output} is not empty; choose a new directory")
        args.output.mkdir(parents=True, exist_ok=True)
        save(args)
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
