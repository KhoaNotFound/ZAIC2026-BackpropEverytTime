"""Lazy GPU backend for any Hugging Face causal LM with a chat template.

Importing the CLI/evaluator never downloads a model. Qwen3.5 checkpoints are
multimodal, so their text-only decoder is loaded with Qwen3_5ForCausalLM; every
other model goes through AutoModelForCausalLM.
"""

import copy
import time

from .formats import first_json_object, strip_reasoning


class ContextLimitError(RuntimeError):
    """Prompt plus output budget exceeds --context-tokens for one question.

    Unlike OOM this leaves the model usable, so the run continues.
    """


def make_split_map(module_names, num_layers, gpu_count, tied_embeddings=False):
    """Place whole decoder blocks contiguously across GPUs, never split a block."""
    if gpu_count < 2:
        raise ValueError(
            "split requires at least two visible GPUs; use --device-map auto for one GPU"
        )
    if gpu_count > num_layers:
        raise ValueError("More GPUs than decoder layers")
    candidates = [
        name
        for name in module_names
        if name.endswith(".layers")
        and all(f"{name}.{i}" in module_names for i in range(num_layers))
    ]
    if len(candidates) != 1:
        raise ValueError("Cannot identify a unique decoder stack for this checkpoint")
    stack = candidates[0]
    parent = stack.rsplit(".", 1)[0]
    # Cover actual text-model modules explicitly instead of using a root-wide map.
    mapping = {f"{parent}.embed_tokens": 0}
    if f"{parent}.rotary_emb" in module_names:
        mapping[f"{parent}.rotary_emb"] = 0
    for i in range(num_layers):
        mapping[f"{stack}.{i}"] = min(i * gpu_count // num_layers, gpu_count - 1)
    norm = f"{parent}.norm"
    if norm in module_names:
        mapping[norm] = gpu_count - 1
    if "lm_head" in module_names:
        mapping["lm_head"] = 0
    return mapping


def reply_complete(text, thinking):
    """True once the reply holds a complete action JSON object (generation can stop)."""
    if thinking and "</think>" not in text:
        return False
    return first_json_object(strip_reasoning(text)) is not None


def make_json_stop(tokenizer, prompt_tokens, thinking):
    """StoppingCriteria that ends generation after the first complete JSON object.

    Without it a finished action is followed by runaway text up to max_new_tokens,
    which costs minutes per step and corrupts the parse.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class JsonStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            done = False
            last = tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True)
            if "}" in last:
                text = tokenizer.decode(
                    input_ids[0, prompt_tokens:], skip_special_tokens=True
                )
                done = reply_complete(text, thinking)
            return torch.full(
                (input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device
            )

    return StoppingCriteriaList([JsonStop()])


def model_class(config):
    """Return (causal LM class, config to load it with) for a checkpoint config."""
    from transformers import AutoModelForCausalLM

    if getattr(config, "model_type", None) in ("qwen3_5", "qwen3_5_text"):
        from transformers import Qwen3_5ForCausalLM

        # Load only the language model of the multimodal checkpoint.
        return Qwen3_5ForCausalLM, getattr(config, "text_config", config)
    return AutoModelForCausalLM, config


def quantization_config(quantization):
    """bitsandbytes settings, shared by online loading and scripts/prepare_model.py."""
    import torch
    from transformers import BitsAndBytesConfig

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def is_prequantized(config):
    """True for checkpoints saved already quantized (e.g. by scripts/prepare_model.py)."""
    return any(
        getattr(c, "quantization_config", None)
        for c in (config, getattr(config, "text_config", None))
        if c is not None
    )


def chat_text(tokenizer, messages, thinking):
    """Apply the model's chat template; fold the system prompt into the first user
    turn for templates that reject a system role."""
    options = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking}
    try:
        return tokenizer.apply_chat_template(messages, **options)
    except Exception:
        if not messages or messages[0]["role"] != "system" or len(messages) < 2:
            raise
        system, first, *rest = messages
        merged = {"role": "user", "content": system["content"] + "\n\n" + first["content"]}
        return tokenizer.apply_chat_template([merged, *rest], **options)


class PrefixCache:
    """Reuse the model state of prompt prefixes that were already processed.

    Each agent step resends the whole conversation, and the system prompt (with
    the rulebook) is identical for every question, so without reuse every step
    re-prefills 10-25k tokens. Hybrid linear-attention caches (Qwen3.5) cannot be
    cropped back to a prefix, so the state is snapshotted at message boundaries
    ("system": end of the system message, shared by all questions; "turn": end of
    the latest user message) and deep-copied before generation, which mutates it.
    """

    def __init__(self, model, chunk_size):
        self.model = model
        self.chunk_size = chunk_size
        self.entries = {}  # key -> (token id list, cache)

    def clear(self):
        self.entries.clear()

    def extend(self, cache, ids, start, stop):
        """Feed ids[start:stop] through the model, appending to cache (None = new)."""
        import torch

        device = self.model.get_input_embeddings().weight.device
        step = self.chunk_size or max(stop - start, 1)
        for i in range(start, stop, step):
            chunk = torch.tensor([ids[i : min(i + step, stop)]], device=device)
            out = self.model(
                input_ids=chunk, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            cache = out.past_key_values
        return cache

    def state_for(self, boundaries):
        """Return the cache for the last boundary, building missing ones from the
        longest stored prefix. boundaries: [(key, ids)] with increasing length."""
        cache = None
        for key, ids in boundaries:
            base_key, base_ids, base = None, [], None
            for k, (k_ids, k_cache) in self.entries.items():
                n = len(k_ids)
                if n > len(base_ids) and n <= len(ids) and ids[:n] == k_ids:
                    base_key, base_ids, base = k, k_ids, k_cache
            if len(base_ids) == len(ids):
                cache = base
            else:
                # Extend in place only an entry this boundary supersedes.
                if base is not None and base_key != key:
                    base = copy.deepcopy(base)
                self.entries.pop(key, None)
                cache = self.extend(base, ids, len(base_ids), len(ids))
            self.entries[key] = (ids, cache)
        return cache


class HFBackend:
    def __init__(
        self,
        model_id,
        quantization="4bit",
        device_map="split",
        context_tokens=16384,
        max_new_tokens=2048,
        thinking=False,
        gpu_memory_gib=13,
        prefill_chunk_size=512,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        prefix_cache=True,
    ):
        import torch
        from transformers import AutoConfig, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required. Enable T4 GPUs in Kaggle settings.")
        self.torch = torch
        self.context_tokens = context_tokens
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.model_id = model_id
        self.prefill_chunk_size = prefill_chunk_size or None
        self.prefix_cache = prefix_cache
        if temperature > 0:
            self.sampling = {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        else:
            self.sampling = {"do_sample": False}
        self.sampling["repetition_penalty"] = repetition_penalty
        config = AutoConfig.from_pretrained(model_id)
        cls, load_config = model_class(config)
        layer_config = getattr(load_config, "text_config", load_config)
        if device_map == "split":
            from accelerate import init_empty_weights

            with init_empty_weights():
                empty = (
                    cls._from_config(load_config)
                    if hasattr(cls, "_from_config")
                    else cls.from_config(load_config)
                )
            names = set(dict(empty.named_modules()))
            device_map = make_split_map(
                names,
                layer_config.num_hidden_layers,
                torch.cuda.device_count(),
                bool(
                    getattr(config, "tie_word_embeddings", False)
                    or getattr(layer_config, "tie_word_embeddings", False)
                ),
            )
            del empty
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        kwargs = {
            "device_map": device_map,
            "dtype": torch.float16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if isinstance(device_map, str) and device_map in {
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        }:
            kwargs["max_memory"] = {
                i: f"{gpu_memory_gib}GiB" for i in range(torch.cuda.device_count())
            }
            kwargs["max_memory"]["cpu"] = "8GiB"
        if is_prequantized(config):
            # Weights were quantized when saved; loading skips quantization entirely.
            print("Loading pre-quantized checkpoint; --quantization is ignored", flush=True)
            self.quantization = "prequantized"
        else:
            self.quantization = quantization
            if quantization_config(quantization) is not None:
                kwargs["quantization_config"] = quantization_config(quantization)
        # For Qwen3.5 the text-only loader maps language-model weights from the
        # multimodal checkpoint without instantiating the unused vision tower.
        self.model = cls.from_pretrained(model_id, config=load_config, **kwargs).eval()
        self.device_map = {
            str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()
        }
        print("Model device map:", self.device_map, flush=True)
        print(
            "GPU memory after loading (GiB):",
            {
                i: round(torch.cuda.memory_allocated(i) / 2**30, 2)
                for i in range(torch.cuda.device_count())
            },
            flush=True,
        )
        self.prefix = PrefixCache(self.model, self.prefill_chunk_size)
        self.reset_peak()

    @classmethod
    def from_loaded(cls, model, tokenizer, **options):
        """Wrap an already-loaded model (tests, custom loaders)."""
        import torch

        backend = cls.__new__(cls)
        backend.torch, backend.model, backend.tokenizer = torch, model, tokenizer
        backend.model_id = getattr(model, "name_or_path", "loaded")
        backend.context_tokens = options.get("context_tokens", 32768)
        backend.max_new_tokens = options.get("max_new_tokens", 2048)
        backend.thinking = options.get("thinking", False)
        backend.prefill_chunk_size = options.get("prefill_chunk_size", 512) or None
        backend.prefix_cache = options.get("prefix_cache", True)
        backend.sampling = options.get("sampling", {"do_sample": False})
        backend.device_map = {}
        backend.prefix = PrefixCache(model, backend.prefill_chunk_size)
        return backend

    def reset_peak(self):
        for i in range(self.torch.cuda.device_count()):
            self.torch.cuda.reset_peak_memory_stats(i)

    def set_seed(self, seed):
        """Seed sampling per question so reruns and resumed runs are reproducible."""
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def memory(self):
        return {
            str(i): {
                "allocated_peak_gib": self.torch.cuda.max_memory_allocated(i) / 2**30,
                "reserved_peak_gib": self.torch.cuda.max_memory_reserved(i) / 2**30,
            }
            for i in range(self.torch.cuda.device_count())
        }

    def boundaries(self, text, ids, messages):
        """Token prefixes of the prompt that end at message boundaries.

        A boundary is kept only if tokenizing the text up to it reproduces the
        prompt's first tokens exactly, so reuse can never change the input.
        """
        found = []
        for key, content, locate in (
            # Chat templates often trim message content, so search the stripped text.
            ("system", messages[0]["content"].strip(), text.find),
            ("turn", messages[-1]["content"].strip(), text.rfind),
        ):
            pos = locate(content) if content else -1
            if pos < 0:
                continue
            prefix = self.tokenizer(
                text[: pos + len(content)], add_special_tokens=False
            )["input_ids"]
            n = len(prefix)
            if 0 < n < len(ids) and ids[:n] == prefix and (
                not found or n > len(found[-1][1])
            ):
                found.append((key, prefix))
        return found

    def generate(self, messages):
        text = chat_text(self.tokenizer, messages, self.thinking)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_tokens = encoded["input_ids"].shape[1]
        if input_tokens + self.max_new_tokens > self.context_tokens:
            raise ContextLimitError(
                f"Context limit: {input_tokens} input + {self.max_new_tokens} output > "
                f"{self.context_tokens}. Increase --context-tokens or reduce tool output/steps."
            )
        device = self.model.get_input_embeddings().weight.device
        encoded = encoded.to(device)
        started = time.perf_counter()
        if self.prefix_cache:
            try:
                return self._generate(encoded, input_tokens, messages, text, started)
            except Exception as exc:  # noqa: BLE001 -- fall back to full prefill
                print(
                    f"Prefix cache disabled after {type(exc).__name__}: {exc}", flush=True
                )
                self.prefix_cache = False
                self.prefix.clear()
                self.torch.cuda.empty_cache()
                started = time.perf_counter()
        return self._generate(encoded, input_tokens, None, text, started)

    def _generate(self, encoded, input_tokens, messages, text, started):
        options = {
            "max_new_tokens": self.max_new_tokens,
            **self.sampling,
            "stopping_criteria": make_json_stop(self.tokenizer, input_tokens, self.thinking),
            "use_cache": True,
            # Avoid allocating [prompt_length, vocabulary_size] logits at prefill.
            "logits_to_keep": 1,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        }
        cached = 0
        with self.torch.inference_mode():
            bounds = []
            if messages is not None:
                ids = encoded["input_ids"][0].tolist()
                bounds = self.boundaries(text, ids, messages)
            if bounds:
                state = self.prefix.state_for(bounds)
                cached = len(bounds[-1][1])
                # Generation mutates the cache: work on a copy, prefilled up to the
                # last prompt token so generate() only processes that token.
                work = self.prefix.extend(copy.deepcopy(state), ids, cached, len(ids) - 1)
                options["past_key_values"] = work
            else:
                options["cache_implementation"] = "dynamic"
                options["prefill_chunk_size"] = self.prefill_chunk_size
            output = self.model.generate(**encoded, **options)
        generated = output[0, input_tokens:]
        reply = self.tokenizer.decode(generated, skip_special_tokens=False)
        # Strip special tokens but keep reasoning tags, which strip_reasoning needs.
        for marker in set(self.tokenizer.all_special_tokens) | {"<|im_end|>", "<|endoftext|>"}:
            if marker and marker not in ("<think>", "</think>"):
                reply = reply.replace(marker, "")
        return reply, {
            "input_tokens": input_tokens,
            "cached_tokens": cached,
            "output_tokens": len(generated),
            "generation_seconds": time.perf_counter() - started,
        }
