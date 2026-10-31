"""Lazy GPU backend: importing CLI/evaluator never downloads a model."""

import time


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
    mapping = {"": 0}
    for i in range(num_layers):
        mapping[f"{stack}.{i}"] = min(i * gpu_count // num_layers, gpu_count - 1)
    norm = f"{parent}.norm"
    if norm in module_names:
        mapping[norm] = gpu_count - 1
    if "lm_head" in module_names:
        mapping["lm_head"] = 0 if tied_embeddings else gpu_count - 1
    return mapping


class QwenBackend:
    def __init__(
        self,
        model_id,
        quantization="4bit",
        device_map="split",
        context_tokens=16384,
        max_new_tokens=2048,
        thinking=False,
        gpu_memory_gib=13,
    ):
        import torch
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            BitsAndBytesConfig,
            Qwen3_5ForConditionalGeneration,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required. Enable T4 GPUs in Kaggle settings.")
        self.torch = torch
        self.context_tokens = context_tokens
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.model_id = model_id
        if device_map == "split":
            from accelerate import init_empty_weights

            config = AutoConfig.from_pretrained(model_id)
            text_config = getattr(config, "text_config", config)
            with init_empty_weights():
                empty = Qwen3_5ForConditionalGeneration(config)
            names = set(dict(empty.named_modules()))
            device_map = make_split_map(
                names,
                text_config.num_hidden_layers,
                torch.cuda.device_count(),
                bool(
                    getattr(config, "tie_word_embeddings", False)
                    or getattr(text_config, "tie_word_embeddings", False)
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
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        # Explicit multimodal class matches the official checkpoint. Text prompts only.
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, **kwargs
        ).eval()
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
        self.reset_peak()

    def reset_peak(self):
        for i in range(self.torch.cuda.device_count()):
            self.torch.cuda.reset_peak_memory_stats(i)

    def memory(self):
        return {
            str(i): {
                "allocated_peak_gib": self.torch.cuda.max_memory_allocated(i) / 2**30,
                "reserved_peak_gib": self.torch.cuda.max_memory_reserved(i) / 2**30,
            }
            for i in range(self.torch.cuda.device_count())
        }

    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_tokens = encoded["input_ids"].shape[1]
        if input_tokens + self.max_new_tokens > self.context_tokens:
            raise RuntimeError(
                f"Context limit: {input_tokens} input + {self.max_new_tokens} output > "
                f"{self.context_tokens}. Increase --context-tokens or reduce tool output/steps."
            )
        device = self.model.get_input_embeddings().weight.device
        encoded = encoded.to(device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                # Avoid allocating [prompt_length, vocabulary_size] logits at prefill.
                logits_to_keep=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, input_tokens:]
        reply = self.tokenizer.decode(generated, skip_special_tokens=False)
        for marker in {
            self.tokenizer.eos_token,
            self.tokenizer.pad_token,
            "<|im_end|>",
            "<|endoftext|>",
        }:
            if marker:
                reply = reply.replace(marker, "")
        return reply, {
            "input_tokens": input_tokens,
            "output_tokens": len(generated),
            "generation_seconds": time.perf_counter() - started,
        }
