"""Lazy GPU backend: importing CLI/evaluator never downloads a model."""

import time


class QwenBackend:
    def __init__(
        self,
        model_id,
        quantization="4bit",
        device_map="auto",
        context_tokens=16384,
        max_new_tokens=2048,
        thinking=False,
        gpu_memory_gib=13,
    ):
        import torch
        from transformers import (
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        kwargs = {
            "device_map": device_map,
            "dtype": torch.float16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if device_map == "auto":
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
