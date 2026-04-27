from __future__ import annotations

"""
Hugging Face actor backend for Yggdrasil.

This module provides a minimal Transformers-based generation backend used by
the interactive CLI. It prefers deterministic settings by default so behavior
differences are more attributable to substrate context than to sampling noise.

The implementation supports:
- local model paths or Hugging Face repo ids
- best-effort deterministic generation setup
- chat-template rendering when available
- a simple fallback prompt format otherwise
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ActorConfig:
    """
    Runtime generation configuration for the Hugging Face actor backend.
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0


class HFActor:
    """
    Minimal Hugging Face Transformers actor.

    Deterministic generation is preferred by default.

    Windows note: if you pass a local path, it must point to the directory
    containing `config.json` and tokenizer files. If the path does not exist,
    Transformers may interpret it as a Hub repo id instead.
    """

    def __init__(
        self,
        model_id_or_path: str,
        cfg: Optional[ActorConfig] = None,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.cfg = cfg or ActorConfig()

        # Normalize local paths, especially on Windows.
        p = os.path.expandvars(os.path.expanduser(model_id_or_path.strip().strip('"').strip("'")))
        looks_like_path = (":" in p) or ("\\" in p) or ("/" in p)
        if looks_like_path:
            pp = Path(p)
            if pp.exists():
                if pp.is_file():
                    pp = pp.parent
                model_id_or_path = str(pp)
            else:
                raise RuntimeError(
                    f"Local model path not found: {p}\n"
                    "Point --model at the folder containing config.json, tokenizer.json, etc.\n"
                    "Tip (PowerShell): try forward slashes, e.g. D:/models/Mistral-7B-Instruct-v0.3"
                )

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "HFActor requires 'transformers' and 'torch'. "
                "Install: pip install -U transformers accelerate torch"
            ) from e

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=(torch.float16 if dtype == "fp16" else None) if dtype != "auto" else "auto",
            device_map=device,
        )
        self.model.eval()

        # Best-effort determinism hardening. Exact bitwise determinism can still
        # vary depending on hardware and low-level kernel behavior.
        if self.cfg.seed is not None:
            try:
                s = int(self.cfg.seed)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)
            except Exception:
                pass

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    def _build_messages(
        self,
        context_text: str,
        user_text: str,
        system_text: Optional[str],
    ) -> list[dict]:
        """
        Build the message list used for prompt rendering.
        """
        sys = system_text or (
            "You are a helpful assistant. Keep responses concise and cite constraints you followed."
        )
        if context_text:
            sys = sys + "\n\n[CONTEXT]\n" + context_text.strip()

        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_text.strip()},
        ]

    def generate(
        self,
        context_text: str,
        user_text: str,
        system_text: Optional[str] = None,
    ) -> str:
        """
        Generate an assistant response from the current context and user input.

        Chat-template rendering is used when supported by the tokenizer.
        Otherwise, a simple fallback prompt format is used.

        Sampling is disabled by default. It is only enabled when temperature is
        greater than zero and top_p is set below 1.0.
        """
        msgs = self._build_messages(
            context_text=context_text,
            user_text=user_text,
            system_text=system_text,
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"<s>[SYSTEM]\n{msgs[0]['content']}\n[/SYSTEM]\n"
                f"[USER]\n{msgs[1]['content']}\n[/USER]\n[ASSISTANT]\n"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = bool(
            self.cfg.temperature
            and self.cfg.temperature > 0.0
            and self.cfg.top_p
            and self.cfg.top_p < 1.0
        )

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=do_sample,
                temperature=float(self.cfg.temperature) if do_sample else None,
                top_p=float(self.cfg.top_p) if do_sample else None,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0][prompt_len:]
        assistant = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if assistant:
            return assistant
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()