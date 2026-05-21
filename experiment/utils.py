"""
Shared utilities for LLM model loading and inference.
Used by client.py, client_metacognitive.py, server.py, server_text.py,
server_cod.py, server_claude_compact.py, etc.
"""

import os
import re
import json
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    HAS_GEMINI = True
    HAS_GENAI_NEW = True
except ImportError:
    HAS_GEMINI = False
    HAS_GENAI_NEW = False

try:
    from openai import OpenAI as _OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Max tokens allowed by backend/model routing policy.
MAX_ALLOWED_INPUT_TOKENS = 1_048_576
# When exceeding MAX_ALLOWED_INPUT_TOKENS, truncate to first 1,000,000 tokens.
TRUNCATED_INPUT_TOKENS = 1_000_000
# Conservative fallback estimate when token counting is unavailable.
_FALLBACK_CHARS_PER_TOKEN_ESTIMATE = 1
TRUNCATED_INPUT_CHARS_FALLBACK = (
    TRUNCATED_INPUT_TOKENS * _FALLBACK_CHARS_PER_TOKEN_ESTIMATE
)


# ---------------------------------------------------------------------------
# CUDA check
# ---------------------------------------------------------------------------
def check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


def _compose_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Combine optional system prompt and user prompt."""
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt


class _GeminiModel:
    """Thin wrapper around google.genai.Client providing a stable interface
    for count_tokens() and generate_content() used by call_gemini()."""

    def __init__(self, client, model_name: str):
        self._client = client
        self._model_name = model_name

    def count_tokens(self, text: str):
        """Return an object with .total_tokens attribute."""
        result = self._client.models.count_tokens(
            model=self._model_name, contents=text
        )
        return result

    def generate_content(
        self,
        prompt,
        generation_config: Optional[Dict] = None,
        request_options: Optional[Dict] = None,
    ):
        """Call generate_content via new SDK."""
        from google.genai import types as _gt
        kwargs: Dict = {}
        if generation_config and generation_config.get("max_output_tokens"):
            kwargs["max_output_tokens"] = generation_config["max_output_tokens"]
        if generation_config and generation_config.get("temperature") is not None:
            kwargs["temperature"] = generation_config["temperature"]
        config = _gt.GenerateContentConfig(**kwargs) if kwargs else None
        return self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )


def _count_gemini_tokens(gemini_model, text: str) -> Optional[int]:
    """Best-effort Gemini token counting. Returns None if unavailable."""
    try:
        token_count_result = gemini_model.count_tokens(text)
        total_tokens = getattr(token_count_result, "total_tokens", None)
        if total_tokens is None:
            return None
        return int(total_tokens)
    except Exception:
        return None


def _truncate_gemini_prompt_to_limit(gemini_model, full_prompt: str) -> Tuple[str, Optional[int], bool]:
    """Apply 1,048,576-token ceiling and truncate to 1,000,000 tokens when exceeded.

    Returns:
        (possibly_truncated_prompt, measured_input_tokens_or_none, was_truncated)
    """
    input_tokens = _count_gemini_tokens(gemini_model, full_prompt)
    if input_tokens is not None:
        if input_tokens <= MAX_ALLOWED_INPUT_TOKENS:
            return full_prompt, input_tokens, False

        # Exceeded allowed limit: truncate to first ~1,000,000 tokens.
        target_tokens = TRUNCATED_INPUT_TOKENS
        truncated_prompt = full_prompt

        for _ in range(6):
            current_tokens = _count_gemini_tokens(gemini_model, truncated_prompt)
            if current_tokens is None:
                break
            if current_tokens <= target_tokens:
                return truncated_prompt, current_tokens, True

            shrink_ratio = target_tokens / max(current_tokens, 1)
            new_char_len = max(1, int(len(truncated_prompt) * shrink_ratio))
            if new_char_len >= len(truncated_prompt):
                new_char_len = len(truncated_prompt) - 1
            truncated_prompt = truncated_prompt[:new_char_len]

        # Final conservative fallback if iterative token counting is unavailable/inconclusive.
        truncated_prompt = truncated_prompt[:TRUNCATED_INPUT_CHARS_FALLBACK]
        final_tokens = _count_gemini_tokens(gemini_model, truncated_prompt)
        return truncated_prompt, final_tokens, True

    # Fallback when token counting is unavailable.
    if len(full_prompt) > TRUNCATED_INPUT_CHARS_FALLBACK:
        return full_prompt[:TRUNCATED_INPUT_CHARS_FALLBACK], None, True

    return full_prompt, None, False


def _resolve_hf_context_limit(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> int:
    """Resolve effective HuggingFace input-token context limit for safe truncation."""
    limits = []

    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and tokenizer_max > 0 and tokenizer_max < 10_000_000:
        limits.append(int(tokenizer_max))

    config = getattr(model, "config", None)
    if config is not None:
        for attr in (
            "max_position_embeddings",
            "max_sequence_length",
            "n_positions",
            "sliding_window",
        ):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                limits.append(int(value))

        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            for key in ("original_max_position_embeddings", "max_position_embeddings"):
                value = rope_scaling.get(key)
                if isinstance(value, int) and value > 0:
                    limits.append(int(value))

    if not limits:
        return MAX_ALLOWED_INPUT_TOKENS

    return min(limits)


# ---------------------------------------------------------------------------
# Gemini setup & call
# ---------------------------------------------------------------------------
def setup_gemini(
    api_key: Optional[str] = None,
    model_name: str = "gemini-3-pro-preview",
) -> "_GeminiModel":
    """Initialize Gemini API and return a model wrapper.

    Args:
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        model_name: Gemini model name.

    Returns:
        _GeminiModel wrapper around google.genai.Client.
    """
    if not HAS_GEMINI:
        raise ImportError(
            "google-genai is required for Gemini API. "
            "Install with: pip install google-genai"
        )
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY env var or pass api_key."
        )
    client = genai_new.Client(api_key=api_key)
    model = _GeminiModel(client, model_name)
    print(f"Gemini model initialized: {model_name}")
    return model


def extract_gemini_response_text(response) -> str:
    """Extract text from Gemini response with robust fallbacks."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    text_parts = [
        part.text for part in parts if hasattr(part, "text") and part.text
    ]
    return "\n".join(text_parts).strip()


def call_gemini(
    gemini_model,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    """Call Gemini API with input truncation.

    Truncates input to ~1,000,000 tokens if it exceeds Gemini's limit.

    Args:
        gemini_model: Initialized GenerativeModel instance.
        prompt: User prompt text.
        system_prompt: Optional system prompt (prepended to prompt).
        max_new_tokens: Max output tokens.

    Returns:
        Tuple of (generated_text, token_info_dict).
        token_info_dict contains: output_tokens, finish_reason, backend.
    """
    try:
        full_prompt = _compose_prompt(prompt, system_prompt)

        full_prompt, input_token_count, was_truncated = _truncate_gemini_prompt_to_limit(
            gemini_model, full_prompt
        )
        if was_truncated:
            if input_token_count is not None:
                print(
                    "Warning: Input prompt exceeded allowed token limit "
                    f"({MAX_ALLOWED_INPUT_TOKENS}). Truncated to first "
                    f"~{TRUNCATED_INPUT_TOKENS} tokens; current input tokens={input_token_count}."
                )
            else:
                print(
                    "Warning: Input prompt likely exceeded allowed token limit "
                    f"({MAX_ALLOWED_INPUT_TOKENS}). Truncated to safe fallback "
                    f"length (~{TRUNCATED_INPUT_TOKENS} tokens)."
                )

        # Configure generation parameters
        generation_config = {}
        if max_new_tokens:
            generation_config["max_output_tokens"] = max_new_tokens

        # Generate response
        if generation_config:
            response = gemini_model.generate_content(
                full_prompt, generation_config=generation_config
            )
        else:
            response = gemini_model.generate_content(full_prompt)

        # Handle response safely
        if not response.candidates:
            raise RuntimeError(
                "Gemini API returned no candidates. Response may have been blocked."
            )

        candidate = response.candidates[0]
        # New SDK returns finish_reason as a FinishReason enum or string;
        # normalize to a lowercase string for consistent handling.
        raw_reason = candidate.finish_reason
        raw_reason_str = str(raw_reason).upper()
        # Support both old int-based (legacy) and new name-based finish reasons.
        _INT_REASON_MAP = {2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION"}
        if isinstance(raw_reason, int):
            raw_reason_str = _INT_REASON_MAP.get(raw_reason, str(raw_reason))
        finish_reason = raw_reason_str.lower()

        token_info = {
            "backend": "gemini",
            "finish_reason": finish_reason,
            "output_tokens": 0,
            "input_tokens": input_token_count,
            "input_truncated": was_truncated,
        }

        if "MAX_TOKENS" in raw_reason_str:
            text_parts = []
            if candidate.content and candidate.content.parts:
                text_parts = [
                    part.text
                    for part in candidate.content.parts
                    if hasattr(part, "text") and part.text
                ]
            if text_parts:
                text = "\n".join(text_parts).strip()
            else:
                # Input was so long that no output tokens were left in the
                # context window.  Return empty string with a warning rather
                # than raising, so callers can handle it gracefully.
                print(
                    "Warning: Gemini hit output token limit with no text generated "
                    "(input likely consumed the entire context window). "
                    f"Input tokens: {input_token_count}. Returning empty string."
                )
                text = ""
            token_info["output_tokens"] = len(text) // 4
            token_info["finish_reason"] = "max_tokens"
            return text, token_info
        elif "SAFETY" in raw_reason_str:
            raise RuntimeError(
                "Gemini API blocked the response due to safety filters."
            )
        elif "RECITATION" in raw_reason_str:
            raise RuntimeError(
                "Gemini API blocked the response due to recitation."
            )

        text = extract_gemini_response_text(response)
        if text:
            token_info["output_tokens"] = len(text) // 4
            return text, token_info
        raise RuntimeError("Failed to extract text from Gemini response.")

    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}")


def call_openrouter(
    api_key: str,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    """Call OpenRouter API (OpenAI-compatible endpoint).

    Args:
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var).
        model_name: Model slug e.g. "anthropic/claude-opus-4.6".
        prompt: User prompt text.
        system_prompt: Optional system prompt.
        max_new_tokens: Max output tokens.

    Returns:
        Tuple of (generated_text, token_info_dict).
    """
    if not HAS_OPENAI:
        raise ImportError(
            "openai package is required for OpenRouter. "
            "Install with: pip install openai"
        )
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY env var or pass api_key."
        )

    client = _OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {}
    if max_new_tokens:
        kwargs["max_tokens"] = max_new_tokens

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs,
        )
    except Exception as e:
        raise RuntimeError(f"Error calling OpenRouter API: {e}")

    choice = response.choices[0]
    text = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

    usage = getattr(response, "usage", None)
    token_info = {
        "backend": "openrouter",
        "model": model_name,
        "finish_reason": finish_reason,
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", len(text) // 4) if usage else len(text) // 4,
    }
    return text, token_info


# ---------------------------------------------------------------------------
# Gemini with ThinkingConfig (new google-genai SDK)
# ---------------------------------------------------------------------------
def call_gemini_thinking(
    api_key: str,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    thinking_level: str = "high",
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    """
    Call Gemini using the new google-genai SDK with ThinkingConfig support.

    Uses:
        from google import genai
        from google.genai import types
        client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
            ),
        )

    Args:
        api_key: Gemini API key.
        model_name: Model name (e.g., "gemini-3.1-pro-preview").
        prompt: User prompt.
        system_prompt: Optional system prompt prepended to prompt.
        thinking_level: "low", "medium", or "high" (default: "high").
        max_new_tokens: Optional max output tokens.

    Returns:
        Tuple of (generated_text, token_info_dict).
    """
    if not HAS_GENAI_NEW:
        raise ImportError(
            "google-genai is required for ThinkingConfig support. "
            "Install with: pip install google-genai"
        )

    full_prompt = _compose_prompt(prompt, system_prompt)

    client = genai_new.Client(api_key=api_key)

    # Only gemini-3.1-pro-preview supports ThinkingConfig; all other models
    # must not include thinking_config in the request.
    _THINKING_MODELS = {"gemini-3.1-pro-preview"}
    supports_thinking = model_name in _THINKING_MODELS

    gen_config_kwargs: Dict = {}
    if supports_thinking:
        gen_config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_level=thinking_level
        )
        print(f"[Gemini] Using ThinkingConfig(thinking_level={thinking_level!r}) for {model_name}")
    else:
        print(
            f"[Gemini] Model {model_name!r} does not support ThinkingConfig — "
            "calling without thinking_config"
        )
    if max_new_tokens:
        gen_config_kwargs["max_output_tokens"] = max_new_tokens

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(**gen_config_kwargs) if gen_config_kwargs else None,
        )
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini (thinking) API: {e}")

    text = getattr(response, "text", None)
    if not text:
        # Fallback: walk candidates
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            text_parts = [
                p.text for p in parts
                if hasattr(p, "text") and p.text and not getattr(p, "thought", False)
            ]
            text = "\n".join(text_parts).strip()

    if not text:
        print(
            f"Warning: Gemini (thinking) returned no text for model={model_name}, "
            f"thinking_level={thinking_level}. Returning empty string."
        )
        text = ""

    token_info = {
        "backend": "gemini_thinking",
        "thinking_level": thinking_level,
        "model": model_name,
        "output_tokens": len(text) // 4,
        "finish_reason": "stop",
    }
    usage = getattr(response, "usage_metadata", None)
    if usage:
        token_info["input_tokens"] = getattr(usage, "prompt_token_count", 0)
        token_info["output_tokens"] = getattr(usage, "candidates_token_count", len(text) // 4)
        token_info["thinking_tokens"] = getattr(usage, "thoughts_token_count", 0)

    print(f"[Gemini thinking={thinking_level}] output_tokens={token_info['output_tokens']}")
    return text, token_info


# ---------------------------------------------------------------------------
# HuggingFace model loading & call
# ---------------------------------------------------------------------------
def load_hf_model(
    model_name: str,
    device: str = "cpu",
    load_in_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: "cuda" or "cpu".
        load_in_8bit: Whether to use 8-bit quantization.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    adapter_config_path = os.path.join(model_name, "adapter_config.json")
    is_lora_adapter_dir = os.path.isdir(model_name) and os.path.exists(adapter_config_path)

    tokenizer_source = model_name
    adapter_base_model: Optional[str] = None
    if is_lora_adapter_dir:
        with open(adapter_config_path, "r", encoding="utf-8") as fp:
            adapter_cfg = json.load(fp)
        adapter_base_model = adapter_cfg.get("base_model_name_or_path")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception:
        if is_lora_adapter_dir and adapter_base_model:
            print(
                "Tokenizer files not found in adapter directory; "
                f"falling back to base model tokenizer: {adapter_base_model}"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                adapter_base_model, trust_remote_code=True
            )
        else:
            raise

    model_kwargs = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if is_lora_adapter_dir:
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Loading LoRA adapter checkpoints requires peft. "
                "Install with: pip install peft"
            ) from exc

        model = AutoPeftModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")
    return model, tokenizer


def call_hf_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[str, Dict]:
    """Call a HuggingFace model.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Loaded tokenizer.
        model_name: Model name (used to detect DeepSeek-R1 settings).
        prompt: User prompt text.
        system_prompt: Optional system prompt (prepended to prompt).
        max_new_tokens: Max output tokens. Defaults to 32768.
        device: Device the model is on.

    Returns:
        Tuple of (generated_text, token_info_dict).
    """
    full_prompt = _compose_prompt(prompt, system_prompt)

    try:
        model_context_limit = _resolve_hf_context_limit(model, tokenizer)
        input_max_length = int(min(model_context_limit, MAX_ALLOWED_INPUT_TOKENS))
        if input_max_length > TRUNCATED_INPUT_TOKENS:
            input_max_length = TRUNCATED_INPUT_TOKENS

        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=input_max_length,
        ).to(device)

        input_token_count = inputs["input_ids"].shape[1]
        if input_token_count >= input_max_length:
            print(
                f"Warning: Input prompt reached model context limit "
                f"({input_max_length} tokens) and may be truncated."
            )

        if max_new_tokens is None:
            max_new_tokens = 32768

        print(f"Input tokens: {input_token_count}, Max new tokens: {max_new_tokens}")

        with torch.no_grad():
            is_deepseek_r1 = "DeepSeek-R1" in model_name

            if is_deepseek_r1:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        output_token_ids = outputs[0][inputs["input_ids"].shape[1]:]
        output_token_count = output_token_ids.shape[0]

        token_info = {
            "backend": "huggingface",
            "output_tokens": output_token_count,
            "input_tokens": int(input_token_count),
            "input_truncated": bool(input_token_count >= input_max_length),
            "input_limit": int(input_max_length),
        }
        return generated_text.strip(), token_info

    except Exception as e:
        print(f"Error calling model: {e}")
        return "", {"output_tokens": 0}
