"""HLE dataset loader and multimodal helper utilities.

HLE: https://huggingface.co/datasets/cais/hle
"""

import base64
import binascii
import io
import re
import time
from typing import Any, Dict, Optional, Tuple, Union

from datasets import load_dataset
from PIL import Image

from utils import extract_gemini_response_text


HLE_MULTIMODAL_INSTRUCTION = """You are solving one HLE multimodal benchmark problem.
Use BOTH the question text and the attached image.
Return only the final answer, with no explanation."""
HLE_DATASET_ID = "cais/hle"


def load_hle_dataset(split: str = "test"):
    """Load HLE from Hugging Face.

    The canonical test-split load is:
        dataset = load_dataset("cais/hle", split="test")
    """
    if split == "test":
        dataset = load_dataset("cais/hle", split="test")
    else:
        dataset = load_dataset("cais/hle", split=split)
    return dataset


def _decode_data_uri_image(data_uri: str) -> Optional[Image.Image]:
    """Decode an image from a `data:image/...;base64,...` URI string."""
    if not data_uri or not isinstance(data_uri, str):
        return None
    if not data_uri.startswith("data:image") or "," not in data_uri:
        return None

    _, payload = data_uri.split(",", 1)
    try:
        image_bytes = base64.b64decode(payload)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except (binascii.Error, ValueError, OSError):
        return None


def extract_hle_image(example: Dict[str, Any]) -> Tuple[Optional[Image.Image], str]:
    """Extract a PIL image from an HLE sample.

    Preferred order:
      1) `image_preview` (already decoded by datasets)
      2) `rationale_image` (decoded image, if present)
      3) `image` base64 data URI string

    Returns:
        (image_or_none, source_label)
    """
    image_preview = example.get("image_preview")
    if isinstance(image_preview, Image.Image):
        return image_preview.convert("RGB"), "image_preview"

    rationale_image = example.get("rationale_image")
    if isinstance(rationale_image, Image.Image):
        return rationale_image.convert("RGB"), "rationale_image"

    image_data_uri = example.get("image")
    decoded = _decode_data_uri_image(image_data_uri)
    if decoded is not None:
        return decoded, "image(data_uri)"

    return None, "missing"


def format_hle_question(example: Dict[str, Any]) -> str:
    """Return a clean question string for prompting."""
    question = example.get("question", "")
    if question is None:
        return ""
    return str(question).strip()


def get_hle_answer(example: Dict[str, Any]) -> str:
    """Return reference answer string from HLE sample."""
    answer = example.get("answer", "")
    if answer is None:
        return ""
    return str(answer).strip()


def normalize_hle_answer(text: str) -> str:
    """Normalize answer text for exact-match style checks."""
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" \t\n\r'\"`.,;:!?")
    return normalized


def is_hle_answer_correct(prediction: str, ground_truth: str) -> bool:
    """Exact-match evaluator after light normalization."""
    return normalize_hle_answer(prediction) == normalize_hle_answer(ground_truth)


def extract_hle_final_answer(response_text: str) -> str:
    """Extract final answer from model response for HLE scoring."""
    if not response_text:
        return ""

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", response_text)
    if boxed:
        return boxed[-1].strip()

    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if not lines:
        return ""

    last_line = lines[-1]
    if ":" in last_line:
        prefix, suffix = last_line.split(":", 1)
        if "answer" in prefix.lower() and suffix.strip():
            return suffix.strip()

    return last_line


def _call_gemini_multimodal(
    gemini_model,
    question: str,
    image=None,
    max_output_tokens: int = 256,
    temperature: float = 0.0,
    request_timeout_sec: int = 180,
    max_retries: int = 3,
    max_timeout_retries: int = 2,
    return_token_info: bool = False,
) -> Union[str, Tuple[str, Dict[str, int]]]:
    """Call Gemini with HLE prompt.

    - If `image` is provided, sends multimodal payload `[prompt, image]`.
    - If `image` is None, sends text-only prompt.
    """
    prompt = (
        f"{HLE_MULTIMODAL_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
        "Final answer:"
    )

    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }

    last_error = None
    timeout_retries = 0
    for attempt in range(max_retries + 1):
        try:
            call_kwargs = {"generation_config": generation_config}
            if _supports_request_options(gemini_model):
                call_kwargs["request_options"] = {"timeout": request_timeout_sec}

            if image is None:
                response = gemini_model.generate_content(prompt, **call_kwargs)
            else:
                response = gemini_model.generate_content([prompt, image], **call_kwargs)

            text = extract_gemini_response_text(response)
            if not text:
                raise RuntimeError("Gemini returned empty text for multimodal request.")

            token_info = _extract_gemini_usage(response, text)
            if return_token_info:
                return text, token_info
            return text
        except BaseException as exc:  # noqa: BLE001
            last_error = exc

            exc_name = type(exc).__name__.lower()
            exc_text = str(exc).lower()
            is_timeout_error = (
                "deadlineexceeded" in exc_name
                or "deadline exceeded" in exc_text
                or "deadline expired" in exc_text
                or "504" in exc_text
            )
            is_retryable = is_timeout_error or any(
                marker in exc_text
                for marker in (
                    "service unavailable",
                    "resource exhausted",
                    "internal server error",
                    "429",
                    "503",
                )
            )

            if is_timeout_error:
                timeout_retries += 1
                if timeout_retries >= max_timeout_retries:
                    raise RuntimeError(
                        f"Gemini timeout after {max_timeout_retries} attempts; skipping question."
                    ) from exc

            if not is_retryable or attempt >= max_retries:
                raise RuntimeError(f"Gemini call failed: {exc}") from exc

            sleep_seconds = 2 ** attempt
            print(
                f"Gemini transient error ({type(exc).__name__}), retry "
                f"{attempt + 1}/{max_retries} in {sleep_seconds}s..."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Gemini call failed after retries: {last_error}")


def _supports_request_options(gemini_model) -> bool:
    """Return whether a Gemini wrapper accepts the legacy request_options kwarg."""
    import inspect

    try:
        signature = inspect.signature(gemini_model.generate_content)
    except (TypeError, ValueError):
        return False

    params = signature.parameters
    return "request_options" in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )


def _extract_gemini_usage(response, text: str) -> Dict[str, int]:
    """Extract Gemini usage metadata with a character-count fallback."""
    usage = getattr(response, "usage_metadata", None)
    fallback_output_tokens = max(len(text) // 4, 1) if text else 0
    if not usage:
        return {
            "input_tokens": 0,
            "output_tokens": fallback_output_tokens,
            "thinking_tokens": 0,
            "total_tokens": fallback_output_tokens,
            "usage_fallback": 1,
        }

    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    if output_tokens <= 0:
        output_tokens = fallback_output_tokens
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens + thinking_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "usage_fallback": 0,
    }
