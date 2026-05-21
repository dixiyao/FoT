"""
Shared utility functions for math datasets
Provides evaluation metrics and answer normalization for various math problem formats.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text (including decimals and negatives).

    Handles formats like: 42, 3.14, -5, 1e-6, 2.5e+3

    Args:
        text: Input text

    Returns:
        List of extracted numbers as floats
    """
    # Pattern: optional minus, digits, optional decimal point and more digits
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return [float(n) for n in numbers if n]


def accuracy(predictions: List[str], answers: List[str]) -> float:
    """Calculate accuracy of predictions.

    Args:
        predictions: List of predicted answers
        answers: List of ground truth answers

    Returns:
        Accuracy ratio (0.0 to 1.0)
    """
    if not predictions or not answers:
        return 0.0

    correct = 0
    total = len(predictions)

    for prediction, answer in zip(predictions, answers):
        if prediction == answer:
            correct += 1

    return correct / total if total > 0 else 0.0


def normalize_numeric(text: str) -> Optional[float]:
    """Extract and normalize single numeric answer from text.

    Useful for datasets with primarily numeric answers (GSM8K, MATH500, AIME).

    Args:
        text: Input text

    Returns:
        Normalized float value or None
    """
    nums = extract_numbers(text)
    return nums[0] if nums else None


def is_numeric_match(pred: str, truth: str, tolerance: float = 1e-6) -> bool:
    """Check if two answers match numerically.

    Args:
        pred: Predicted answer
        truth: Ground truth answer
        tolerance: Floating point tolerance

    Returns:
        True if numeric values match within tolerance
    """
    pred_nums = extract_numbers(pred)
    truth_nums = extract_numbers(truth)

    if not pred_nums or not truth_nums:
        return False

    # Single number comparison
    if len(pred_nums) == 1 and len(truth_nums) == 1:
        return abs(pred_nums[0] - truth_nums[0]) < tolerance

    # Multiple numbers: exact length match required
    if len(pred_nums) == len(truth_nums):
        return all(abs(p - t) < tolerance for p, t in zip(pred_nums, truth_nums))

    return False


def is_exact_match(pred: str, truth: str, case_sensitive: bool = False) -> bool:
    """Check for exact string match after normalization.

    Args:
        pred: Predicted answer
        truth: Ground truth answer
        case_sensitive: Whether to perform case-sensitive comparison

    Returns:
        True if strings match
    """
    pred_normalized = pred.strip()
    truth_normalized = truth.strip()

    if case_sensitive:
        return pred_normalized == truth_normalized
    else:
        return pred_normalized.lower() == truth_normalized.lower()


def remove_boxing_notation(text: str) -> str:
    """Remove LaTeX boxing notation from answer.

    Handles: \\boxed{}, boxed(), $...$

    Args:
        text: Input text

    Returns:
        Text with boxing notation removed
    """
    # Remove \boxed{...}
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)

    # Remove boxed(...)
    text = re.sub(r"boxed\(([^)]*)\)", r"\1", text)

    # Remove $...$
    text = re.sub(r"\$([^$]*)\$", r"\1", text)

    return text.strip()


def remove_units(text: str) -> str:
    """Remove common unit suffixes from numeric answers.

    Args:
        text: Input text

    Returns:
        Text with units removed
    """
    units = [
        # Length
        "cm",
        "m",
        "mm",
        "km",
        "inches",
        "feet",
        "yards",
        "miles",
        # Weight
        "kg",
        "g",
        "lb",
        "ounce",
        # Temperature
        "celsius",
        "°c",
        "fahrenheit",
        "°f",
        "kelvin",
        # Angles
        "degrees",
        "°",
        "radians",
        "rad",
        # Area/Volume
        "square units",
        "sq",
        "cm²",
        "m²",
        "cubic units",
        "cu",
        "cm³",
        "m³",
        # Other
        "units",
        "dollars",
        "$",
        "percent",
        "%",
    ]

    text_lower = text.lower()
    for unit in units:
        # Match unit at end of string with optional space
        pattern = r"\s*" + re.escape(unit) + r"\s*$"
        text_lower = re.sub(pattern, "", text_lower, flags=re.IGNORECASE)

    return text_lower.strip()


def batch_accuracy(
    predictions: List[str],
    answers: List[str],
    evaluator: Optional[Callable[[str, str], bool]] = None,
) -> Dict[str, Any]:
    """Calculate batch evaluation metrics with optional custom evaluator.

    Args:
        predictions: List of predicted answers
        answers: List of ground truth answers
        evaluator: Optional custom evaluation function(pred, ans) -> bool

    Returns:
        Dict with accuracy, correct_count, total_count
    """
    if not predictions or not answers:
        return {"accuracy": 0.0, "correct_count": 0, "total_count": 0, "errors": []}

    correct_count = 0
    total_count = len(predictions)
    errors = []

    for idx, (pred, ans) in enumerate(zip(predictions, answers)):
        try:
            if evaluator:
                is_correct = evaluator(pred, ans)
            else:
                # Default: exact match
                is_correct = is_exact_match(pred, ans, case_sensitive=False)

            if is_correct:
                correct_count += 1
        except Exception as e:
            errors.append(
                {"index": idx, "prediction": pred, "answer": ans, "error": str(e)}
            )

    accuracy_score = correct_count / total_count if total_count > 0 else 0.0

    return {
        "accuracy": accuracy_score,
        "correct_count": correct_count,
        "total_count": total_count,
        "errors": errors,
    }
