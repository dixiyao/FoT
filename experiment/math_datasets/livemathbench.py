"""
LiveMathBench Dataset Handler
Unified interface for all 6 LiveMathBench variants from OpenCompass
https://huggingface.co/datasets/opencompass/LiveMathBench

Variants:
- v202412_AMC_en: AMC (American Mathematics Competitions)
- v202412_CCEE_en: CCEE (Chinese College Entrance Examination)
- v202412_CNMO_en: CNMO (Chinese National Mathematics Olympiad)
- v202412_WLPMC_en: WLPMC (World Logic/Proof Mathematics Competition)
- v202412_hard_en: Hard problems from 2024
- v202505_hard_en: Hard problems from 2025
"""

from typing import Any, Dict, List, Optional, Tuple

from .utils import accuracy, extract_numbers

# Prompt instruction for competition math problems
livemathbench_prompt = (
    '\nSolve the problem step by step. Wrap your final answer in "\\boxed{}".'
)

# Configuration for each LiveMathBench variant
LIVEMATHBENCH_CONFIGS = {
    "livemathbench_amc": {
        "hf_name": "opencompass/LiveMathBench",
        "config": "v202412_AMC_en",
        "split": "test",
        "description": "American Mathematics Competitions (AMC)",
    },
    "livemathbench_ccee": {
        "hf_name": "opencompass/LiveMathBench",
        "config": "v202412_CCEE_en",
        "split": "test",
        "description": "Chinese College Entrance Examination (CCEE)",
    },
    "livemathbench_cnmo": {
        "hf_name": "opencompass/LiveMathBench",
        "config": "v202412_CNMO_en",
        "split": "test",
        "description": "Chinese National Mathematics Olympiad (CNMO)",
    },
    "livemathbench_wlpmc": {
        "hf_name": "opencompass/LiveMathBench",
        "config": "v202412_WLPMC_en",
        "split": "test",
        "description": "World Logic/Proof Mathematics Competition (WLPMC)",
    },
    "livemathbench_hard_2024": {
        "hf_name": "opencompass/LiveMathBench",
        "config": "v202412_hard_en",
        "split": "test",
        "description": "Hard problems from 2024",
    },
    "livemathbench_hard_2025": {
        "hf_name": "opencompass/LiveMathBench",
        "config": "v202505_hard_en",
        "split": "test",
        "description": "Hard problems from 2025",
    },
}


def livemathbench_formatter(
    example: Dict[str, Any], variant: Optional[str] = None
) -> Tuple[str, str]:
    """
    Format example from LiveMathBench dataset.

    LiveMathBench uses "question" and "answer" fields, with optional "question_type".

    Args:
        example: Dataset example dict
        variant: Optional variant name for context-specific formatting

    Returns:
        Tuple of (question_text, answer_text)
    """
    # Extract question (LiveMathBench uses "question" field, not "problem")
    question_text = example.get("question") or example.get("problem", "")
    if not question_text:
        raise ValueError("Example missing 'question' or 'problem' field")

    # Add prompt
    question_text = question_text + livemathbench_prompt

    # Extract answer
    answer_text = example.get("answer", "")

    return question_text, answer_text


def livemathbench_evaluator(
    prediction: str,
    ground_truth: str,
    dataset_name: Optional[str] = None,
    problem_text: Optional[str] = None,
) -> bool:
    """
    Evaluate LiveMathBench prediction against ground truth.

    LiveMathBench answers can be:
    - Numeric (42, 3.14, -5)
    - Expressions ($\frac{2n+2}{3}$, $\sqrt{2}$)
    - Multiple choice (A, B, C, D)
    - Yes/No answers

    Evaluation strategy:
    1. Exact string match (case-insensitive for text answers)
    2. Numeric comparison (extract and compare numbers with tolerance)
    3. Substring matching for expressions

    Args:
        prediction: Model's predicted answer
        ground_truth: Correct answer from dataset
        dataset_name: Optional dataset variant name
        problem_text: Optional problem statement for context

    Returns:
        True if prediction is correct, False otherwise
    """
    if not prediction or not ground_truth:
        return False

    pred = prediction.strip()
    truth = ground_truth.strip()

    # Exact match (case-insensitive)
    if pred.lower() == truth.lower():
        return True

    # Remove common boxing notation
    boxing_patterns = [r"\\boxed\{([^}]*)\}", r"boxed\(([^)]*)\)", r"\$([^$]*)\$"]
    for pattern in boxing_patterns:
        import re

        pred_match = re.search(pattern, pred)
        truth_match = re.search(pattern, truth)
        if pred_match:
            pred = pred_match.group(1)
        if truth_match:
            truth = truth_match.group(1)

    # Re-check after unboxing
    if pred.strip().lower() == truth.strip().lower():
        return True

    # Numeric extraction and comparison
    pred_nums = extract_numbers(pred)
    truth_nums = extract_numbers(truth)

    # If both have numbers, try numeric comparison
    if pred_nums and truth_nums:
        # Single number comparison
        if len(pred_nums) == 1 and len(truth_nums) == 1:
            return abs(pred_nums[0] - truth_nums[0]) < 1e-6

        # Multiple numbers: check if all match (in order)
        if len(pred_nums) == len(truth_nums):
            return all(abs(p - t) < 1e-6 for p, t in zip(pred_nums, truth_nums))

        # Try first number match as fallback
        if len(pred_nums) > 0 and len(truth_nums) > 0:
            return abs(pred_nums[0] - truth_nums[0]) < 1e-6

    # Substring match for expressions (only if both are meaningful)
    if len(pred) > 1 and len(truth) > 1:
        pred_lower = pred.lower()
        truth_lower = truth.lower()
        if pred_lower in truth_lower or truth_lower in pred_lower:
            return True

    # Multiple choice answer matching (A, B, C, D, etc.)
    mc_answers = ["a", "b", "c", "d", "e"]
    if pred.lower() in mc_answers and truth.lower() in mc_answers:
        return pred.lower() == truth.lower()

    return False


def livemathbench_scorer(
    predictions: List[str],
    answers: List[str],
    dataset_name: Optional[str] = None,
    problem_texts: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Score predictions for LiveMathBench dataset.

    Args:
        predictions: List of model predictions
        answers: List of ground truth answers
        dataset_name: Optional dataset variant name
        problem_texts: Optional list of problem statements

    Returns:
        Dict with evaluation metrics: accuracy, correct_count, total_count
    """
    if not predictions or not answers:
        return {"accuracy": 0.0, "correct_count": 0, "total_count": 0}

    correct_count = 0
    total_count = len(predictions)

    for pred, ans in zip(predictions, answers):
        if livemathbench_evaluator(pred, ans, dataset_name):
            correct_count += 1

    accuracy_score = correct_count / total_count if total_count > 0 else 0.0

    return {
        "accuracy": accuracy_score,
        "correct_count": correct_count,
        "total_count": total_count,
    }


def get_livemathbench_config(variant_name: str) -> Optional[Dict[str, str]]:
    """
    Get configuration for a LiveMathBench variant.

    Args:
        variant_name: Variant name (e.g., 'livemathbench_amc')

    Returns:
        Config dict or None if not found
    """
    return LIVEMATHBENCH_CONFIGS.get(variant_name)


def list_livemathbench_variants() -> List[str]:
    """List all available LiveMathBench variants."""
    return list(LIVEMATHBENCH_CONFIGS.keys())
