"""
GPQA (Graduate-Level Google-Proof Q&A) Dataset Handler
Supports both gpqa and gpqa_diamond variants

GPQA is a challenging dataset of multiple-choice questions in:
- Physics
- Chemistry
- Biology

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa
Diamond subset: https://huggingface.co/datasets/Idavidrein/gpqa (config: gpqa_diamond)

Format:
- Question: Text question with multiple choice options
- Choices: List of answer choices (typically A, B, C, D)
- Answer: Index or letter of correct answer
"""

from typing import Any, Dict, Tuple

# Prompt for GPQA multiple choice questions
gpqa_prompt = """

Please solve this graduate-level science question step by step.

Instructions:
1. Analyze the question carefully
2. Consider each option systematically
3. Explain your reasoning
4. Provide your final answer as a single letter (A, B, C, or D)
5. Wrap your final answer letter in \\boxed{}, for example: \\boxed{A}
"""


def gpqa_formatter(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Format example from GPQA dataset.

    GPQA datasets typically have these fields:
    - Question: The question text
    - choices: List of answer choices OR separate fields (A, B, C, D)
    - Answer: The correct answer (as index or letter)

    Args:
        example: Dataset example dict

    Returns:
        Tuple of (formatted_question, correct_answer)
    """
    # Extract question text - try multiple field name variations
    question_text = (
        example.get("Question")
        or example.get("question")
        or example.get("Problem")
        or example.get("problem")
        or example.get("query")
        or ""
    )
    if not question_text:
        # Provide helpful error message with available fields
        available_fields = list(example.keys())
        raise ValueError(
            f"Example missing question field. Available fields: {available_fields}"
        )

    # Format multiple choice options
    # GPQA datasets may have different formats for choices
    choices = None

    # Try different field name patterns
    if "choices" in example:
        choices = example["choices"]
    elif "Choices" in example:
        choices = example["Choices"]
    elif all(key in example for key in ["A", "B", "C", "D"]):
        # Choices stored as separate fields
        choices = [
            example["A"],
            example["B"],
            example["C"],
            example["D"],
        ]
    elif all(key in example for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]):
        # GPQA format: Correct + 3 Incorrect answers
        # Shuffle them deterministically to create A/B/C/D options
        import random
        import hashlib

        correct_ans = example["Correct Answer"]
        incorrect_answers = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]

        # Combine all answers
        all_answers = [correct_ans] + incorrect_answers

        # Create deterministic shuffle based on question text
        shuffled_indices = list(range(4))
        rng = random.Random(int(hashlib.sha256(question_text.encode("utf-8")).hexdigest()[:16], 16))
        rng.shuffle(shuffled_indices)

        # Apply shuffle
        choices = [all_answers[i] for i in shuffled_indices]

        # Find where the correct answer ended up - store for later
        correct_position = shuffled_indices.index(0)

    # Build formatted question with choices
    formatted_question = question_text.strip()

    if choices:
        formatted_question += "\n\nOptions:"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            formatted_question += f"\n{letter}. {choice}"

    # Add prompt
    formatted_question += gpqa_prompt

    # Determine correct answer letter
    # If we have GPQA format with shuffled answers, use the correct position
    if 'correct_position' in locals():
        answer_letter = chr(65 + correct_position)
    else:
        # Extract correct answer from Answer field
        # Answer might be stored as letter (A/B/C/D) or index (0/1/2/3)
        answer = example.get("Answer") or example.get("answer", "")

        # Normalize answer to letter format
        if isinstance(answer, int):
            # If answer is index, convert to letter
            answer_letter = chr(65 + answer)  # 0->A, 1->B, etc
        elif isinstance(answer, str):
            # If already a letter, keep it; if it's a number string, convert it
            answer = answer.strip().upper()
            if answer.isdigit():
                answer_letter = chr(65 + int(answer))
            elif len(answer) == 1 and answer in "ABCD":
                answer_letter = answer
            else:
                # If answer is the full text, try to match it to choices
                if choices:
                    for i, choice in enumerate(choices):
                        if answer.lower() == choice.lower():
                            answer_letter = chr(65 + i)
                            break
                    else:
                        # Default to A if we can't match
                        answer_letter = "A"
                else:
                    answer_letter = "A"
        else:
            answer_letter = "A"  # Default fallback

    return formatted_question, answer_letter


def gpqa_evaluator(
    prediction: str,
    ground_truth: str,
    dataset_name: str = None,
    problem_text: str = None,
) -> bool:
    """
    Evaluate GPQA prediction against ground truth.

    For multiple choice questions, we check if the predicted letter (A/B/C/D)
    matches the ground truth letter.

    Args:
        prediction: Model's predicted answer
        ground_truth: Correct answer (letter A/B/C/D)
        dataset_name: Name of dataset (for context)
        problem_text: Original problem text (unused, for API compatibility)

    Returns:
        True if prediction matches ground truth, False otherwise
    """
    if not prediction or not ground_truth:
        return False

    # Extract letter from prediction
    # Look for boxed answer first
    import re

    # Pattern 1: \boxed{A}, \boxed{B}, etc
    boxed_match = re.search(r'\\boxed\{([A-D])\}', prediction, re.IGNORECASE)
    if boxed_match:
        pred_letter = boxed_match.group(1).upper()
    else:
        # Pattern 2: Look for standalone letter near end of response
        # Check last 200 characters for "answer is A" or similar
        search_text = prediction[-200:] if len(prediction) > 200 else prediction

        # Common answer patterns
        patterns = [
            r'(?:answer|choice|option)\s+(?:is\s+)?([A-D])',
            r'\b([A-D])\s+is\s+(?:correct|right)',
            r'(?:select|choose)\s+([A-D])',
            r'\(([A-D])\)',  # (A), (B), etc
            r'^([A-D])$',  # Just the letter alone
        ]

        pred_letter = None
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
            if match:
                pred_letter = match.group(1).upper()
                break

        if not pred_letter:
            # Last resort: find any A/B/C/D letter in the text
            letters = re.findall(r'\b([A-D])\b', search_text, re.IGNORECASE)
            if letters:
                pred_letter = letters[-1].upper()  # Take the last occurrence
            else:
                return False  # No valid letter found

    # Normalize ground truth
    gt_letter = ground_truth.strip().upper()
    if gt_letter not in "ABCD":
        # If ground truth is not a letter, try to extract it
        gt_match = re.search(r'([A-D])', gt_letter, re.IGNORECASE)
        if gt_match:
            gt_letter = gt_match.group(1).upper()
        else:
            return False

    # Compare
    return pred_letter == gt_letter


def gpqa_scorer(predictions: list, answers: list) -> Dict[str, float]:
    """
    Score multiple GPQA predictions.

    Args:
        predictions: List of predicted answers
        answers: List of ground truth answers

    Returns:
        Dictionary with accuracy score
    """
    if len(predictions) != len(answers):
        raise ValueError(
            f"Predictions ({len(predictions)}) and answers ({len(answers)}) must have same length"
        )

    correct = sum(
        1 for pred, ans in zip(predictions, answers)
        if gpqa_evaluator(pred, ans)
    )

    accuracy = correct / len(predictions) if predictions else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
    }
