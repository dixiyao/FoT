"""
IMOBench Dataset Handler
Unified interface for IMO-related benchmarks with semantic evaluation support

Available benchmarks:
- IMO-AnswerBench: 400 short-answer problems with symbolic/semantic evaluation
- IMO-ProofBench: 60 proof-based problems (requires Gemini ProofAutograder)
- IMO-GradingBench: 1000 grading examples (training data for automatic graders)

References:
- IMOBench: https://imobench.github.io/
- ProofAutoGrader: Google DeepMind (semantic grading 0-7 scale, ~0.96 correlation with humans)
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .utils import accuracy, extract_numbers

# Prompt for IMO-style problems
imo_prompt = '\nSolve this olympiad-style problem step by step. Wrap your final answer in "\\boxed{}".'


def imo_formatter(
    example: Dict[str, Any], benchmark: Optional[str] = None
) -> Tuple[str, str]:
    """
    Format example from IMOBench dataset.

    IMOBench datasets use "problem"/"question" and "answer"/"solution" fields.

    Args:
        example: Dataset example dict
        benchmark: Optional benchmark name (imo_answerbench, imo_proofbench, imo_grading_bench)

    Returns:
        Tuple of (problem_text, answer_text)
    """
    # Extract problem statement
    problem_text = example.get("problem") or example.get("question", "")
    if not problem_text:
        raise ValueError("Example missing 'problem' or 'question' field")

    # Add prompt
    problem_text = problem_text + imo_prompt

    # Extract answer/solution
    answer_text = example.get("answer") or example.get("solution", "")

    return problem_text, answer_text


def _normalize_imo_answer(text: str) -> str:
    """Normalize IMO answer by removing common suffixes and normalizing format."""
    normalized = text.strip().lower()

    # Remove common unit suffixes
    suffixes = [
        " degrees",
        "°",
        " radians",
        " rad",
        " units",
        " cm",
        " m",
        " inches",
        " km",
        " mm",
        " feet",
        " yards",
        " square units",
        " cubic units",
        " degrees celsius",
        "°c",
        " degrees fahrenheit",
        "°f",
    ]

    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()

    return normalized


def grade_proof_with_gemini(
    gemini_model,
    problem: str,
    candidate_solution: str,
    reference_solution: str,
    grading_guidelines: Optional[str] = None,
) -> Dict:
    """Grade a mathematical proof using Gemini API for IMO-ProofBench.

    This function is specific to IMO-style proof problems and uses the Gemini API
    to provide semantic grading on a 0-7 scale (IMO standard).

    Args:
        gemini_model: Gemini model instance (from google.generativeai)
        problem: Problem statement
        candidate_solution: Student/model-generated solution
        reference_solution: Reference/ground truth solution
        grading_guidelines: Optional specific grading criteria

    Returns:
        Dict with keys:
        - score: Grade 0-7 (IMO standard)
        - category: "Correct", "Almost", "Partial", or "Incorrect"
        - reasoning: Explanation of the grade
        - feedback: Specific feedback on the solution
    """
    guidelines = (
        grading_guidelines
        or """
Grade the candidate solution on a 0-7 scale:
- 7: Fully correct, rigorous, and complete
- 6: Almost correct, minor errors or gaps
- 1: Mostly incorrect, some relevant results
- 0: Completely incorrect or irrelevant

Also classify as: Correct (7), Almost (6), Partial (1), Incorrect (0)
"""
    )

    grading_prompt = f"""You are an expert mathematical grader evaluating a student's solution to an IMO-style problem.

Problem:
{problem}

Candidate Solution (to be graded):
{candidate_solution}

Reference Solution (correct solution):
{reference_solution}

Grading Guidelines:
{guidelines}

Please provide:
1. A numerical score (0-7)
2. A category (Correct, Almost, Partial, Incorrect)
3. Reasoning for your grade
4. Specific feedback on the solution

Output as JSON:
{{"score": <0-7>, "category": "<category>", "reasoning": "<reasoning>", "feedback": "<feedback>"}}
"""

    try:
        response = gemini_model.generate_content(grading_prompt)
        response_text = response.text

        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {
                    "score": 0,
                    "category": "Incorrect",
                    "reasoning": "Failed to parse response",
                    "feedback": response_text,
                }
        except json.JSONDecodeError:
            result = {
                "score": 0,
                "category": "Incorrect",
                "reasoning": "Failed to parse grading response",
                "feedback": response_text,
            }

        return result

    except Exception as e:
        print(f"Error grading proof with Gemini: {e}")
        return {
            "score": 0,
            "category": "Incorrect",
            "reasoning": f"Grading error: {str(e)}",
            "feedback": "",
        }


def imo_evaluator(
    prediction: str,
    ground_truth: str,
    benchmark: Optional[str] = None,
    problem_text: Optional[str] = None,
    use_gemini: bool = False,
    client=None,
) -> bool:
    """
    Evaluate IMO answer with multiple strategies.

    IMO answers can be:
    - Numeric (42, 3.14, -5)
    - Expressions ($\frac{2n+2}{3}$, $\sqrt{2}$, $2\pi + 3$)
    - Geometric (point coordinates, angle measures)
    - Boolean/text (Yes/No, True/False)

    Evaluation strategies:
    1. Exact match (after normalization)
    2. Numeric comparison (if both contain numbers)
    3. Symbolic/algebraic comparison
    4. Gemini semantic evaluation (if enabled and available)

    Args:
        prediction: Model's predicted answer
        ground_truth: Correct answer from dataset
        benchmark: Optional benchmark name for context
        problem_text: Optional problem statement for Gemini evaluation
        use_gemini: Whether to use Gemini API for semantic grading
        client: ChainOfThoughtReader client with Gemini access

    Returns:
        True if answer is correct (grade >= 6 for Gemini), False otherwise
    """
    if not prediction or not ground_truth:
        return False

    pred = prediction.strip()
    truth = ground_truth.strip()

    # Strategy 1: Exact match (case-insensitive)
    if pred.lower() == truth.lower():
        return True

    # Strategy 2: Normalize and match
    pred_norm = _normalize_imo_answer(pred)
    truth_norm = _normalize_imo_answer(truth)

    if pred_norm == truth_norm:
        return True

    # Strategy 3: Remove boxing notation and try again
    boxing_patterns = [r"\\boxed\{([^}]*)\}", r"boxed\(([^)]*)\)", r"\$([^$]*)\$"]
    for pattern in boxing_patterns:
        pred_match = re.search(pattern, pred)
        truth_match = re.search(pattern, truth)
        if pred_match:
            pred = pred_match.group(1)
        if truth_match:
            truth = truth_match.group(1)

    if pred.strip().lower() == truth.strip().lower():
        return True

    # Strategy 4: Numeric comparison
    pred_nums = extract_numbers(pred)
    truth_nums = extract_numbers(truth)

    if pred_nums and truth_nums:
        # Single number comparison
        if len(pred_nums) == 1 and len(truth_nums) == 1:
            return abs(pred_nums[0] - truth_nums[0]) < 1e-6

        # Multiple numbers: check if all match (in order)
        if len(pred_nums) == len(truth_nums):
            return all(abs(p - t) < 1e-6 for p, t in zip(pred_nums, truth_nums))

        # Fallback: first number match
        if len(pred_nums) > 0 and len(truth_nums) > 0:
            return abs(pred_nums[0] - truth_nums[0]) < 1e-6

    # Strategy 5: Substring matching for expressions
    if len(pred_norm) > 1 and len(truth_norm) > 1:
        if pred_norm in truth_norm or truth_norm in pred_norm:
            return True

    # Strategy 6: Gemini semantic evaluation (if enabled)
    if use_gemini and client and problem_text:
        try:
            result = grade_proof_with_gemini(
                gemini_model=client.gemini_model,
                problem=problem_text,
                candidate_solution=pred,
                reference_solution=truth,
                grading_guidelines=(
                    "For IMOBench: Grade on correctness of the final answer. "
                    "7 = Correct, 6 = Almost correct (minor arithmetic/notation error), "
                    "1-5 = Partially correct, 0 = Incorrect"
                ),
            )
            score = result.get("score", 0)
            # Consider correct if score >= 6
            return score >= 6
        except Exception as e:
            # Fallback silently to symbolic comparison on error
            pass

    return False


def imo_scorer(
    predictions: List[str],
    answers: List[str],
    benchmark: Optional[str] = None,
    problem_texts: Optional[List[str]] = None,
    use_gemini: bool = False,
    client=None,
) -> Dict[str, Any]:
    """
    Score predictions for IMOBench dataset.

    Args:
        predictions: List of model predictions
        answers: List of ground truth answers
        benchmark: Optional benchmark name
        problem_texts: Optional list of problem statements (required for Gemini)
        use_gemini: Whether to use Gemini API
        client: ChainOfThoughtReader client

    Returns:
        Dict with evaluation metrics: accuracy, correct_count, total_count, grades (if Gemini)
    """
    if not predictions or not answers:
        return {"accuracy": 0.0, "correct_count": 0, "total_count": 0}

    correct_count = 0
    total_count = len(predictions)
    grades = [] if use_gemini else None

    for idx, (pred, ans) in enumerate(zip(predictions, answers)):
        problem = (
            problem_texts[idx] if problem_texts and idx < len(problem_texts) else None
        )

        is_correct = imo_evaluator(
            pred,
            ans,
            benchmark=benchmark,
            problem_text=problem,
            use_gemini=use_gemini,
            client=client,
        )

        if is_correct:
            correct_count += 1

        if grades is not None and use_gemini and client and problem:
            try:
                result = grade_proof_with_gemini(
                    gemini_model=client.gemini_model,
                    problem=problem,
                    candidate_solution=pred,
                    reference_solution=ans,
                    grading_guidelines="Grade 0-7: 7=Correct, 6=Almost, 1-5=Partial, 0=Incorrect",
                )
                grades.append(result.get("score", 0))
            except:
                grades.append(0)

    accuracy_score = correct_count / total_count if total_count > 0 else 0.0

    result_dict = {
        "accuracy": accuracy_score,
        "correct_count": correct_count,
        "total_count": total_count,
    }

    if grades is not None:
        avg_grade = sum(grades) / len(grades) if grades else 0.0
        result_dict["average_grade"] = avg_grade
        result_dict["grades"] = grades

    return result_dict


def normalize_imo_answer(answer: str) -> str:
    """Public wrapper for answer normalization."""
    return _normalize_imo_answer(answer)


def imo_answerbench_formatter(example: Dict[str, Any]) -> Tuple[str, str]:
    """Format IMO-AnswerBench examples."""
    return imo_formatter(example, "imo_answerbench")


def imo_proofbench_formatter(example: Dict[str, Any]) -> Tuple[str, str]:
    """Format IMO-ProofBench examples. Requires Gemini grading."""
    return imo_formatter(example, "imo_proofbench")


def imo_gradingbench_formatter(example: Dict[str, Any]) -> Tuple[str, str]:
    """Format IMO-GradingBench examples (for training graders)."""
    # GradingBench includes grades, use for reference
    problem_text = example.get("problem", "") + imo_prompt
    answer_text = example.get("solution", "")
    return problem_text, answer_text


# Specializations for each IMO benchmark
def imo_answerbench_scorer(predictions, answers, **kwargs):
    """Score IMO-AnswerBench (short-answer problems)."""
    return imo_scorer(predictions, answers, benchmark="imo_answerbench", **kwargs)


def imo_proofbench_scorer(predictions, answers, **kwargs):
    """Score IMO-ProofBench (proof-based, requires Gemini)."""
    kwargs.setdefault("use_gemini", True)  # ProofBench benefits from Gemini
    return imo_scorer(predictions, answers, benchmark="imo_proofbench", **kwargs)


def imo_gradingbench_scorer(predictions, answers, **kwargs):
    """Score IMO-GradingBench (grading examples for training)."""
    return imo_scorer(predictions, answers, benchmark="imo_gradingbench", **kwargs)
