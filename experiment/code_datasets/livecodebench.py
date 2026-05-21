"""
LiveCodeBench Code Generation Dataset Handler

LiveCodeBench is a benchmark for evaluating code generation on real-world programming problems.
Dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite

The dataset contains programming problems with:
- Problem description
- Public test cases (input/output pairs)
- Private test cases (hidden)
- Starter code (optional)

Evaluation requires executing generated code with test inputs and comparing outputs.
"""

import json
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Optional
import os
import re


# Prompt for code generation
livecodebench_code_prompt = """

Please solve this programming problem by writing clean, efficient code.

Requirements:
1. Read the problem description carefully
2. Understand the input/output format
3. Write a complete solution
4. Include proper input/output handling
5. Wrap your final code solution in markdown code blocks with triple backticks (```)

Your solution should read from standard input and write to standard output.
"""


class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timing out code execution"""
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out")

    # Set the signal handler
    if sys.platform != "win32":
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Windows doesn't support SIGALRM, use subprocess timeout instead
        yield


def livecodebench_formatter(example: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Format example from LiveCodeBench dataset.

    LiveCodeBench format:
    - question_title: Title of the problem
    - question_content: Full problem description
    - starter_code: Optional starter code
    - public_test_cases: List of {"input": "...", "output": "..."} dicts
    - metadata: Additional info

    Args:
        example: Dataset example dict

    Returns:
        Tuple of (formatted_question, test_cases_dict)
    """
    # Extract question content
    question_title = example.get("question_title", "")
    question_content = example.get("question_content") or example.get("question", "")

    if not question_content:
        raise ValueError("Example missing 'question_content' or 'question' field")

    # Build formatted question
    formatted_question = ""
    if question_title:
        formatted_question += f"## {question_title}\n\n"

    formatted_question += question_content.strip()

    # Add starter code if available
    starter_code = example.get("starter_code", "")
    if starter_code and starter_code.strip():
        formatted_question += f"\n\n### Starter Code:\n```python\n{starter_code.strip()}\n```"

    # Add prompt
    formatted_question += livecodebench_code_prompt

    # Extract test cases for evaluation
    # Note: bzantium/livecodebench stores test_cases as JSON strings, not parsed objects
    public_test_cases = example.get("public_test_cases", [])
    private_test_cases = example.get("private_test_cases", [])

    # Parse JSON strings if needed
    if isinstance(public_test_cases, str):
        try:
            public_test_cases = json.loads(public_test_cases)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse public_test_cases: {public_test_cases[:100]}")
            public_test_cases = []

    if isinstance(private_test_cases, str):
        try:
            private_test_cases = json.loads(private_test_cases)
        except json.JSONDecodeError:
            # Private test cases might be encoded/compressed, skip parsing
            private_test_cases = []

    test_cases = {
        "public_test_cases": public_test_cases,
        "private_test_cases": private_test_cases,
        "starter_code": starter_code,
    }

    return formatted_question, test_cases


def extract_code_from_response(response: str) -> Optional[str]:
    """
    Extract code from model response.

    Looks for code in markdown blocks or other common patterns.

    Args:
        response: Model's text response

    Returns:
        Extracted code string or None
    """
    if not response:
        return None

    # Pattern 1: Markdown code blocks with language
    # ```python\ncode\n``` or ```\ncode\n```
    patterns = [
        r'```python\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
        r'<code>(.*?)</code>',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the last code block (usually the final solution)
            code = matches[-1].strip()
            if code:
                return code

    # Pattern 2: If no code blocks, try to find code-like content
    # Look for lines that start with 'def', 'class', 'import', etc.
    code_indicators = ['def ', 'class ', 'import ', 'from ', 'if __name__']
    lines = response.split('\n')
    code_lines = []
    in_code = False

    for line in lines:
        if any(line.strip().startswith(indicator) for indicator in code_indicators):
            in_code = True

        if in_code:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    return None


def execute_code_with_input(
    code: str,
    test_input: str,
    timeout_seconds: int = 5,
    language: str = "python"
) -> Tuple[bool, str, str]:
    """
    Execute code with given input and return output.

    Args:
        code: Code string to execute
        test_input: Input string to pass to the code
        timeout_seconds: Maximum execution time in seconds
        language: Programming language (default: python)

    Returns:
        Tuple of (success: bool, stdout: str, stderr: str)
    """
    if language != "python":
        return False, "", f"Unsupported language: {language}"

    try:
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute code with input
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
            )

            # Communicate with timeout
            stdout, stderr = process.communicate(
                input=test_input,
                timeout=timeout_seconds
            )

            success = process.returncode == 0
            return success, stdout, stderr

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except:
            pass
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", f"Execution error: {str(e)}"


def normalize_output(output: str) -> str:
    """
    Normalize output for comparison.

    Args:
        output: Output string

    Returns:
        Normalized output string
    """
    if not output:
        return ""

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in output.split('\n')]

    # Remove trailing empty lines
    while lines and not lines[-1]:
        lines.pop()

    # Join back
    normalized = '\n'.join(lines)

    return normalized


def livecodebench_evaluator(
    prediction: str,
    test_cases: Dict[str, Any],
    dataset_name: Optional[str] = None,
    problem_text: Optional[str] = None,
) -> bool:
    """
    Evaluate LiveCodeBench code generation prediction.

    Executes the generated code with test cases and checks if outputs match.

    Args:
        prediction: Model's generated code (as text response)
        test_cases: Dict with public_test_cases and private_test_cases
        dataset_name: Name of dataset (for context)
        problem_text: Original problem text (unused, for API compatibility)

    Returns:
        True if all test cases pass, False otherwise
    """
    if not prediction or not test_cases:
        return False

    # Extract code from prediction
    code = extract_code_from_response(prediction)
    if not code:
        print("    [LiveCodeBench] Failed to extract code from response")
        return False

    # Get test cases
    public_tests = test_cases.get("public_test_cases", [])

    # If no public tests, cannot evaluate
    if not public_tests:
        print("    [LiveCodeBench] No test cases available for evaluation")
        return False

    # Run each test case
    passed_tests = 0
    total_tests = len(public_tests)

    for i, test_case in enumerate(public_tests):
        test_input = test_case.get("input", "")
        expected_output = test_case.get("output", "")

        # Execute code
        success, actual_output, stderr = execute_code_with_input(
            code,
            test_input,
            timeout_seconds=5
        )

        if not success:
            print(f"    [LiveCodeBench] Test {i+1}/{total_tests} failed: {stderr[:100]}")
            continue

        # Normalize outputs for comparison
        actual_normalized = normalize_output(actual_output)
        expected_normalized = normalize_output(expected_output)

        # Compare outputs
        if actual_normalized == expected_normalized:
            passed_tests += 1
        else:
            print(f"    [LiveCodeBench] Test {i+1}/{total_tests} output mismatch")
            print(f"      Expected: {expected_normalized[:100]}")
            print(f"      Got:      {actual_normalized[:100]}")

    # All tests must pass
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    print(f"    [LiveCodeBench] Passed {passed_tests}/{total_tests} tests ({success_rate:.1%})")

    return passed_tests == total_tests


def livecodebench_scorer(
    predictions: List[str],
    test_cases_list: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Score multiple LiveCodeBench predictions.

    Args:
        predictions: List of predicted code responses
        test_cases_list: List of test case dicts

    Returns:
        Dictionary with accuracy and pass@1 metrics
    """
    if len(predictions) != len(test_cases_list):
        raise ValueError(
            f"Predictions ({len(predictions)}) and test cases ({len(test_cases_list)}) must have same length"
        )

    correct = 0
    total = len(predictions)

    for pred, test_cases in zip(predictions, test_cases_list):
        if livecodebench_evaluator(pred, test_cases):
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "pass@1": accuracy,  # Same as accuracy for single sample
        "correct": correct,
        "total": total,
    }
