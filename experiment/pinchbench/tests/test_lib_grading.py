from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_grading import (  # noqa: E402
    _combine_grades,
    _normalize_judge_response,
    _parse_judge_response,
    GradeResult,
)


class JudgeNormalizationTests(unittest.TestCase):
    def test_normalize_judge_response_averages_summed_total_when_breakdown_is_unit_scale(
        self,
    ) -> None:
        parsed = {
            "scores": {
                "coverage": 0.75,
                "synthesis": 0.75,
                "structure": 0.75,
                "tone": 0.8,
                "conciseness": 0.8,
            },
            "total": 3.85,
            "notes": "Summed by mistake",
        }

        normalized = _normalize_judge_response(parsed)

        self.assertAlmostEqual(normalized["total"], 0.77)

    def test_hybrid_score_uses_normalized_judge_total(self) -> None:
        auto = GradeResult(
            task_id="task_email_triage",
            score=0.7062937062937062,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="",
        )
        judge = GradeResult(
            task_id="task_email_triage",
            score=0.87,
            max_score=1.0,
            grading_type="llm_judge",
            breakdown={},
            notes="",
        )

        class _Task:
            task_id = "task_email_triage"
            grading_weights = {"automated": 0.4, "llm_judge": 0.6}

        combined = _combine_grades(_Task(), auto, judge)

        self.assertAlmostEqual(combined.score, 0.8045174825174824)

    def test_parse_judge_response_prefers_latest_assistant_json_over_embedded_tool_json(
        self,
    ) -> None:
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                'Tool: web_search({"count": 10, "query": "WWDC 2025"})\n'
                                'Result: {"query": "WWDC 2025", "count": 10}'
                            ),
                        }
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "NO_REPLY"}],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '{"scores": {"accuracy": 0.75, "completeness": 1.0}, '
                                '"total": 0.875, "notes": "Final judgment"}'
                            ),
                        }
                    ],
                },
            },
        ]

        parsed = _parse_judge_response(transcript)

        self.assertEqual(parsed["scores"]["accuracy"], 0.75)
        self.assertEqual(parsed["scores"]["completeness"], 1.0)
        self.assertEqual(parsed["total"], 0.875)

    def test_parse_judge_response_ignores_waiting_messages_before_final_json(self) -> None:
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Waiting for remaining parts (6-7)."}
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '{"scores": {"clarity": 0.75, "accuracy": 0.85}, '
                                '"total": 0.8, "notes": "Looks good"}'
                            ),
                        }
                    ],
                },
            },
        ]

        parsed = _parse_judge_response(transcript)

        self.assertEqual(parsed["scores"]["clarity"], 0.75)
        self.assertEqual(parsed["total"], 0.8)


if __name__ == "__main__":
    unittest.main()
