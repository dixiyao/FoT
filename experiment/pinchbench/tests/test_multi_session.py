"""Tests for multi-session task support in lib_agent."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_agent import _archive_transcript  # noqa: E402
from lib_tasks import Task  # noqa: E402


def _make_task(
    task_id: str = "task_test",
    sessions: list | None = None,
    **kwargs,
) -> Task:
    """Create a Task with optional sessions in frontmatter."""
    frontmatter = kwargs.pop("frontmatter", {})
    if sessions is not None:
        frontmatter["sessions"] = sessions
    return Task(
        task_id=task_id,
        name="Test Task",
        category="test",
        grading_type="automated",
        timeout_seconds=120,
        workspace_files=[],
        prompt="Default prompt",
        expected_behavior="",
        grading_criteria=[],
        frontmatter=frontmatter,
        **kwargs,
    )


class TestMultiSessionFrontmatterParsing(unittest.TestCase):
    """Test that multi-session frontmatter is correctly parsed from task files."""

    def test_sessions_list_parsed(self) -> None:
        sessions = [
            {"id": "s1", "prompt": "Hello"},
            {"id": "s2", "prompt": "Follow up"},
        ]
        task = _make_task(sessions=sessions)
        self.assertEqual(task.frontmatter.get("sessions"), sessions)

    def test_new_session_flag_parsed(self) -> None:
        sessions = [
            {"id": "s1", "prompt": "Hello"},
            {"id": "s2", "new_session": True, "prompt": "Fresh start"},
        ]
        task = _make_task(sessions=sessions)
        parsed = task.frontmatter.get("sessions", [])
        self.assertTrue(parsed[1].get("new_session", False))

    def test_no_sessions_default_empty(self) -> None:
        task = _make_task()
        self.assertEqual(task.frontmatter.get("sessions", []), [])

    def test_string_session_entry(self) -> None:
        sessions = ["Just a string prompt"]
        task = _make_task(sessions=sessions)
        self.assertEqual(task.frontmatter.get("sessions"), sessions)


class TestNewSessionHandling(unittest.TestCase):
    """Test that new_session: true triggers session isolation logic."""

    def test_new_session_flag_detected(self) -> None:
        """new_session: true should be detected from session entry."""
        sessions = [
            {"id": "s1", "prompt": "Hello"},
            {"id": "s2", "new_session": True, "prompt": "Fresh start"},
        ]
        task = _make_task(sessions=sessions)
        parsed = task.frontmatter.get("sessions", [])
        self.assertEqual(len(parsed), 2)

        # First session: no new_session
        self.assertFalse(bool(parsed[0].get("new_session", False)))

        # Second session: new_session = True
        is_new_session = bool(parsed[1].get("new_session", False))
        self.assertTrue(is_new_session)

    def test_sessions_without_new_session(self) -> None:
        """Sessions without new_session should have is_new_session=False."""
        sessions = [
            {"id": "s1", "prompt": "Hello"},
            {"id": "s2", "prompt": "Follow up"},
        ]
        task = _make_task(sessions=sessions)
        parsed = task.frontmatter.get("sessions", [])
        for entry in parsed:
            self.assertFalse(bool(entry.get("new_session", False)))

    @patch("lib_agent.subprocess.run")
    @patch("lib_agent._load_transcript")
    @patch("lib_agent.cleanup_agent_sessions")
    @patch("lib_agent._archive_transcript")
    @patch("lib_agent.prepare_task_workspace")
    @patch("lib_agent._get_agent_workspace")
    @patch("lib_agent.is_fws_task", return_value=False)
    def test_new_session_triggers_archive_and_cleanup(
        self,
        mock_is_fws: MagicMock,
        mock_get_ws: MagicMock,
        mock_prepare: MagicMock,
        mock_archive: MagicMock,
        mock_cleanup: MagicMock,
        mock_load: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """execute_openclaw_task should call archive then cleanup for new_session."""
        from lib_agent import execute_openclaw_task

        mock_get_ws.return_value = Path("/tmp/test_workspace")
        mock_prepare.return_value = Path("/tmp/test_workspace")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_load.return_value = ([], None)

        sessions = [
            {"id": "s1", "prompt": "Hello"},
            {"id": "s2", "new_session": True, "prompt": "Fresh start"},
        ]
        task = _make_task(sessions=sessions)

        execute_openclaw_task(
            task=task,
            agent_id="test-agent",
            model_id="test-model",
            run_id="test-run",
            timeout_multiplier=1.0,
            skill_dir=ROOT,
        )

        # _archive_transcript should have been called (before cleanup)
        self.assertTrue(mock_archive.called, "_archive_transcript should be called for new_session")
        # cleanup_agent_sessions should have been called (for initial cleanup + new_session)
        self.assertTrue(mock_cleanup.called, "cleanup_agent_sessions should be called")


class TestArchiveTranscript(unittest.TestCase):
    """Test the _archive_transcript helper function."""

    @patch("lib_agent._load_transcript")
    def test_archive_copies_transcript(self, mock_load: MagicMock) -> None:
        """_archive_transcript should copy transcript to session-specific file."""
        transcript_data = [
            {"type": "message", "message": {"role": "assistant", "content": "hello"}}
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake transcript file
            transcript_file = Path(tmpdir) / "original.jsonl"
            transcript_file.write_text(json.dumps(transcript_data[0]) + "\n")
            mock_load.return_value = (transcript_data, transcript_file)

            output_dir = Path(tmpdir) / "output"
            _archive_transcript(
                agent_id="test-agent",
                current_session_id="session_1",
                start_time=0.0,
                output_dir=output_dir,
                task_id="task_test",
                session_index=0,
            )

            archive_path = output_dir / "task_test_session0.jsonl"
            self.assertTrue(archive_path.exists())

    @patch("lib_agent._load_transcript")
    def test_archive_no_transcript_path(self, mock_load: MagicMock) -> None:
        """_archive_transcript should handle missing transcript gracefully."""
        mock_load.return_value = ([], None)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            # Should not raise
            _archive_transcript(
                agent_id="test-agent",
                current_session_id="session_1",
                start_time=0.0,
                output_dir=output_dir,
                task_id="task_test",
                session_index=0,
            )


class TestMultiSessionTaskLoading(unittest.TestCase):
    """Test that multi-session tasks load correctly from the tasks directory."""

    def test_task_second_brain_has_sessions(self) -> None:
        """task_second_brain.md should have sessions with new_session: true."""
        from lib_tasks import TaskLoader

        tasks_dir = ROOT / "tasks"
        if not (tasks_dir / "task_second_brain.md").exists():
            self.skipTest("task_second_brain.md not found")

        loader = TaskLoader(tasks_dir)
        task = loader.load_task(tasks_dir / "task_second_brain.md")
        sessions = task.frontmatter.get("sessions", [])
        self.assertGreater(len(sessions), 0, "task_second_brain should have sessions")

        # At least one session should have new_session: true
        has_new_session = any(
            isinstance(s, dict) and s.get("new_session") for s in sessions
        )
        self.assertTrue(has_new_session, "At least one session should have new_session: true")

    def test_task_iterative_code_refine_has_sessions(self) -> None:
        """task_iterative_code_refine.md should have sessions with new_session: true."""
        from lib_tasks import TaskLoader

        tasks_dir = ROOT / "tasks"
        if not (tasks_dir / "task_iterative_code_refine.md").exists():
            self.skipTest("task_iterative_code_refine.md not found")

        loader = TaskLoader(tasks_dir)
        task = loader.load_task(tasks_dir / "task_iterative_code_refine.md")
        sessions = task.frontmatter.get("sessions", [])
        self.assertEqual(len(sessions), 3, "task_iterative_code_refine should have 3 sessions")
        self.assertTrue(sessions[2].get("new_session", False), "3rd session should have new_session: true")


if __name__ == "__main__":
    unittest.main()
