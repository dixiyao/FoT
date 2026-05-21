"""
Axiom logging integration for PinchBench.

Sends structured events to Axiom for observability and debugging of benchmark runs.
Requires AXIOM_API_TOKEN and optionally AXIOM_DATASET (defaults to 'pinchbench').
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("benchmark.axiom")

AXIOM_INGEST_URL = "https://api.axiom.co/v1/datasets/{dataset}/ingest"
DEFAULT_DATASET = "pinchbench"


@dataclass
class AxiomEvent:
    """Base event structure for Axiom logging."""
    event: str
    run_id: str
    instance_id: Optional[str] = None
    instance_ip: Optional[str] = None
    model: Optional[str] = None
    benchmark_version: Optional[str] = None
    _time: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    
    # Optional fields populated by specific events
    task_id: Optional[str] = None
    task_num: Optional[int] = None
    total_tasks: Optional[int] = None
    score: Optional[float] = None
    max_score: Optional[float] = None
    score_pct: Optional[float] = None
    grading_type: Optional[str] = None
    execution_time_sec: Optional[float] = None
    timed_out: Optional[bool] = None
    error: Optional[str] = None
    overall_score_pct: Optional[float] = None
    overall_earned: Optional[float] = None
    overall_possible: Optional[float] = None
    total_cost_usd: Optional[float] = None
    total_time_sec: Optional[float] = None
    submission_id: Optional[str] = None
    leaderboard_url: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class AxiomLogger:
    """
    Sends structured events to Axiom for benchmark observability.
    
    Silently no-ops if AXIOM_API_TOKEN is not set, allowing the benchmark
    to run without Axiom integration.
    """
    
    def __init__(
        self,
        run_id: str,
        model: Optional[str] = None,
        benchmark_version: Optional[str] = None,
        instance_id: Optional[str] = None,
        instance_ip: Optional[str] = None,
    ):
        self.token = os.environ.get("AXIOM_API_TOKEN", "")
        self.dataset = os.environ.get("AXIOM_DATASET", DEFAULT_DATASET)
        self.enabled = bool(self.token)
        
        # Common fields for all events in this run
        self.run_id = run_id
        self.model = model
        self.benchmark_version = benchmark_version
        self.instance_id = instance_id or os.environ.get("VULTR_INSTANCE_ID", "")
        self.instance_ip = instance_ip or os.environ.get("VULTR_INSTANCE_IP", "")
        
        # Batch events for efficiency
        self._batch: List[Dict[str, Any]] = []
        self._batch_size = 10  # Flush every N events
        
        if self.enabled:
            logger.info("Axiom logging enabled (dataset: %s)", self.dataset)
        else:
            logger.debug("Axiom logging disabled (no AXIOM_API_TOKEN)")
    
    def _make_event(self, event_type: str, **kwargs) -> AxiomEvent:
        """Create an event with common fields populated."""
        return AxiomEvent(
            event=event_type,
            run_id=self.run_id,
            model=self.model,
            benchmark_version=self.benchmark_version,
            instance_id=self.instance_id or None,
            instance_ip=self.instance_ip or None,
            **kwargs,
        )

    def heartbeat(self, current_task: Optional[str] = None, uptime_ms: Optional[int] = None) -> None:
        """Log a heartbeat event to indicate instance is still alive."""
        event = self._make_event(
            "heartbeat",
            task_id=current_task,
            extra={"uptime_ms": uptime_ms} if uptime_ms else None,
        )
        self._send_immediate(event)
    
    def _send_batch(self, events: List[Dict[str, Any]]) -> bool:
        """Send a batch of events to Axiom. Returns True on success."""
        if not self.enabled or not events:
            return True
        
        url = AXIOM_INGEST_URL.format(dataset=self.dataset)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = json.dumps(events).encode("utf-8")
        
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status >= 200 and resp.status < 300:
                    logger.debug("Sent %d events to Axiom", len(events))
                    return True
                else:
                    logger.warning("Axiom returned status %d", resp.status)
                    return False
        except urllib.error.URLError as e:
            logger.warning("Failed to send to Axiom: %s", e)
            return False
        except Exception as e:
            logger.warning("Axiom send error: %s", e)
            return False
    
    def _queue_event(self, event: AxiomEvent) -> None:
        """Queue an event for batched sending."""
        if not self.enabled:
            return
        self._batch.append(event.to_dict())
        if len(self._batch) >= self._batch_size:
            self.flush()
    
    def _send_immediate(self, event: AxiomEvent) -> None:
        """Send an event immediately (for important lifecycle events)."""
        if not self.enabled:
            return
        self._send_batch([event.to_dict()])
    
    def flush(self) -> None:
        """Flush any queued events to Axiom."""
        if self._batch:
            self._send_batch(self._batch)
            self._batch = []
    
    # ─── Event Methods ───────────────────────────────────────────────────────
    
    def run_start(self, total_tasks: int, suite: str = "all") -> None:
        """Log benchmark run start."""
        event = self._make_event(
            "run_start",
            total_tasks=total_tasks,
            extra={"suite": suite},
        )
        self._send_immediate(event)
    
    def task_start(self, task_id: str, task_num: int, total_tasks: int) -> None:
        """Log task execution start."""
        event = self._make_event(
            "task_start",
            task_id=task_id,
            task_num=task_num,
            total_tasks=total_tasks,
        )
        self._queue_event(event)
    
    def task_complete(
        self,
        task_id: str,
        task_num: int,
        total_tasks: int,
        score: float,
        max_score: float,
        grading_type: str,
        execution_time_sec: float,
        timed_out: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Log task completion with grading results."""
        score_pct = (score / max_score * 100) if max_score > 0 else 0
        event = self._make_event(
            "task_complete",
            task_id=task_id,
            task_num=task_num,
            total_tasks=total_tasks,
            score=score,
            max_score=max_score,
            score_pct=score_pct,
            grading_type=grading_type,
            execution_time_sec=execution_time_sec,
            timed_out=timed_out,
            error=error,
        )
        self._queue_event(event)
    
    def sanity_failed(self, score: float) -> None:
        """Log sanity check failure (fail-fast trigger)."""
        event = self._make_event(
            "sanity_failed",
            task_id="task_sanity",
            score=score,
            error="Sanity check scored 0%, triggering fail-fast",
        )
        self._send_immediate(event)
    
    def run_complete(
        self,
        overall_score_pct: float,
        overall_earned: float,
        overall_possible: float,
        total_cost_usd: Optional[float] = None,
        total_time_sec: Optional[float] = None,
        submission_id: Optional[str] = None,
        leaderboard_url: Optional[str] = None,
    ) -> None:
        """Log successful benchmark run completion."""
        self.flush()  # Ensure all task events are sent first
        event = self._make_event(
            "run_complete",
            overall_score_pct=overall_score_pct,
            overall_earned=overall_earned,
            overall_possible=overall_possible,
            total_cost_usd=total_cost_usd,
            total_time_sec=total_time_sec,
            submission_id=submission_id,
            leaderboard_url=leaderboard_url,
        )
        self._send_immediate(event)
    
    def run_failed(self, error: str, task_id: Optional[str] = None) -> None:
        """Log benchmark run failure."""
        self.flush()  # Ensure all prior events are sent
        event = self._make_event(
            "run_failed",
            error=error,
            task_id=task_id,
        )
        self._send_immediate(event)
    
    def upload_failed(self, error: str) -> None:
        """Log upload failure (run completed but couldn't submit)."""
        event = self._make_event(
            "upload_failed",
            error=error,
        )
        self._send_immediate(event)


# Module-level singleton for easy access
_logger: Optional[AxiomLogger] = None


def init_axiom(
    run_id: str,
    model: Optional[str] = None,
    benchmark_version: Optional[str] = None,
    instance_id: Optional[str] = None,
    instance_ip: Optional[str] = None,
) -> AxiomLogger:
    """Initialize the global Axiom logger. Call once at benchmark start."""
    global _logger
    _logger = AxiomLogger(
        run_id=run_id,
        model=model,
        benchmark_version=benchmark_version,
        instance_id=instance_id,
        instance_ip=instance_ip,
    )
    return _logger


def get_axiom() -> Optional[AxiomLogger]:
    """Get the global Axiom logger, or None if not initialized."""
    return _logger
