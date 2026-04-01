"""Metrics capture for nnsight remote execution.

This module provides utilities for capturing timestamps from nnsight remote
execution without modifying the nnsight client. It works by intercepting
stdout and parsing the status updates that JobStatusDisplay emits.

Usage:
    from src.metrics import capture_remote_metrics

    with capture_remote_metrics() as metrics:
        with model.trace("Hello", remote=True):
            output = model.output.save()

    print(metrics.summary())
"""

from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class RemoteMetrics:
    """Metrics collected from a remote nnsight execution."""

    timestamps: dict[str, float] = field(default_factory=dict)
    job_id: str | None = None

    @property
    def queue_duration(self) -> float | None:
        """Time spent waiting in queue (QUEUED -> RUNNING)."""
        if "QUEUED" in self.timestamps and "RUNNING" in self.timestamps:
            return self.timestamps["RUNNING"] - self.timestamps["QUEUED"]
        return None

    @property
    def execution_duration(self) -> float | None:
        """Time spent executing (RUNNING -> COMPLETED)."""
        if "RUNNING" in self.timestamps and "COMPLETED" in self.timestamps:
            return self.timestamps["COMPLETED"] - self.timestamps["RUNNING"]
        return None

    @property
    def total_duration(self) -> float | None:
        """Total time from first status to completion."""
        if self.timestamps:
            first = min(self.timestamps.values())
            last = max(self.timestamps.values())
            return last - first
        return None

    def summary(self) -> str:
        """Return a formatted summary of the metrics."""
        lines = []
        lines.append(f"  Job ID: {self.job_id or 'unknown'}")
        lines.append(f"  Statuses: {' -> '.join(self.timestamps.keys())}")

        if self.queue_duration is not None:
            lines.append(f"  Queue time:     {self.queue_duration:.3f}s")
        if self.execution_duration is not None:
            lines.append(f"  Execution time: {self.execution_duration:.3f}s")
        if self.total_duration is not None:
            lines.append(f"  Total time:     {self.total_duration:.3f}s")

        return "\n".join(lines)


@dataclass
class AggregateMetrics:
    """Aggregated metrics from multiple remote executions."""

    executions: list[RemoteMetrics] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.executions)

    @property
    def total_queue_time(self) -> float:
        return sum(m.queue_duration or 0 for m in self.executions)

    @property
    def total_execution_time(self) -> float:
        return sum(m.execution_duration or 0 for m in self.executions)

    @property
    def total_time(self) -> float:
        return sum(m.total_duration or 0 for m in self.executions)

    @property
    def avg_queue_time(self) -> float | None:
        times = [m.queue_duration for m in self.executions if m.queue_duration is not None]
        return sum(times) / len(times) if times else None

    @property
    def avg_execution_time(self) -> float | None:
        times = [m.execution_duration for m in self.executions if m.execution_duration is not None]
        return sum(times) / len(times) if times else None

    def summary(self) -> str:
        lines = [
            f"Remote Executions: {self.count}",
            f"Total queue time:     {self.total_queue_time:.3f}s",
            f"Total execution time: {self.total_execution_time:.3f}s",
            f"Total time:           {self.total_time:.3f}s",
        ]
        if self.avg_queue_time is not None:
            lines.append(f"Avg queue time:       {self.avg_queue_time:.3f}s")
        if self.avg_execution_time is not None:
            lines.append(f"Avg execution time:   {self.avg_execution_time:.3f}s")
        return "\n".join(lines)


class _StdoutCapture:
    """Wraps stdout to capture nnsight status updates and record timestamps."""

    # Known statuses from nnsight ResponseModel.JobStatus
    STATUSES = {"RECEIVED", "QUEUED", "DISPATCHED", "RUNNING", "COMPLETED", "ERROR"}

    # Regex to extract job_id and status from output (handles ANSI codes)
    STATUS_RE = re.compile(
        r"\[([\w-]+)\].*?(RECEIVED|QUEUED|DISPATCHED|RUNNING|COMPLETED|ERROR)\s"
    )

    # ANSI escape code pattern for stripping colors
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, original: TextIO, aggregate: AggregateMetrics):
        self._original = original
        self._aggregate = aggregate
        self._current: RemoteMetrics | None = None

    def write(self, text: str) -> int:
        # 1. Pass through to real stdout (transparent logging)
        result = self._original.write(text)

        # 2. Parse for status updates (strip ANSI codes first)
        clean = self.ANSI_RE.sub("", text)
        match = self.STATUS_RE.search(clean)

        if match:
            job_id, status = match.groups()

            # New job? Start tracking
            if self._current is None or self._current.job_id != job_id:
                if self._current is not None:
                    self._aggregate.executions.append(self._current)
                self._current = RemoteMetrics(job_id=job_id)

            # Record timestamp for first occurrence of each status
            if status not in self._current.timestamps:
                self._current.timestamps[status] = time.time()

            # Job finished? Finalize it
            if status in ("COMPLETED", "ERROR"):
                self._aggregate.executions.append(self._current)
                self._current = None

        return result

    def flush(self) -> None:
        self._original.flush()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


@contextmanager
def capture_remote_metrics():
    """Context manager that captures metrics from nnsight remote execution.

    Yields:
        AggregateMetrics: Object containing all captured execution metrics.

    Example:
        with capture_remote_metrics() as metrics:
            with model.session(remote=True):
                with model.trace("Hello"):
                    output = model.output.save()

        print(metrics.summary())
    """
    aggregate = AggregateMetrics()
    capture = _StdoutCapture(sys.stdout, aggregate)

    old_stdout = sys.stdout
    sys.stdout = capture  # type: ignore

    try:
        yield aggregate
    finally:
        sys.stdout = old_stdout
