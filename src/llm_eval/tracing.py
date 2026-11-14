"""Lightweight tracing: JSONL spans on disk, plus an optional OpenTelemetry bridge.

JSONL keeps the framework dependency-free and easy to diff in CI. If
``opentelemetry-api`` is installed and ``otel=True``, every span is mirrored
to the active OpenTelemetry tracer (export to Jaeger, Tempo, Langfuse, etc.).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Span:
    def __init__(self, name: str, trace_id: str, attributes: dict[str, Any]):
        self.name = name
        self.trace_id = trace_id
        self.span_id = uuid.uuid4().hex[:16]
        self.attributes = dict(attributes)
        self.status = "ok"
        self.error: str | None = None
        self._start = time.perf_counter()
        self.started_at = time.time()
        self.duration_ms = 0.0

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "started_at": round(self.started_at, 3),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class Tracer:
    def __init__(self, path: str | Path | None = None, *, otel: bool = False):
        self.path = Path(path) if path else None
        self.spans: list[dict[str, Any]] = []
        self._local = threading.local()
        self._lock = threading.Lock()
        self._otel = None
        if otel:
            try:
                from opentelemetry import trace  # type: ignore

                self._otel = trace.get_tracer("llm_eval")
            except ImportError:  # pragma: no cover - optional dependency
                self._otel = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    @property
    def current_trace_id(self) -> str:
        return getattr(self._local, "trace_id", None) or uuid.uuid4().hex

    @contextmanager
    def trace(self, trace_id: str | None = None) -> Iterator[str]:
        self._local.trace_id = trace_id or uuid.uuid4().hex
        try:
            yield self._local.trace_id
        finally:
            self._local.trace_id = None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name, self.current_trace_id, attributes)
        otel_cm = self._otel.start_as_current_span(name) if self._otel else None
        otel_span = otel_cm.__enter__() if otel_cm else None
        try:
            yield span
        except Exception as exc:
            span.fail(repr(exc))
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - span._start) * 1000, 2)
            if otel_span is not None:
                for key, value in span.attributes.items():
                    otel_span.set_attribute(f"llm.{key}", value if isinstance(value, (str, int, float, bool)) else str(value))
                if span.error:
                    otel_span.set_attribute("error", True)
                otel_cm.__exit__(None, None, None)
            self._record(span.to_dict())

    def _record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.spans.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
