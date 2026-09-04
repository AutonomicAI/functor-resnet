from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class TemporalLogEntry:
    """
    Minimal temporal log entry for SLM governance and reification.

    Notes:
    - Keep payloads structured and small; store large artifacts externally and reference them by id/hash.
    """

    timestamp: float
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
        }


class TemporalLog(Protocol):
    """Protocol for an append-only temporal log."""

    def append(self, entry: TemporalLogEntry) -> None: ...
    def tail(self, n: int = 100) -> List[TemporalLogEntry]: ...


@dataclass
class InMemoryTemporalLog:
    """Simple in-memory temporal log. Swap with Kafka, SQLite, etc. in production."""

    _entries: List[TemporalLogEntry] = field(default_factory=list)

    def append(self, entry: TemporalLogEntry) -> None:
        self._entries.append(entry)

    def tail(self, n: int = 100) -> List[TemporalLogEntry]:
        return self._entries[-n:] if n > 0 else []


class CanonicalDSLBinding(Protocol):
    """
    Domain binding for SLMs.

    Typical implementation:
    - parse incoming domain DSL to canonical form
    - validate canonical form
    - (optional) invert/emit domain DSL from canonical
    """

    def parse_to_canonical(self, dsl: str) -> Any: ...
    def validate_canonical(self, canonical: Any) -> None: ...
    def emit_from_canonical(self, canonical: Any) -> str: ...


@dataclass
class NoOpDSLBinding:
    """Default DSL binding: accepts any string, no canonicalization."""

    def parse_to_canonical(self, dsl: str) -> Any:
        return dsl

    def validate_canonical(self, canonical: Any) -> None:
        return None

    def emit_from_canonical(self, canonical: Any) -> str:
        return str(canonical)


@dataclass
class GovernancePolicy:
    """
    Governance surface for an SLM.

    Keep this lightweight: it is the hook point for SLAs, compliance checks,
    and resource budgeting (latency/energy/token budgets).
    """

    require_audit: bool = True
    allow_external_calls: bool = True
    max_latency_ms: Optional[int] = None
    max_steps: Optional[int] = None  # orchestration step budget
    allowed_experts: Optional[List[str]] = None

    def check_expert_allowed(self, expert_name: str) -> None:
        if self.allowed_experts is not None and expert_name not in self.allowed_experts:
            raise PermissionError(f"Expert '{expert_name}' is not allowed by policy.")

    def check_step_budget(self, steps_used: int) -> None:
        if self.max_steps is not None and steps_used > self.max_steps:
            raise RuntimeError(
                f"Exceeded orchestration step budget: {steps_used} > {self.max_steps}"
            )


@dataclass
class OperationLog:
    enabled: bool = True
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, operation: str, details: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.entries.append(
            {
                "timestamp": datetime.now().timestamp(),
                "operation": operation,
                "details": details,
            }
        )


@dataclass
class SLMConfig:
    """Configuration for SLM behavior."""

    enable_audit_logging: bool = True
    enable_orchestration: bool = True
    max_experts: Optional[int] = None
    default_tail_log_n: int = 50
