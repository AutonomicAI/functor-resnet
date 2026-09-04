from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib

import numpy as np

from .governance import (
    CanonicalDSLBinding,
    GovernancePolicy,
    InMemoryTemporalLog,
    NoOpDSLBinding,
    SLMConfig,
    TemporalLog,
    TemporalLogEntry,
)


@dataclass
class EnsembleSpec:
    """
    Declarative pipeline-as-ensemble spec.

    pattern:
      - 'chain'    : sequential composition over ordered experts
      - 'parallel' : evaluate all experts then aggregate
      - 'moe'      : route to one expert (hard) or many (soft) using router/weights
      - 'cascade'  : cheap-first then optionally heavy expert based on predicate
    """

    pattern: str
    experts: List[str]
    aggregator: Optional[Callable[[List[Any]], Any]] = None
    router: Optional[Callable[[Any], Union[str, List[Tuple[str, float]]]]] = None
    predicate: Optional[Callable[[Any], bool]] = None  # for cascade

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "experts": list(self.experts),
            "has_aggregator": self.aggregator is not None,
            "has_router": self.router is not None,
            "has_predicate": self.predicate is not None,
        }


@dataclass
class SLM:
    """
    Small Language Model (SLM) control-plane wrapper.

    This is intentionally lightweight and deterministic:
    - It orchestrates *experts* (often MicroModels or other callables).
    - It binds to a canonical DSL (optional) for domain-specific operations.
    - It exposes governance hooks (policy checks, audit trail, budgets).
    - It records to a temporal log for reification / provenance.

    The SLM does not assume "language" in the LLM sense; it is a small,
    domain-focused decision module (often CPU-friendly) that composes
    micro-experts into reliable outcomes.
    """

    name: str
    policy: GovernancePolicy = field(default_factory=GovernancePolicy)
    dsl: CanonicalDSLBinding = field(default_factory=NoOpDSLBinding)
    log: TemporalLog = field(default_factory=InMemoryTemporalLog)
    config: SLMConfig = field(default_factory=SLMConfig)

    # Expert registry: each expert maps input -> output
    experts: Dict[str, Callable[[Any], Any]] = field(default_factory=dict)

    # Audit trail for orchestration decisions
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def register_expert(self, name: str, expert: Callable[[Any], Any]) -> None:
        if (
            self.config.max_experts is not None
            and len(self.experts) >= self.config.max_experts
        ):
            raise RuntimeError("SLM expert registry is full (max_experts reached).")
        self.experts[name] = expert
        self._audit("register_expert", {"expert": name})

    def get_expert(self, name: str) -> Callable[[Any], Any]:
        if name not in self.experts:
            raise KeyError(f"Unknown expert: {name}")
        self.policy.check_expert_allowed(name)
        return self.experts[name]

    def run(
        self, spec: EnsembleSpec, x: Any, *, context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a declarative ensemble spec.

        This method is the "simple pipeline runner". For complex workflows with I/O,
        retries, or human-in-the-loop steps, wrap this with an external orchestrator.
        """
        if not self.config.enable_orchestration:
            raise RuntimeError("Orchestration is disabled in SLMConfig.")

        context = dict(context or {})
        steps_used = 0

        self._log_event("ensemble_start", {"slm": self.name, "spec": spec.to_dict()})

        pattern = spec.pattern.lower().strip()
        if pattern == "chain":
            y = x
            for expert_name in spec.experts:
                steps_used += 1
                self.policy.check_step_budget(steps_used)
                y = self._call_expert(expert_name, y, context=context)
            self._log_event("ensemble_end", {"pattern": "chain", "steps": steps_used})
            return y

        if pattern == "parallel":
            results: List[Any] = []
            for expert_name in spec.experts:
                steps_used += 1
                self.policy.check_step_budget(steps_used)
                results.append(self._call_expert(expert_name, x, context=context))
            if spec.aggregator is None:
                raise ValueError("parallel pattern requires an aggregator.")
            y = spec.aggregator(results)
            self._log_event(
                "ensemble_end", {"pattern": "parallel", "steps": steps_used}
            )
            return y

        if pattern == "moe":
            if spec.router is None:
                raise ValueError("moe pattern requires a router.")
            route = spec.router(x)

            # Hard route: router returns a single expert name
            if isinstance(route, str):
                steps_used += 1
                self.policy.check_step_budget(steps_used)
                y = self._call_expert(route, x, context=context)
                self._log_event(
                    "ensemble_end",
                    {"pattern": "moe_hard", "steps": steps_used, "expert": route},
                )
                return y

            # Soft route: router returns list of (expert, weight)
            if not route:
                raise ValueError("router returned empty route list.")
            weighted: List[Any] = []
            weights: List[float] = []
            for expert_name, w in route:
                steps_used += 1
                self.policy.check_step_budget(steps_used)
                weighted.append(self._call_expert(expert_name, x, context=context))
                weights.append(float(w))
            if spec.aggregator is None:
                # Default: weighted sum when outputs are numeric vectors
                def _default_weighted_sum(vals: List[Any]) -> Any:
                    vs = [np.asarray(v) for v in vals]
                    ws = np.asarray(weights, dtype=float)
                    ws = ws / (ws.sum() if ws.sum() != 0.0 else 1.0)
                    out = np.zeros_like(vs[0], dtype=float)
                    for v, w in zip(vs, ws):
                        out = out + w * v
                    return out

                y = _default_weighted_sum(weighted)
            else:
                # Let caller's aggregator incorporate weights if desired (via closure)
                y = spec.aggregator(weighted)
            self._log_event(
                "ensemble_end",
                {
                    "pattern": "moe_soft",
                    "steps": steps_used,
                    "experts": [e for e, _ in route],
                },
            )
            return y

        if pattern == "cascade":
            if len(spec.experts) < 2:
                raise ValueError(
                    "cascade pattern requires at least two experts: [cheap, heavy]."
                )
            cheap, heavy = spec.experts[0], spec.experts[1]
            steps_used += 1
            self.policy.check_step_budget(steps_used)
            y0 = self._call_expert(cheap, x, context=context)

            if spec.predicate is None:
                raise ValueError(
                    "cascade pattern requires a predicate to decide escalation."
                )

            if spec.predicate(y0):
                steps_used += 1
                self.policy.check_step_budget(steps_used)
                y = self._call_expert(heavy, x, context=context)
                self._log_event(
                    "ensemble_end",
                    {
                        "pattern": "cascade_escalated",
                        "steps": steps_used,
                        "cheap": cheap,
                        "heavy": heavy,
                    },
                )
                return y

            self._log_event(
                "ensemble_end",
                {
                    "pattern": "cascade_shortcircuit",
                    "steps": steps_used,
                    "cheap": cheap,
                },
            )
            return y0

        raise ValueError(f"Unknown ensemble pattern: {spec.pattern}")

    def apply_domain_dsl(self, dsl_text: str) -> Any:
        """
        Parse and validate a domain DSL string.

        Returns the canonical representation (caller decides how to execute).
        """
        canonical = self.dsl.parse_to_canonical(dsl_text)
        self.dsl.validate_canonical(canonical)
        self._log_event(
            "dsl_parsed",
            {
                "slm": self.name,
                "dsl_hash": hashlib.sha256(dsl_text.encode("utf-8")).hexdigest(),
            },
        )
        return canonical

    # -------- internals --------

    def _call_expert(self, expert_name: str, x: Any, *, context: Dict[str, Any]) -> Any:
        self.policy.check_expert_allowed(expert_name)
        expert = self.get_expert(expert_name)
        self._audit("call_expert", {"expert": expert_name})
        return expert(x)

    def _audit(self, op: str, data: Dict[str, Any]) -> None:
        if not self.config.enable_audit_logging and not self.policy.require_audit:
            return
        record = {"timestamp": datetime.now().timestamp(), "op": op, **data}
        self.audit_log.append(record)

    def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.log.append(
            TemporalLogEntry(
                timestamp=datetime.now().timestamp(),
                event_type=event_type,
                payload=payload,
            )
        )
