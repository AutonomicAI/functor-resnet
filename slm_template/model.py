from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import json
import pickle

import numpy as np

from .delta import ConstantDelta, DeltaPatch, HardGate, PatchMetadata, TrainingEvent
from .f import MaterializedLearnedFunction
from .invariants import Invariants, ModelConfig
from .types import Function, Vector


@dataclass
class SLMModel:
    """
    Functor SLM model  with committed functional updates.

    Implements streaming learning by maintaining self.f as a materialized learned function.
    Retained patches are rollback/provenance records, not a nested inference-time closure chain.
    It is assumed a production model would use Kafka or another streaming log that is immutable so
    handling of the patches list here is purely for demonstrative purposes. The model should only be 
    aware of the current function and incoming patch for learning.
    """

    f: Function  # current learned function fₙ(x)
    invariants: Invariants
    config: ModelConfig = field(default_factory=ModelConfig)

    # Core state
    patches: List[DeltaPatch] = field(default_factory=list)
    version: int = 0
    _base_f: Optional[Function] = field(default=None, init=False, repr=False)

    # Training set tracking (Set S from paper)
    trained_inputs: set = field(default_factory=set)  # Set S

    # Skip/delay training
    skipped_events: List[TrainingEvent] = field(default_factory=list)
    delayed_events: List[TrainingEvent] = field(default_factory=list)

    # Audit trail
    patch_history: List[Tuple[DeltaPatch, PatchMetadata]] = field(default_factory=list)
    operation_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Wrap the base function in a materialized learned function."""
        if isinstance(self.f, MaterializedLearnedFunction):
            self._base_f = self.f.base_f
            return
        if self._base_f is None:
            self._base_f = self.f
        self.f = MaterializedLearnedFunction(base_f=self._base_f)

    def predict(self, x: Vector) -> Vector:
        """Run inference against the current learned function fₙ."""
        return self.f(x)

    def propose(self, event: TrainingEvent) -> DeltaPatch:
        """
        Generate a one-shot delta patch from a training event.

        Direct-sum semantics: f_new(x0) = f_old(x0) + Δf(x0) = y0 exactly, applied
        only at x0 via an exact-match (hard) gate. No interpolation/proximity blending.

        Args:
            event: Training event with x, y, metadata

        Returns:
            DeltaPatch ready for validation
        """
        x0 = event.x
        y0 = event.y

        # Additive-delta semantics: interpret event.y as a target output and compute Δf at the anchor
        y_current = self.predict(x0)
        df0 = y0 - y_current

        gate = HardGate(anchor=x0)
        delta = ConstantDelta(df0=df0)
        return DeltaPatch(gate=gate, delta=delta, anchor=x0, target=y0, gate_type="hard")

    def validate(self, delta: DeltaPatch, samples: Iterable[Vector]) -> bool:
        """Validate patch against invariants."""
        return self.invariants.check(self.predict, delta, list(samples))

    def commit(
        self,
        delta: DeltaPatch,
        event: Optional[TrainingEvent] = None,
        samples: Optional[List[Vector]] = None,
        profile: bool = False,
    ) -> None:
        """
        Commit validated patch to model.

        Args:
            delta: Validated DeltaPatch
            event: Original training event (for audit trail)
            samples: Validation samples used (for audit trail)
            profile: Enable timing profiling
        """
        start_time = datetime.now().timestamp()

        # Check max patches limit
        if self.config.max_patches and len(self.patches) >= self.config.max_patches:
            self._log_operation("commit_rejected", {"reason": "max_patches_reached"})
            raise ValueError(f"Maximum patches ({self.config.max_patches}) reached")

        # Create metadata
        metadata = PatchMetadata(
            timestamp=datetime.now().timestamp(),
            event=event if event else TrainingEvent(x=np.array([]), y=np.array([])),
            version=self.version + 1,
            validation_samples=samples if samples else [],
            validation_passed=True,
        )
        delta.metadata = metadata

        # Prepare a shadow materialized function first. Publish it only after the
        # learn step succeeds; do not construct a composed closure here.
        if not isinstance(self.f, MaterializedLearnedFunction):
            if self._base_f is None:
                self._base_f = self.f
            current_f = MaterializedLearnedFunction(base_f=self._base_f)
        else:
            current_f = self.f

        shadow_f = current_f.clone()
        shadow_f.learn(delta)

        self.f = shadow_f
        self.patches.append(delta)
        self.version += 1

        # Update trained inputs set (Set S from paper)
        if event:
            # Store hash of input for tracking
            x_hash = MaterializedLearnedFunction.hash_input(event.x)
            self.trained_inputs.add(x_hash)

        # Update audit trail
        if self.config.enable_audit_logging:
            self.patch_history.append((delta, metadata))

        self._log_operation(
            "commit",
            {"version": self.version, "event": event.to_dict() if event else None},
        )

        if profile or self.config.enable_profiling:
            elapsed = datetime.now().timestamp() - start_time
            self._log_operation("commit_profile", {"elapsed_seconds": elapsed})

    def rollback(self, to_version: Optional[int] = None) -> None:
        """
        Rollback to previous version or specified version.

        Args:
            to_version: Target version (default: current - 1)
        """
        if to_version is None:
            to_version = max(0, self.version - 1)

        if to_version < 0 or to_version > self.version:
            raise ValueError(
                f"Invalid target version: {to_version} (current: {self.version})"
            )

        # Calculate number of patches to remove
        patches_to_remove = self.version - to_version

        if patches_to_remove > 0:
            # Rollback is the only place that replays prior patches.
            # In normal inference, self.f is already the latest learned function.
            self.patches = self.patches[:to_version]
            self.version = to_version

            if self._base_f is None:
                raise RuntimeError("Cannot rollback: original base function is unavailable.")

            self.f = MaterializedLearnedFunction(base_f=self._base_f)
            self.f.reset_to(self.patches)

            # Rebuild trained_inputs from retained patches, not audit-gated history.
            self.trained_inputs = set()
            for patch in self.patches:
                if patch.anchor is not None:
                    self.trained_inputs.add(MaterializedLearnedFunction.hash_input(patch.anchor))

            self._log_operation(
                "rollback",
                {"to_version": to_version, "patches_removed": patches_to_remove},
            )

    def skip(self, event: TrainingEvent, reason: str = "") -> None:
        """
        Skip training event (log but don't apply).

        Args:
            event: Training event to skip
            reason: Optional reason for skipping
        """
        if not self.config.enable_skip_training:
            raise ValueError("Skip training is disabled in config")

        self.skipped_events.append(event)
        self._log_operation("skip", {"event": event.to_dict(), "reason": reason})

    def delay(self, event: TrainingEvent, reason: str = "") -> None:
        """
        Queue training event for later application.

        Args:
            event: Training event to delay
            reason: Optional reason for delaying
        """
        if not self.config.enable_delay_training:
            raise ValueError("Delay training is disabled in config")

        self.delayed_events.append(event)
        self._log_operation("delay", {"event": event.to_dict(), "reason": reason})

    def apply_delayed(
        self, event: TrainingEvent, samples: List[Vector]
    ) -> bool:
        """
        Apply a previously delayed training event.

        Args:
            event: Delayed event to apply
            samples: Validation samples

        Returns:
            True if applied successfully, False if validation failed
        """
        if event not in self.delayed_events:
            raise ValueError("Event not in delayed queue")

        # Propose and validate
        delta = self.propose(event)

        if self.validate(delta, samples):
            self.commit(delta, event=event, samples=samples)
            self.delayed_events.remove(event)
            self._log_operation(
                "apply_delayed", {"event": event.to_dict(), "success": True}
            )
            return True
        else:
            self._log_operation(
                "apply_delayed",
                {
                    "event": event.to_dict(),
                    "success": False,
                    "reason": "validation_failed",
                },
            )
            return False

    @staticmethod
    def _canonical_vector_digest(x: Optional[Vector]) -> Optional[str]:
        """Digest a vector using the same canonical encoding as hard-gate keys."""
        if x is None:
            return None
        arr = np.ascontiguousarray(np.asarray(x, dtype="<f8"))
        shape = repr(arr.shape).encode("utf-8")
        return hashlib.sha256(shape + b"|" + arr.tobytes()).hexdigest()

    def get_state_hash(self) -> str:
        """
        Return deterministic hash of model state.

        Useful for verifying determinism: same training sequence → same hash.
        """
        trained_outputs: Dict[str, str] = {}
        if isinstance(self.f, MaterializedLearnedFunction):
            trained_outputs = {
                key: self._canonical_vector_digest(value) or ""
                for key, value in sorted(self.f.trained_outputs.items())
            }

        state_data = {
            "version": self.version,
            "trained_inputs": sorted(self.trained_inputs),
            "num_patches": len(self.patches),
            "trained_outputs": trained_outputs,
        }
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def save(self, path: str) -> None:
        """
        Serialize model state to disk.

        Saves:
        - Base function (pickled)
        - All patches with metadata
        - Audit trail
        - Configuration

        Security:
        - Only load pickle files from trusted sources. Pickle can execute arbitrary code.

        Note:
        - Standard `pickle` cannot serialize lambdas or locally-defined functions.
        - This template includes pickle-friendly callables (ScopeAll, HardGate, ConstantDelta).
        - For production, keep scope_fn/custom_validators as named top-level callables or small callable classes.
        """
        try:
            state = {
                "base_function": pickle.dumps(self._base_f if self._base_f is not None else self.f),
                "invariants": pickle.dumps(self.invariants),
                "config": pickle.dumps(self.config),
                "patches": pickle.dumps(self.patches),
                "version": self.version,
                "trained_inputs": list(self.trained_inputs),
                "skipped_events": [e.to_dict() for e in self.skipped_events],
                "delayed_events": [e.to_dict() for e in self.delayed_events],
                "patch_history": [
                    (pickle.dumps(p), m.to_dict()) for p, m in self.patch_history
                ],
                "operation_log": self.operation_log,
                "state_hash": self.get_state_hash(),
            }

            with open(path, "wb") as f:
                pickle.dump(state, f)

            self._log_operation("save", {"path": path})
        except (pickle.PicklingError, AttributeError) as e:
            print(f"Warning: Model contains non-pickle-friendly callables: {e}")
            print(
                "Fix: avoid lambdas/local functions for scope_fn/custom_validators and patch components."
            )
            self._log_operation("save_failed", {"path": path, "error": str(e)})

    @classmethod
    def load(cls, path: str) -> "SLMModel":
        """
        Restore model from saved state.

        Returns:
            SLMModel instance with full state restored

        Security:
            Only load pickle files from trusted sources. Pickle can execute arbitrary code.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Reconstruct model
        base_f = pickle.loads(state["base_function"])
        model = cls(
            f=base_f,
            invariants=pickle.loads(state["invariants"]),
            config=pickle.loads(state["config"]),
        )
        model._base_f = base_f

        model.patches = pickle.loads(state["patches"])
        model.version = state["version"]
        model.trained_inputs = set(state["trained_inputs"])

        # Restore as a materialized learned function; do not rebuild a closure chain.
        model.f = MaterializedLearnedFunction(base_f=base_f)
        model.f.reset_to(model.patches)

        # Restore skip/delay queues
        model.skipped_events = [
            TrainingEvent(
                x=np.array(e["x"]),
                y=np.array(e["y"]),
                timestamp=e["timestamp"],
                metadata=e["metadata"],
            )
            for e in state["skipped_events"]
        ]

        model.delayed_events = [
            TrainingEvent(
                x=np.array(e["x"]),
                y=np.array(e["y"]),
                timestamp=e["timestamp"],
                metadata=e["metadata"],
            )
            for e in state["delayed_events"]
        ]

        # Restore audit trail
        model.patch_history = [
            (
                pickle.loads(p),
                PatchMetadata(
                    timestamp=m["timestamp"],
                    event=TrainingEvent(
                        x=np.array(m["event"]["x"]),
                        y=np.array(m["event"]["y"]),
                        timestamp=m["event"]["timestamp"],
                        metadata=m["event"]["metadata"],
                    ),
                    version=m["version"],
                    validation_samples=[np.array(s) for s in m["validation_samples"]],
                    validation_passed=m["validation_passed"],
                ),
            )
            for p, m in state["patch_history"]
        ]

        model.operation_log = state["operation_log"]

        # Verify integrity
        loaded_hash = model.get_state_hash()
        if loaded_hash != state["state_hash"]:
            print(
                f"Warning: State hash mismatch (expected {state['state_hash']}, got {loaded_hash})"
            )

        model._log_operation("load", {"path": path})
        return model

    def generate_validation_samples(
        self, x0: Vector, n_samples: int = 10, radius: float = 1.0
    ) -> List[Vector]:
        """
        Generate validation samples around training point.

        Args:
            x0: Center point
            n_samples: Number of samples to generate
            radius: Sampling radius

        Returns:
            List of sample vectors
        """
        samples = [x0]  # Include the training point

        for _ in range(n_samples - 1):
            # Random perturbation
            perturbation = np.random.randn(*x0.shape) * radius
            sample = x0 + perturbation
            samples.append(sample)

        return samples

    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Internal logging of operations."""
        if self.config.enable_audit_logging:
            log_entry = {
                "timestamp": datetime.now().timestamp(),
                "operation": operation,
                "details": details,
            }
            self.operation_log.append(log_entry)

    def get_stats(self) -> Dict[str, Any]:
        """Return model statistics."""
        return {
            "version": self.version,
            "num_patches": len(self.patches),
            "num_trained_inputs": len(self.trained_inputs),
            "num_skipped": len(self.skipped_events),
            "num_delayed": len(self.delayed_events),
            "state_hash": self.get_state_hash(),
        }
