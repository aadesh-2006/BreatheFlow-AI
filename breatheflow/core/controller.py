"""Deterministic Breathing Session Controller.

Controls high-precision timing, phase transitions, and lifecycle states
for guided breathing protocols.

Fully decoupled from LLM inference latency and external sensor jitter:
neither the conversational brain nor camera signals may directly advance
phases or alter durations mid-breath.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional


class BreathingPhase(str, Enum):
    """Lifecycle and breathing cycle phases."""

    IDLE = "idle"
    INHALE = "inhale"
    HOLD_IN = "hold_in"
    EXHALE = "exhale"
    HOLD_OUT = "hold_out"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class BreathingProtocol:
    """Immutable specification of a breathing technique."""

    name: str
    inhale_s: float
    hold_in_s: float
    exhale_s: float
    hold_out_s: float
    cycles: int

    def __post_init__(self) -> None:
        """Validate protocol durations and cycles."""
        if not self.name or not self.name.strip():
            raise ValueError("Protocol name must be a non-empty string.")
        if self.inhale_s <= 0 or math.isnan(self.inhale_s):
            raise ValueError(f"Inhale duration must be > 0, got {self.inhale_s}")
        if self.exhale_s <= 0 or math.isnan(self.exhale_s):
            raise ValueError(f"Exhale duration must be > 0, got {self.exhale_s}")
        if self.hold_in_s < 0 or math.isnan(self.hold_in_s):
            raise ValueError(f"Hold-in duration must be >= 0, got {self.hold_in_s}")
        if self.hold_out_s < 0 or math.isnan(self.hold_out_s):
            raise ValueError(f"Hold-out duration must be >= 0, got {self.hold_out_s}")
        if self.cycles < 1:
            raise ValueError(f"Cycle count must be >= 1, got {self.cycles}")

    @property
    def cycle_duration_s(self) -> float:
        """Total duration of a single breath cycle in seconds."""
        return self.inhale_s + self.hold_in_s + self.exhale_s + self.hold_out_s

    @property
    def total_duration_s(self) -> float:
        """Total duration of all cycles in seconds."""
        return self.cycle_duration_s * self.cycles

    # Preset factory methods
    @classmethod
    def box_breathing(cls, cycles: int = 4, duration_s: float = 4.0) -> "BreathingProtocol":
        """Box Breathing (Sama Vritti): equal inhale, hold, exhale, hold."""
        return cls(
            name="Box Breathing",
            inhale_s=duration_s,
            hold_in_s=duration_s,
            exhale_s=duration_s,
            hold_out_s=duration_s,
            cycles=cycles,
        )

    @classmethod
    def four_seven_eight(cls, cycles: int = 4) -> "BreathingProtocol":
        """4-7-8 Breathing: 4s inhale, 7s hold, 8s exhale for parasympathetic calming."""
        return cls(
            name="4-7-8 Relaxing Breath",
            inhale_s=4.0,
            hold_in_s=7.0,
            exhale_s=8.0,
            hold_out_s=0.0,
            cycles=cycles,
        )

    @classmethod
    def coherent_breathing(cls, cycles: int = 5, duration_s: float = 5.5) -> "BreathingProtocol":
        """Coherent / Resonant Breathing: smooth equal inhale and exhale (~5.5-6 BPM)."""
        return cls(
            name="Coherent Breathing",
            inhale_s=duration_s,
            hold_in_s=0.0,
            exhale_s=duration_s,
            hold_out_s=0.0,
            cycles=cycles,
        )

    @classmethod
    def physiological_sigh(cls, cycles: int = 3) -> "BreathingProtocol":
        """Physiological Sigh: two inhales (modeled as 2.5s inhale + 1s top-up hold) and 6s exhale."""
        return cls(
            name="Physiological Sigh",
            inhale_s=2.5,
            hold_in_s=1.0,
            exhale_s=6.0,
            hold_out_s=0.0,
            cycles=cycles,
        )

    @classmethod
    def from_name(cls, name: str) -> Optional["BreathingProtocol"]:
        """Resolve a protocol identifier or common name to a standard BreathingProtocol preset.

        Returns None if the protocol name is unrecognized.
        """
        if not name or not isinstance(name, str):
            return None
        normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in ("box_breathing", "box", "box_breath"):
            return cls.box_breathing()
        if normalized in ("4_7_8", "four_seven_eight", "4_7_8_breathing", "4_7_8_relaxing_breath"):
            return cls.four_seven_eight()
        if normalized in ("coherent_breathing", "coherent", "resonant_breathing"):
            return cls.coherent_breathing()
        if normalized in ("physiological_sigh", "sigh"):
            return cls.physiological_sigh()
        return None


@dataclass(frozen=True)
class ControllerTick:
    """Deterministic snapshot of session state at a specific point in time."""

    phase: BreathingPhase
    phase_elapsed_s: float
    phase_remaining_s: float
    phase_duration_s: float
    current_cycle: int
    total_cycles: int
    is_phase_start: bool = False
    is_cycle_start: bool = False

    @property
    def is_active(self) -> bool:
        """True if currently in an active breathing phase (inhale, hold, exhale)."""
        return self.phase in (
            BreathingPhase.INHALE,
            BreathingPhase.HOLD_IN,
            BreathingPhase.EXHALE,
            BreathingPhase.HOLD_OUT,
        )


class DeterministicBreathingController:
    """State machine that executes breathing cycles deterministically."""

    def __init__(self) -> None:
        self._phase: BreathingPhase = BreathingPhase.IDLE
        self._protocol: Optional[BreathingProtocol] = None
        self._current_cycle: int = 0
        self._phase_elapsed: float = 0.0
        self._phase_before_pause: Optional[BreathingPhase] = None
        self._just_transitioned_phase: bool = False
        self._just_transitioned_cycle: bool = False

    @property
    def phase(self) -> BreathingPhase:
        """Current lifecycle or breathing phase."""
        return self._phase

    @property
    def protocol(self) -> Optional[BreathingProtocol]:
        """Currently active protocol, if any."""
        return self._protocol

    @property
    def current_cycle(self) -> int:
        """Current cycle index (1-indexed), or 0 if idle."""
        return self._current_cycle

    @property
    def is_active(self) -> bool:
        """True if currently in an active breathing phase."""
        return self._phase in (
            BreathingPhase.INHALE,
            BreathingPhase.HOLD_IN,
            BreathingPhase.EXHALE,
            BreathingPhase.HOLD_OUT,
        )

    def _get_phase_duration(self, phase: BreathingPhase) -> float:
        """Get the configured duration for a specific phase."""
        if not self._protocol:
            return 0.0
        if phase == BreathingPhase.INHALE:
            return self._protocol.inhale_s
        if phase == BreathingPhase.HOLD_IN:
            return self._protocol.hold_in_s
        if phase == BreathingPhase.EXHALE:
            return self._protocol.exhale_s
        if phase == BreathingPhase.HOLD_OUT:
            return self._protocol.hold_out_s
        return 0.0

    def _build_tick(
        self,
        is_phase_start: bool = False,
        is_cycle_start: bool = False,
    ) -> ControllerTick:
        """Construct the immutable ControllerTick snapshot for current state."""
        duration = self._get_phase_duration(self._phase)
        if self._phase == BreathingPhase.PAUSED and self._phase_before_pause:
            duration = self._get_phase_duration(self._phase_before_pause)

        remaining = max(0.0, duration - self._phase_elapsed)

        return ControllerTick(
            phase=self._phase,
            phase_elapsed_s=round(self._phase_elapsed, 4),
            phase_remaining_s=round(remaining, 4),
            phase_duration_s=round(duration, 4),
            current_cycle=self._current_cycle,
            total_cycles=self._protocol.cycles if self._protocol else 0,
            is_phase_start=is_phase_start,
            is_cycle_start=is_cycle_start,
        )

    def start(self, protocol: BreathingProtocol) -> ControllerTick:
        """Initialize and start a breathing session.

        Args:
            protocol: Validated BreathingProtocol configuration.

        Returns:
            Initial ControllerTick at start of cycle 1, inhale phase.
        """
        if self.is_active or self._phase == BreathingPhase.PAUSED:
            raise RuntimeError(
                f"Cannot start a new session while current session is {self._phase.value}. "
                "Stop or finish current session first."
            )

        self._protocol = protocol
        self._current_cycle = 1
        self._phase = BreathingPhase.INHALE
        self._phase_elapsed = 0.0
        self._phase_before_pause = None
        self._just_transitioned_phase = True
        self._just_transitioned_cycle = True

        return self._build_tick(is_phase_start=True, is_cycle_start=True)

    def _get_next_phase(self, current: BreathingPhase) -> Optional[BreathingPhase]:
        """Determine next non-zero phase in protocol."""
        if not self._protocol:
            return None

        if current == BreathingPhase.INHALE:
            if self._protocol.hold_in_s > 0:
                return BreathingPhase.HOLD_IN
            return BreathingPhase.EXHALE

        if current == BreathingPhase.HOLD_IN:
            return BreathingPhase.EXHALE

        if current == BreathingPhase.EXHALE:
            if self._protocol.hold_out_s > 0:
                return BreathingPhase.HOLD_OUT
            return self._advance_or_complete_cycle()

        if current == BreathingPhase.HOLD_OUT:
            return self._advance_or_complete_cycle()

        return None

    def _advance_or_complete_cycle(self) -> BreathingPhase:
        """Check if all cycles completed, or advance to next cycle inhale."""
        if not self._protocol:
            return BreathingPhase.COMPLETED

        if self._current_cycle >= self._protocol.cycles:
            return BreathingPhase.COMPLETED

        self._current_cycle += 1
        self._just_transitioned_cycle = True
        return BreathingPhase.INHALE

    def step(self, delta_s: float) -> ControllerTick:
        """Advance session time deterministically by delta_s seconds.

        Args:
            delta_s: Non-negative elapsed time to advance.

        Returns:
            ControllerTick snapshot after stepping.
        """
        if delta_s < 0 or math.isnan(delta_s):
            raise ValueError(f"Step delta_s must be non-negative, got {delta_s}")

        # Non-active states do not advance internal time
        if not self.is_active:
            return self._build_tick()

        self._just_transitioned_phase = False
        self._just_transitioned_cycle = False

        self._phase_elapsed += delta_s

        # Process phase transitions if phase duration elapsed
        while self.is_active:
            current_duration = self._get_phase_duration(self._phase)
            if self._phase_elapsed < current_duration:
                # Still within current phase
                break

            # Phase completed, calculate overflow and move to next phase
            excess = self._phase_elapsed - current_duration
            next_phase = self._get_next_phase(self._phase)

            if next_phase == BreathingPhase.COMPLETED or next_phase is None:
                self._phase = BreathingPhase.COMPLETED
                self._phase_elapsed = 0.0
                self._just_transitioned_phase = True
                break

            self._phase = next_phase
            self._phase_elapsed = excess
            self._just_transitioned_phase = True

        return self._build_tick(
            is_phase_start=self._just_transitioned_phase,
            is_cycle_start=self._just_transitioned_cycle,
        )

    def pause(self) -> ControllerTick:
        """Pause an active session, preserving exact phase elapsed progress."""
        if not self.is_active:
            raise RuntimeError(f"Cannot pause session when in state '{self._phase.value}'.")

        self._phase_before_pause = self._phase
        self._phase = BreathingPhase.PAUSED
        return self._build_tick(is_phase_start=False, is_cycle_start=False)

    def resume(self) -> ControllerTick:
        """Resume a paused session in the exact phase and timestamp it was paused."""
        if self._phase != BreathingPhase.PAUSED or not self._phase_before_pause:
            raise RuntimeError("Cannot resume session because it is not currently paused.")

        self._phase = self._phase_before_pause
        self._phase_before_pause = None
        return self._build_tick(is_phase_start=False, is_cycle_start=False)

    def stop(self) -> ControllerTick:
        """Prematurely abort or stop the session."""
        self._phase = BreathingPhase.TERMINATED
        self._phase_elapsed = 0.0
        self._phase_before_pause = None
        return self._build_tick(is_phase_start=False, is_cycle_start=False)
