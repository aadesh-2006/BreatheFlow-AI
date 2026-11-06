"""Unit tests for the Deterministic Breathing Session Controller."""

import unittest
from breatheflow.core.controller import (
    BreathingPhase,
    BreathingProtocol,
    ControllerTick,
    DeterministicBreathingController,
)


class TestBreathingProtocolValidation(unittest.TestCase):
    """Test protocol validation rules and presets."""

    def test_valid_custom_protocol(self) -> None:
        proto = BreathingProtocol(
            name="Custom",
            inhale_s=3.0,
            hold_in_s=1.5,
            exhale_s=4.0,
            hold_out_s=2.0,
            cycles=3,
        )
        self.assertEqual(proto.name, "Custom")
        self.assertEqual(proto.cycle_duration_s, 10.5)
        self.assertEqual(proto.total_duration_s, 31.5)

    def test_invalid_name(self) -> None:
        with self.assertRaises(ValueError):
            BreathingProtocol("", 4.0, 4.0, 4.0, 4.0, 2)

    def test_invalid_inhale(self) -> None:
        with self.assertRaises(ValueError):
            BreathingProtocol("Zero Inhale", 0.0, 2.0, 4.0, 2.0, 2)
        with self.assertRaises(ValueError):
            BreathingProtocol("Negative Inhale", -1.0, 2.0, 4.0, 2.0, 2)

    def test_invalid_exhale(self) -> None:
        with self.assertRaises(ValueError):
            BreathingProtocol("Zero Exhale", 4.0, 2.0, 0.0, 2.0, 2)

    def test_invalid_holds(self) -> None:
        with self.assertRaises(ValueError):
            BreathingProtocol("Neg Hold In", 4.0, -0.5, 4.0, 0.0, 2)
        with self.assertRaises(ValueError):
            BreathingProtocol("Neg Hold Out", 4.0, 0.0, 4.0, -1.0, 2)

    def test_invalid_cycles(self) -> None:
        with self.assertRaises(ValueError):
            BreathingProtocol("Zero Cycles", 4.0, 4.0, 4.0, 4.0, 0)

    def test_presets(self) -> None:
        box = BreathingProtocol.box_breathing(cycles=4, duration_s=4.0)
        self.assertEqual(box.inhale_s, 4.0)
        self.assertEqual(box.hold_in_s, 4.0)
        self.assertEqual(box.exhale_s, 4.0)
        self.assertEqual(box.hold_out_s, 4.0)
        self.assertEqual(box.cycle_duration_s, 16.0)

        fse = BreathingProtocol.four_seven_eight(cycles=3)
        self.assertEqual(fse.inhale_s, 4.0)
        self.assertEqual(fse.hold_in_s, 7.0)
        self.assertEqual(fse.exhale_s, 8.0)
        self.assertEqual(fse.hold_out_s, 0.0)

        coherent = BreathingProtocol.coherent_breathing(cycles=5, duration_s=5.5)
        self.assertEqual(coherent.inhale_s, 5.5)
        self.assertEqual(coherent.hold_in_s, 0.0)
        self.assertEqual(coherent.exhale_s, 5.5)
        self.assertEqual(coherent.hold_out_s, 0.0)


class TestDeterministicControllerTransitions(unittest.TestCase):
    """Test phase transitions and cycle advancement."""

    def setUp(self) -> None:
        self.controller = DeterministicBreathingController()

    def test_initial_state(self) -> None:
        self.assertEqual(self.controller.phase, BreathingPhase.IDLE)
        self.assertFalse(self.controller.is_active)
        self.assertEqual(self.controller.current_cycle, 0)

    def test_start_box_breathing(self) -> None:
        protocol = BreathingProtocol.box_breathing(cycles=2, duration_s=4.0)
        tick = self.controller.start(protocol)

        self.assertEqual(tick.phase, BreathingPhase.INHALE)
        self.assertEqual(tick.current_cycle, 1)
        self.assertEqual(tick.total_cycles, 2)
        self.assertEqual(tick.phase_elapsed_s, 0.0)
        self.assertEqual(tick.phase_remaining_s, 4.0)
        self.assertTrue(tick.is_phase_start)
        self.assertTrue(tick.is_cycle_start)

    def test_box_breathing_phase_sequence(self) -> None:
        protocol = BreathingProtocol.box_breathing(cycles=1, duration_s=4.0)
        self.controller.start(protocol)

        # Inhale: 4 seconds total
        t1 = self.controller.step(2.0)
        self.assertEqual(t1.phase, BreathingPhase.INHALE)
        self.assertEqual(t1.phase_elapsed_s, 2.0)
        self.assertEqual(t1.phase_remaining_s, 2.0)

        # Transition to HOLD_IN
        t2 = self.controller.step(2.0)
        self.assertEqual(t2.phase, BreathingPhase.HOLD_IN)
        self.assertEqual(t2.phase_elapsed_s, 0.0)
        self.assertEqual(t2.phase_remaining_s, 4.0)
        self.assertTrue(t2.is_phase_start)

        # Transition to EXHALE
        t3 = self.controller.step(4.0)
        self.assertEqual(t3.phase, BreathingPhase.EXHALE)
        self.assertEqual(t3.phase_elapsed_s, 0.0)
        self.assertTrue(t3.is_phase_start)

        # Transition to HOLD_OUT
        t4 = self.controller.step(4.0)
        self.assertEqual(t4.phase, BreathingPhase.HOLD_OUT)
        self.assertEqual(t4.phase_elapsed_s, 0.0)
        self.assertTrue(t4.is_phase_start)

        # Complete cycle 1 of 1
        t5 = self.controller.step(4.0)
        self.assertEqual(t5.phase, BreathingPhase.COMPLETED)
        self.assertFalse(self.controller.is_active)

    def test_multi_cycle_advancement(self) -> None:
        protocol = BreathingProtocol.box_breathing(cycles=2, duration_s=4.0)
        self.controller.start(protocol)

        # Step full 16s of cycle 1 in 1s increments
        for _ in range(16):
            self.controller.step(1.0)

        # At 16s boundary, should enter cycle 2 INHALE
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)
        self.assertEqual(self.controller.current_cycle, 2)

        # Step full 16s of cycle 2
        for _ in range(16):
            self.controller.step(1.0)

        self.assertEqual(self.controller.phase, BreathingPhase.COMPLETED)
        self.assertFalse(self.controller.is_active)

    def test_zero_hold_coherent_breathing(self) -> None:
        """Verify protocols with zero holds bypass hold phases cleanly."""
        protocol = BreathingProtocol.coherent_breathing(cycles=2, duration_s=5.0)
        self.controller.start(protocol)

        # Inhale 5s
        t_inhale = self.controller.step(5.0)
        # Directly goes to EXHALE (skips HOLD_IN since hold_in_s=0)
        self.assertEqual(t_inhale.phase, BreathingPhase.EXHALE)
        self.assertEqual(t_inhale.current_cycle, 1)

        # Exhale 5s
        t_cycle2 = self.controller.step(5.0)
        # Directly goes to Cycle 2 INHALE (skips HOLD_OUT since hold_out_s=0)
        self.assertEqual(t_cycle2.phase, BreathingPhase.INHALE)
        self.assertEqual(t_cycle2.current_cycle, 2)


class TestDeterministicControllerControlFlow(unittest.TestCase):
    """Test pause, resume, stop, and safety edge cases."""

    def setUp(self) -> None:
        self.controller = DeterministicBreathingController()

    def test_pause_and_resume_preserves_timestamp(self) -> None:
        protocol = BreathingProtocol.box_breathing(cycles=2, duration_s=4.0)
        self.controller.start(protocol)

        # Step 2.5s into Inhale (4s duration)
        self.controller.step(2.5)
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)

        # Pause
        paused_tick = self.controller.pause()
        self.assertEqual(paused_tick.phase, BreathingPhase.PAUSED)
        self.assertEqual(paused_tick.phase_elapsed_s, 2.5)
        self.assertEqual(paused_tick.phase_remaining_s, 1.5)

        # Stepping while paused does NOT advance elapsed time
        step_paused = self.controller.step(10.0)
        self.assertEqual(step_paused.phase, BreathingPhase.PAUSED)
        self.assertEqual(step_paused.phase_elapsed_s, 2.5)
        self.assertEqual(step_paused.phase_remaining_s, 1.5)

        # Resume
        resumed_tick = self.controller.resume()
        self.assertEqual(resumed_tick.phase, BreathingPhase.INHALE)
        self.assertEqual(resumed_tick.phase_elapsed_s, 2.5)
        self.assertEqual(resumed_tick.phase_remaining_s, 1.5)

        # Step the remaining 1.5s -> should transition into HOLD_IN exactly
        next_tick = self.controller.step(1.5)
        self.assertEqual(next_tick.phase, BreathingPhase.HOLD_IN)
        self.assertEqual(next_tick.phase_elapsed_s, 0.0)

    def test_stop_terminates_session(self) -> None:
        protocol = BreathingProtocol.box_breathing(cycles=2, duration_s=4.0)
        self.controller.start(protocol)
        self.controller.step(2.0)

        term_tick = self.controller.stop()
        self.assertEqual(term_tick.phase, BreathingPhase.TERMINATED)
        self.assertFalse(self.controller.is_active)

        # Stepping while terminated does nothing
        after_term = self.controller.step(5.0)
        self.assertEqual(after_term.phase, BreathingPhase.TERMINATED)

    def test_invalid_controller_actions(self) -> None:
        # Cannot pause when IDLE
        with self.assertRaises(RuntimeError):
            self.controller.pause()

        # Cannot resume when not paused
        with self.assertRaises(RuntimeError):
            self.controller.resume()

        # Start a session
        proto = BreathingProtocol.box_breathing(cycles=1)
        self.controller.start(proto)

        # Cannot start again while running
        with self.assertRaises(RuntimeError):
            self.controller.start(proto)

        # Negative delta_s rejected
        with self.assertRaises(ValueError):
            self.controller.step(-1.0)


if __name__ == "__main__":
    unittest.main()
