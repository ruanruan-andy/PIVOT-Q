import random
import unittest

import torch

from gr00t.experiment.pivot_q_selection import PIVOTQSelectionConfig, select_phase_balanced_states


class PIVOTQSelectionTest(unittest.TestCase):
    def test_balanced_selection_has_expected_budget_and_gap(self) -> None:
        q_values = torch.linspace(0.0, 1.0, 160)
        r_values = torch.linspace(1.0, 0.0, 160)
        result = select_phase_balanced_states(
            q_values,
            r_values,
            PIVOTQSelectionConfig(),
            rng=random.Random(7),
        )
        self.assertEqual(len(result.indices), 16)
        self.assertEqual([result.phases.count(phase) for phase in range(4)], [4, 4, 4, 4])
        self.assertFalse(result.gap_relaxed)
        self.assertEqual(result.phase_effective_gaps, (4, 4, 4, 4))
        self.assertGreaterEqual(result.actual_min_gap, 4)
        self.assertTrue(
            all(right - left >= 4 for left, right in zip(result.indices, result.indices[1:]))
        )
        for phase in range(4):
            self.assertEqual(result.reasons.count(f"priority_phase_{phase}"), 2)
            self.assertEqual(result.reasons.count(f"random_phase_{phase}"), 2)

    def test_seed_reproduces_random_half(self) -> None:
        q_values = torch.rand(120, generator=torch.Generator().manual_seed(3))
        r_values = torch.rand(120, generator=torch.Generator().manual_seed(4))
        first = select_phase_balanced_states(
            q_values, r_values, PIVOTQSelectionConfig(), rng=random.Random(11)
        )
        second = select_phase_balanced_states(
            q_values, r_values, PIVOTQSelectionConfig(), rng=random.Random(11)
        )
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(first.reasons, second.reasons)
        torch.testing.assert_close(first.weights, second.weights)

    def test_short_trajectory_relaxes_gap_inside_each_phase(self) -> None:
        q_values = torch.arange(20, dtype=torch.float32)
        result = select_phase_balanced_states(
            q_values,
            q_values.flip(0),
            PIVOTQSelectionConfig(min_temporal_gap=4),
            rng=random.Random(0),
        )
        self.assertEqual(len(result.indices), 16)
        self.assertEqual(len(set(result.indices)), 16)
        self.assertEqual([result.phases.count(phase) for phase in range(4)], [4, 4, 4, 4])
        for phase in range(4):
            self.assertEqual(result.reasons.count(f"priority_phase_{phase}"), 2)
            self.assertEqual(result.reasons.count(f"random_phase_{phase}"), 2)
        self.assertTrue(result.gap_relaxed)
        self.assertNotIn("fallback_relaxed_gap", result.reasons)

    def test_formal_trajectory_lengths_keep_exact_quotas(self) -> None:
        for length in (16, 32, 61, 75, 102, 124, 125, 220, 520):
            generator = torch.Generator().manual_seed(length)
            result = select_phase_balanced_states(
                torch.rand(length, generator=generator),
                torch.rand(length, generator=generator),
                PIVOTQSelectionConfig(),
                rng=random.Random(2026),
            )
            self.assertEqual(len(result.indices), 16, length)
            self.assertEqual(len(set(result.indices)), 16, length)
            self.assertEqual(
                [result.phases.count(phase) for phase in range(4)],
                [4, 4, 4, 4],
                length,
            )
            self.assertGreaterEqual(result.actual_min_gap, min(result.phase_effective_gaps))
            for phase in range(4):
                self.assertEqual(result.reasons.count(f"priority_phase_{phase}"), 2)
                self.assertEqual(result.reasons.count(f"random_phase_{phase}"), 2)

    def test_too_short_trajectory_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 16"):
            select_phase_balanced_states(
                torch.ones(15),
                torch.ones(15),
                PIVOTQSelectionConfig(),
                rng=random.Random(0),
            )

    def test_empty_trajectory(self) -> None:
        result = select_phase_balanced_states(
            torch.empty(0),
            torch.empty(0),
            PIVOTQSelectionConfig(),
            rng=random.Random(0),
        )
        self.assertEqual(result.indices, ())


if __name__ == "__main__":
    unittest.main()
