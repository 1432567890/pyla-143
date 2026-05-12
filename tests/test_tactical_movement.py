import unittest

from tactical_movement import (
    candidate_dodge_angles,
    classify_dodge_mode,
    movement_keys_to_angle,
    score_dodge_angle,
    threat_level_from_distance,
)


class TacticalMovementTests(unittest.TestCase):
    def test_movement_keys_to_angle_uses_screen_coordinates(self):
        self.assertEqual(movement_keys_to_angle("d"), 0.0)
        self.assertEqual(movement_keys_to_angle("s"), 90.0)
        self.assertEqual(movement_keys_to_angle("a"), 180.0)
        self.assertEqual(movement_keys_to_angle("w"), 270.0)

    def test_threat_level_ignores_far_enemy(self):
        self.assertEqual(threat_level_from_distance(800, 500, 250), 0.0)
        self.assertGreater(threat_level_from_distance(200, 500, 250), 0.5)

    def test_dodge_mode_kites_close_enemy(self):
        mode = classify_dodge_mode(0.9, 80, 250)
        self.assertEqual(mode, "kite")

    def test_score_rejects_blocked_direction(self):
        score, reasons = score_dodge_angle(
            90,
            base_angle=0,
            threat_angle=0,
            closest_enemy_distance=100,
            safe_range=250,
            attack_range=500,
            is_blocked=lambda _angle: True,
            points_into_fog=lambda _angle: False,
        )
        self.assertLess(score, -900)
        self.assertIn("blocked_by_wall", reasons)

    def test_candidates_include_lateral_angles(self):
        candidates = candidate_dodge_angles(0, 0)
        self.assertIn(90.0, candidates)
        self.assertIn(270.0, candidates)


if __name__ == "__main__":
    unittest.main()
