import unittest

from auto_aim import choose_auto_aim


class AutoAimTests(unittest.TestCase):
    def test_rejects_target_outside_attack_range(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[200, -10, 220, 10]],
            walls=[],
            attack_range=100,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
        )

        self.assertFalse(decision.should_fire)
        self.assertEqual(decision.reason, "enemy_out_of_range")

    def test_rejects_wall_blocked_target(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[80, -10, 100, 10]],
            walls=[[40, -20, 60, 20]],
            attack_range=120,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: True,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
        )

        self.assertFalse(decision.should_fire)
        self.assertEqual(decision.reason, "los_blocked")

    def test_close_target_can_override_noisy_los_block(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[55, -10, 75, 10]],
            walls=[[20, -25, 35, 25]],
            attack_range=180,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: True,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
            min_confidence=0.95,
            close_tap_range=80,
            dangerous_close_range=120,
            close_los_override_range=90,
            close_range_override=True,
        )

        self.assertTrue(decision.should_fire)
        self.assertIn(decision.reason, {"close_tap", "close_range_override"})
        self.assertEqual(decision.los_status, "close_override")

    def test_close_target_ignores_stale_aim_line_mismatch(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[60, -12, 80, 12]],
            walls=[],
            attack_range=180,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
            aim_line_angle=180,
            min_confidence=0.80,
            close_tap_range=50,
            dangerous_close_range=120,
            close_range_override=True,
        )

        self.assertTrue(decision.should_fire)
        self.assertTrue(decision.close_range_override)

    def test_predicts_ahead_of_moving_target(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[875, -25, 925, 25]],
            walls=[],
            attack_range=1200,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 250.0),
            velocity_confidence=1.0,
            projectile_speed=900,
            current_time=1.0,
        )

        self.assertTrue(decision.should_fire)
        self.assertGreater(decision.predicted[1], 0)
        self.assertGreater(decision.aim_angle, 0)

    def test_allows_only_close_tap_when_confidence_is_low(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[20, -4, 24, 4]],
            walls=[],
            attack_range=200,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
            min_confidence=0.95,
            close_tap_range=40,
        )

        self.assertTrue(decision.should_fire)
        self.assertTrue(decision.use_tap)
        self.assertIn(decision.reason, {"close_tap", "close_range_override"})

    def test_close_range_override_can_fire_with_low_confidence(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[35, -2, 39, 2]],
            walls=[],
            attack_range=200,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
            min_confidence=0.99,
            close_tap_range=20,
            dangerous_close_range=60,
            close_range_override=True,
        )

        self.assertTrue(decision.should_fire)
        self.assertTrue(decision.close_range_override)

    def test_prioritizes_close_threat_over_far_confident_target(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[380, -25, 430, 25], [30, -8, 42, 8]],
            walls=[],
            attack_range=500,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
            min_confidence=0.62,
            dangerous_close_range=80,
        )

        self.assertTrue(decision.should_fire)
        self.assertLess(decision.distance, 80)

    def test_close_range_override_lowers_effective_confidence_floor(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[42, -2, 46, 2]],
            walls=[],
            attack_range=240,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
            min_confidence=0.80,
            close_tap_range=20,
            dangerous_close_range=75,
            close_range_override=True,
        )

        self.assertTrue(decision.should_fire)
        self.assertEqual(decision.reason, "close_range_override")
        self.assertGreaterEqual(decision.confidence, 0.42)


if __name__ == "__main__":
    unittest.main()
