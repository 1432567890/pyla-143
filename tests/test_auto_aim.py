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

    def test_close_target_uses_clear_bbox_hit_lane_when_center_is_noisy(self):
        def center_line_blocked(_p1, p2, _walls):
            return abs(p2[1]) <= 2

        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[55, -20, 75, 20]],
            walls=[[20, -25, 35, 25]],
            attack_range=180,
            can_ignore_walls=False,
            walls_block_line_of_sight=center_line_blocked,
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
        self.assertIn(decision.reason, {"ok", "close_tap", "close_range_override"})
        self.assertEqual(decision.los_status, "clear")
        self.assertNotEqual(decision.target, (65.0, 0.0))

    def test_close_target_does_not_override_real_wall_block(self):
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
            close_tap_range=80,
            dangerous_close_range=120,
            close_range_override=True,
        )

        self.assertFalse(decision.should_fire)
        self.assertTrue(decision.close_threat)
        self.assertEqual(decision.denied_by, "los_blocked")

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

    def test_close_target_snaps_prediction_to_live_center(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[70, -4, 78, 4]],
            walls=[],
            attack_range=260,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (900.0, 450.0),
            velocity_confidence=1.0,
            projectile_speed=500,
            current_time=1.0,
            min_confidence=0.99,
            close_tap_range=20,
            dangerous_close_range=120,
            close_range_override=True,
        )

        self.assertTrue(decision.should_fire)
        self.assertEqual(decision.predicted, decision.target)
        self.assertEqual(decision.aim_fallback_reason, "close_snap_to_target")

    def test_mid_range_prediction_clamps_to_hit_lane(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[250, -10, 270, 10]],
            walls=[],
            attack_range=520,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 1200.0),
            velocity_confidence=1.0,
            projectile_speed=700,
            current_time=1.0,
            min_confidence=0.50,
            close_tap_range=40,
            dangerous_close_range=120,
            close_range_override=True,
        )

        self.assertTrue(decision.should_fire)
        self.assertLess(decision.predicted[1], 90)
        self.assertEqual(decision.aim_fallback_reason, "lead_clamped")

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

    def test_melee_snaps_prediction_when_lead_overshoots_attack_range(self):
        """Lead must not reject point-blank shots when bbox is still in range."""

        def track_enemy_velocity(_target, _current_time):
            return (420.0, 0.0)

        decision = choose_auto_aim(
            player_pos=(0.0, 0.0),
            enemy_data=[[72.0, -6.0, 88.0, 6.0]],
            walls=[],
            attack_range=320,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=track_enemy_velocity,
            velocity_confidence=1.0,
            projectile_speed=900.0,
            current_time=1.0,
            min_confidence=0.99,
            close_tap_range=60,
            dangerous_close_range=200,
            close_range_override=True,
        )

        self.assertTrue(decision.should_fire)
        self.assertLessEqual(decision.distance or 0.0, 200.0)

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

    def test_attack_window_keeps_borderline_visible_target(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[75, -25, 127, 25]],
            walls=[],
            attack_range=100,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (0.0, 0.0),
            velocity_confidence=0.0,
            projectile_speed=900,
            current_time=1.0,
            min_confidence=0.30,
        )

        self.assertTrue(decision.in_range)
        self.assertTrue(decision.should_fire)

    def test_close_prediction_stays_on_live_bbox_center(self):
        decision = choose_auto_aim(
            player_pos=(0, 0),
            enemy_data=[[60, -8, 80, 8]],
            walls=[],
            attack_range=220,
            can_ignore_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            track_enemy_velocity=lambda *_args: (1200.0, 800.0),
            velocity_confidence=1.0,
            projectile_speed=500,
            current_time=1.0,
            close_tap_range=20,
            dangerous_close_range=120,
            close_range_override=True,
        )

        self.assertTrue(decision.should_fire)
        self.assertEqual(decision.predicted, decision.target)
        self.assertGreaterEqual(decision.predicted[0], 60)
        self.assertLessEqual(decision.predicted[0], 80)
        self.assertGreaterEqual(decision.predicted[1], -8)
        self.assertLessEqual(decision.predicted[1], 8)


if __name__ == "__main__":
    unittest.main()
