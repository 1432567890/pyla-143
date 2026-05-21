import unittest

from movement_intent import (
    MovementIntent,
    MovementIntentMemory,
    build_movement_intent,
    build_threat_state,
    smooth_intent,
)


class MovementIntentTests(unittest.TestCase):
    def test_threat_model_scores_close_projectile_and_attack_lane(self):
        threat = build_threat_state(
            closest_enemy_distance=90,
            safe_range=220,
            attack_range=520,
            enemy_velocity=(-120, 0),
            vector_to_enemy=(100, 0),
            enemy_has_line=True,
            projectile={"danger": 0.8},
            fog_danger=False,
            nearby_enemy_count=2,
            health_ratio=0.35,
            teammate_distance=260,
            attack_lane_available=True,
        )

        self.assertGreater(threat.total_score, 0.75)
        self.assertTrue(threat.enemy_close)
        self.assertTrue(threat.projectile_incoming)
        self.assertTrue(threat.low_hp)
        self.assertIn("projectile_crossing_path", threat.reasons)
        self.assertIn("attack_lane_available", threat.reasons)

    def test_projectile_intent_uses_lateral_dodge_but_keeps_attack_allowed(self):
        threat = build_threat_state(
            closest_enemy_distance=180,
            safe_range=220,
            attack_range=520,
            projectile={"danger": 0.9},
            attack_lane_available=True,
        )

        intent = build_movement_intent(
            threat=threat,
            base_angle=0,
            enemy_visible=True,
            enemy_distance=180,
            safe_range=220,
            attack_range=520,
            toward_enemy_angle=0,
            away_enemy_angle=180,
            strafe_angle=90,
            projectile_escape_angle=90,
        )

        self.assertEqual(intent.mode, "dodge_projectile")
        self.assertTrue(intent.attack_allowed)
        self.assertLess(abs((intent.angle - 90 + 180) % 360 - 180), 45)
        self.assertIn("projectile_lateral", intent.reasons)

    def test_fog_has_priority_over_projectile_and_blocks_non_close_attack(self):
        threat = build_threat_state(
            closest_enemy_distance=500,
            safe_range=220,
            attack_range=520,
            projectile={"danger": 1.0},
            fog_danger=True,
            attack_lane_available=True,
        )

        intent = build_movement_intent(
            threat=threat,
            base_angle=90,
            enemy_visible=True,
            enemy_distance=500,
            safe_range=220,
            attack_range=520,
            projectile_escape_angle=0,
            fog_escape_angle=180,
        )

        self.assertEqual(intent.mode, "escape_fog")
        self.assertFalse(intent.attack_allowed)
        self.assertLess(abs((intent.angle - 180 + 180) % 360 - 180), 35)

    def test_low_health_threshold_comes_from_config(self):
        threat = build_threat_state(
            closest_enemy_distance=240,
            safe_range=220,
            attack_range=520,
            health_ratio=0.50,
            low_health_threshold=0.55,
            attack_lane_available=True,
        )

        self.assertTrue(threat.low_hp)
        self.assertIn("low_hp", threat.reasons)

    def test_retreat_heal_allows_attack_only_inside_heal_panic_range(self):
        threat = build_threat_state(
            closest_enemy_distance=180,
            safe_range=220,
            attack_range=520,
            health_ratio=0.25,
            attack_lane_available=True,
        )

        intent = build_movement_intent(
            threat=threat,
            base_angle=180,
            enemy_visible=True,
            enemy_distance=180,
            safe_range=220,
            attack_range=520,
            away_enemy_angle=180,
            heal_retreat_angle=180,
            heal_attack_range=150,
        )
        close_intent = build_movement_intent(
            threat=threat,
            base_angle=180,
            enemy_visible=True,
            enemy_distance=120,
            safe_range=220,
            attack_range=520,
            away_enemy_angle=180,
            heal_retreat_angle=180,
            heal_attack_range=150,
        )

        self.assertEqual(intent.mode, "retreat_heal")
        self.assertFalse(intent.attack_allowed)
        self.assertTrue(close_intent.attack_allowed)

    def test_smoothing_keeps_previous_intent_until_hold_expires(self):
        previous = MovementIntentMemory(
            MovementIntent("strafe", 90, 0.55, ["old"], True, hold_ms=500),
            started_at=10.0,
        )
        new = MovementIntent("approach", 0, 0.60, ["new"], True, hold_ms=400)

        memory, intent, reason = smooth_intent(
            previous,
            new,
            now=10.2,
            min_hold_ms=350,
            max_hold_ms=650,
            switch_score_threshold=0.18,
            angle_smoothing=0.35,
        )

        self.assertIs(memory, previous)
        self.assertEqual(intent.mode, "strafe")
        self.assertEqual(reason, "min_hold_active")

    def test_smoothing_switches_for_serious_priority_upgrade(self):
        previous = MovementIntentMemory(
            MovementIntent("approach", 0, 0.40, ["old"], True, hold_ms=500),
            started_at=10.0,
        )
        new = MovementIntent("escape_fog", 180, 0.70, ["fog"], False, hold_ms=520)

        _, intent, reason = smooth_intent(
            previous,
            new,
            now=10.1,
            min_hold_ms=350,
            max_hold_ms=650,
            switch_score_threshold=0.18,
            angle_smoothing=0.0,
        )

        self.assertEqual(intent.mode, "escape_fog")
        self.assertEqual(reason, "switched")


if __name__ == "__main__":
    unittest.main()
