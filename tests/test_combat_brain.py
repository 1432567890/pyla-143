import unittest

from combat_brain import (
    HealthState,
    build_threat_model,
    choose_ability_plan,
    choose_attack_gate,
    choose_target,
)


class CombatBrainTests(unittest.TestCase):
    def test_target_selection_prefers_close_threat_over_far_target(self):
        target = choose_target(
            player_pos=(0, 0),
            enemy_data=[[430, -20, 470, 20], [80, -10, 100, 10]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=150,
        )

        self.assertLess(target.distance, 120)
        self.assertTrue(target.close_threat)

    def test_defensive_gate_blocks_low_health_attack(self):
        target = type("Target", (), {
            "close_threat": False,
            "line_of_sight": True,
            "distance": 260,
        })()

        allowed, reason = choose_attack_gate(
            mode="retreat_heal",
            target=target,
            health=HealthState(ratio=0.25, confidence=1.0, heal_active=True),
            panic_shot_range=150,
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "retreat_heal")

    def test_defensive_gate_allows_panic_shot(self):
        target = type("Target", (), {
            "close_threat": True,
            "line_of_sight": True,
            "distance": 90,
        })()

        allowed, reason = choose_attack_gate(
            mode="retreat_heal",
            target=target,
            health=HealthState(ratio=0.20, confidence=1.0, heal_active=True),
            panic_shot_range=150,
        )

        self.assertTrue(allowed)
        self.assertEqual(reason, "panic_shot")

    def test_damage_super_uses_hypercharge_for_valuable_close_target(self):
        target = type("Target", (), {
            "stale": False,
            "distance": 90,
            "line_of_sight": True,
            "in_attack_range": True,
            "close_threat": True,
        })()
        threat = build_threat_model(
            target=target,
            enemy_count_in_range=2,
            health=HealthState(ratio=0.80, confidence=1.0),
            safe_range=180,
        )

        plan = choose_ability_plan(
            target=target,
            threat=threat,
            health=HealthState(ratio=0.80, confidence=1.0),
            super_type="damage",
            super_ready=True,
            hypercharge_ready=True,
            gadget_ready=False,
            super_hittable=True,
            attack_hittable=True,
            enemy_count_in_range=2,
            teammate_near=True,
        )

        self.assertTrue(plan.use_super)
        self.assertTrue(plan.use_hypercharge)
        self.assertEqual(plan.hypercharge_reason, "combo_super")

    def test_retreat_heal_does_not_waste_hypercharge_without_panic(self):
        target = type("Target", (), {
            "stale": False,
            "distance": 300,
            "line_of_sight": True,
            "in_attack_range": True,
            "close_threat": False,
        })()
        health = HealthState(ratio=0.20, confidence=1.0, heal_active=True)
        threat = build_threat_model(target=target, health=health, safe_range=180)

        plan = choose_ability_plan(
            target=target,
            threat=threat,
            health=health,
            super_type="damage",
            super_ready=True,
            hypercharge_ready=True,
            super_hittable=True,
            attack_hittable=True,
        )

        self.assertFalse(plan.use_hypercharge)
        self.assertFalse(plan.use_super)

    def test_hypercharge_standalone_requires_committed_high_value_fight(self):
        target = type("Target", (), {
            "stale": False,
            "distance": 90,
            "line_of_sight": True,
            "in_attack_range": True,
            "close_threat": True,
        })()
        threat = build_threat_model(
            target=target,
            enemy_count_in_range=2,
            health=HealthState(ratio=0.90, confidence=1.0),
            safe_range=180,
        )

        plan = choose_ability_plan(
            target=target,
            threat=threat,
            health=HealthState(ratio=0.90, confidence=1.0),
            super_ready=False,
            hypercharge_ready=True,
            attack_hittable=True,
            enemy_count_in_range=2,
        )

        self.assertTrue(plan.use_hypercharge)
        self.assertEqual(plan.hypercharge_reason, "committed_fight")

    def test_charge_super_rejects_unsafe_path(self):
        target = type("Target", (), {
            "stale": False,
            "distance": 120,
            "line_of_sight": True,
            "in_attack_range": True,
            "close_threat": True,
        })()

        plan = choose_ability_plan(
            target=target,
            threat=build_threat_model(target=target, safe_range=180),
            health=HealthState(ratio=0.90, confidence=1.0),
            super_type="charge",
            super_ready=True,
            super_hittable=True,
            attack_hittable=True,
            charge_path_safe=False,
        )

        self.assertFalse(plan.use_super)
        self.assertEqual(plan.super_reason, "unsafe_charge_path")

    def test_gadget_uses_only_for_valuable_close_hittable_target(self):
        target = type("Target", (), {
            "stale": False,
            "distance": 80,
            "line_of_sight": True,
            "in_attack_range": True,
            "close_threat": True,
        })()

        plan = choose_ability_plan(
            target=target,
            threat=build_threat_model(target=target, enemy_count_in_range=2, safe_range=180),
            health=HealthState(ratio=0.90, confidence=1.0),
            gadget_ready=True,
            gadget_enabled=True,
            attack_hittable=True,
            enemy_count_in_range=2,
        )

        self.assertTrue(plan.use_gadget)


if __name__ == "__main__":
    unittest.main()
