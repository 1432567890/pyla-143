import unittest

from combat_brain import (
    CombatFrame,
    HealthState,
    SafetyResult,
    TargetMemory,
    build_threat_model,
    choose_ability_plan,
    choose_attack_gate,
    choose_combat_intent,
    choose_tactical_plan,
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

    def test_hold_cover_allows_visible_in_range_pressure_shot(self):
        target = type("Target", (), {
            "bbox": (240, -20, 280, 20),
            "center": (260, 0),
            "distance": 260,
            "score": 0.55,
            "line_of_sight": True,
            "in_attack_range": True,
            "close_threat": False,
            "stale": False,
        })()
        tactical_plan = type("Plan", (), {
            "objective": "hold_cover",
            "selected_angle": 0.0,
            "fire_window": True,
            "survival_score": 0.72,
            "engagement_score": 0.55,
            "kill_confirm_score": 0.0,
            "reasons": ["fire_window"],
        })()

        intent = choose_combat_intent(
            frame=CombatFrame(
                player_pos=(0, 0),
                enemy_data=[[240, -20, 280, 20]],
                health=HealthState(ratio=0.85, confidence=1.0),
                safe_range=180,
                attack_range=520,
            ),
            target=target,
            safety=SafetyResult(angle=0.0, safe=True, status="clear"),
            tactical_plan=tactical_plan,
            tactical_planner_enabled=True,
        )

        self.assertTrue(intent.attack_allowed)
        self.assertEqual(intent.attack_denied_reason, "pressure_shot")

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

    def test_target_lock_does_not_switch_to_slightly_better_target(self):
        memory = TargetMemory()
        target = memory.choose(
            now=1.0,
            memory_seconds=0.75,
            switch_margin=0.18,
            player_pos=(0, 0),
            enemy_data=[[95, -10, 115, 10], [180, -10, 200, 10]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=80,
        )
        self.assertEqual(target.center, (105.0, 0.0))

        target = memory.choose(
            now=1.1,
            memory_seconds=0.75,
            switch_margin=0.18,
            player_pos=(0, 0),
            enemy_data=[[100, -10, 120, 10], [85, -10, 105, 10]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=80,
        )

        self.assertEqual(target.center, (110.0, 0.0))

    def test_close_enemy_overrides_target_lock(self):
        memory = TargetMemory()
        memory.choose(
            now=1.0,
            memory_seconds=0.75,
            switch_margin=0.50,
            player_pos=(0, 0),
            enemy_data=[[280, -10, 300, 10]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=80,
        )

        target = memory.choose(
            now=1.1,
            memory_seconds=0.75,
            switch_margin=0.50,
            player_pos=(0, 0),
            enemy_data=[[285, -10, 305, 10], [45, -10, 65, 10]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=80,
        )

        self.assertEqual(target.center, (55.0, 0.0))
        self.assertTrue(target.close_threat)

    def test_stale_target_blocks_attack(self):
        memory = TargetMemory()
        memory.choose(
            now=1.0,
            memory_seconds=0.75,
            switch_margin=0.18,
            player_pos=(0, 0),
            enemy_data=[[120, -10, 140, 10]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=80,
        )
        target = memory.choose(
            now=1.2,
            memory_seconds=0.75,
            switch_margin=0.18,
            player_pos=(0, 0),
            enemy_data=[],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=80,
        )
        intent = choose_combat_intent(
            frame=CombatFrame(
                player_pos=(0, 0),
                health=HealthState(ratio=0.9, confidence=1.0),
                desired_angle=90,
                safe_range=180,
                attack_range=520,
            ),
            target=target,
            safety=SafetyResult(angle=90, safe=True, status="clear"),
        )

        self.assertTrue(target.stale)
        self.assertFalse(intent.attack_allowed)
        self.assertEqual(intent.attack_denied_reason, "stale_target")

    def test_projectile_dodge_without_enemy_blocks_normal_attack(self):
        intent = choose_combat_intent(
            frame=CombatFrame(
                player_pos=(0, 0),
                enemy_data=[],
                health=HealthState(ratio=0.9, confidence=1.0),
                desired_angle=180,
                safe_range=180,
                attack_range=520,
                projectile_incoming=True,
            ),
            target=None,
            safety=SafetyResult(angle=180, safe=True, status="clear"),
        )

        self.assertEqual(intent.mode, "dodge_projectile")
        self.assertFalse(intent.attack_allowed)
        self.assertIn("projectile_without_enemy", intent.reasons)

    def test_wall_no_safe_angle_enables_wall_escape(self):
        intent = choose_combat_intent(
            frame=CombatFrame(
                player_pos=(0, 0),
                health=HealthState(ratio=0.9, confidence=1.0),
                desired_angle=45,
                safe_range=180,
                attack_range=520,
            ),
            target=None,
            safety=SafetyResult(angle=45, safe=False, status="no_safe_angle"),
        )

        self.assertEqual(intent.mode, "wall_escape")
        self.assertFalse(intent.attack_allowed)
        self.assertEqual(intent.attack_denied_reason, "wall_escape_blocks_attack")

    def test_ability_plan_does_not_press_three_buttons_in_one_frame(self):
        target = type("Target", (), {
            "stale": False,
            "distance": 80,
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
            super_type="damage",
            super_ready=True,
            hypercharge_ready=True,
            gadget_ready=True,
            super_hittable=True,
            attack_hittable=True,
            enemy_count_in_range=2,
            teammate_near=True,
        )

        self.assertLessEqual(sum([plan.use_hypercharge, plan.use_super, plan.use_gadget]), 2)
        self.assertEqual(plan.gadget_reason, "combo_limit")

    def test_tactical_plan_chooses_safe_los_angle_over_blocked_retreat(self):
        target = choose_target(
            player_pos=(0, 0),
            enemy_data=[[250, -20, 290, 20]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=120,
        )

        plan = choose_tactical_plan(
            frame=CombatFrame(
                player_pos=(0, 0),
                enemy_data=[[250, -20, 290, 20]],
                health=HealthState(ratio=0.90, confidence=1.0),
                desired_angle=180,
                safe_range=180,
                attack_range=520,
                preferred_distance=220,
            ),
            target=target,
            safety=SafetyResult(angle=180, safe=True, status="clear"),
            is_angle_blocked=lambda angle: abs(angle - 180) < 1,
            points_into_fog=lambda _angle: False,
            line_of_sight_after_move=lambda angle: abs(angle - 180) >= 1,
        )

        self.assertNotAlmostEqual(plan.selected_angle, 180)
        self.assertTrue(plan.fire_window)
        self.assertGreater(plan.survival_score, 0.5)

    def test_tactical_plan_low_hp_blocks_commit_chase(self):
        target = choose_target(
            player_pos=(0, 0),
            enemy_data=[[260, -20, 300, 20]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=120,
        )

        plan = choose_tactical_plan(
            frame=CombatFrame(
                player_pos=(0, 0),
                enemy_data=[[260, -20, 300, 20]],
                health=HealthState(ratio=0.20, confidence=1.0, heal_active=True),
                desired_angle=0,
                safe_range=180,
                attack_range=520,
                preferred_distance=200,
            ),
            target=target,
            safety=SafetyResult(angle=0, safe=True, status="clear"),
            line_of_sight_after_move=lambda _angle: True,
        )

        self.assertFalse(plan.commit_allowed)
        self.assertIn(plan.objective, {"hold_cover", "regroup"})

    def test_tactical_plan_projectile_dodge_angle_beats_attack_angle_when_urgent(self):
        target = choose_target(
            player_pos=(0, 0),
            enemy_data=[[370, -20, 410, 20]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=120,
        )

        plan = choose_tactical_plan(
            frame=CombatFrame(
                player_pos=(0, 0),
                enemy_data=[[370, -20, 410, 20]],
                health=HealthState(ratio=0.90, confidence=1.0),
                desired_angle=0,
                safe_range=180,
                attack_range=520,
                preferred_distance=300,
                projectile_incoming=True,
                projectile_dodge_angle=90,
            ),
            target=target,
            safety=SafetyResult(angle=0, safe=True, status="clear"),
            projectile_danger_for_angle=lambda angle: 0.1 if abs(angle - 90) <= 1 else 1.0,
            line_of_sight_after_move=lambda _angle: True,
        )

        self.assertAlmostEqual(plan.selected_angle, 90)
        self.assertEqual(plan.objective, "survive")

    def test_tactical_plan_teammate_pressure_does_not_override_fog(self):
        target = choose_target(
            player_pos=(0, 0),
            enemy_data=[[250, -20, 290, 20]],
            safe_range=180,
            attack_range=520,
            walls=[],
            can_attack_through_walls=False,
            walls_block_line_of_sight=lambda *_args: False,
            dangerous_close_range=120,
        )

        plan = choose_tactical_plan(
            frame=CombatFrame(
                player_pos=(0, 0),
                enemy_data=[[250, -20, 290, 20]],
                health=HealthState(ratio=0.90, confidence=1.0),
                desired_angle=0,
                safe_range=180,
                attack_range=520,
                teammate_near=True,
                teammate_angle=0,
                fog_danger=True,
            ),
            target=target,
            safety=SafetyResult(angle=0, safe=True, status="clear"),
            points_into_fog=lambda angle: abs(angle) <= 1,
            line_of_sight_after_move=lambda _angle: True,
        )

        self.assertEqual(plan.objective, "survive")
        self.assertNotAlmostEqual(plan.selected_angle, 0)


if __name__ == "__main__":
    unittest.main()
