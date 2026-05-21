import json
import math
import os
import sys
import tempfile
import time
import types
import unittest

import cv2
import numpy as np

sys.modules.setdefault("onnxruntime", types.SimpleNamespace(InferenceSession=None))

from auto_aim import AttackDecision
from play import Play


class CombatAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.movement = object.__new__(Play)
        self.movement._strafe_started_at = 0.0
        self.movement._strafe_side = 1
        self.movement._strafe_current_interval = 0.0
        self.movement.strafe_interval = 1.0
        self.movement.strafe_enabled = True
        self.movement.combat_dodge_blend = 0.65
        self.movement.combat_dodge_jitter_degrees = 0.0
        self.movement.projectile_speed_px_s = 900.0

    def test_strafe_angle_smoothly_flips_after_interval(self):
        first = self.movement.get_strafe_angle(0, 10.0)
        second = self.movement.get_strafe_angle(0, 11.2)
        self.assertAlmostEqual(first, 49.5)
        self.assertGreater(second, 180)
        self.assertLess(second, 360)

    def test_lead_shot_falls_back_to_direct_when_unsolvable(self):
        angle = self.movement.lead_shot_angle((0, 0), (100, 0), (3000, 0), projectile_speed_px_s=100)
        self.assertAlmostEqual(angle, 0.0)

    def test_lead_shot_aims_ahead_of_moving_target(self):
        angle = self.movement.lead_shot_angle((0, 0), (900, 0), (0, 300), projectile_speed_px_s=900)
        self.assertGreater(angle, 0.0)
        self.assertLess(angle, 45.0)
        self.assertFalse(math.isnan(angle))

    def test_combat_dodge_biases_shooting_movement_sideways(self):
        desired = self.movement.apply_combat_dodge(
            desired_angle=0,
            toward_enemy_angle=0,
            current_time=10.0,
            enemy_distance=180,
            safe_range=120,
        )

        self.assertGreater(desired, 25)
        self.assertLess(desired, 90)

    def test_movement_to_vector_converts_legacy_keys(self):
        self.assertEqual(self.movement.movement_to_vector("wd"), (1, -1))
        self.assertEqual(self.movement.movement_to_vector("as"), (-1, 1))

    def test_combat_recorder_saves_death_snapshot(self):
        play = object.__new__(Play)
        play.combat_snapshot_enabled = True
        play.combat_snapshot_seconds = 8
        play.combat_brain_debug = False
        play.current_brawler = "shelly"
        with tempfile.TemporaryDirectory() as directory:
            play.combat_snapshot_dir = directory
            play.record_combat_decision({
                "mode": "retreat_heal",
                "health_ratio": 0.20,
                "attack_allowed": False,
                "attack_denied": "retreat_heal",
            })
            path = play.save_combat_snapshot("player_lost", extra={"state": "match"})

            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            self.assertEqual(snapshot["reason"], "player_lost")
            self.assertEqual(snapshot["recent_decisions"][0]["mode"], "retreat_heal")

    def make_auto_aim_play(self, window_controller):
        play = object.__new__(Play)
        play.window_controller = window_controller
        play.brawlers_info = {"shelly": {"ignore_walls_for_attacks": False}}
        play.attack_cooldown = 0.0
        play.close_range_attack_cooldown_multiplier = 0.55
        play.attack_spam_enabled = True
        play.attack_spam_cooldown_multiplier = 0.12
        play.attack_spam_requires_los = True
        play.aim_attack_duration = 0.02
        play.force_release_movement_on_close_threat = False
        play.last_attack_time = 0.0
        play._suppress_attack_until = 0.0
        play._last_aim_attempt_time = 0.0
        play.auto_aim_debug = False
        play.attack_decision_debug = False
        play.current_frame = None
        play.projectile_speed_px_s = 700.0
        play.auto_aim_min_confidence = 0.54
        play.auto_aim_close_tap_range = 0
        play.auto_aim_close_los_override_range = 0
        play.close_range_attack_override = True
        play.dangerous_close_range = 120
        play.friendly_fire_guard_enabled = True
        play.friendly_fire_iou_threshold = 0.18
        play.friendly_fire_center_distance_px = 70
        play.friendly_fire_lane_guard_enabled = True
        play.friendly_fire_lane_padding_ratio = 0.25
        play.friendly_fire_lane_min_padding = 8
        play.friendly_fire_lane_max_padding = 28
        play.attack_wall_guard_enabled = True
        play.auto_aim_wall_lane_guard_enabled = True
        play.auto_aim_wall_lane_padding_ratio = 0.08
        play.auto_aim_wall_lane_min_padding = 6
        play.auto_aim_wall_lane_max_padding = 18
        play.lead_shots_enabled = True
        play.aimed_attacks_enabled = True
        play.enemy_velocity_confidence = 1.0
        play.track_enemy_velocity = lambda *_args: (0.0, 0.0)
        return play

    def make_marker_play(self):
        play = object.__new__(Play)
        play.entity_marker_min_ratio = 0.012
        play.entity_marker_min_pixels = 12
        play.entity_marker_below_box_ratio = 0.22
        play.entity_marker_blue_min_ratio = 0.012
        play.entity_marker_enemy_min_ratio = 0.012
        play.entity_marker_decision_margin = 1.25
        play._entity_marker_cache_frame_id = None
        play._entity_marker_score_cache = {}
        play._perf_entity_marker_scores = 0
        play._perf_entity_marker_cache_hits = 0
        return play

    def draw_hsv_circle_below_box(self, frame, box, hsv_color):
        x1, y1, x2, y2 = box
        rgb = cv2.cvtColor(np.array([[hsv_color]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        center = (int((x1 + x2) / 2), int(y2 + 8))
        cv2.circle(frame, center, 10, rgb.tolist(), -1)

    def test_auto_aim_attack_falls_back_to_tap_when_aim_drag_unavailable(self):
        class TapOnlyWindow:
            def __init__(self):
                self.keys = []

            def press_key(self, key, **kwargs):
                self.keys.append((key, kwargs))

        window = TapOnlyWindow()
        play = self.make_auto_aim_play(window)
        play.aimed_attacks_enabled = False

        fired = play.auto_aim_attack("shelly", (0, 0), [[80, -12, 104, 12]], [], attack_range=260)

        self.assertTrue(fired)
        self.assertEqual(window.keys[0][0], "M")

    def test_auto_aim_attack_disables_prediction_when_lead_shots_off(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.lead_shots_enabled = False
        play.track_enemy_velocity = lambda *_args: (0.0, 900.0)
        play.enemy_velocity_confidence = 1.0

        fired = play.auto_aim_attack("shelly", (0, 0), [[250, -10, 270, 10]], [], attack_range=520)

        self.assertTrue(fired)
        self.assertAlmostEqual(window.angle, 0.0)

    def test_close_visible_enemy_with_clear_los_fires_while_moving_toward_enemy(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.last_movement = "D"
        play.keys_hold = ["d"]

        fired = play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)

        self.assertTrue(fired)
        self.assertIsNotNone(window.angle)
        self.assertAlmostEqual(window.angle, 0.0)

    def test_close_visible_enemy_overrides_defensive_suppression_only_with_clear_los(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play._suppress_attack_until = time.time() + 1.0

        fired = play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)

        self.assertTrue(fired)
        self.assertIsNotNone(window.angle)

        blocked_window = AimWindow()
        blocked_play = self.make_auto_aim_play(blocked_window)
        blocked_play._suppress_attack_until = time.time() + 1.0
        blocked_play.walls_block_line_of_sight = lambda *_args, **_kwargs: True

        blocked = blocked_play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)

        self.assertFalse(blocked)
        self.assertIsNone(blocked_window.angle)

    def test_auto_aim_attack_blocks_friendly_overlap_target(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)

        fired = play.auto_aim_attack(
            "shelly",
            (0, 0),
            [[100, -20, 140, 20]],
            [],
            attack_range=300,
            excluded_boxes=[[102, -18, 138, 18]],
        )

        self.assertFalse(fired)
        self.assertIsNone(window.angle)

    def test_auto_aim_attack_rechecks_precomputed_decision_against_friendly_boxes(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        decision = AttackDecision(
            True,
            aim_angle=0.0,
            target=(120, 0),
            predicted=(120, 0),
            distance=120,
            attack_range=300,
            in_range=True,
            line_of_sight=True,
            target_bbox=(100, -20, 140, 20),
        )

        fired = play.auto_aim_attack(
            "shelly",
            (0, 0),
            [[100, -20, 140, 20]],
            [],
            attack_range=300,
            decision=decision,
            excluded_boxes=[[102, -18, 138, 18]],
        )

        self.assertFalse(fired)
        self.assertIsNone(window.angle)
        self.assertEqual(decision.denied_by, "friendly_excluded")

    def test_auto_aim_attack_rechecks_precomputed_decision_against_friendly_lane(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        decision = AttackDecision(
            True,
            aim_angle=0.0,
            target=(210, 0),
            predicted=(210, 0),
            distance=210,
            attack_range=300,
            in_range=True,
            line_of_sight=True,
            target_bbox=(190, -20, 230, 20),
        )

        fired = play.auto_aim_attack(
            "shelly",
            (0, 0),
            [[190, -20, 230, 20]],
            [],
            attack_range=300,
            decision=decision,
            excluded_boxes=[{"box": [90, -14, 125, 14], "kind": "teammate"}],
        )

        self.assertFalse(fired)
        self.assertIsNone(window.angle)
        self.assertEqual(decision.denied_by, "friendly_lane_blocked")
        self.assertIn("teammate_lane", decision.friendly_lane_status)

    def test_auto_aim_attack_rechecks_precomputed_decision_against_wall_lane(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        decision = AttackDecision(
            True,
            aim_angle=0.0,
            target=(210, 0),
            predicted=(210, 0),
            distance=210,
            attack_range=300,
            in_range=True,
            line_of_sight=True,
            target_bbox=(190, -20, 230, 20),
        )

        fired = play.auto_aim_attack(
            "shelly",
            (0, 0),
            [[190, -20, 230, 20]],
            [[90, -8, 120, 8]],
            attack_range=300,
            decision=decision,
        )

        self.assertFalse(fired)
        self.assertIsNone(window.angle)
        self.assertEqual(decision.denied_by, "wall_blocked_final_hitpoint")
        self.assertIn("wall_lane", decision.wall_lane_status)

    def test_playstyle_auto_aim_wrapper_passes_friendly_exclusions(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.playstyle_code = compile("auto_aim_attack(300)", "<test_playstyle>", "exec")
        play.time_since_holding_attack = None
        play.TILE_SIZE = 60
        play.game_mode = 3
        play.seconds_to_hold_attack_after_reaching_max = 1.5
        play.is_hypercharge_ready = False
        play.should_use_gadget = False
        play.is_gadget_ready = False
        play.is_super_ready = False
        play._playstyle_error_reported = False
        play.attack = lambda *args, **kwargs: True
        play.use_hypercharge = lambda: True
        play.use_gadget = lambda: True
        play.use_super = lambda: True
        play.clear_ability_ready = lambda _ability: None
        play.last_playstyle_teammate_data = [[90, -14, 125, 14]]

        movement = play.run_playstyle(
            [0, -20, 40, 20],
            [[190, -20, 230, 20]],
            [],
            "shelly",
        )

        self.assertIsNone(movement)
        self.assertIsNone(window.angle)

    def test_playstyle_auto_aim_wrapper_blocks_wall_lane(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.playstyle_code = compile("auto_aim_attack(300)", "<test_playstyle>", "exec")
        play.time_since_holding_attack = None
        play.TILE_SIZE = 60
        play.game_mode = 3
        play.seconds_to_hold_attack_after_reaching_max = 1.5
        play.is_hypercharge_ready = False
        play.should_use_gadget = False
        play.is_gadget_ready = False
        play.is_super_ready = False
        play._playstyle_error_reported = False
        play.attack = lambda *args, **kwargs: True
        play.use_hypercharge = lambda: True
        play.use_gadget = lambda: True
        play.use_super = lambda: True
        play.clear_ability_ready = lambda _ability: None
        play.last_playstyle_teammate_data = []

        movement = play.run_playstyle(
            [0, -20, 40, 20],
            [[190, -20, 230, 20]],
            [[90, -8, 120, 8]],
            "shelly",
        )

        self.assertIsNone(movement)
        self.assertIsNone(window.angle)

    def test_legacy_get_movement_passes_friendly_exclusions(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.playstyle_code = None
        play.brawlers_info = {"shelly": {"hold_attack": 0, "ignore_walls_for_attacks": False}}
        play.current_brawler = "shelly"
        play.time_since_holding_attack = None
        play.seconds_to_hold_attack_after_reaching_max = 1.5
        play.get_brawler_range = lambda _brawler: (100, 300, 300)
        play.last_movement = "D"
        play.last_movement_time = 0.0
        play.minimum_movement_delay = 0.0
        play.game_mode = 3
        play.strafe_enabled = False
        play.no_enemy_movement = lambda *_args, **_kwargs: "W"
        play.is_path_blocked = lambda *_args, **_kwargs: False
        play.try_use_super_on_enemy = lambda *_args, **_kwargs: False
        play.should_use_gadget_on_enemy = lambda *_args, **_kwargs: False

        movement = play.get_movement(
            [0, -20, 40, 20],
            [[190, -20, 230, 20]],
            [],
            "shelly",
            teammate_data=[[90, -14, 125, 14]],
        )

        self.assertEqual(movement, "DW")
        self.assertIsNone(window.angle)

    def test_legacy_get_movement_can_move_toward_wall_blocked_enemy_without_firing(self):
        class AimWindow:
            def __init__(self):
                self.angle = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.playstyle_code = None
        play.brawlers_info = {"shelly": {"hold_attack": 0, "ignore_walls_for_attacks": False}}
        play.current_brawler = "shelly"
        play.time_since_holding_attack = None
        play.seconds_to_hold_attack_after_reaching_max = 1.5
        play.get_brawler_range = lambda _brawler: (100, 300, 300)
        play.last_movement = "D"
        play.last_movement_time = 0.0
        play.minimum_movement_delay = 0.0
        play.game_mode = 3
        play.strafe_enabled = False
        play.no_enemy_movement = lambda *_args, **_kwargs: "W"
        play.is_path_blocked = lambda *_args, **_kwargs: False
        play.try_use_super_on_enemy = lambda *_args, **_kwargs: False
        play.should_use_gadget_on_enemy = lambda *_args, **_kwargs: False

        movement = play.get_movement(
            [0, -20, 40, 20],
            [[190, -20, 230, 20]],
            [[90, -8, 120, 8]],
            "shelly",
            teammate_data=[],
        )

        self.assertEqual(movement, "DW")
        self.assertIsNone(window.angle)

    def test_sanitize_enemy_targets_removes_teammate_overlap_but_keeps_clear_enemy(self):
        play = self.make_auto_aim_play(types.SimpleNamespace())
        sanitized, excluded = play.sanitize_enemy_targets(
            [[100, -20, 140, 20], [220, -20, 260, 20]],
            [[102, -18, 138, 18]],
        )

        self.assertEqual(sanitized, [[220, -20, 260, 20]])
        self.assertEqual(excluded[0]["bbox"], (100, -20, 140, 20))

    def test_self_exclusion_does_not_block_adjacent_close_enemy(self):
        play = self.make_auto_aim_play(types.SimpleNamespace())
        excluded_boxes = play.build_attack_excluded_boxes(
            player_data=[0, -20, 40, 20],
            teammate_data=[],
        )
        sanitized, excluded = play.sanitize_enemy_targets(
            [[60, -20, 100, 20]],
            excluded_boxes,
        )

        self.assertEqual(sanitized, [[60, -20, 100, 20]])
        self.assertEqual(excluded, [])

    def test_self_iou_overlap_does_not_sanitize_close_enemy(self):
        play = self.make_auto_aim_play(types.SimpleNamespace())
        excluded_boxes = play.build_attack_excluded_boxes(
            player_data=[0, -30, 80, 70],
            teammate_data=[],
        )
        sanitized, excluded = play.sanitize_enemy_targets(
            [[45, -20, 125, 60]],
            excluded_boxes,
        )

        self.assertEqual(sanitized, [[45, -20, 125, 60]])
        self.assertEqual(excluded, [])

    def test_self_center_duplicate_is_still_sanitized(self):
        play = self.make_auto_aim_play(types.SimpleNamespace())
        excluded_boxes = play.build_attack_excluded_boxes(
            player_data=[0, -30, 80, 70],
            teammate_data=[],
        )
        sanitized, excluded = play.sanitize_enemy_targets(
            [[0, -30, 80, 70]],
            excluded_boxes,
        )

        self.assertEqual(sanitized, [])
        self.assertEqual(excluded[0]["reason"], "player_center:0.0")

    def test_valid_in_range_shot_is_not_delayed_by_defensive_suppression(self):
        class AimWindow:
            def __init__(self):
                self.angle = None
                self.kwargs = None

            def aim_attack_angle(self, angle, **kwargs):
                self.angle = angle
                self.kwargs = kwargs

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.attack_cooldown = 0.0
        play.dangerous_close_range = 120
        play._suppress_attack_until = time.time() + 1.0

        fired = play.auto_aim_attack("shelly", (0, 0), [[430, -18, 470, 18]], [], attack_range=520)

        self.assertTrue(fired)
        self.assertIsNotNone(window.angle)
        self.assertFalse(window.kwargs.get("force_release_movement"))

    def test_auto_aim_cooldown_still_blocks_repeated_shots(self):
        class AimWindow:
            def __init__(self):
                self.angles = []

            def aim_attack_angle(self, angle, **kwargs):
                self.angles.append((angle, kwargs))

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.attack_cooldown = 0.5
        play.last_attack_time = 0.0

        first = play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)
        second = play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(window.angles), 1)

    def test_attack_spam_shortens_cooldown_for_valid_in_range_enemy(self):
        class AimWindow:
            def __init__(self):
                self.angles = []

            def aim_attack_angle(self, angle, **kwargs):
                self.angles.append((angle, kwargs))

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.attack_cooldown = 0.5
        play.attack_spam_enabled = True
        play.attack_spam_cooldown_multiplier = 0.12

        first = play.auto_aim_attack("shelly", (0, 0), [[430, -18, 470, 18]], [], attack_range=520)
        play.last_attack_time -= 0.07
        second = play.auto_aim_attack("shelly", (0, 0), [[430, -18, 470, 18]], [], attack_range=520)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(len(window.angles), 2)

    def test_attack_spam_disabled_preserves_normal_cooldown(self):
        class AimWindow:
            def __init__(self):
                self.angles = []

            def aim_attack_angle(self, angle, **kwargs):
                self.angles.append((angle, kwargs))

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.attack_cooldown = 0.5
        play.attack_spam_enabled = False

        first = play.auto_aim_attack("shelly", (0, 0), [[430, -18, 470, 18]], [], attack_range=520)
        play.last_attack_time -= 0.24
        second = play.auto_aim_attack("shelly", (0, 0), [[430, -18, 470, 18]], [], attack_range=520)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(window.angles), 1)

    def test_attack_spam_requires_line_of_sight(self):
        class AimWindow:
            def __init__(self):
                self.angles = []

            def aim_attack_angle(self, angle, **kwargs):
                self.angles.append((angle, kwargs))

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.attack_cooldown = 0.0
        play.attack_spam_enabled = True
        play.attack_spam_requires_los = True
        play.walls_block_line_of_sight = lambda *_args, **_kwargs: True

        fired = play.auto_aim_attack("shelly", (0, 0), [[430, -18, 470, 18]], [], attack_range=520)

        self.assertFalse(fired)
        self.assertEqual(window.angles, [])

    def test_low_confidence_retry_merges_missing_enemy_boxes(self):
        class FakeDetector:
            def __init__(self):
                self.calls = []

            def detect_objects(self, frame, conf_tresh=0.6):
                self.calls.append(conf_tresh)
                if len(self.calls) == 1:
                    return {"player": [[90, 90, 130, 150]]}
                return {
                    "player": [[90, 90, 130, 150]],
                    "enemy": [[300, 90, 340, 150]],
                }

        play = object.__new__(Play)
        play.Detect_main_info = FakeDetector()
        play.entity_detection_confidence = 0.6
        play.entity_detection_retry_confidence = 0.35
        play.entity_retry_when_enemy_missing = True
        play.entity_marker_min_ratio = 0.012
        play.entity_marker_min_pixels = 12
        play.entity_marker_decision_margin = 1.25
        play.player_center_bias_radius = 420
        play.player_green_pixel_weight = 0.03
        play.player_red_pixel_penalty = 0.05
        play.window_controller = types.SimpleNamespace(scale_factor=1.0)

        frame = np.zeros((240, 480, 3), dtype=np.uint8)
        data = play.get_main_data(frame)

        self.assertEqual(play.Detect_main_info.calls, [0.6, 0.35])
        self.assertEqual(data["player"], [[90, 90, 130, 150]])
        self.assertEqual(data["enemy"], [[300, 90, 340, 150]])
        self.assertEqual(play._perf_entity_retry_count, 1)

    def test_marker_role_reads_blue_circle_below_box_as_teammate(self):
        play = self.make_marker_play()
        frame = np.zeros((220, 260, 3), dtype=np.uint8)
        box = [90, 80, 130, 130]
        self.draw_hsv_circle_below_box(frame, box, [105, 220, 220])

        self.assertEqual(play._marker_role(frame, box), "teammate")
        self.assertEqual(play._marker_role(frame, box), "teammate")
        self.assertEqual(play._perf_entity_marker_scores, 1)
        self.assertEqual(play._perf_entity_marker_cache_hits, 1)

    def test_marker_role_reads_enemy_circle_below_box_colors(self):
        colors = ([30, 230, 230], [16, 230, 230], [2, 230, 230])
        for hsv_color in colors:
            with self.subTest(hsv_color=hsv_color):
                play = self.make_marker_play()
                frame = np.zeros((220, 260, 3), dtype=np.uint8)
                box = [90, 80, 130, 130]
                self.draw_hsv_circle_below_box(frame, box, hsv_color)

                self.assertEqual(play._marker_role(frame, box), "enemy")

    def test_marker_role_stays_neutral_on_weak_or_mixed_signal(self):
        play = self.make_marker_play()
        frame = np.zeros((220, 260, 3), dtype=np.uint8)
        box = [90, 80, 130, 130]
        self.draw_hsv_circle_below_box(frame, box, [105, 220, 35])

        self.assertIsNone(play._marker_role(frame, box))

    def test_close_threat_keeps_parallel_movement_for_repeated_aimed_attacks(self):
        class AimWindow:
            def __init__(self):
                self.calls = []
                self.enable_parallel_movement_attack = True
                self.input_backend_supports_parallel_drag = True

            def aim_attack_angle(self, angle, **kwargs):
                self.calls.append((angle, kwargs))

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.attack_cooldown = 0.05
        play.dangerous_close_range = 180

        first = play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)
        play.last_attack_time -= 0.06
        second = play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(len(window.calls), 2)
        self.assertTrue(all(not call[1].get("force_release_movement") for call in window.calls))
        self.assertTrue(all(call[1].get("duration") == 0.02 for call in window.calls))

    def test_close_threat_releases_movement_when_parallel_attack_disabled(self):
        class AimWindow:
            def __init__(self):
                self.calls = []
                self.enable_parallel_movement_attack = False
                self.input_backend_supports_parallel_drag = True

            def aim_attack_angle(self, angle, **kwargs):
                self.calls.append((angle, kwargs))

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.attack_cooldown = 0.0
        play.dangerous_close_range = 180

        fired = play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)

        self.assertTrue(fired)
        self.assertTrue(window.calls[0][1].get("force_release_movement"))

    def test_close_threat_can_force_release_with_config_flag(self):
        class AimWindow:
            def __init__(self):
                self.calls = []
                self.enable_parallel_movement_attack = True
                self.input_backend_supports_parallel_drag = True

            def aim_attack_angle(self, angle, **kwargs):
                self.calls.append((angle, kwargs))

        window = AimWindow()
        play = self.make_auto_aim_play(window)
        play.attack_cooldown = 0.0
        play.dangerous_close_range = 180
        play.force_release_movement_on_close_threat = True

        fired = play.auto_aim_attack("shelly", (0, 0), [[70, -12, 94, 12]], [], attack_range=260)

        self.assertTrue(fired)
        self.assertTrue(window.calls[0][1].get("force_release_movement"))

    def test_playstyle_env_exposes_biomistik_helpers(self):
        play = object.__new__(Play)
        play.playstyle_code = compile(
            "movement = 270.0 if angle_to_keys(270) == 'W' and get_distance((0, 0), (3, 4)) == 5.0 else None",
            "<test_playstyle>",
            "exec",
        )
        play.time_since_holding_attack = None
        play.TILE_SIZE = 60
        play.brawlers_info = {}
        play.game_mode = 3
        play.seconds_to_hold_attack_after_reaching_max = 1.5
        play.is_hypercharge_ready = False
        play.should_use_gadget = False
        play.is_gadget_ready = False
        play.is_super_ready = False
        play.attack = lambda *args, **kwargs: True
        play.use_hypercharge = lambda: True
        play.use_gadget = lambda: True
        play.use_super = lambda: True
        play.clear_ability_ready = lambda _ability: None
        play.should_use_super_on_enemy = lambda *args, **kwargs: False
        play.must_brawler_hold_attack = lambda *args, **kwargs: False
        play.get_brawler_range = lambda _brawler: (100, 200, 300)
        play.get_player_pos = lambda _player: (50, 50)
        play.get_entity_pos = lambda _entity: (50, 50)
        play.is_there_enemy = lambda _enemy: False
        play.is_there_poison_gas = lambda *_args, **_kwargs: False
        play.no_enemy_movement = lambda *_args, **_kwargs: "W"
        play.find_closest_enemy = lambda *_args, **_kwargs: (None, None)
        play.find_closest_teammate = lambda *_args, **_kwargs: (None, None)
        play.get_horizontal_move_key = lambda *_args, **_kwargs: "D"
        play.get_vertical_move_key = lambda *_args, **_kwargs: "W"
        play.is_path_blocked = lambda *_args, **_kwargs: False
        play.is_path_blocked_angle = lambda *_args, **_kwargs: False
        play.is_enemy_hittable = lambda *_args, **_kwargs: False
        play.walls_block_line_of_sight = lambda *_args, **_kwargs: False
        play.aimed_attack = lambda *_args, **_kwargs: True
        play.get_distance = Play.get_distance
        play.angle_from_direction = lambda *_args, **_kwargs: 0.0
        play.find_best_angle = lambda _player, angle, _walls: angle
        play.blend_angles = lambda primary, *_args, **_kwargs: primary
        play.lead_shot_angle = lambda *_args, **_kwargs: 0.0
        play.track_enemy_velocity = lambda *_args, **_kwargs: (0.0, 0.0)
        play.detect_wall_stuck = lambda *_args, **_kwargs: False
        play.start_semicircle_escape = lambda *_args, **_kwargs: None
        play.semicircle_escape_step = lambda *_args, **_kwargs: None
        play._playstyle_error_reported = False

        movement = play.run_playstyle([0, 0, 100, 100], [], [], "shelly")

        self.assertEqual(movement, 270.0)

    def test_showdown_hide_mode_roams_when_no_enemy_and_teammate_visible(self):
        play = object.__new__(Play)
        play.brawlers_info = {"shelly": {"hold_attack": 0, "super_type": "damage"}}
        play.must_brawler_hold_attack = lambda *_args, **_kwargs: False
        play.time_since_holding_attack = None
        play.seconds_to_hold_attack_after_reaching_max = 1.5
        play.get_brawler_range = lambda _brawler: (100, 200, 300)
        play.get_player_pos = lambda _player: (50, 50)
        play._fog_check_counter = 0
        play.fog_check_every_n_frames = 999
        play._fog_direction_escape_cached = None
        play._fog_threat_cached = None
        play.detect_fog_threat = lambda *_args, **_kwargs: None
        play.detect_fog_direction_escape = lambda *_args, **_kwargs: None
        play.current_frame = None
        play.is_there_enemy = lambda _enemy: False
        play.showdown_follow_teammate = lambda *_args, **_kwargs: 45.0
        play.showdown_roam = lambda *_args, **_kwargs: 270.0
        play.showdown_playstyle_mode = "hide"

        movement = play.get_showdown_movement([0, 0, 100, 100], [], [[100, 100, 120, 120]], [], "shelly")

        self.assertEqual(movement, 270.0)

    def test_showdown_follow_mode_follows_teammate_when_no_enemy_visible(self):
        play = object.__new__(Play)
        play.brawlers_info = {"shelly": {"hold_attack": 0, "super_type": "damage"}}
        play.must_brawler_hold_attack = lambda *_args, **_kwargs: False
        play.time_since_holding_attack = None
        play.seconds_to_hold_attack_after_reaching_max = 1.5
        play.get_brawler_range = lambda _brawler: (100, 200, 300)
        play.get_player_pos = lambda _player: (50, 50)
        play._fog_check_counter = 0
        play.fog_check_every_n_frames = 999
        play._fog_direction_escape_cached = None
        play._fog_threat_cached = None
        play.detect_fog_threat = lambda *_args, **_kwargs: None
        play.detect_fog_direction_escape = lambda *_args, **_kwargs: None
        play.current_frame = None
        play.is_there_enemy = lambda _enemy: False
        play.showdown_follow_teammate = lambda *_args, **_kwargs: 45.0
        play.showdown_roam = lambda *_args, **_kwargs: 270.0
        play.showdown_playstyle_mode = "follow"

        movement = play.get_showdown_movement([0, 0, 100, 100], [], [[100, 100, 120, 120]], [], "shelly")

        self.assertEqual(movement, 45.0)

    def test_showdown_follow_mode_uses_alive_marker_when_teammate_box_missing(self):
        play = object.__new__(Play)
        play.brawlers_info = {"shelly": {"hold_attack": 0, "super_type": "damage"}}
        play.must_brawler_hold_attack = lambda *_args, **_kwargs: False
        play.time_since_holding_attack = None
        play.get_brawler_range = lambda _brawler: (100, 200, 300)
        play.get_player_pos = lambda _player: (150, 150)
        play.is_there_enemy = lambda _enemy: False
        play.detect_fog_threat = lambda *_args, **_kwargs: None
        play.detect_fog_direction_escape = lambda *_args, **_kwargs: None
        play.angle_points_into_fog = lambda *_args, **_kwargs: False
        play.showdown_roam = lambda *_args, **_kwargs: 270.0
        play.showdown_playstyle_mode = "follow"
        play._fog_check_counter = 0
        play.fog_check_every_n_frames = 3
        play._fog_threat_cached = None
        play._fog_direction_escape_cached = None
        play.teammate_marker_follow_enabled = True
        play.teammate_marker_edge_margin = 0.28
        play.locked_teammate = None
        play.locked_teammate_distance = float("inf")
        play.teammate_hysteresis = 0.75
        play.teammate_lock_max_jump = 320
        play.teammate_lock_lost_since = 0.0
        play.teammate_follow_step_distance = 8
        play.get_enemy_pos = lambda entity: entity
        play.get_distance = Play.get_distance
        play.angle_from_direction = Play.angle_from_direction
        play.find_closest_teammate = Play.find_closest_teammate.__get__(play, Play)
        play.choose_locked_teammate = Play.choose_locked_teammate.__get__(play, Play)
        play.find_teammate_alive_marker = Play.find_teammate_alive_marker.__get__(play, Play)
        play.teammate_marker_follow_angle = Play.teammate_marker_follow_angle.__get__(play, Play)
        play._count_mask_pixels = Play._count_mask_pixels

        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        frame[:, :] = (35, 35, 45)
        cv2.rectangle(frame, (238, 112), (298, 188), (20, 110, 245), -1)
        cv2.circle(frame, (268, 150), 23, (245, 245, 245), -1)
        cv2.circle(frame, (258, 145), 5, (25, 105, 220), -1)
        cv2.circle(frame, (278, 145), 5, (25, 105, 220), -1)
        play.current_frame = frame

        movement = play.get_showdown_movement([135, 135, 165, 165], [], [], [], "shelly")

        self.assertLess(abs((movement - 0.0 + 180) % 360 - 180), 8.0)

    def test_showdown_follow_mode_does_not_follow_into_fog(self):
        play = object.__new__(Play)
        play.brawlers_info = {"shelly": {"hold_attack": 0, "super_type": "damage"}}
        play.must_brawler_hold_attack = lambda *_args, **_kwargs: False
        play.time_since_holding_attack = None
        play.seconds_to_hold_attack_after_reaching_max = 1.5
        play.get_brawler_range = lambda _brawler: (100, 200, 300)
        play.get_player_pos = lambda _player: (50, 50)
        play._fog_check_counter = 0
        play.fog_check_every_n_frames = 999
        play._fog_direction_escape_cached = None
        play._fog_threat_cached = None
        play.detect_fog_threat = lambda *_args, **_kwargs: None
        play.detect_fog_direction_escape = lambda *_args, **_kwargs: None
        play.current_frame = object()
        play.is_there_enemy = lambda _enemy: False
        play.showdown_follow_teammate = lambda *_args, **_kwargs: 0.0
        play.showdown_roam = lambda *_args, **_kwargs: 270.0
        play.angle_points_into_fog = lambda *_args, **_kwargs: True
        play.angle_opposite = Play.angle_opposite
        play.find_best_angle = lambda _player, angle, _walls: angle
        play.showdown_playstyle_mode = "follow"

        movement = play.get_showdown_movement([0, 0, 100, 100], [], [[100, 100, 120, 120]], [], "shelly")

        self.assertEqual(movement, 180.0)

    def test_showdown_follow_teammate_moves_directly_toward_closest_teammate(self):
        play = object.__new__(Play)
        play.locked_teammate = None
        play.locked_teammate_distance = float("inf")
        play.teammate_hysteresis = 0.75
        play.teammate_lock_max_jump = 320
        play.teammate_lock_lost_since = 0.0
        play.teammate_follow_min_distance = 180
        play.teammate_follow_step_distance = 8
        play.teammate_follow_force_direct = True
        play.get_player_pos = lambda _player: (100, 100)
        play.get_enemy_pos = lambda entity: entity
        play.get_distance = Play.get_distance
        play.angle_from_direction = Play.angle_from_direction
        play.is_path_blocked_angle = lambda *_args, **_kwargs: False

        movement = play.showdown_follow_teammate(
            [90, 90, 110, 110],
            [(200, 100), (120, 220)],
            [],
        )

        self.assertEqual(movement, 0.0)

    def test_showdown_follow_teammate_uses_axis_option_when_diagonal_blocked(self):
        play = object.__new__(Play)
        play.locked_teammate = None
        play.locked_teammate_distance = float("inf")
        play.teammate_hysteresis = 0.75
        play.teammate_lock_max_jump = 320
        play.teammate_lock_lost_since = 0.0
        play.teammate_follow_min_distance = 180
        play.teammate_follow_step_distance = 8
        play.teammate_follow_force_direct = False
        play.teammate_follow_wall_avoid_enabled = True
        play.teammate_follow_detour_angles = [25, 45, 70, 95]
        play.teammate_follow_direct_probe_multiplier = 1.35
        play.teammate_follow_detour_hysteresis = 0.25
        play.teammate_follow_blocked_angle_memory_seconds = 0.8
        play.teammate_follow_no_safe_escape_enabled = True
        play._follow_blocked_angles = []
        play._last_follow_angle = None
        play._last_follow_wall_debug = {}
        play.get_player_pos = lambda _player: (100, 100)
        play.get_enemy_pos = lambda entity: entity
        play.get_distance = Play.get_distance
        play.angle_from_direction = Play.angle_from_direction
        blocked = {45.0}
        play.is_path_blocked_angle = lambda _player, angle, _walls: round(angle, 1) in blocked

        movement = play.showdown_follow_teammate(
            [90, 90, 110, 110],
            [(200, 200)],
            [[130, 130, 150, 150]],
        )

        self.assertNotEqual(round(movement, 1), 45.0)
        self.assertEqual(play._last_follow_wall_debug["status"], "detour_clear")

    def test_showdown_follow_teammate_force_direct_ignores_blocked_unknown_path(self):
        play = object.__new__(Play)
        play.locked_teammate = None
        play.locked_teammate_distance = float("inf")
        play.teammate_hysteresis = 0.75
        play.teammate_lock_max_jump = 320
        play.teammate_lock_lost_since = 0.0
        play.teammate_follow_min_distance = 80
        play.teammate_follow_step_distance = 8
        play.teammate_follow_force_direct = True
        play.teammate_follow_wall_avoid_enabled = True
        play.teammate_follow_detour_angles = [25, 45, 70, 95]
        play.teammate_follow_direct_probe_multiplier = 1.35
        play.teammate_follow_detour_hysteresis = 0.25
        play.teammate_follow_blocked_angle_memory_seconds = 0.8
        play.teammate_follow_no_safe_escape_enabled = True
        play._follow_blocked_angles = []
        play._last_follow_angle = None
        play._last_follow_wall_debug = {}
        play.get_player_pos = lambda _player: (100, 100)
        play.get_enemy_pos = lambda entity: entity
        play.get_distance = Play.get_distance
        play.angle_from_direction = Play.angle_from_direction
        play.is_path_blocked_angle = lambda *_args, **_kwargs: True

        movement = play.showdown_follow_teammate(
            [90, 90, 110, 110],
            [(220, 220)],
            [[130, 130, 150, 150]],
        )

        self.assertNotEqual(round(movement, 1), 45.0)
        self.assertEqual(round(movement, 1), 225.0)
        self.assertEqual(play._last_follow_wall_debug["status"], "no_safe_angle")

    def test_wall_aware_follow_keeps_direct_when_clear(self):
        play = object.__new__(Play)
        play.teammate_follow_wall_avoid_enabled = True
        play._last_follow_wall_debug = {}
        play._follow_blocked_angles = []
        play.angle_from_direction = Play.angle_from_direction
        play.is_path_blocked_angle = lambda *_args, **_kwargs: False

        angle, _reason, status = play.choose_wall_aware_follow_angle(
            (100, 100),
            (220, 100),
            [[140, 180, 170, 220]],
        )

        self.assertEqual(angle, 0.0)
        self.assertEqual(status, "direct_clear")

    def test_wall_aware_follow_uses_nearest_detour_when_direct_blocked(self):
        play = object.__new__(Play)
        play.teammate_follow_wall_avoid_enabled = True
        play.teammate_follow_detour_angles = [25, 45, 70, 95]
        play.teammate_follow_direct_probe_multiplier = 1.35
        play.teammate_follow_detour_hysteresis = 0.25
        play.teammate_follow_blocked_angle_memory_seconds = 0.8
        play._follow_blocked_angles = []
        play._last_follow_angle = None
        play._last_follow_wall_debug = {}
        play.angle_from_direction = Play.angle_from_direction
        play.is_path_blocked_angle = lambda _player, angle, _walls, **_kwargs: round(angle, 1) == 0.0

        angle, _reason, status = play.choose_wall_aware_follow_angle(
            (100, 100),
            (220, 100),
            [[130, 90, 160, 120]],
        )

        self.assertEqual(round(angle, 1), 25.0)
        self.assertEqual(status, "detour_clear")

    def test_wall_aware_follow_blocked_memory_prefers_other_side(self):
        play = object.__new__(Play)
        play.teammate_follow_wall_avoid_enabled = True
        play.teammate_follow_detour_angles = [25, 45]
        play.teammate_follow_direct_probe_multiplier = 1.35
        play.teammate_follow_detour_hysteresis = 0.25
        play.teammate_follow_blocked_angle_memory_seconds = 0.8
        play._follow_blocked_angles = [{"angle": 25.0, "time": time.time(), "reason": "recent_stuck"}]
        play._last_follow_angle = None
        play._last_follow_wall_debug = {}
        play.angle_from_direction = Play.angle_from_direction
        play.is_path_blocked_angle = lambda _player, angle, _walls, **_kwargs: round(angle, 1) == 0.0

        angle, _reason, status = play.choose_wall_aware_follow_angle(
            (100, 100),
            (220, 100),
            [[130, 90, 160, 120]],
        )

        self.assertEqual(round(angle, 1), 335.0)
        self.assertEqual(status, "detour_clear")

    def test_showdown_follow_teammate_keeps_locked_mate_over_new_closer_mate(self):
        play = object.__new__(Play)
        play.locked_teammate = (300, 100)
        play.locked_teammate_distance = 200
        play.teammate_hysteresis = 0.75
        play.teammate_lock_max_jump = 320
        play.teammate_lock_lost_since = 0.0
        play.teammate_follow_min_distance = 180
        play.teammate_follow_step_distance = 8
        play.teammate_follow_force_direct = True
        play.get_player_pos = lambda _player: (100, 100)
        play.get_enemy_pos = lambda entity: entity
        play.get_distance = Play.get_distance
        play.angle_from_direction = Play.angle_from_direction
        play.is_path_blocked_angle = lambda *_args, **_kwargs: False

        movement = play.showdown_follow_teammate(
            [90, 90, 110, 110],
            [(305, 100), (160, 100)],
            [],
        )

        self.assertEqual(movement, 0.0)
        self.assertEqual(play.locked_teammate, (305, 100))


if __name__ == "__main__":
    unittest.main()
