import math
import sys
import time
import types
import unittest

import cv2
import numpy as np

sys.modules.setdefault("onnxruntime", types.SimpleNamespace(InferenceSession=None))

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

    def make_auto_aim_play(self, window_controller):
        play = object.__new__(Play)
        play.window_controller = window_controller
        play.brawlers_info = {"shelly": {"ignore_walls_for_attacks": False}}
        play.attack_cooldown = 0.0
        play.close_range_attack_cooldown_multiplier = 0.55
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
        play.lead_shots_enabled = True
        play.aimed_attacks_enabled = True
        play.enemy_velocity_confidence = 1.0
        play.track_enemy_velocity = lambda *_args: (0.0, 0.0)
        return play

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

    def test_close_threat_forces_movement_pause_for_repeated_aimed_attacks(self):
        class AimWindow:
            def __init__(self):
                self.calls = []
                self.enable_parallel_movement_attack = True

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
        self.assertTrue(all(call[1].get("force_release_movement") for call in window.calls))

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
        play.get_player_pos = lambda _player: (100, 100)
        play.get_enemy_pos = lambda entity: entity
        play.get_distance = Play.get_distance
        play.angle_from_direction = Play.angle_from_direction
        blocked = {45.0}
        play.is_path_blocked_angle = lambda _player, angle, _walls: round(angle, 1) in blocked

        movement = play.showdown_follow_teammate(
            [90, 90, 110, 110],
            [(200, 200)],
            [],
        )

        self.assertEqual(movement, 0.0)

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
        play.get_player_pos = lambda _player: (100, 100)
        play.get_enemy_pos = lambda entity: entity
        play.get_distance = Play.get_distance
        play.angle_from_direction = Play.angle_from_direction
        play.is_path_blocked_angle = lambda *_args, **_kwargs: True

        movement = play.showdown_follow_teammate(
            [90, 90, 110, 110],
            [(220, 220)],
            [],
        )

        self.assertEqual(round(movement, 1), 45.0)

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
