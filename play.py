import math
import json
import os
import random
import threading
import time
from collections import deque

import cv2
import numpy as np
from state_finder import get_state
from auto_aim import choose_auto_aim, detect_aim_line_angle
from combat_brain import (
    CombatFrame,
    HealthState,
    SafetyResult,
    TargetScore,
    TargetMemory,
    build_threat_model as build_combat_threat_model,
    choose_ability_plan,
    choose_attack_gate,
    choose_combat_intent,
    choose_target as choose_combat_target,
)
from detect import Detect
from movement_intent import (
    MovementIntentMemory,
    build_movement_intent,
    build_threat_state,
    smooth_intent,
)
from tactical_movement import (
    candidate_dodge_angles,
    classify_dodge_mode,
    movement_keys_to_angle,
    projectile_threat,
    score_dodge_angle,
    score_projectile_dodge_angle,
    should_seek_healing,
    threat_level_from_distance,
)
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info

brawl_stars_width, brawl_stars_height = 1920, 1080
debug = load_toml_as_dict("cfg/general_config.toml").get('super_debug', 'no') == "yes"
visual_debug = load_toml_as_dict("cfg/general_config.toml").get('visual_debug', 'no') == "yes"

def vlog(*args):
    if visual_debug:
        print("[DBG]", *args)
DEFAULT_PIXEL_COUNTER_CROP_AREA = {
    "super": [1460, 830, 1560, 930],
    "gadget": [1580, 930, 1700, 1050],
    "hypercharge": [1350, 940, 1450, 1050],
}


def _pixel_counter_crop_area():
    lobby_config = load_toml_as_dict("./cfg/lobby_config.toml")
    return {**DEFAULT_PIXEL_COUNTER_CROP_AREA, **lobby_config.get("pixel_counter_crop_area", {})}


_crop_area = _pixel_counter_crop_area()
super_crop_area = _crop_area["super"]
gadget_crop_area = _crop_area["gadget"]
hypercharge_crop_area = _crop_area["hypercharge"]

class Movement:

    def __init__(self, window_controller):
        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        self.fix_movement_keys = {
            "delay_to_trigger": bot_config.get("unstuck_movement_delay", 3.0),
            "duration": bot_config.get("unstuck_movement_hold_time", 1.4),
            "toggled": False,
            "started_at": time.time(),
            "fixed": ""
        }
        self.game_mode = bot_config.get("gamemode_type", 3)
        gadget_value = bot_config.get("bot_uses_gadgets", "yes")
        self.should_use_gadget = str(gadget_value).lower() in ("yes", "true", "1")
        self.gadget_cooldown = float(bot_config.get("gadget_cooldown", 1.0))
        self.last_gadget_time = 0.0
        self.super_cooldown = float(bot_config.get("super_cooldown", 0.25))
        self.super_retry_cooldown_multiplier = float(bot_config.get("super_retry_cooldown_multiplier", 0.5))
        self.last_super_time = 0.0
        self.super_treshold = time_config.get("super", 0.1)
        self.gadget_treshold = time_config.get("gadget", 0.1)
        self.hypercharge_treshold = time_config.get("hypercharge", 0.1)
        self.walls_treshold = time_config.get("wall_detection", 0.75)
        self.keep_walls_in_memory = self.walls_treshold <= 1
        self.last_walls_data = []
        self.keys_hold = []
        self.time_since_different_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False
        self.window_controller = window_controller
        self.attack_cooldown = float(bot_config.get("attack_cooldown", 0.12))
        self.close_range_attack_cooldown_multiplier = float(bot_config.get("close_range_attack_cooldown_multiplier", 0.55))
        self.attack_spam_enabled = str(bot_config.get("attack_spam_enabled", "true")).lower() in ("yes", "true", "1")
        self.attack_spam_cooldown_multiplier = float(bot_config.get("attack_spam_cooldown_multiplier", 0.12))
        self.attack_spam_requires_los = str(bot_config.get("attack_spam_requires_los", "true")).lower() in ("yes", "true", "1")
        self.aim_attack_duration = float(bot_config.get("aim_attack_duration", 0.02))
        self.force_release_movement_on_close_threat = str(
            bot_config.get("force_release_movement_on_close_threat", "false")
        ).lower() in ("yes", "true", "1")
        self.last_attack_time = 0.0
        self.TILE_SIZE = 60
        # Wall-based stuck detector: samples wall bboxes on an interval, ignores
        # walls near the player (they flicker as he overlaps them), and flags
        # "stuck" when walls don't move for wall_stuck_timeout seconds while the
        # bot is trying to move. Triggers a semicircle escape maneuver.
        self.wall_stuck_enabled = str(bot_config.get("wall_stuck_enabled", "yes")).lower() in ("yes", "true", "1")
        general_config = load_toml_as_dict("cfg/general_config.toml")
        self.wall_stuck_debug = str(general_config.get("wall_stuck_debug", "no")).lower() in ("yes", "true", "1")
        self.wall_stuck_ignore_radius = float(bot_config.get("wall_stuck_ignore_radius", 150))
        self.wall_stuck_sample_interval = float(bot_config.get("wall_stuck_sample_interval", 0.2))
        self.wall_stuck_shift_threshold = float(bot_config.get("wall_stuck_shift_threshold", 3.0))
        self.wall_stuck_timeout = float(bot_config.get("wall_stuck_timeout", 3.0))
        self.wall_stuck_min_walls = int(bot_config.get("wall_stuck_min_walls", 3))
        self.wall_path_padding = float(bot_config.get("wall_path_padding", 28))
        self.wall_path_probe_tiles = float(bot_config.get("wall_path_probe_tiles", 1.5))
        self.wall_box_min_size = float(bot_config.get("wall_box_min_size", 20))
        self.wall_box_merge_iou = float(bot_config.get("wall_box_merge_iou", 0.25))
        self.wall_box_merge_center_distance = float(bot_config.get("wall_box_merge_center_distance", 35))
        self.wall_history_min_hits = int(bot_config.get("wall_history_min_hits", 1))
        self.jump_pad_detection_enabled = str(bot_config.get("jump_pad_detection_enabled", "yes")).lower() in ("yes", "true", "1")
        self.jump_pad_escape_distance = float(bot_config.get("jump_pad_escape_distance", 620))
        self.jump_pad_escape_min_distance = float(bot_config.get("jump_pad_escape_min_distance", 55))
        self.jump_pad_escape_requires_edge = str(bot_config.get("jump_pad_escape_requires_edge", "yes")).lower() in ("yes", "true", "1")
        self.jump_pad_escape_edge_margin = float(bot_config.get("jump_pad_escape_edge_margin", 0.22))
        self.jump_pad_escape_teammate_safe_distance = float(bot_config.get("jump_pad_escape_teammate_safe_distance", 360))
        self.jump_pad_smoke_early_distance = float(bot_config.get("jump_pad_smoke_early_distance", 230))
        self.wall_stuck_state = {
            "last_sample_time": 0.0,
            "last_wall_centers": None,   # np.ndarray (N, 2) of filtered wall centers
            "stationary_since": None,    # when walls first went stationary; None = not stationary
        }

        # Semicircle escape state. Alternates side globally between triggers.
        self.escape_retreat_duration = float(bot_config.get("escape_retreat_duration", 0.4))
        self.escape_arc_duration = float(bot_config.get("escape_arc_duration", 1.2))
        self.escape_arc_degrees = float(bot_config.get("escape_arc_degrees", 135.0))
        self.escape_state = {
            "phase": None,            # "retreat" | "arc" | None
            "started_at": 0.0,
            "retreat_angle": 0.0,
            "arc_side": 1,            # +1 = CCW, -1 = CW; flipped each trigger
        }
        self._next_arc_side = 1
        self.adaptive_safe_range_multiplier = 1.0
        self.strafe_enabled = str(bot_config.get("strafe_while_attacking", "yes")).lower() in ("yes", "true", "1")
        self.strafe_interval = float(bot_config.get("strafe_interval", 1.6))
        self.strafe_blend = float(bot_config.get("strafe_blend", 0.35))
        self._strafe_started_at = 0.0
        self._strafe_side = 1
        self.combat_dodge_blend = float(bot_config.get("combat_dodge_blend", 0.45))
        self.combat_dodge_jitter_degrees = float(bot_config.get("combat_dodge_jitter_degrees", 18.0))
        self.enemy_pressure_move_range_multiplier = float(bot_config.get("enemy_pressure_move_range_multiplier", 1.15))
        self.lead_shots_enabled = str(bot_config.get("lead_shots", "yes")).lower() in ("yes", "true", "1")
        self.aimed_attacks_enabled = str(bot_config.get("aimed_attacks", "no")).lower() in ("yes", "true", "1")
        self.projectile_speed_px_s = float(bot_config.get("projectile_speed_px_s", 900.0))
        self.auto_aim_min_confidence = float(bot_config.get("auto_aim_min_confidence", 0.54))
        self.auto_aim_close_tap_range = float(bot_config.get("auto_aim_close_tap_range", 0))
        self.auto_aim_close_los_override_range = float(bot_config.get("auto_aim_close_los_override_range", 0))
        self.close_range_attack_override = str(bot_config.get("close_range_attack_override", "true")).lower() in ("yes", "true", "1")
        self.attack_decision_debug = str(bot_config.get("attack_decision_debug", bot_config.get("auto_aim_debug", "yes"))).lower() in ("yes", "true", "1")
        self.auto_aim_debug = str(bot_config.get("auto_aim_debug", "yes")).lower() in ("yes", "true", "1")
        self.enable_flicker_retreat = str(bot_config.get("enable_flicker_retreat", "true")).lower() in ("yes", "true", "1")
        self.enable_combat_mans = str(bot_config.get("enable_combat_mans", "true")).lower() in ("yes", "true", "1")
        self.mans_threat_threshold = float(bot_config.get("mans_threat_threshold", 0.42))
        self.mans_hysteresis_seconds = float(bot_config.get("mans_hysteresis_ms", 450)) / 1000.0
        self.flicker_retreat_cooldown_seconds = float(bot_config.get("flicker_retreat_cooldown_ms", 900)) / 1000.0
        self.flicker_retreat_hold_seconds = float(bot_config.get("flicker_retreat_hold_ms", 650)) / 1000.0
        self.dangerous_close_range = float(bot_config.get("dangerous_close_range", 150))
        self.heal_retreat_enabled = str(bot_config.get("heal_retreat_enabled", "true")).lower() in ("yes", "true", "1")
        self.heal_low_health_threshold = float(bot_config.get("heal_low_health_threshold", 0.42))
        self.heal_resume_health_threshold = float(bot_config.get("heal_resume_health_threshold", 0.72))
        self.heal_retreat_hold_ms = float(bot_config.get("heal_retreat_hold_ms", 2400))
        self.heal_attack_only_close_range = float(bot_config.get("heal_attack_only_close_range", 150))
        self._player_bar_history = []
        self._flicker_state = {"active_until": 0.0, "last_trigger": 0.0, "confidence": 0.0}
        self._heal_state = {"active_until": 0.0, "last_health_ratio": None, "reason": ""}
        self._dodge_state = {"angle": None, "mode": "no_dodge", "score": 0.0, "until": 0.0}
        self._projectile_track = {}
        self._last_projectile_threat = None
        self._last_projectile_threat_until = 0.0
        self._suppress_attack_until = 0.0
        self._last_aim_attempt_time = 0.0
        self._enemy_track = {}
        self.enemy_velocity = (0.0, 0.0)
        self.velocity_ema_alpha = float(bot_config.get("velocity_ema_alpha", 0.40))
        self._enemy_velocity_smooth = {}
        self._enemy_velocity_confidence = {}
        self.enemy_velocity_confidence = 0.0
        self._strafe_current_interval = 0.0
        self.roam_direction_hold_time = float(bot_config.get("roam_direction_hold_time", 1.5))
        self.roam_center_bias = float(bot_config.get("roam_center_bias", 0.25))
        self._roam_angle = random.uniform(0, 360)
        self._roam_last_changed = 0.0
        self.retreat_strafe_fraction = float(bot_config.get("retreat_strafe_fraction", 0.5))
        self.approach_flank_blend = float(bot_config.get("approach_flank_blend", 0.12))
        self.multi_enemy_flee_weight = float(bot_config.get("multi_enemy_flee_weight", 0.45))
        self.angle_smooth_factor = float(bot_config.get("angle_smooth_factor", 0.28))
        self.movement_input_mode = str(bot_config.get("movement_input_mode", "auto")).strip().lower()
        self.enable_joystick_movement = str(bot_config.get("enable_joystick_movement", "true")).lower() in ("yes", "true", "1")
        self._movement_fallback_to_wasd = False
        self.projectile_dodge_enabled = str(bot_config.get("projectile_dodge_enabled", "true")).lower() in ("yes", "true", "1")
        self.projectile_dodge_horizon = float(bot_config.get("projectile_dodge_horizon", 0.75))
        self.projectile_dodge_player_radius = float(bot_config.get("projectile_dodge_player_radius", 38.0))
        self.movement_intent_enabled = str(bot_config.get("movement_intent_enabled", "true")).lower() in ("yes", "true", "1")
        self.movement_intent_min_hold_ms = float(bot_config.get("movement_intent_min_hold_ms", 350))
        self.movement_intent_max_hold_ms = float(bot_config.get("movement_intent_max_hold_ms", 650))
        self.movement_intent_switch_score_threshold = float(bot_config.get("movement_intent_switch_score_threshold", 0.18))
        self.movement_intent_angle_smoothing = float(bot_config.get("movement_intent_angle_smoothing", 0.35))
        self.movement_intent_debug = str(bot_config.get("movement_intent_debug", "yes")).lower() in ("yes", "true", "1")
        self._movement_intent_memory = MovementIntentMemory()
        self.combat_brain_enabled = str(bot_config.get("combat_brain_enabled", "true")).lower() in ("yes", "true", "1")
        self.combat_brain_debug = str(bot_config.get("combat_brain_debug", "yes")).lower() in ("yes", "true", "1")
        self.ability_brain_enabled = str(bot_config.get("ability_brain_enabled", "true")).lower() in ("yes", "true", "1")
        self.ability_brain_debug = str(bot_config.get("ability_brain_debug", "yes")).lower() in ("yes", "true", "1")
        self.defensive_attack_gate_enabled = str(bot_config.get("defensive_attack_gate_enabled", "true")).lower() in ("yes", "true", "1")
        self.panic_shot_range = float(bot_config.get("panic_shot_range", 150))
        self.panic_super_range = float(bot_config.get("panic_super_range", 180))
        self.target_memory_seconds = float(bot_config.get("target_memory_seconds", 0.75))
        self.target_switch_margin = float(bot_config.get("target_switch_margin", 0.18))
        self.projectile_dodge_without_enemy = str(bot_config.get("projectile_dodge_without_enemy", "true")).lower() in ("yes", "true", "1")
        self.projectile_threat_memory_ms = float(bot_config.get("projectile_threat_memory_ms", 300))
        self.super_min_value_score = float(bot_config.get("super_min_value_score", 0.55))
        self.gadget_min_value_score = float(bot_config.get("gadget_min_value_score", 0.50))
        self.hypercharge_min_value_score = float(bot_config.get("hypercharge_min_value_score", 0.70))
        self.ability_retry_log_interval_ms = float(bot_config.get("ability_retry_log_interval_ms", 500))
        self.wall_angle_fail_escape_enabled = str(bot_config.get("wall_angle_fail_escape_enabled", "true")).lower() in ("yes", "true", "1")
        self.wall_escape_blocks_attack = str(bot_config.get("wall_escape_blocks_attack", "true")).lower() in ("yes", "true", "1")
        self.tactical_planner_enabled = str(bot_config.get("tactical_planner_enabled", "true")).lower() in ("yes", "true", "1")
        self.tactical_angle_samples = int(bot_config.get("tactical_angle_samples", 16))
        self.survival_score_min_to_commit = float(bot_config.get("survival_score_min_to_commit", 0.62))
        self.kill_confirm_score_threshold = float(bot_config.get("kill_confirm_score_threshold", 0.68))
        self.preferred_target_lock_for_aim = str(bot_config.get("preferred_target_lock_for_aim", "true")).lower() in ("yes", "true", "1")
        self.adaptive_aggression_enabled = str(bot_config.get("adaptive_aggression_enabled", "true")).lower() in ("yes", "true", "1")
        self.damage_penalty_window_seconds = float(bot_config.get("damage_penalty_window_seconds", 3.0))
        self.missed_kill_window_decay = float(bot_config.get("missed_kill_window_decay", 0.04))
        self.friendly_fire_guard_enabled = str(bot_config.get("friendly_fire_guard_enabled", "true")).lower() in ("yes", "true", "1")
        self.friendly_fire_iou_threshold = float(bot_config.get("friendly_fire_iou_threshold", 0.18))
        self.friendly_fire_center_distance_px = float(bot_config.get("friendly_fire_center_distance_px", 70))
        self.close_attack_requires_clear_hit_point = str(bot_config.get("close_attack_requires_clear_hit_point", "true")).lower() in ("yes", "true", "1")
        self.attack_wall_guard_enabled = str(bot_config.get("attack_wall_guard_enabled", "true")).lower() in ("yes", "true", "1")
        self._last_ability_plan_log = 0.0
        self._target_memory = TargetMemory()
        self._tactical_adaptation = {
            "pressure_damage_events": 0,
            "aggression_penalty_until": 0.0,
            "aggression_penalty": 0.0,
            "fire_threshold_delta": 0.0,
            "last_objective": None,
            "last_objective_time": 0.0,
        }
        self.combat_snapshot_enabled = str(bot_config.get("combat_snapshot_enabled", "true")).lower() in ("yes", "true", "1")
        self.combat_snapshot_dir = str(bot_config.get("combat_snapshot_dir", "debug_frames/combat"))
        self.combat_snapshot_seconds = float(bot_config.get("combat_snapshot_seconds", 8))
        self._combat_decision_history = deque(maxlen=max(30, int(self.combat_snapshot_seconds * 30)))
        self.brawler_combat_profiles = load_toml_as_dict("cfg/brawler_combat_profiles.toml")
        
    @staticmethod
    def get_enemy_pos(enemy):
        return (enemy[0] + enemy[2]) / 2, (enemy[1] + enemy[3]) / 2

    @staticmethod
    def get_player_pos(player_data):
        return (player_data[0] + player_data[2]) / 2, (player_data[1] + player_data[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        return math.hypot(enemy_coords[0] - player_coords[0], enemy_coords[1] - player_coords[1])

    def closest_enemy_distance(self, player_pos, enemy_data):
        if not player_pos or not enemy_data:
            return None
        distances = []
        for enemy in enemy_data:
            try:
                distances.append(self.get_distance(self.get_enemy_pos(enemy), player_pos))
            except (TypeError, IndexError):
                continue
        return min(distances) if distances else None

    def close_attack_threat_threshold(self, attack_range):
        range_part = float(attack_range or 0) * 0.60
        return max(
            float(getattr(self, "dangerous_close_range", 0) or 0),
            float(getattr(self, "auto_aim_close_tap_range", 0) or 0),
            range_part,
        )

    def is_close_attack_threat(self, enemy_distance, attack_range):
        if not getattr(self, "close_range_attack_override", True):
            return False
        if enemy_distance is None:
            return False
        return float(enemy_distance) <= self.close_attack_threat_threshold(attack_range)

    @staticmethod
    def box_iou(box_a, box_b):
        if not box_a or not box_b:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
        ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
        ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
        bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
        by1, by2 = min(by1, by2), max(by1, by2)
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union

    def friendly_overlap_reason(self, box, excluded_boxes):
        if not getattr(self, "friendly_fire_guard_enabled", True):
            return ""
        center = self.get_enemy_pos(box)
        best_reason = ""
        best_score = 0.0
        for excluded in excluded_boxes or []:
            if isinstance(excluded, dict):
                excluded_box = excluded.get("box")
                if excluded_box is None:
                    excluded_box = excluded.get("bbox")
                kind = excluded.get("kind", "friendly")
            else:
                excluded_box = excluded
                kind = "friendly"
            if excluded_box is None:
                continue
            iou = self.box_iou(box, excluded_box)
            excluded_center = self.get_enemy_pos(excluded_box)
            center_dist = self.get_distance(center, excluded_center)
            if iou >= getattr(self, "friendly_fire_iou_threshold", 0.18) and iou > best_score:
                best_score = iou
                best_reason = f"{kind}_iou:{iou:.2f}"
            center_threshold = getattr(self, "friendly_fire_center_distance_px", 70)
            if kind == "player":
                width = abs(float(excluded_box[2]) - float(excluded_box[0]))
                height = abs(float(excluded_box[3]) - float(excluded_box[1]))
                center_threshold = min(center_threshold, max(12.0, min(width, height) * 0.25))
            if center_dist <= center_threshold:
                score = 1.0 - center_dist / max(1.0, center_threshold)
                if score > best_score:
                    best_score = score
                    best_reason = f"{kind}_center:{center_dist:.1f}"
        return best_reason

    def build_attack_excluded_boxes(self, player_data=None, teammate_data=None):
        boxes = []
        if teammate_data:
            boxes.extend({"box": teammate, "kind": "teammate"} for teammate in teammate_data)
        if player_data:
            boxes.append({"box": player_data, "kind": "player"})
        return boxes

    def sanitize_enemy_targets(self, enemy_data, excluded_boxes):
        if not enemy_data:
            return [], []
        if not getattr(self, "friendly_fire_guard_enabled", True):
            return list(enemy_data or []), []
        sanitized = []
        excluded = []
        for enemy in enemy_data or []:
            reason = self.friendly_overlap_reason(enemy, excluded_boxes)
            if reason:
                excluded.append({"bbox": tuple(enemy), "reason": reason})
                continue
            sanitized.append(enemy)
        return sanitized, excluded

    @staticmethod
    def is_there_enemy(enemy_data):
        if not enemy_data:
            return False
        return True

    @staticmethod
    def get_horizontal_move_key(direction_x, opposite=False):
        if opposite:
            return "A" if direction_x > 0 else "D"
        return "D" if direction_x > 0 else "A"

    @staticmethod
    def get_vertical_move_key(direction_y, opposite=False):
        if opposite:
            return "W" if direction_y > 0 else "S"
        return "S" if direction_y > 0 else "W"

    def attack(self, touch_up=True, touch_down=True, cooldown_multiplier=1.0, force_release_movement=False):
        effective_cooldown = max(0.0, self.attack_cooldown * float(cooldown_multiplier))
        if touch_up and touch_down and effective_cooldown > 0:
            current_time = time.time()
            if current_time - self.last_attack_time < effective_cooldown:
                return False
            self.last_attack_time = current_time
        self.window_controller.press_key("M", touch_up=touch_up, touch_down=touch_down, force_release_movement=force_release_movement)
        return True

    def aimed_attack(self, angle_degrees):
        if not self.aimed_attacks_enabled:
            return self.attack()
        if self.attack_cooldown > 0:
            current_time = time.time()
            if current_time - self.last_attack_time < self.attack_cooldown:
                return False
            self.last_attack_time = current_time
        if hasattr(self.window_controller, "aim_attack_angle"):
            self.window_controller.aim_attack_angle(angle_degrees)
            return True
        return self.attack()

    def _aimlog(self, *args):
        if getattr(self, "auto_aim_debug", False) or getattr(self, "attack_decision_debug", False) or visual_debug:
            print("[AIM]", *args)

    def choose_attack_decision(
            self,
            brawler,
            player_pos,
            enemy_data,
            walls,
            attack_range=None,
            current_time=None,
            excluded_boxes=None):
        if current_time is None:
            current_time = time.time()
        if attack_range is None:
            _, attack_range, _ = self.get_brawler_range(brawler)
        can_ignore_walls = self.can_attack_through_walls(brawler, "attack", self.brawlers_info)
        aim_line_angle = detect_aim_line_angle(getattr(self, "current_frame", None), player_pos)
        close_tap_range = getattr(self, "auto_aim_close_tap_range", 0)
        close_tap_range = close_tap_range if close_tap_range > 0 else None
        close_los_override_range = getattr(self, "auto_aim_close_los_override_range", 0)
        close_los_override_range = close_los_override_range if close_los_override_range > 0 else None
        lead_shots_enabled = getattr(self, "lead_shots_enabled", True)
        profile = self.get_combat_profile(brawler)
        preferred_target_bbox = None
        if getattr(self, "preferred_target_lock_for_aim", False):
            locked = getattr(getattr(self, "_target_memory", None), "locked_target", None)
            if locked and not getattr(locked, "stale", False):
                preferred_target_bbox = locked.bbox
        return choose_auto_aim(
            player_pos=player_pos,
            enemy_data=enemy_data,
            walls=walls,
            attack_range=attack_range,
            can_ignore_walls=can_ignore_walls,
            walls_block_line_of_sight=self.walls_block_line_of_sight,
            track_enemy_velocity=self.track_enemy_velocity if lead_shots_enabled else (lambda *_args: (0.0, 0.0)),
            velocity_confidence=(lambda: getattr(self, "enemy_velocity_confidence", 0.0)) if lead_shots_enabled else 0.0,
            projectile_speed=float(profile.get("projectile_speed_px_s", self.projectile_speed_px_s)),
            current_time=current_time,
            aim_line_angle=aim_line_angle,
            min_confidence=float(profile.get("min_confidence", getattr(self, "auto_aim_min_confidence", 0.62))),
            close_tap_range=float(profile.get("close_tap_range", close_tap_range)) if profile.get("close_tap_range", close_tap_range) is not None else None,
            close_range_override=getattr(self, "close_range_attack_override", True),
            dangerous_close_range=getattr(self, "dangerous_close_range", None),
            close_los_override_range=close_los_override_range,
            preferred_target_bbox=preferred_target_bbox,
            excluded_boxes=excluded_boxes,
            friendly_iou_threshold=getattr(self, "friendly_fire_iou_threshold", 0.18),
            friendly_center_distance_px=getattr(self, "friendly_fire_center_distance_px", 70),
            close_attack_requires_clear_hit_point=getattr(self, "close_attack_requires_clear_hit_point", True),
            attack_wall_guard_enabled=getattr(self, "attack_wall_guard_enabled", True),
        )

    def is_attack_spam_active(self, brawler, decision):
        if not getattr(self, "attack_spam_enabled", True):
            return False
        if not decision.should_fire or not decision.in_range:
            return False
        if not getattr(self, "attack_spam_requires_los", True):
            return True
        if decision.line_of_sight:
            return True
        return self.can_attack_through_walls(brawler, "attack", self.brawlers_info)

    def attack_cooldown_multiplier_for_decision(self, brawler, decision):
        multiplier = 1.0
        if self.is_attack_spam_active(brawler, decision):
            multiplier = min(
                multiplier,
                max(0.02, min(1.0, getattr(self, "attack_spam_cooldown_multiplier", 0.12))),
            )
        if decision.close_threat or decision.use_tap:
            multiplier = min(
                multiplier,
                max(0.25, min(1.0, getattr(self, "close_range_attack_cooldown_multiplier", 0.55))),
            )
        return multiplier

    def log_attack_decision(
            self,
            decision,
            attack_range,
            aim_frequency_hz=0.0,
            input_mode=None,
            input_busy=False,
            denied_by=None,
            attack_spam_active=False,
            effective_cooldown=None,
            force_release_movement=False,
            aim_duration=None):
        target_s = tuple(map(int, decision.target)) if decision.target else None
        predicted_s = tuple(map(int, decision.predicted)) if decision.predicted else None
        angle_s = None if decision.aim_angle is None else round(decision.aim_angle, 1)
        dist_s = None if decision.distance is None else int(decision.distance)
        closest_s = None if decision.closest_enemy_distance is None else int(decision.closest_enemy_distance)
        cooldown_s = self.attack_cooldown if effective_cooldown is None else effective_cooldown
        duration_s = getattr(self, "aim_attack_duration", 0.04) if aim_duration is None else aim_duration
        parallel_attack = bool(getattr(self.window_controller, "enable_parallel_movement_attack", True))
        movement_active = bool(getattr(self.window_controller, "are_we_moving", False))
        if input_mode is None:
            input_mode = "tap" if decision.use_tap else "aimed_drag"
            if not getattr(self, "aimed_attacks_enabled", False) or not hasattr(self.window_controller, "aim_attack_angle"):
                input_mode = "tap_fallback"
        reason = denied_by or decision.denied_by or decision.reason
        self._aimlog(
            "attack_decision "
            f"attack_allowed={decision.should_fire and not denied_by} "
            f"attack_denied_reason={reason if (denied_by or not decision.should_fire) else 'none'} "
            f"visible_enemy_count={decision.visible_enemy_count} "
            f"target={target_s} target_bbox={decision.target_bbox} selected_target={target_s} "
            f"closest_enemy_distance={closest_s} selected_distance={dist_s} attack_range={int(attack_range)} "
            f"in_range={decision.in_range} line_of_sight={decision.los_status} "
            f"confidence={decision.confidence:.2f} confidence_threshold={decision.threshold:.2f} "
            f"close_threat={decision.close_threat} close_range_override={decision.close_range_override} "
            f"attack_spam_active={attack_spam_active} effective_cooldown_ms={int(max(0.0, cooldown_s) * 1000)} "
            f"parallel_movement_attack={parallel_attack} force_release_movement={force_release_movement} "
            f"aim_duration_ms={int(max(0.0, duration_s) * 1000)} movement_active={movement_active} "
            f"cooldown_remaining_ms={int(decision.cooldown_remaining_ms or 0)} "
            f"input_busy={input_busy} aim_frequency_hz={aim_frequency_hz:.2f} input_mode={input_mode} tap={decision.use_tap} "
            f"predicted_point={predicted_s} angle={angle_s} "
            f"fallback_reason={decision.aim_fallback_reason or 'none'} reason={decision.reason}"
        )

    def _combatlog(self, *args):
        if getattr(self, "combat_brain_debug", False) or visual_debug:
            print("[COMBAT]", *args)

    def _abilitylog(self, *args):
        if getattr(self, "ability_brain_debug", False) or visual_debug:
            print("[ABILITY]", *args)

    def record_combat_decision(self, event):
        if not getattr(self, "combat_snapshot_enabled", False):
            return
        history = getattr(self, "_combat_decision_history", None)
        if history is None:
            self._combat_decision_history = deque(maxlen=240)
            history = self._combat_decision_history
        payload = dict(event or {})
        payload["time"] = time.time()
        history.append(payload)

    def save_combat_snapshot(self, reason, extra=None, brawler=None):
        if not getattr(self, "combat_snapshot_enabled", False):
            return None
        now = time.time()
        history = list(getattr(self, "_combat_decision_history", []))
        cutoff = now - float(getattr(self, "combat_snapshot_seconds", 8))
        recent = [item for item in history if item.get("time", 0.0) >= cutoff]
        snapshot = {
            "time": now,
            "reason": reason,
            "brawler": brawler or getattr(self, "current_brawler", None),
            "recent_decisions": recent,
            "extra": extra or {},
        }
        try:
            directory = getattr(self, "combat_snapshot_dir", "debug_frames/combat")
            os.makedirs(directory, exist_ok=True)
            filename = f"combat_{reason}_{int(now * 1000)}.json".replace(os.sep, "_")
            path = os.path.join(directory, filename)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False, indent=2)
            self._combatlog(f"snapshot_saved reason={reason} path={path}")
            return path
        except Exception as exc:
            self._combatlog(f"snapshot_failed reason={reason} error={exc}")
            return None

    def current_tactical_adaptation(self, current_time=None):
        current_time = time.time() if current_time is None else current_time
        state = getattr(self, "_tactical_adaptation", None) or {}
        if current_time >= state.get("aggression_penalty_until", 0.0):
            state["aggression_penalty"] = 0.0
            state["pressure_damage_events"] = 0
        return {
            "aggression_penalty": float(state.get("aggression_penalty", 0.0)),
            "fire_threshold_delta": float(state.get("fire_threshold_delta", 0.0)),
        }

    def update_tactical_adaptation(self, intent, flicker_active=False, attack_denied_by=None, current_time=None):
        if not getattr(self, "adaptive_aggression_enabled", False) or not intent:
            return
        current_time = time.time() if current_time is None else current_time
        state = getattr(self, "_tactical_adaptation", None)
        if state is None:
            self._tactical_adaptation = {}
            state = self._tactical_adaptation
        last_objective = state.get("last_objective")
        last_time = state.get("last_objective_time", 0.0)
        if flicker_active and last_objective in {"pressure", "finish_kill"} and current_time - last_time <= self.damage_penalty_window_seconds:
            state["pressure_damage_events"] = int(state.get("pressure_damage_events", 0)) + 1
            if state["pressure_damage_events"] >= 2:
                state["aggression_penalty"] = 0.18
                state["aggression_penalty_until"] = current_time + self.damage_penalty_window_seconds
                self._combatlog("adaptive_aggression penalty=0.18 reason=damage_after_pressure")
        plan = getattr(intent, "tactical_plan", None)
        if plan and plan.kill_confirm_score >= getattr(self, "kill_confirm_score_threshold", 0.68) and attack_denied_by:
            state["fire_threshold_delta"] = max(
                -0.12,
                float(state.get("fire_threshold_delta", 0.0)) - getattr(self, "missed_kill_window_decay", 0.04),
            )
        elif plan and plan.fire_window and not attack_denied_by:
            state["fire_threshold_delta"] = min(0.0, float(state.get("fire_threshold_delta", 0.0)) + 0.02)
        if plan:
            state["last_objective"] = plan.objective
            state["last_objective_time"] = current_time

    def reset_tactical_adaptation(self):
        if hasattr(self, "_tactical_adaptation"):
            self._tactical_adaptation.update({
                "pressure_damage_events": 0,
                "aggression_penalty_until": 0.0,
                "aggression_penalty": 0.0,
                "fire_threshold_delta": 0.0,
                "last_objective": None,
                "last_objective_time": 0.0,
            })

    def get_combat_profile(self, brawler):
        profiles = getattr(self, "brawler_combat_profiles", {}) or {}
        profile = profiles.get(str(brawler or "").strip().lower(), {})
        return profile if isinstance(profile, dict) else {}

    def target_score_from_attack_decision(self, decision, player_pos):
        if not decision or not decision.target_bbox:
            return None
        center = self.get_enemy_pos(decision.target_bbox)
        return TargetScore(
            bbox=tuple(decision.target_bbox),
            center=center,
            distance=decision.distance,
            score=max(0.0, min(1.0, float(decision.confidence or 0.0))),
            line_of_sight=bool(decision.line_of_sight),
            in_attack_range=bool(decision.in_range),
            close_threat=bool(decision.close_threat or decision.use_tap),
            stale=False,
            reasons=[decision.reason] if decision.reason else [],
        )

    def build_health_state(self, health_ratio, heal_active, flicker_active=False, flicker_confidence=0.0):
        confidence = 0.0
        source = "unknown"
        if health_ratio is not None:
            confidence = 0.85
            source = "color_bar"
        if flicker_active:
            confidence = max(confidence, min(1.0, 0.45 + float(flicker_confidence or 0.0) * 0.45))
            source = "color_bar+flicker" if health_ratio is not None else "flicker"
        return HealthState(
            ratio=health_ratio,
            confidence=confidence,
            recent_damage=bool(flicker_active and flicker_confidence >= 0.50),
            heal_active=bool(heal_active),
            source=source,
        )

    def choose_combat_target_score(self, player_pos, enemy_data, walls, brawler, safe_range, attack_range, attack_decision=None):
        if attack_decision and attack_decision.target_bbox:
            return self.target_score_from_attack_decision(attack_decision, player_pos)
        try:
            attack_ignores_walls = self.can_attack_through_walls(brawler, "attack", self.brawlers_info)
        except (KeyError, TypeError):
            attack_ignores_walls = False
        memory = getattr(self, "_target_memory", None)
        if memory is not None and getattr(self, "combat_brain_enabled", False):
            return memory.choose(
                now=time.time(),
                memory_seconds=getattr(self, "target_memory_seconds", 0.75),
                switch_margin=getattr(self, "target_switch_margin", 0.18),
                player_pos=player_pos,
                enemy_data=enemy_data or [],
                safe_range=safe_range,
                attack_range=attack_range,
                walls=walls or [],
                can_attack_through_walls=attack_ignores_walls,
                walls_block_line_of_sight=self.walls_block_line_of_sight,
                dangerous_close_range=self.close_attack_threat_threshold(attack_range),
            )
        return choose_combat_target(
            player_pos=player_pos,
            enemy_data=enemy_data or [],
            safe_range=safe_range,
            attack_range=attack_range,
            walls=walls or [],
            can_attack_through_walls=attack_ignores_walls,
            walls_block_line_of_sight=self.walls_block_line_of_sight,
            dangerous_close_range=self.close_attack_threat_threshold(attack_range),
        )

    def is_charge_path_safe(self, player_pos, target_score, walls):
        if not target_score or target_score.center is None:
            return False
        angle = self.angle_from_direction(target_score.center[0] - player_pos[0], target_score.center[1] - player_pos[1])
        if self.is_path_blocked_angle(player_pos, angle, walls, distance=max(60, min(target_score.distance or 60, self.TILE_SIZE * 2))):
            return False
        return not self.angle_points_into_fog(self.current_frame, player_pos, angle)

    def choose_combat_ability_plan(
            self,
            brawler,
            brawler_info,
            player_pos,
            enemy_data,
            teammate_data,
            walls,
            safe_range,
            attack_range,
            target_score,
            health_state,
            intent_mode,
            fog_flee_angle=None,
            projectile_incoming=False,
            tactical_plan=None):
        if not getattr(self, "ability_brain_enabled", False):
            return None
        enemy_count_in_range = sum(
            1 for enemy in (enemy_data or [])
            if self.get_distance(self.get_enemy_pos(enemy), player_pos) <= attack_range
        )
        closest_teammate, teammate_distance = self.get_closest_teammate(
            (player_pos[0], player_pos[1], player_pos[0], player_pos[1]),
            teammate_data,
        )
        teammate_near = closest_teammate is not None and teammate_distance <= getattr(self, "teammate_combat_regroup_distance", 650)
        threat = build_combat_threat_model(
            target=target_score,
            enemy_count_in_range=enemy_count_in_range,
            health=health_state,
            fog_danger=fog_flee_angle is not None or intent_mode == "escape_fog",
            projectile_incoming=bool(projectile_incoming or intent_mode == "dodge_projectile"),
            wall_trap=intent_mode in {"wall_escape", "unstuck"},
            teammate_near=teammate_near,
            safe_range=safe_range,
        )
        if intent_mode and intent_mode != threat.mode:
            threat.mode = intent_mode
        super_hittable = bool(
            target_score and target_score.center is not None and self.is_enemy_hittable(player_pos, target_score.center, walls, "super")
        )
        attack_hittable = bool(target_score and target_score.line_of_sight)
        profile = self.get_combat_profile(brawler)
        plan = choose_ability_plan(
            target=target_score,
            threat=threat,
            health=health_state,
            super_type=profile.get("super_mode", brawler_info.get("super_type", "damage")),
            super_ready=bool(getattr(self, "is_super_ready", False)),
            gadget_ready=bool(getattr(self, "is_gadget_ready", False)),
            hypercharge_ready=bool(getattr(self, "is_hypercharge_ready", False)),
            gadget_enabled=bool(getattr(self, "should_use_gadget", True)) and profile.get("gadget_mode", "generic") != "disabled",
            holding_attack=self.time_since_holding_attack is not None,
            super_hittable=super_hittable,
            attack_hittable=attack_hittable,
            enemy_count_in_range=enemy_count_in_range,
            teammate_near=teammate_near,
            super_cooldown_remaining_ms=self.super_cooldown_remaining_ms(),
            gadget_cooldown_remaining_ms=0,
            super_min_value_score=float(profile.get("super_min_value_score", getattr(self, "super_min_value_score", 0.55))),
            gadget_min_value_score=float(profile.get("gadget_min_value_score", getattr(self, "gadget_min_value_score", 0.50))),
            hypercharge_min_value_score=float(profile.get("hypercharge_min_value_score", getattr(self, "hypercharge_min_value_score", 0.70))),
            panic_super_range=float(profile.get("panic_super_range", getattr(self, "panic_super_range", 180))),
            charge_path_safe=self.is_charge_path_safe(player_pos, target_score, walls),
            tactical_plan=tactical_plan,
            gadget_mode=profile.get("gadget_mode", "generic"),
            finisher_super=bool(profile.get("finisher_super", False)),
        )
        self.log_ability_plan(plan, threat, target_score)
        return plan

    def log_ability_plan(self, plan, threat, target_score):
        if not plan or not (getattr(self, "ability_brain_debug", False) or visual_debug):
            return
        now = time.time()
        interval = max(0.0, getattr(self, "ability_retry_log_interval_ms", 500) / 1000.0)
        if now - getattr(self, "_last_ability_plan_log", 0.0) < interval and not (
                plan.use_super or plan.use_gadget or plan.use_hypercharge):
            return
        self._last_ability_plan_log = now
        distance_s = None if not target_score or target_score.distance is None else int(target_score.distance)
        self._abilitylog(
            "ability_plan "
            f"use_hypercharge={plan.use_hypercharge} reason_hypercharge={plan.hypercharge_reason} "
            f"value_hypercharge={plan.hypercharge_value:.2f} "
            f"use_super={plan.use_super} reason_super={plan.super_reason} value_super={plan.super_value:.2f} "
            f"use_gadget={plan.use_gadget} reason_gadget={plan.gadget_reason} value_gadget={plan.gadget_value:.2f} "
            f"threat_mode={threat.mode} threat_score={threat.score:.2f} target_distance={distance_s} "
            f"denies={','.join(plan.denies) if plan.denies else 'none'}"
        )

    def execute_ability_plan(self, plan):
        if not plan:
            return False
        used = False
        if plan.use_hypercharge and getattr(self, "is_hypercharge_ready", False):
            if self.use_hypercharge():
                self.time_since_hypercharge_checked = time.time()
                self.clear_ability_ready("hypercharge")
                used = True
        if plan.use_gadget and getattr(self, "is_gadget_ready", False):
            if self.use_gadget():
                self.time_since_gadget_checked = time.time()
                self.clear_ability_ready("gadget")
                used = True
        if plan.use_super and getattr(self, "is_super_ready", False):
            self.release_held_attack_for_super()
            if self.use_super():
                self.time_since_super_checked = time.time()
                self.clear_ability_ready("super")
                used = True
        return used

    def auto_aim_attack(self, brawler, player_pos, enemy_data, walls, attack_range=None, decision=None, excluded_boxes=None):
        now = time.time()
        elapsed = now - getattr(self, "_last_aim_attempt_time", 0.0)
        aim_frequency_hz = 0.0 if elapsed <= 0 else min(99.0, 1.0 / elapsed)
        self._last_aim_attempt_time = now
        if attack_range is None:
            _, attack_range, _ = self.get_brawler_range(brawler)
        if decision is None:
            decision = self.choose_attack_decision(
                brawler,
                player_pos,
                enemy_data,
                walls,
                attack_range=attack_range,
                current_time=now,
                excluded_boxes=excluded_boxes,
            )
        elif getattr(self, "friendly_fire_guard_enabled", True) and decision.target_bbox:
            reason = self.friendly_overlap_reason(decision.target_bbox, excluded_boxes or [])
            if reason:
                decision.should_fire = False
                decision.denied_by = "friendly_excluded"
                decision.reason = "friendly_excluded"
                decision.los_status = reason
        input_mode = "tap" if decision.use_tap else "aimed_drag"
        if not getattr(self, "aimed_attacks_enabled", False) or not hasattr(self.window_controller, "aim_attack_angle"):
            input_mode = "tap_fallback"
        attack_spam_active = self.is_attack_spam_active(brawler, decision)
        cooldown_multiplier = self.attack_cooldown_multiplier_for_decision(brawler, decision)
        effective_cooldown = max(0.0, self.attack_cooldown * cooldown_multiplier)
        parallel_attack = bool(
            getattr(self.window_controller, "enable_parallel_movement_attack", True)
            and getattr(self.window_controller, "input_backend_supports_parallel_drag", True)
        )
        force_release_movement = bool(
            not parallel_attack
            or (
                decision.close_threat
                and getattr(self, "force_release_movement_on_close_threat", False)
            )
        )
        self.log_attack_decision(
            decision,
            attack_range,
            aim_frequency_hz=aim_frequency_hz,
            input_mode=input_mode,
            attack_spam_active=attack_spam_active,
            effective_cooldown=effective_cooldown,
            force_release_movement=force_release_movement,
            aim_duration=getattr(self, "aim_attack_duration", 0.04),
        )
        if not decision.should_fire:
            return False
        if (
            decision.use_tap
            or not getattr(self, "aimed_attacks_enabled", False)
            or not hasattr(self.window_controller, "aim_attack_angle")
        ):
            return self.attack(cooldown_multiplier=cooldown_multiplier, force_release_movement=force_release_movement)
        if effective_cooldown > 0:
            current_time = time.time()
            if current_time - self.last_attack_time < effective_cooldown:
                remaining = max(0.0, effective_cooldown - (current_time - self.last_attack_time))
                decision.cooldown_remaining_ms = int(remaining * 1000)
                self.log_attack_decision(
                    decision,
                    attack_range,
                    aim_frequency_hz=aim_frequency_hz,
                    input_mode=input_mode,
                    input_busy=True,
                    denied_by="attack_on_cooldown",
                    attack_spam_active=attack_spam_active,
                    effective_cooldown=effective_cooldown,
                    force_release_movement=force_release_movement,
                    aim_duration=getattr(self, "aim_attack_duration", 0.04),
                )
                return False
            self.last_attack_time = current_time
        self.window_controller.aim_attack_angle(
            decision.aim_angle,
            duration=getattr(self, "aim_attack_duration", 0.02),
            force_release_movement=force_release_movement,
        )
        return True

    def use_hypercharge(self):
        print("Using hypercharge")
        self.window_controller.press_key("H", delay=0.035)
        return True

    def use_gadget(self):
        if self.gadget_cooldown > 0:
            current_time = time.time()
            if current_time - self.last_gadget_time < self.gadget_cooldown:
                return False
            self.last_gadget_time = current_time
        print("Using gadget")
        self.window_controller.press_key("G", delay=0.035)
        return True

    def use_super(self, cooldown_multiplier=1.0):
        effective_cooldown = max(0.0, self.super_cooldown * float(cooldown_multiplier))
        if effective_cooldown > 0:
            current_time = time.time()
            if current_time - self.last_super_time < effective_cooldown:
                return False
            self.last_super_time = current_time
        print("Using super")
        self.window_controller.press_key("E", delay=0.035)
        return True

    def super_cooldown_remaining_ms(self, cooldown_multiplier=1.0):
        effective_cooldown = max(0.0, self.super_cooldown * float(cooldown_multiplier))
        if effective_cooldown <= 0:
            return 0
        elapsed = time.time() - getattr(self, "last_super_time", 0.0)
        return int(max(0.0, effective_cooldown - elapsed) * 1000)

    @staticmethod
    def should_use_super_on_enemy(brawler, super_type, enemy_distance, attack_range, super_range, enemy_hittable):
        utility_super = super_type in {"spawnable", "other", "other_target"}
        charge_super = super_type == "charge"
        near_range = max(super_range, attack_range * 0.75)
        near_range = min(near_range, attack_range)
        if enemy_hittable and enemy_distance <= min(super_range, near_range):
            return True
        if enemy_hittable and super_type == "damage" and enemy_distance <= near_range:
            return True
        if enemy_hittable and utility_super and enemy_distance <= near_range:
            return True
        if (
                charge_super
                and enemy_distance <= near_range
                and (enemy_hittable or brawler in {"stu", "surge"})
        ):
            return True
        return False

    @staticmethod
    def get_random_attack_key():
        random_movement = random.choice(["A", "W", "S", "D"])
        random_movement += random.choice(["A", "W", "S", "D"])
        return random_movement

    @staticmethod
    def angle_from_direction(dx: float, dy: float) -> float:
        """Return joystick angle in degrees from a direction vector.

        Uses screen coordinates: 0° = right, 90° = down, 180° = left, 270° = up.
        """
        return math.degrees(math.atan2(dy, dx)) % 360

    @staticmethod
    def angle_opposite(angle_degrees: float) -> float:
        """Return the opposite direction angle (retreat)."""
        return (angle_degrees + 180) % 360

    @staticmethod
    def reverse_movement(movement):
        # Create a translation table
        movement = movement.lower()
        translation_table = str.maketrans("wasd", "sdwa")
        return movement.translate(translation_table)

    @staticmethod
    def movement_to_vector(movement):
        dx = 0
        dy = 0
        movement = str(movement or "").lower()
        if "a" in movement:
            dx -= 1
        if "d" in movement:
            dx += 1
        if "w" in movement:
            dy -= 1
        if "s" in movement:
            dy += 1
        return dx, dy

    def unstuck_movement_if_needed(self, movement, current_time=None):
        if current_time is None:
            current_time = time.time()
        movement = movement.lower()
        if self.fix_movement_keys['toggled']:
            if current_time - self.fix_movement_keys['started_at'] > self.fix_movement_keys['duration']:
                self.fix_movement_keys['toggled'] = False
                vlog("unstuck: finished")
            else:
                vlog(f"unstuck: active → {self.fix_movement_keys['fixed']}")

            return self.fix_movement_keys['fixed']

        if "".join(self.keys_hold) != movement and movement[::-1] != "".join(self.keys_hold):
            self.time_since_different_movement = current_time

        # print(f"Last change: {self.time_since_different_movement}", f" self.hold: {self.keys_hold}",f" c movement: {movement}")
        if current_time - self.time_since_different_movement > self.fix_movement_keys["delay_to_trigger"]:
            reversed_movement = self.reverse_movement(movement)

            if reversed_movement == "s":
                reversed_movement = random.choice(['aw', 'dw'])
            elif reversed_movement == "w":
                reversed_movement = random.choice(['as', 'ds'])

            """
            If reverse movement is either "w" or "s" it means the bot is stuck
            going forward or backward. This happens when it doesn't detect a wall in front
            so to go around it it could either go to the left diagonal or right
            """

            self.fix_movement_keys['fixed'] = reversed_movement
            self.fix_movement_keys['toggled'] = True
            self.fix_movement_keys['started_at'] = current_time
            vlog(f"unstuck triggered: {movement} → {reversed_movement}")
            return reversed_movement

        return movement

    def _wslog(self, *args):
        """Dedicated logger for wall-stuck / escape — independent of vlog/visual_debug
        so the new unstuck machinery can be traced without dumping the full debug stream.
        """
        if self.wall_stuck_debug:
            print("[WS]", *args)

    def _wall_centers_filtered(self, walls, player_pos):
        """Return (N, 2) float array of wall centers, excluding walls whose
        center lies within wall_stuck_ignore_radius of the player (those
        flicker as the player overlaps them).
        """
        import numpy as np
        if not walls:
            return np.empty((0, 2), dtype=np.float32)
        centers = []
        px, py = player_pos
        r2 = self.wall_stuck_ignore_radius * self.wall_stuck_ignore_radius
        for box in walls:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            dx, dy = cx - px, cy - py
            if dx * dx + dy * dy >= r2:
                centers.append((cx, cy))
        return np.asarray(centers, dtype=np.float32) if centers else np.empty((0, 2), dtype=np.float32)

    def _avg_wall_shift(self, prev_centers, curr_centers):
        """Greedy nearest-neighbor match between two sets of wall centers.
        Returns mean pairwise distance (px). Returns None if either set is too
        small (can't form a reliable metric).
        """
        import numpy as np
        if prev_centers is None or len(prev_centers) < self.wall_stuck_min_walls:
            return None
        if len(curr_centers) < self.wall_stuck_min_walls:
            return None
        # For each prev center, find nearest curr center (O(N*M), fine for N~20)
        diffs = prev_centers[:, None, :] - curr_centers[None, :, :]
        d2 = (diffs * diffs).sum(axis=2)
        nearest = np.sqrt(d2.min(axis=1))
        return float(nearest.mean())

    def detect_wall_stuck(self, walls, player_pos, is_trying_to_move, current_time):
        """Wall-based stuck detector. Returns True if the walls around the
        player have been stationary longer than wall_stuck_timeout while the
        bot was issuing movement commands — meaning the bot is pressed against
        something and not actually moving.
        """
        if not self.wall_stuck_enabled or player_pos is None:
            return False
        state = self.wall_stuck_state
        if current_time - state["last_sample_time"] < self.wall_stuck_sample_interval:
            # Between samples: just honor the latest stationary flag
            if state["stationary_since"] is None or not is_trying_to_move:
                return False
            return (current_time - state["stationary_since"]) >= self.wall_stuck_timeout

        curr_centers = self._wall_centers_filtered(walls, player_pos)
        shift = self._avg_wall_shift(state["last_wall_centers"], curr_centers)
        state["last_wall_centers"] = curr_centers
        state["last_sample_time"] = current_time

        if shift is None:
            # Not enough walls to judge — treat as "unknown", don't advance timer
            state["stationary_since"] = None
            return False

        if shift < self.wall_stuck_shift_threshold:
            if state["stationary_since"] is None:
                state["stationary_since"] = current_time
            self._wslog(f"walls shift={shift:.2f}px, stationary for "
                        f"{current_time - state['stationary_since']:.2f}s "
                        f"(trying_to_move={is_trying_to_move})")
        else:
            if state["stationary_since"] is not None:
                self._wslog(f"walls moved again: shift={shift:.2f}px, resetting timer")
            state["stationary_since"] = None

        if state["stationary_since"] is None or not is_trying_to_move:
            return False
        return (current_time - state["stationary_since"]) >= self.wall_stuck_timeout

    def _reset_wall_stuck_state(self, current_time):
        """Clear the wall-stuck timer. Call after triggering an escape to
        avoid retriggering during/just after the maneuver.
        """
        self.wall_stuck_state["stationary_since"] = None
        self.wall_stuck_state["last_wall_centers"] = None
        self.wall_stuck_state["last_sample_time"] = current_time

    def start_semicircle_escape(self, angle, current_time):
        """Begin the retreat+arc escape maneuver. arc_side alternates globally
        between triggers.
        """
        side = self._next_arc_side
        self._next_arc_side = -side
        self.escape_state["phase"] = "retreat"
        self.escape_state["started_at"] = current_time
        self.escape_state["retreat_angle"] = self.angle_opposite(angle)
        self.escape_state["arc_side"] = side
        self._wslog(f"semicircle escape START: angle={angle:.1f}° "
                    f"retreat={self.escape_state['retreat_angle']:.1f}° "
                    f"side={'CCW' if side > 0 else 'CW'}")

    def semicircle_escape_step(self, current_time):
        """Return the current commanded angle for the active escape maneuver,
        or None if no maneuver is active / it just finished.
        """
        state = self.escape_state
        phase = state["phase"]
        if phase is None:
            return None
        elapsed = current_time - state["started_at"]

        if phase == "retreat":
            if elapsed < self.escape_retreat_duration:
                return state["retreat_angle"]
            # Transition: arc starts from retreat angle and sweeps arc_degrees
            state["phase"] = "arc"
            state["started_at"] = current_time
            self._wslog("semicircle escape: retreat done, starting arc")
            elapsed = 0.0
            phase = "arc"

        if phase == "arc":
            if elapsed >= self.escape_arc_duration:
                state["phase"] = None
                self._wslog("semicircle escape: finished")
                return None
            t = elapsed / self.escape_arc_duration  # 0..1
            sweep = self.escape_arc_degrees * t * state["arc_side"]
            return (state["retreat_angle"] + sweep) % 360

        return None


class Play(Movement):

    def __init__(self, main_info_model, tile_detector_model, window_controller):
        super().__init__(window_controller)

        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")

        self.Detect_main_info = Detect(main_info_model, classes=['enemy', 'teammate', 'player'])
        self.tile_detector_model_classes = bot_config.get("wall_model_classes", ["wall", "bush", "close_bush"])
        self.Detect_tile_detector = Detect(
            tile_detector_model,
            classes=self.tile_detector_model_classes
        )

        self.time_since_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.time_since_hypercharge_checked = time.time()
        self.time_since_super_checked = time.time()
        self.time_since_walls_checked = 0
        self.time_since_movement_change = time.time()
        self.time_since_player_last_found = time.time()
        self.last_jump_pad_data = []
        self.current_brawler = None
        self.is_hypercharge_ready = False
        self.is_gadget_ready = False
        self.is_super_ready = False
        self.ability_ready_memory_seconds = float(bot_config.get("ability_ready_memory_seconds", 1.25))
        self._hypercharge_ready_seen_at = 0.0
        self._gadget_ready_seen_at = 0.0
        self._super_ready_seen_at = 0.0
        self.brawlers_info = load_brawlers_info()
        self.brawler_ranges = None
        self.time_since_detections = {
            "player": time.time(),
            "enemy": time.time(),
        }
        self.time_since_last_proceeding = time.time()
        self.time_since_last_no_detection_q = time.time()

        self.last_movement = None
        self.last_movement_time = time.time()
        self.locked_teammate = None
        self.locked_teammate_distance = float('inf')
        self.teammate_hysteresis = 0.75  # Switch only if another teammate is dramatically closer
        self.teammate_lock_max_jump = float(bot_config.get("teammate_lock_max_jump", 320))
        self.teammate_lock_lost_since = 0.0
        self.trio_grouping_enabled = str(bot_config.get("trio_grouping_enabled", "yes")).lower() in ("yes", "true", "1")
        self.showdown_playstyle_mode = str(bot_config.get("showdown_playstyle_mode", "follow")).strip().lower()
        self.teammate_follow_min_distance = float(bot_config.get("teammate_follow_min_distance", 180))
        self.teammate_follow_max_distance = float(bot_config.get("teammate_follow_max_distance", 520))
        self.teammate_follow_step_distance = float(bot_config.get("teammate_follow_step_distance", 8))
        self.teammate_combat_regroup_distance = float(bot_config.get("teammate_combat_regroup_distance", 650))
        self.teammate_combat_bias = float(bot_config.get("teammate_combat_bias", 0.75))
        self.teammate_follow_force_direct = str(bot_config.get("teammate_follow_force_direct", "yes")).lower() in ("yes", "true", "1")
        self.teammate_marker_follow_enabled = str(bot_config.get("teammate_marker_follow_enabled", "yes")).lower() in ("yes", "true", "1")
        self.teammate_marker_edge_margin = float(bot_config.get("teammate_marker_edge_margin", 0.28))
        self.wall_history = []
        self.wall_history_length = int(bot_config.get("wall_history_length", 3))
        self.scene_data = []
        self.should_detect_walls = bot_config.get("gamemode", "showdown") == "showdown"
        self.is_showdown = bot_config.get("gamemode", "showdown") == "showdown"
        self.minimum_movement_delay = bot_config.get("minimum_movement_delay", 0.1)
        self.no_detection_proceed_delay = time_config.get("no_detection_proceed", 8.5)
        self.no_detection_q_press_interval = float(time_config.get("no_detection_q_press_interval", 15.0))
        self.gadget_pixels_minimum = bot_config.get("gadget_pixels_minimum", 1100.0)
        self.hypercharge_pixels_minimum = bot_config.get("hypercharge_pixels_minimum", 1800.0)
        self.super_pixels_minimum = bot_config.get("super_pixels_minimum", 1800.0)
        self.wall_detection_confidence = bot_config.get("wall_detection_confidence", 0.9)
        self.entity_detection_confidence = bot_config.get("entity_detection_confidence", 0.6)
        self.entity_detection_retry_confidence = float(
            bot_config.get("entity_detection_retry_confidence", max(0.35, self.entity_detection_confidence - 0.20))
        )
        self.entity_retry_when_enemy_missing = str(bot_config.get("entity_retry_when_enemy_missing", "yes")).lower() in ("yes", "true", "1")
        self.entity_marker_min_ratio = float(bot_config.get("entity_marker_min_ratio", 0.012))
        self.entity_marker_min_pixels = int(bot_config.get("entity_marker_min_pixels", 12))
        self.entity_marker_below_box_ratio = float(bot_config.get("entity_marker_below_box_ratio", 0.22))
        self.entity_marker_blue_min_ratio = float(bot_config.get("entity_marker_blue_min_ratio", 0.012))
        self.entity_marker_enemy_min_ratio = float(bot_config.get("entity_marker_enemy_min_ratio", 0.012))
        self.entity_marker_decision_margin = float(bot_config.get("entity_marker_decision_margin", 1.25))
        self.player_center_bias_radius = float(bot_config.get("player_center_bias_radius", 420))
        self.player_green_pixel_weight = float(bot_config.get("player_green_pixel_weight", 0.03))
        self.player_red_pixel_penalty = float(bot_config.get("player_red_pixel_penalty", 0.05))
        self.time_since_holding_attack = None
        self.seconds_to_hold_attack_after_reaching_max = load_toml_as_dict("cfg/bot_config.toml")["seconds_to_hold_attack_after_reaching_max"]
        self.current_frame = None
        general_config = load_toml_as_dict("cfg/general_config.toml")
        crop_area = _pixel_counter_crop_area()
        self.super_crop_area = crop_area["super"]
        self.gadget_crop_area = crop_area["gadget"]
        self.hypercharge_crop_area = crop_area["hypercharge"]
        global debug, visual_debug
        debug = str(general_config.get("super_debug", "no")).lower() in ("yes", "true", "1")
        visual_debug = str(general_config.get("visual_debug", "no")).lower() in ("yes", "true", "1")
        self.visual_debug_scale = max(0.25, min(1.0, float(general_config.get("visual_debug_scale", 0.6))))
        self.visual_debug_max_fps = max(1.0, float(general_config.get("visual_debug_max_fps", 30)))
        self.visual_debug_max_boxes = max(20, int(general_config.get("visual_debug_max_boxes", 120)))
        self._visual_debug_next_frame_at = 0.0
        self._visual_debug_next_enqueue_at = 0.0
        self._visual_debug_lock = threading.Lock()
        self._visual_debug_payload = None
        self._visual_debug_thread = None
        self._visual_debug_stop = False
        self.capture_bad_vision_frames = str(general_config.get("capture_bad_vision_frames", "no")).lower() in ("yes", "true", "1")
        self.bad_vision_capture_dir = general_config.get("bad_vision_capture_dir", "debug_frames/vision")
        self.bad_vision_capture_interval = float(general_config.get("bad_vision_capture_interval", 2.0))
        self.bad_vision_capture_max = int(general_config.get("bad_vision_capture_max", 500))
        self._bad_vision_last_capture = {}
        self._bad_vision_capture_count = 0
        # Fog color (poison gas in showdown) — sampled from images/fog_sample.png.
        # Narrow range because the fog fully overlays whatever is under it.
        self.fog_hsv_low = (50, 95, 215)
        self.fog_hsv_high = (60, 125, 245)
        # Fog proximity override: movement flees fog when a real fog front is
        # within this distance. Attack logic is untouched.
        self.fog_flee_distance = 130
        # Confidence filters to avoid reacting to stray pixels:
        #   - morph opening kernel removes speckle noise
        #   - only connected fog blobs ≥ this many pixels are trusted
        #   - need at least this many trusted fog pixels inside the flee
        #     radius before the override kicks in
        self.fog_min_blob_pixels = 300
        self.fog_min_pixels_in_radius = 50
        # Run the fog-threat check once every N calls to get_showdown_movement.
        # Between checks the previous decision is reused.
        self.fog_check_every_n_frames = 3
        self._fog_check_counter = 0
        self._fog_threat_cached = None
        self._fog_direction_escape_cached = None
        # Per-frame cache of the trusted fog mask, keyed by id(frame).
        # Cache covers one pipeline run so the mask is not rebuilt when both
        # detect_fog_threat and detect_fog_direction are called on the same frame.
        self._fog_mask_cache_frame_id = None
        self._fog_mask_cache_value = None
        self._fog_mask_cache_origin = None
        self._entity_marker_cache_frame_id = None
        self._entity_marker_score_cache = {}
        self._perf_entity_marker_scores = 0
        self._perf_entity_marker_cache_hits = 0
        self._perf_entity_retry_count = 0
        self.playstyle_name = str(bot_config.get("current_playstyle", "")).strip()
        self.playstyle_meta = {}
        self.playstyle_code = None
        self._playstyle_error_reported = False
        self.load_playstyle()

    def load_playstyle(self):
        if not self.playstyle_name:
            return
        safe_name = os.path.basename(self.playstyle_name)
        path = os.path.join("playstyles", safe_name)
        if not os.path.exists(path):
            print(f"Playstyle '{safe_name}' was not found. Falling back to built-in logic.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                try:
                    self.playstyle_meta = json.loads(first_line) if first_line.startswith("{") else {}
                    source = f.read()
                except json.JSONDecodeError:
                    self.playstyle_meta = {}
                    source = first_line + "\n" + f.read()
            self.playstyle_code = compile(source, path, "exec")
            print(f"Loaded playstyle: {safe_name}")
        except Exception as e:
            print(f"Could not load playstyle '{safe_name}': {e}. Falling back to built-in logic.")
            self.playstyle_code = None

    def run_playstyle(self, player_data, enemy_data, walls, brawler):
        if self.playstyle_code is None:
            return None

        persistent_data = {
            "time_since_holding_attack": self.time_since_holding_attack,
        }

        def use_hypercharge_wrapper():
            if self.use_hypercharge():
                self.time_since_hypercharge_checked = time.time()
                self.clear_ability_ready("hypercharge")

        def use_gadget_wrapper():
            if self.should_use_gadget_on_enemy(brawler, player_data, enemy_data, walls):
                if self.use_gadget():
                    self.time_since_gadget_checked = time.time()
                    self.clear_ability_ready("gadget")

        def use_super_wrapper():
            if self.use_super():
                self.time_since_super_checked = time.time()
                self.clear_ability_ready("super")

        env = {
            "__builtins__": {
                "abs": abs,
                "bool": bool,
                "float": float,
                "int": int,
                "len": len,
                "max": max,
                "min": min,
                "print": print,
                "range": range,
                "str": str,
                "ValueError": ValueError,
            },
            "time": time,
            "random": random,
            "debug": debug,
            "brawler": brawler,
            "brawlers_info": self.brawlers_info,
            "player_data": player_data,
            "enemy_data": enemy_data,
            "teammate_data": getattr(self, "last_playstyle_teammate_data", None),
            "walls": walls,
            "game_mode": self.game_mode,
            "persistent_data": persistent_data,
            "seconds_to_hold_attack_after_reaching_max": self.seconds_to_hold_attack_after_reaching_max,
            "is_hypercharge_ready": self.is_hypercharge_ready,
            "is_gadget_ready": self.should_use_gadget and self.is_gadget_ready,
            "is_super_ready": self.is_super_ready,
            "movement": None,
            "attack": self.attack,
            "auto_aim_attack": lambda attack_range=None: self.auto_aim_attack(
                brawler,
                self.get_player_pos(player_data),
                enemy_data,
                walls,
                attack_range=attack_range,
            ),
            "should_attack_enemy": lambda attack_range=None: self.choose_attack_decision(
                brawler,
                self.get_player_pos(player_data),
                enemy_data,
                walls,
                attack_range=attack_range,
                current_time=time.time(),
            ).should_fire,
            "use_hypercharge": use_hypercharge_wrapper,
            "use_gadget": use_gadget_wrapper,
            "use_super": use_super_wrapper,
            "should_use_super_on_enemy": self.should_use_super_on_enemy,
            "must_brawler_hold_attack": self.must_brawler_hold_attack,
            "get_brawler_range": self.get_brawler_range,
            "get_player_pos": self.get_player_pos,
            "get_entity_pos": self.get_entity_pos,
            "is_there_enemy": self.is_there_enemy,
            "is_there_poison_gas": self.is_there_poison_gas,
            "no_enemy_movement": self.no_enemy_movement,
            "find_closest_enemy": self.find_closest_enemy,
            "find_closest_teammate": self.find_closest_teammate,
            "get_horizontal_move_key": self.get_horizontal_move_key,
            "get_vertical_move_key": self.get_vertical_move_key,
            "is_path_blocked": self.is_path_blocked,
            "is_path_blocked_angle": self.is_path_blocked_angle,
            "is_enemy_hittable": self.is_enemy_hittable,
            "walls_block_line_of_sight": self.walls_block_line_of_sight,
            "aimed_attack": self.aimed_attack,
            "get_distance": self.get_distance,
            "get_random_movement": lambda: random.choice(["WA", "WD", "SA", "SD", "W", "A", "S", "D"]),
            "TILE_SIZE": self.TILE_SIZE,
            "width": brawl_stars_width,
            "height": brawl_stars_height,
            "math": math,
            "angle_from_direction": self.angle_from_direction,
            "find_best_angle": self.find_best_angle,
            "blend_angles": self.blend_angles,
            "lead_shot_angle": self.lead_shot_angle,
            "track_enemy_velocity": self.track_enemy_velocity,
            "angle_to_keys": lambda angle: [
                "D", "SD", "S", "SA", "A", "WA", "W", "WD"
            ][int((float(angle) % 360 + 22.5) / 45) % 8],
            "detect_wall_stuck": lambda is_moving: self.detect_wall_stuck(
                walls,
                self.get_player_pos(player_data),
                bool(is_moving),
                time.time(),
            ),
            "start_escape": lambda angle: self.start_semicircle_escape(float(angle), time.time()),
            "escape_step": lambda: self.semicircle_escape_step(time.time()),
        }

        try:
            exec(self.playstyle_code, env, env)
        except Exception as e:
            if not self._playstyle_error_reported:
                print(f"Playstyle '{self.playstyle_name}' failed: {e}. Falling back to built-in logic.")
                self._playstyle_error_reported = True
            return None

        self.time_since_holding_attack = persistent_data.get("time_since_holding_attack")
        return env.get("movement")

    def capture_vision_frame(self, reason, frame, data=None, brawler=None, extra=None):
        if not self.capture_bad_vision_frames or frame is None:
            return
        if self._bad_vision_capture_count >= self.bad_vision_capture_max:
            return
        now = time.time()
        last = self._bad_vision_last_capture.get(reason, 0.0)
        if now - last < self.bad_vision_capture_interval:
            return

        safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(reason))
        folder = os.path.join(self.bad_vision_capture_dir, safe_reason)
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((now % 1) * 1000):03d}"
        image_path = os.path.join(folder, f"{stamp}.png")
        meta_path = os.path.join(folder, f"{stamp}.json")

        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        metadata = {
            "reason": reason,
            "brawler": brawler or self.current_brawler,
            "time": now,
            "data": data or {},
            "extra": extra or {},
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        self._bad_vision_last_capture[reason] = now
        self._bad_vision_capture_count += 1
        print(f"Captured vision frame: {image_path}")

    def reset_match_control_state(self):
        self.window_controller.keys_up(list("wasd"))
        self.keys_hold = []
        self.last_movement = None
        self.last_movement_time = time.time()
        self.time_since_movement = 0
        self.time_since_different_movement = time.time()
        self.time_since_player_last_found = time.time()
        self.time_since_last_proceeding = time.time()
        self.fix_movement_keys['toggled'] = False
        self.time_since_holding_attack = None

    def load_brawler_ranges(self, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        screen_size_ratio = self.window_controller.scale_factor
        ranges = {}
        for brawler, info in brawlers_info.items():
            attack_range = info['attack_range']
            safe_range = info['safe_range']
            super_range = info['super_range']
            v = [safe_range, attack_range, super_range]
            ranges[brawler] = [int(v[0] * screen_size_ratio), int(v[1] * screen_size_ratio), int(v[2] * screen_size_ratio)]
        return ranges

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        if skill_type == "attack":
            return brawlers_info[brawler]['ignore_walls_for_attacks']
        elif skill_type == "super":
            return brawlers_info[brawler]['ignore_walls_for_supers']
        raise ValueError("skill_type must be either 'attack' or 'super'")

    @staticmethod
    def must_brawler_hold_attack(brawler, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        return brawlers_info[brawler]['hold_attack'] > 0

    @staticmethod
    def walls_block_line_of_sight(p1, p2, walls, padding=0):
        if not walls:
            return False

        p1_t = (int(p1[0]), int(p1[1]))
        p2_t = (int(p2[0]), int(p2[1]))
        min_x, max_x = min(p1_t[0], p2_t[0]), max(p1_t[0], p2_t[0])
        min_y, max_y = min(p1_t[1], p2_t[1]), max(p1_t[1], p2_t[1])
        padding = int(max(0, padding))
        for wall in walls:
            x1, y1, x2, y2 = wall
            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding

            if max_x < x1 or min_x > x2 or max_y < y1 or min_y > y2:
                continue

            rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if cv2.clipLine(rect, p1_t, p2_t)[0]:
                return True
        return False

    def no_enemy_movement(self, player_data, walls):
        player_position = self.get_player_pos(player_data)
        preferred_movement = 'W' if self.game_mode == 3 else 'D'  # Adjust based on game mode

        if not self.is_path_blocked(player_position, preferred_movement, walls):
            return preferred_movement
        else:
            # Try alternative movements
            alternative_moves = ['W', 'A', 'S', 'D']
            alternative_moves.remove(preferred_movement)
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_position, move, walls):
                    return move
            print("no movement possible ?")
            # If no movement is possible, return empty string
            return preferred_movement

    def get_entity_pos(self, entity):
        return self.get_enemy_pos(entity)

    def find_closest_teammate(self, teammate_data, player_coords, walls=None):
        closest_distance = float('inf')
        closest_teammate = None
        for teammate in teammate_data or []:
            teammate_pos = self.get_enemy_pos(teammate)
            distance = self.get_distance(teammate_pos, player_coords)
            if distance < closest_distance:
                closest_distance = distance
                closest_teammate = teammate_pos
        return closest_teammate, closest_distance

    def find_teammate_alive_marker(self, frame):
        """Find the blue off-screen teammate marker used when an alive mate is far away."""
        if not self.teammate_marker_follow_enabled or frame is None:
            return None

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        blue = cv2.inRange(
            hsv,
            np.array((92, 90, 85), dtype=np.uint8),
            np.array((125, 255, 255), dtype=np.uint8),
        )
        blue = cv2.morphologyEx(
            blue,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        edge_x = max(24, int(w * self.teammate_marker_edge_margin))
        edge_y = max(24, int(h * self.teammate_marker_edge_margin))
        scale = max(0.4, min(1.2, w / brawl_stars_width))
        min_area = max(180, int(500 * scale * scale))
        max_area = max(min_area + 1, int(50000 * scale * scale))
        best = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 18 * scale or bh < 18 * scale or bw > 240 * scale or bh > 240 * scale:
                continue
            cx, cy = x + bw * 0.5, y + bh * 0.5
            near_edge = cx <= edge_x or cx >= w - edge_x or cy <= edge_y or cy >= h - edge_y
            if not near_edge:
                continue

            pad = int(max(bw, bh) * 0.45)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            roi = hsv[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_area = max(1, roi.shape[0] * roi.shape[1])
            white_ratio = self._count_mask_pixels(roi, (0, 0, 185), (179, 70, 255)) / roi_area
            blue_ratio = self._count_mask_pixels(roi, (92, 90, 85), (125, 255, 255)) / roi_area
            if white_ratio < 0.055 or blue_ratio < 0.08:
                continue

            edge_score = min(cx, w - cx, cy, h - cy)
            score = (edge_score, -area)
            if best is None or score < best[0]:
                best = (score, (cx, cy), (x1, y1, x2, y2))

        if best is None:
            return None
        _, marker_pos, marker_box = best
        vlog(f"teammate marker detected -> pos={tuple(map(int, marker_pos))} box={tuple(map(int, marker_box))}")
        return marker_pos

    def teammate_marker_follow_angle(self, player_pos):
        marker_pos = self.find_teammate_alive_marker(self.current_frame)
        if marker_pos is None or player_pos is None:
            return None
        dx = marker_pos[0] - player_pos[0]
        dy = marker_pos[1] - player_pos[1]
        if math.hypot(dx, dy) < 8:
            return None
        angle = self.angle_from_direction(dx, dy)
        vlog(f"follow teammate marker -> angle={angle:.1f}°")
        return angle

    def _build_trusted_fog_mask(self, frame, roi_center, roi_radius):
        """Return (mask, (ox, oy)) or None.

        Only processes an ROI of side 2*roi_radius+1 around roi_center —
        we only care about fog that's close to the player.
        Mask contains only fog pixels that belong to a large, morphologically
        clean blob — not stray color noise. (ox, oy) is the ROI's top-left
        offset in frame coordinates so callers can translate back.

        Result is cached per-frame (keyed by id(frame) and ROI tuple).
        """
        if frame is None:
            return None

        roi_radius = int(max(1, roi_radius))
        cache_key = (id(frame), int(roi_center[0]), int(roi_center[1]), int(roi_radius))
        if self._fog_mask_cache_frame_id == cache_key:
            return self._fog_mask_cache_value

        import numpy as np
        h, w = frame.shape[:2]
        cx, cy = int(roi_center[0]), int(roi_center[1])
        x0, y0 = max(0, cx - roi_radius), max(0, cy - roi_radius)
        x1, y1 = min(w, cx + roi_radius + 1), min(h, cy + roi_radius + 1)
        if x0 >= x1 or y0 >= y1:
            self._fog_mask_cache_frame_id = cache_key
            self._fog_mask_cache_value = None
            return None
        region = frame[y0:y1, x0:x1]
        origin = (x0, y0)

        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        low = np.array(self.fog_hsv_low, dtype=np.uint8)
        high = np.array(self.fog_hsv_high, dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        result = None
        if num_labels > 1:
            trusted = np.zeros_like(mask)
            any_kept = False
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= self.fog_min_blob_pixels:
                    trusted[labels == label] = 255
                    any_kept = True
            if any_kept and cv2.countNonZero(trusted) > 0:
                result = (trusted, origin)

        self._fog_mask_cache_frame_id = cache_key
        self._fog_mask_cache_value = result
        return result

    def detect_fog_threat(self, frame, player_position):
        """Check whether a real fog front is within self.fog_flee_distance of
        the player. Returns the flee angle (away from local fog mass) if so,
        else None.

        Confidence pipeline:
          1. HSV threshold → raw mask.
          2. Morph open + size-filtered connected components → trusted mask.
          3. Count trusted fog pixels inside a disk of radius fog_flee_distance
             around the player. If count ≥ fog_min_pixels_in_radius, it's a
             real incoming front — not a stray artifact.
        The flee direction is the angle opposite to the centroid of the
        trusted fog pixels *inside the radius*, so we run away from the
        closest wall of fog, not from fog on the far side of the map.
        """
        r = self.fog_flee_distance
        built = self._build_trusted_fog_mask(frame, roi_center=player_position, roi_radius=r)
        if built is None:
            return None
        mask, (ox, oy) = built

        import numpy as np
        px, py = int(player_position[0]), int(player_position[1])
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None

        # Translate ROI-local coords to frame coords, then filter to circle
        dx_all = (xs + ox) - px
        dy_all = (ys + oy) - py
        dist_sq = dx_all * dx_all + dy_all * dy_all
        inside = dist_sq <= r * r
        count = int(inside.sum())
        if count < self.fog_min_pixels_in_radius:
            return None

        # Centroid of the nearby fog mass, then flee opposite direction
        cx = float(dx_all[inside].mean())
        cy = float(dy_all[inside].mean())
        if math.hypot(cx, cy) < 1:
            return None
        toward_fog = self.angle_from_direction(cx, cy)
        flee = self.angle_opposite(toward_fog)
        vlog(f"fog threat: {count}px within {r}px → flee angle={flee:.1f}° (fog at {toward_fog:.1f}°)")
        return flee

    def detect_fog_direction_escape(self, frame, player_position):
        """Return an escape angle if poison gas is touching a side of player.

        This mirrors the official v0.8.3 playstyle idea: check up/down/left/right
        close to the player and move in the opposite direction. It complements
        the centroid-based fog detector, which can miss thin gas edges.
        """
        r = int(max(self.fog_flee_distance, 120))
        built = self._build_trusted_fog_mask(frame, roi_center=player_position, roi_radius=r)
        if built is None:
            return None
        mask, (ox, oy) = built

        import numpy as np
        px, py = int(player_position[0]), int(player_position[1])
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None

        dx = (xs + ox) - px
        dy = (ys + oy) - py
        band = max(35, int(r * 0.45))
        min_pixels = max(20, int(self.fog_min_pixels_in_radius * 0.55))

        direction_counts = {
            "up": int(((dy < 0) & (dy >= -r) & (np.abs(dx) <= band)).sum()),
            "down": int(((dy > 0) & (dy <= r) & (np.abs(dx) <= band)).sum()),
            "left": int(((dx < 0) & (dx >= -r) & (np.abs(dy) <= band)).sum()),
            "right": int(((dx > 0) & (dx <= r) & (np.abs(dy) <= band)).sum()),
        }

        escape_x = 0.0
        escape_y = 0.0
        if direction_counts["up"] >= min_pixels and direction_counts["up"] > direction_counts["down"] + min_pixels:
            escape_y += 1.0
        if direction_counts["down"] >= min_pixels and direction_counts["down"] > direction_counts["up"] + min_pixels:
            escape_y -= 1.0
        if direction_counts["left"] >= min_pixels and direction_counts["left"] > direction_counts["right"] + min_pixels:
            escape_x += 1.0
        if direction_counts["right"] >= min_pixels and direction_counts["right"] > direction_counts["left"] + min_pixels:
            escape_x -= 1.0

        if math.hypot(escape_x, escape_y) < 0.01:
            return None

        angle = self.angle_from_direction(escape_x, escape_y)
        vlog(f"directional fog escape: counts={direction_counts} -> angle={angle:.1f} deg")
        return angle

    def detect_jump_pad_smoke_escape(self, frame, player_position):
        """Return a softer fog escape angle for jump-pad decisions only.

        Normal fog fleeing stays tight to avoid jitter. This detects smoke a
        little earlier, but callers only use it when the bot is alone at the
        map edge and a reachable jump pad exists.
        """
        if frame is None or player_position is None:
            return None
        r = int(max(self.jump_pad_smoke_early_distance, self.fog_flee_distance))
        built = self._build_trusted_fog_mask(frame, roi_center=player_position, roi_radius=r)
        if built is None:
            return None
        mask, (ox, oy) = built

        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None

        px, py = int(player_position[0]), int(player_position[1])
        dx_all = (xs + ox) - px
        dy_all = (ys + oy) - py
        dist_sq = dx_all * dx_all + dy_all * dy_all
        inside = dist_sq <= r * r
        count = int(inside.sum())
        min_pixels = max(20, int(self.fog_min_pixels_in_radius * 0.45))
        if count < min_pixels:
            return None

        cx = float(dx_all[inside].mean())
        cy = float(dy_all[inside].mean())
        if math.hypot(cx, cy) < 1:
            return None
        toward_fog = self.angle_from_direction(cx, cy)
        flee = self.angle_opposite(toward_fog)
        vlog(f"jump pad smoke check: {count}px within {r}px -> flee angle={flee:.1f}°")
        return flee

    def angle_points_into_fog(self, frame, player_position, angle_degrees, lookahead=None):
        """Return True when moving at angle_degrees would drive into nearby fog.

        This is intentionally stricter than the generic fog threat check: it
        only rejects the selected movement path, so teammate-following can still
        work beside smoke without blindly walking toward a teammate inside it.
        """
        if frame is None or player_position is None or angle_degrees is None:
            return False
        r = int(max(140, lookahead or self.fog_flee_distance * 1.7))
        built = self._build_trusted_fog_mask(frame, roi_center=player_position, roi_radius=r)
        if built is None:
            return False
        mask, (ox, oy) = built

        import numpy as np
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return False

        px, py = player_position
        dx = (xs + ox) - px
        dy = (ys + oy) - py
        ux, uy = self.angle_to_vector(float(angle_degrees))
        forward = dx * ux + dy * uy
        lateral = np.abs(dx * uy - dy * ux)
        corridor_width = max(42, int(r * 0.28))
        min_pixels = max(18, int(self.fog_min_pixels_in_radius * 0.5))
        in_path = (forward > 0) & (forward <= r) & (lateral <= corridor_width)
        count = int(in_path.sum())
        if count >= min_pixels:
            vlog(f"fog path guard: {count}px ahead at angle={float(angle_degrees):.1f}°")
            return True
        return False

    def is_there_poison_gas(self, direction, player_data):
        if self.current_frame is None or player_data is None:
            return False
        player_pos = self.get_player_pos(player_data)
        r = int(max(80, min(self.fog_flee_distance, 150)))
        built = self._build_trusted_fog_mask(self.current_frame, roi_center=player_pos, roi_radius=r)
        if built is None:
            return False
        mask, (ox, oy) = built
        import numpy as np
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return False

        px, py = player_pos
        dx = (xs + ox) - px
        dy = (ys + oy) - py
        band = max(30, int(r * 0.45))
        min_pixels = max(12, int(self.fog_min_pixels_in_radius * 0.45))
        direction = str(direction).lower()
        try:
            player_half_width = max(8, abs(float(player_data[2]) - float(player_data[0])) * 0.55)
            player_half_height = max(8, abs(float(player_data[3]) - float(player_data[1])) * 0.55)
        except (TypeError, ValueError, IndexError):
            player_half_width = player_half_height = max(12, band * 0.35)
        player_area = (np.abs(dx) <= player_half_width) & (np.abs(dy) <= player_half_height)
        if int(player_area.sum()) >= min_pixels:
            return True
        checks = {
            "up": (dy < 0) & (dy >= -r) & (np.abs(dx) <= band),
            "down": (dy > 0) & (dy <= r) & (np.abs(dx) <= band),
            "left": (dx < 0) & (dx >= -r) & (np.abs(dy) <= band),
            "right": (dx > 0) & (dx <= r) & (np.abs(dy) <= band),
        }
        if direction not in checks:
            return False
        return int(checks[direction].sum()) >= min_pixels

    def showdown_roam(self, player_data, walls):
        """Idle roam movement that travels instead of spinning in place.

        Close-fog avoidance is still handled by the uniform fog override in
        get_showdown_movement, but this keeps ordinary no-enemy movement away
        from walls and lightly biased toward screen center.
        """
        now = time.time()
        player_pos = self.get_player_pos(player_data)
        current_blocked = self.is_path_blocked_angle(player_pos, self._roam_angle, walls)
        time_expired = (now - self._roam_last_changed) > self.roam_direction_hold_time

        if current_blocked or time_expired:
            new_angle = None
            for _ in range(16):
                candidate = random.uniform(0, 360)
                if not self.is_path_blocked_angle(player_pos, candidate, walls):
                    new_angle = candidate
                    break
            if new_angle is None:
                new_angle = self.find_best_angle(player_pos, (self._roam_angle + 180) % 360, walls)

            if self.roam_center_bias > 0:
                screen_cx, screen_cy = 960.0, 540.0
                dx = screen_cx - player_pos[0]
                dy = screen_cy - player_pos[1]
                if math.hypot(dx, dy) > 160:
                    toward_center = self.angle_from_direction(dx, dy)
                    blended = self.blend_angles(new_angle, toward_center, self.roam_center_bias)
                    if not self.is_path_blocked_angle(player_pos, blended, walls):
                        new_angle = blended

            self._roam_angle = new_angle % 360
            self._roam_last_changed = now
            vlog(f"roam: new direction -> {self._roam_angle:.1f}°")

        vlog(f"roam: holding -> angle={self._roam_angle:.1f}°")
        return self._roam_angle

    @staticmethod
    def angle_to_vector(angle_degrees):
        angle_rad = math.radians(angle_degrees)
        return math.cos(angle_rad), math.sin(angle_rad)

    def blend_angles(self, primary_angle, secondary_angle, secondary_weight):
        primary_weight = max(0.0, 1.0 - secondary_weight)
        sx = max(0.0, secondary_weight)
        ax, ay = self.angle_to_vector(primary_angle)
        bx, by = self.angle_to_vector(secondary_angle)
        dx = ax * primary_weight + bx * sx
        dy = ay * primary_weight + by * sx
        if math.hypot(dx, dy) < 0.01:
            return primary_angle
        return self.angle_from_direction(dx, dy)

    def get_strafe_angle(self, toward_enemy_angle, current_time, enemy_distance=None, safe_range=None):
        if self._strafe_started_at == 0.0:
            self._strafe_started_at = current_time
            self._strafe_current_interval = self.strafe_interval

        elapsed = current_time - self._strafe_started_at
        if elapsed >= self._strafe_current_interval:
            self._strafe_side *= -1
            self._strafe_started_at = current_time
            jitter = random.uniform(-0.3, 0.3) * self.strafe_interval
            self._strafe_current_interval = max(0.5, self.strafe_interval + jitter)
            elapsed = 0.0

        if enemy_distance is not None and safe_range is not None and enemy_distance < safe_range * 0.6:
            random_kick = random.uniform(65.0, 90.0) * self._strafe_side + random.uniform(-15.0, 15.0)
            return (toward_enemy_angle + random_kick) % 360

        t = elapsed / max(0.001, self._strafe_current_interval)
        sine_factor = math.sin(t * math.pi)
        strafe_offset = 90.0 * self._strafe_side * max(0.55, sine_factor)
        return (toward_enemy_angle + strafe_offset) % 360

    def get_combat_dodge_angle(self, toward_enemy_angle, current_time, enemy_distance=None, safe_range=None):
        """Sideways movement used while shooting so the bot does not become an easy target."""
        strafe_angle = self.get_strafe_angle(toward_enemy_angle, current_time, enemy_distance, safe_range)
        jitter = float(getattr(self, "combat_dodge_jitter_degrees", 0.0))
        if jitter > 0:
            strafe_angle = (strafe_angle + random.uniform(-jitter, jitter)) % 360
        return strafe_angle

    def apply_combat_dodge(self, desired_angle, toward_enemy_angle, current_time, enemy_distance, safe_range):
        if not self.strafe_enabled:
            return desired_angle
        dodge_angle = self.get_combat_dodge_angle(toward_enemy_angle, current_time, enemy_distance, safe_range)
        blend = max(0.0, min(1.0, float(getattr(self, "combat_dodge_blend", 0.0))))
        if enemy_distance is not None and safe_range is not None and enemy_distance <= safe_range:
            blend = max(blend, min(0.85, blend + 0.15))
        return self.blend_angles(desired_angle, dodge_angle, blend)

    def _manslog(self, *args):
        if visual_debug or getattr(self, "auto_aim_debug", False):
            print("[MANS]", *args)

    def detect_player_bar_flicker(self, frame, player_data, current_time):
        if not getattr(self, "enable_flicker_retreat", False) or frame is None or player_data is None:
            return False, 0.0
        h, w = frame.shape[:2]
        x1, y1, x2, _ = map(int, self.normalize_box(player_data))
        bw = max(1, x2 - x1)
        rx1 = max(0, x1 - int(bw * 0.35))
        rx2 = min(w, x2 + int(bw * 0.35))
        ry1 = max(0, y1 - int(bw * 0.75))
        ry2 = max(0, y1 - int(bw * 0.08))
        if rx1 >= rx2 or ry1 >= ry2:
            return False, 0.0
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        area = max(1, roi.shape[0] * roi.shape[1])
        bright = self._count_mask_pixels(hsv, (0, 0, 145), (179, 105, 255))
        red = (
            self._count_mask_pixels(hsv, (0, 75, 75), (18, 255, 255))
            + self._count_mask_pixels(hsv, (170, 75, 75), (179, 255, 255))
        )
        green = self._count_mask_pixels(hsv, (35, 65, 65), (88, 255, 255))
        signal = (bright + red + green) / area
        history = self._player_bar_history
        history.append((current_time, signal))
        cutoff = current_time - 0.45
        while history and history[0][0] < cutoff:
            history.pop(0)
        if len(history) < 4:
            return False, 0.0
        values = [item[1] for item in history]
        amplitude = max(values) - min(values)
        toggles = sum(
            1
            for prev, cur in zip(values, values[1:])
            if abs(cur - prev) >= 0.035
        )
        confidence = min(1.0, amplitude * 5.0 + toggles * 0.12)
        detected = confidence >= 0.48 and toggles >= 2
        if detected and current_time - self._flicker_state["last_trigger"] >= self.flicker_retreat_cooldown_seconds:
            self._flicker_state["active_until"] = current_time + self.flicker_retreat_hold_seconds
            self._flicker_state["last_trigger"] = current_time
            self._flicker_state["confidence"] = confidence
        active = current_time < self._flicker_state["active_until"]
        if detected or active:
            self._manslog(f"flicker_detected={detected} flicker_confidence={confidence:.2f}")
        return active, max(confidence, self._flicker_state.get("confidence", 0.0) if active else 0.0)

    def estimate_player_health_ratio(self, frame, player_data):
        if frame is None or player_data is None or not hasattr(frame, "shape"):
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, _ = map(int, self.normalize_box(player_data))
        bw = max(1, x2 - x1)
        rx1 = max(0, x1 - int(bw * 0.38))
        rx2 = min(w, x2 + int(bw * 0.38))
        ry1 = max(0, y1 - int(bw * 0.72))
        ry2 = max(0, y1 - int(bw * 0.16))
        if rx1 >= rx2 or ry1 >= ry2:
            return None
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, np.array((35, 70, 70), dtype=np.uint8), np.array((88, 255, 255), dtype=np.uint8))
        ys, xs = np.nonzero(green_mask)
        if xs.size < 8:
            return None
        width = max(1, rx2 - rx1)
        green_span = float(xs.max() - xs.min() + 1)
        return max(0.0, min(1.0, green_span / width))

    def update_heal_state(self, health_ratio, flicker_active, flicker_confidence, current_time):
        if not getattr(self, "heal_retreat_enabled", False):
            return False, "disabled"
        if not hasattr(self, "_heal_state"):
            self._heal_state = {"active_until": 0.0, "last_health_ratio": None, "reason": ""}
        active_until = float(self._heal_state.get("active_until", 0.0) or 0.0)
        active = should_seek_healing(
            health_ratio,
            recent_damage=flicker_active and flicker_confidence >= 0.65,
            active_until=active_until,
            now=current_time,
            low_threshold=getattr(self, "heal_low_health_threshold", 0.42),
        )
        if health_ratio is not None and health_ratio >= getattr(self, "heal_resume_health_threshold", 0.72):
            active = False
        if active:
            reason = "low_health" if health_ratio is not None and health_ratio <= getattr(self, "heal_low_health_threshold", 0.42) else "recent_damage"
            self._heal_state["active_until"] = max(
                active_until,
                current_time + getattr(self, "heal_retreat_hold_ms", 2400) / 1000.0,
            )
            self._heal_state["last_health_ratio"] = health_ratio
            self._heal_state["reason"] = reason
            self._manslog(
                f"heal_retreat active=True reason={reason} health_ratio={health_ratio} "
                f"flicker_confidence={flicker_confidence:.2f}"
            )
            return True, reason
        self._heal_state["last_health_ratio"] = health_ratio
        if current_time >= active_until:
            self._heal_state["reason"] = ""
        return False, "healthy"

    def choose_heal_retreat_angle(self, player_pos, enemy_data, teammate_data, walls, current_angle):
        enemies = [
            (self.get_distance(self.get_enemy_pos(enemy), player_pos), self.get_enemy_pos(enemy))
            for enemy in (enemy_data or [])
        ]
        enemies.sort(key=lambda item: item[0])
        if enemies:
            threat_angle = self.angle_from_direction(enemies[0][1][0] - player_pos[0], enemies[0][1][1] - player_pos[1])
            base = (threat_angle + 180.0) % 360.0
        else:
            base = current_angle
            closest_teammate, _ = self.get_closest_teammate((player_pos[0], player_pos[1], player_pos[0], player_pos[1]), teammate_data)
            if closest_teammate is not None:
                base = self.angle_from_direction(closest_teammate[0] - player_pos[0], closest_teammate[1] - player_pos[1])

        candidates = [base, (base + 30.0) % 360.0, (base - 30.0) % 360.0, (base + 65.0) % 360.0, (base - 65.0) % 360.0]
        for candidate in candidates:
            safe, reason = self._angle_safe_for_tactical_move(player_pos, candidate, walls)
            if safe:
                return candidate, "heal_retreat"
        return current_angle, "heal_no_safe_angle"

    def _angle_safe_for_tactical_move(self, player_pos, angle, walls):
        if self.is_path_blocked_angle(player_pos, angle, walls):
            return False, "blocked_by_wall"
        if self.angle_points_into_fog(self.current_frame, player_pos, angle):
            return False, "blocked_by_poison"
        return True, "ok"

    def choose_tactical_dodge_angle(
            self,
            base_angle,
            player_pos,
            enemy_data,
            teammate_data,
            walls,
            safe_range,
            attack_range,
            current_time,
            flicker_active=False,
            projectile_data=None,
    ):
        if not getattr(self, "enable_combat_mans", True):
            return base_angle, {"mode": "no_dodge", "threat": 0.0, "reason": "disabled_or_no_enemy"}

        projectile = self.select_incoming_projectile_threat(
            projectile_data or [],
            player_pos,
            current_time,
        )
        if not enemy_data:
            if projectile and getattr(self, "projectile_dodge_without_enemy", True):
                threat_angle = self.angle_from_direction(
                    projectile["projectile_velocity"][0],
                    projectile["projectile_velocity"][1],
                )
                candidates = candidate_dodge_angles(base_angle, threat_angle)
                scored = []
                rejected = []
                for candidate in candidates:
                    score, reasons = score_dodge_angle(
                        candidate,
                        base_angle=base_angle,
                        threat_angle=threat_angle,
                        closest_enemy_distance=float("inf"),
                        safe_range=safe_range,
                        attack_range=attack_range,
                        is_blocked=lambda angle: self.is_path_blocked_angle(player_pos, angle, walls),
                        points_into_fog=lambda angle: self.angle_points_into_fog(self.current_frame, player_pos, angle),
                        current_angle=self._dodge_state.get("angle"),
                        teammate_angle=None,
                    )
                    projectile_score, projectile_reasons = score_projectile_dodge_angle(candidate, projectile)
                    score += projectile_score + 2.0
                    reasons.extend(projectile_reasons)
                    if score < -900:
                        rejected.append((round(candidate, 1), reasons[0] if reasons else "rejected"))
                        continue
                    scored.append((score, candidate, reasons))
                if scored:
                    scored.sort(key=lambda item: item[0], reverse=True)
                    best_score, best_angle, reasons = scored[0]
                    self._dodge_state = {
                        "angle": best_angle,
                        "mode": "projectile_dodge",
                        "score": best_score,
                        "until": current_time + self.mans_hysteresis_seconds,
                    }
                    self._manslog(
                        f"dodge_mode=projectile_dodge threat_level={0.72 + projectile['danger'] * 0.24:.2f} "
                        f"selected_dodge_vector={best_angle:.1f} dodge_score={best_score:.2f} "
                        f"reason={'+'.join(reasons)} rejected_directions={rejected} enemy_visible=False "
                        f"projectile_tti={round(projectile['time_to_impact'], 2)}"
                    )
                    return best_angle, {
                        "mode": "projectile_dodge",
                        "threat": 0.72 + projectile["danger"] * 0.24,
                        "score": best_score,
                        "reason": "+".join(reasons),
                        "rejected": rejected,
                        "projectile": projectile,
                    }
                self._manslog(
                    f"dodge_mode=projectile_dodge threat_level={0.72 + projectile['danger'] * 0.24:.2f} "
                    f"enemy_visible=False rejected_directions={rejected} reason=no_safe_candidate"
                )
                return base_angle, {
                    "mode": "projectile_dodge",
                    "threat": 0.72 + projectile["danger"] * 0.24,
                    "reason": "no_safe_candidate",
                    "rejected": rejected,
                    "projectile": projectile,
                }
            return base_angle, {"mode": "no_dodge", "threat": 0.0, "reason": "no_enemy_no_projectile"}

        closest_enemy = None
        closest_distance = float("inf")
        for enemy in enemy_data or []:
            pos = self.get_enemy_pos(enemy)
            dist = self.get_distance(pos, player_pos)
            if dist < closest_distance:
                closest_enemy = pos
                closest_distance = dist
        if closest_enemy is None:
            return base_angle, {"mode": "no_dodge", "threat": 0.0, "reason": "no_enemy_pos"}

        threat = threat_level_from_distance(closest_distance, attack_range, safe_range)
        if flicker_active:
            threat = max(threat, 0.50)
        if projectile:
            threat = max(threat, 0.72 + projectile["danger"] * 0.24)
        if threat < self.mans_threat_threshold and not flicker_active:
            return base_angle, {
                "mode": "no_dodge",
                "threat": threat,
                "closest_enemy_distance": closest_distance,
                "reason": "below_threshold",
            }

        threat_angle = self.angle_from_direction(closest_enemy[0] - player_pos[0], closest_enemy[1] - player_pos[1])
        mode = classify_dodge_mode(threat, closest_distance, safe_range, flicker_active=flicker_active)
        teammate_angle = None
        closest_teammate, teammate_distance = self.get_closest_teammate((player_pos[0], player_pos[1], player_pos[0], player_pos[1]), teammate_data)
        if closest_teammate is not None and teammate_distance > self.teammate_follow_step_distance:
            teammate_angle = self.angle_from_direction(closest_teammate[0] - player_pos[0], closest_teammate[1] - player_pos[1])

        candidates = candidate_dodge_angles(base_angle, threat_angle)
        rejected = []
        scored = []
        for candidate in candidates:
            score, reasons = score_dodge_angle(
                candidate,
                base_angle=base_angle,
                threat_angle=threat_angle,
                closest_enemy_distance=closest_distance,
                safe_range=safe_range,
                attack_range=attack_range,
                is_blocked=lambda angle: self.is_path_blocked_angle(player_pos, angle, walls),
                points_into_fog=lambda angle: self.angle_points_into_fog(self.current_frame, player_pos, angle),
                current_angle=self._dodge_state.get("angle"),
                teammate_angle=teammate_angle,
            )
            projectile_score, projectile_reasons = score_projectile_dodge_angle(candidate, projectile)
            score += projectile_score
            reasons.extend(projectile_reasons)
            if score < -900:
                rejected.append((round(candidate, 1), reasons[0] if reasons else "rejected"))
                continue
            scored.append((score, candidate, reasons))

        if not scored:
            self._manslog(
                f"dodge_mode={mode} threat_level={threat:.2f} rejected_directions={rejected} "
                f"closest_enemy_distance={int(closest_distance)} reason=no_safe_candidate"
            )
            return base_angle, {
                "mode": mode,
                "threat": threat,
                "closest_enemy_distance": closest_distance,
                "reason": "no_safe_candidate",
                "rejected": rejected,
            }

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_angle, reasons = scored[0]
        current_angle = self._dodge_state.get("angle")
        if current_angle is not None and current_time < self._dodge_state.get("until", 0.0):
            current_safe, _ = self._angle_safe_for_tactical_move(player_pos, current_angle, walls)
            current_score, _ = score_dodge_angle(
                current_angle,
                base_angle=base_angle,
                threat_angle=threat_angle,
                closest_enemy_distance=closest_distance,
                safe_range=safe_range,
                attack_range=attack_range,
                is_blocked=lambda angle: self.is_path_blocked_angle(player_pos, angle, walls),
                points_into_fog=lambda angle: self.angle_points_into_fog(self.current_frame, player_pos, angle),
                current_angle=current_angle,
                teammate_angle=teammate_angle,
            )
            projectile_score, _ = score_projectile_dodge_angle(current_angle, projectile)
            current_score += projectile_score
            if current_safe and best_score - current_score < 1.2:
                self._manslog(
                    f"dodge_mode={mode} threat_level={threat:.2f} selected_dodge_vector={current_angle:.1f} "
                    f"dodge_score={current_score:.2f} closest_enemy_distance={int(closest_distance)} "
                    f"incoming_projectile_detected={bool(projectile)} "
                    "keeping_current_direction_due_to_hysteresis=True combined_with_auto_aim=True"
                )
                return current_angle, {
                    "mode": mode,
                    "threat": threat,
                    "closest_enemy_distance": closest_distance,
                    "score": current_score,
                    "reason": "hysteresis",
                    "rejected": rejected,
                }

        self._dodge_state = {
            "angle": best_angle,
            "mode": mode,
            "score": best_score,
            "until": current_time + self.mans_hysteresis_seconds,
        }
        self._manslog(
            f"dodge_mode={mode} threat_level={threat:.2f} selected_dodge_vector={best_angle:.1f} "
            f"dodge_score={best_score:.2f} reason={'+'.join(reasons)} rejected_directions={rejected} "
            f"closest_enemy_distance={int(closest_distance)} incoming_projectile_detected={bool(projectile)} "
            f"projectile_tti={None if not projectile else round(projectile['time_to_impact'], 2)} "
            "combined_with_auto_aim=True"
        )
        return best_angle, {
            "mode": mode,
            "threat": threat,
            "closest_enemy_distance": closest_distance,
            "score": best_score,
            "reason": "+".join(reasons),
            "rejected": rejected,
            "projectile": projectile,
        }

    def select_incoming_projectile_threat(self, projectile_data, player_pos, current_time):
        if not getattr(self, "projectile_dodge_enabled", False):
            return None
        if not projectile_data:
            if current_time < getattr(self, "_last_projectile_threat_until", 0.0):
                cached = getattr(self, "_last_projectile_threat", None)
                if cached:
                    cached = dict(cached)
                    cached["from_memory"] = True
                    return cached
            return None
        best = None
        for projectile in projectile_data:
            center = self.get_enemy_pos(projectile)
            velocity = self.track_projectile_velocity(center, current_time)
            threat = projectile_threat(
                center,
                velocity,
                player_pos,
                player_radius=self.projectile_dodge_player_radius,
                horizon_seconds=self.projectile_dodge_horizon,
            )
            if not threat:
                continue
            threat["projectile_pos"] = center
            threat["projectile_velocity"] = velocity
            if best is None or (threat["danger"], -threat["time_to_impact"]) > (best["danger"], -best["time_to_impact"]):
                best = threat
        if best:
            self._last_projectile_threat = dict(best)
            self._last_projectile_threat_until = current_time + max(
                0.0,
                float(getattr(self, "projectile_threat_memory_ms", 300)) / 1000.0,
            )
            self._manslog(
                "projectile_threat "
                f"pos={tuple(map(int, best['projectile_pos']))} "
                f"velocity=({best['projectile_velocity'][0]:.1f},{best['projectile_velocity'][1]:.1f}) "
                f"time_to_impact={best['time_to_impact']:.2f} miss_distance={best['miss_distance']:.1f} "
                f"danger={best['danger']:.2f}"
            )
        elif current_time < getattr(self, "_last_projectile_threat_until", 0.0):
            cached = getattr(self, "_last_projectile_threat", None)
            if cached:
                best = dict(cached)
                best["from_memory"] = True
        return best

    def track_projectile_velocity(self, projectile_coords, current_time):
        if not hasattr(self, "_projectile_track"):
            self._projectile_track = {}
        grid = 18
        rounded_key = (round(projectile_coords[0] / grid) * grid, round(projectile_coords[1] / grid) * grid)
        best_key = None
        best_dist = float("inf")
        for key, item in list(self._projectile_track.items()):
            age = current_time - item["time"]
            if age > 0.8:
                self._projectile_track.pop(key, None)
                continue
            dist = (key[0] - rounded_key[0]) ** 2 + (key[1] - rounded_key[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_key = key
        if best_key is None:
            self._projectile_track[rounded_key] = {"pos": projectile_coords, "time": current_time}
            return 0.0, 0.0
        previous = self._projectile_track.pop(best_key)
        dt = max(0.001, current_time - previous["time"])
        vx = max(-2200.0, min(2200.0, (projectile_coords[0] - previous["pos"][0]) / dt))
        vy = max(-2200.0, min(2200.0, (projectile_coords[1] - previous["pos"][1]) / dt))
        self._projectile_track[rounded_key] = {"pos": projectile_coords, "time": current_time}
        return vx, vy

    def _intentlog(self, tag, *parts):
        if visual_debug or getattr(self, "movement_intent_debug", False):
            print(f"[{tag}]", *parts)

    def choose_projectile_escape_angle(self, projectile, player_pos, walls, fallback_angle):
        if not projectile:
            return None
        candidates = list(projectile.get("escape_angles") or [])
        if fallback_angle is not None:
            candidates.extend([(fallback_angle + 25.0) % 360.0, (fallback_angle - 25.0) % 360.0])
        best = None
        for candidate in candidates:
            if candidate is None:
                continue
            safe, reason = self._angle_safe_for_tactical_move(player_pos, candidate, walls)
            dodge_score, dodge_reasons = score_projectile_dodge_angle(candidate, projectile)
            if not safe:
                self._intentlog(
                    "DODGE",
                    f"will_intersect=True dodge_angle={candidate:.1f} safe=False reason={reason}",
                )
                continue
            lane_bonus = 0.0
            if fallback_angle is not None:
                lane_bonus = max(0.0, 60.0 - abs((candidate - fallback_angle + 180.0) % 360.0 - 180.0)) * 0.01
            score = dodge_score + lane_bonus
            if best is None or score > best[0]:
                best = (score, candidate, "+".join(dodge_reasons) or "projectile_lateral")
        if best is None:
            return None
        _, angle, reason = best
        self._intentlog(
            "DODGE",
            f"will_intersect=True dodge_angle={angle:.1f} safe=True reason={reason}",
        )
        return angle

    def apply_movement_intent(
            self,
            *,
            base_angle,
            player_pos,
            enemy_coords,
            enemy_distance,
            enemy_data,
            teammate_data,
            walls,
            safe_range,
            attack_range,
            fog_flee_angle,
            health_ratio,
            heal_active,
            projectile_data,
            current_time,
    ):
        if not getattr(self, "movement_intent_enabled", False):
            return base_angle, True, None

        enemy_visible = enemy_coords is not None and enemy_distance is not None
        vector_to_enemy = (0.0, 0.0)
        toward_enemy_angle = None
        away_enemy_angle = None
        strafe_angle = None
        enemy_has_line = False
        attack_lane_available = False
        nearby_enemy_count = 0
        if enemy_visible:
            vector_to_enemy = (enemy_coords[0] - player_pos[0], enemy_coords[1] - player_pos[1])
            toward_enemy_angle = self.angle_from_direction(vector_to_enemy[0], vector_to_enemy[1])
            away_enemy_angle = self.angle_opposite(toward_enemy_angle)
            strafe_angle = self.get_strafe_angle(toward_enemy_angle, current_time, enemy_distance, safe_range)
            enemy_has_line = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
            attack_lane_available = enemy_has_line and enemy_distance <= attack_range * 1.035
            nearby_enemy_count = sum(
                1 for enemy in (enemy_data or [])
                if self.get_distance(self.get_enemy_pos(enemy), player_pos) <= attack_range
            )

        closest_teammate, teammate_distance = self.get_closest_teammate(
            (player_pos[0], player_pos[1], player_pos[0], player_pos[1]),
            teammate_data,
        )
        teammate_angle = None
        if closest_teammate is not None:
            teammate_angle = self.angle_from_direction(
                closest_teammate[0] - player_pos[0],
                closest_teammate[1] - player_pos[1],
            )

        projectile = self.select_incoming_projectile_threat(projectile_data or [], player_pos, current_time)
        projectile_escape_angle = self.choose_projectile_escape_angle(projectile, player_pos, walls, base_angle)
        threat = build_threat_state(
            closest_enemy_distance=enemy_distance,
            safe_range=safe_range,
            attack_range=attack_range,
            enemy_velocity=getattr(self, "enemy_velocity", (0.0, 0.0)),
            vector_to_enemy=vector_to_enemy,
            enemy_has_line=enemy_has_line,
            projectile=projectile if projectile_escape_angle is not None else None,
            fog_danger=fog_flee_angle is not None,
            nearby_enemy_count=nearby_enemy_count,
            health_ratio=health_ratio,
            teammate_distance=teammate_distance,
            teammate_near_range=getattr(self, "teammate_combat_regroup_distance", 650),
            wall_pressure=self.is_path_blocked_angle(player_pos, base_angle, walls),
            attack_lane_available=attack_lane_available,
        )
        self._intentlog(
            "THREAT_MODEL",
            f"total={threat.total_score:.2f}",
            f"projectile={threat.projectile_incoming}",
            f"fog={threat.fog_danger}",
            f"low_hp={threat.low_hp}",
            f"enemy_close={threat.enemy_close}",
            f"reasons={','.join(threat.reasons)}",
        )

        heal_angle = None
        if heal_active:
            heal_angle, _ = self.choose_heal_retreat_angle(player_pos, enemy_data, teammate_data, walls, base_angle)

        raw_intent = build_movement_intent(
            threat=threat,
            base_angle=base_angle,
            enemy_visible=enemy_visible,
            enemy_distance=enemy_distance,
            safe_range=safe_range,
            attack_range=attack_range,
            toward_enemy_angle=toward_enemy_angle,
            away_enemy_angle=away_enemy_angle,
            strafe_angle=strafe_angle,
            projectile_escape_angle=projectile_escape_angle,
            fog_escape_angle=fog_flee_angle,
            teammate_angle=teammate_angle,
            heal_retreat_angle=heal_angle,
        )

        safe_angle = self.find_best_angle(player_pos, raw_intent.angle, walls)
        if self.angle_points_into_fog(self.current_frame, player_pos, safe_angle) and fog_flee_angle is None:
            fallback = self.find_best_angle(player_pos, self.angle_opposite(safe_angle), walls)
            raw_intent.angle = fallback
            raw_intent.reasons.append("intent_fog_guard")
        else:
            raw_intent.angle = safe_angle

        memory = getattr(self, "_movement_intent_memory", MovementIntentMemory())
        new_memory, intent, smoothing_reason = smooth_intent(
            memory,
            raw_intent,
            now=current_time,
            min_hold_ms=getattr(self, "movement_intent_min_hold_ms", 350),
            max_hold_ms=getattr(self, "movement_intent_max_hold_ms", 650),
            switch_score_threshold=getattr(self, "movement_intent_switch_score_threshold", 0.18),
            angle_smoothing=getattr(self, "movement_intent_angle_smoothing", 0.35),
        )
        self._movement_intent_memory = new_memory
        if smoothing_reason != "switched" and smoothing_reason != "new_intent":
            self._intentlog("SMOOTHING", f"keeping_previous_intent reason={smoothing_reason}")
        self._intentlog(
            "MOVEMENT_INTENT",
            f"mode={intent.mode}",
            f"angle={intent.angle:.1f}",
            f"score={intent.score:.2f}",
            f"attack_allowed={str(intent.attack_allowed).lower()}",
            f"reasons={','.join(intent.reasons)}",
        )
        return intent.angle, intent.attack_allowed, intent

    def choose_flicker_retreat_angle(
            self,
            player_pos,
            enemy_data,
            teammate_data,
            walls,
            safe_range,
            attack_range,
            current_angle,
            flicker_confidence,
            fog_flee_angle=None,
    ):
        if not self.enable_flicker_retreat:
            return None, "disabled"
        if fog_flee_angle is not None:
            return None, "escape_poison_priority"
        enemies = [
            (self.get_distance(self.get_enemy_pos(enemy), player_pos), self.get_enemy_pos(enemy))
            for enemy in (enemy_data or [])
        ]
        enemies.sort(key=lambda item: item[0])
        closest_distance = enemies[0][0] if enemies else None
        nearby_count = sum(1 for dist, _ in enemies if dist <= self.dangerous_close_range)
        forced_close = closest_distance is not None and closest_distance <= max(75.0, safe_range * 0.50)
        if nearby_count > 0:
            return None, f"nearby_enemy_count={nearby_count}"
        if forced_close:
            return None, "forced_close_combat"
        if closest_distance is not None and closest_distance <= attack_range * 0.72:
            return None, "attack_better_than_retreat"

        base = current_angle
        if enemies:
            threat_angle = self.angle_from_direction(enemies[0][1][0] - player_pos[0], enemies[0][1][1] - player_pos[1])
            base = (threat_angle + 180.0) % 360.0
        elif teammate_data:
            closest_teammate, _ = self.get_closest_teammate((player_pos[0], player_pos[1], player_pos[0], player_pos[1]), teammate_data)
            if closest_teammate is not None:
                base = self.angle_from_direction(closest_teammate[0] - player_pos[0], closest_teammate[1] - player_pos[1])

        candidates = [base, (base + 28.0) % 360.0, (base - 28.0) % 360.0, (base + 55.0) % 360.0, (base - 55.0) % 360.0]
        for candidate in candidates:
            safe, reason = self._angle_safe_for_tactical_move(player_pos, candidate, walls)
            if safe:
                self._manslog(
                    f"flicker_retreat flicker_confidence={flicker_confidence:.2f} "
                    f"nearby_enemy_count={nearby_count} closest_enemy_distance={closest_distance} "
                    f"retreat_allowed=True selected={candidate:.1f}"
                )
                return candidate, "retreat_allowed"
        return None, "no_safe_retreat_direction"

    def track_enemy_velocity(self, enemy_coords, current_time):
        grid = 25
        rounded_key = (round(enemy_coords[0] / grid) * grid, round(enemy_coords[1] / grid) * grid)
        best_key = None
        best_dist = (grid * 4) ** 2
        for key, item in list(self._enemy_track.items()):
            age = current_time - item["time"]
            if age > 2.5:
                self._enemy_track.pop(key, None)
                self._enemy_velocity_smooth.pop(key, None)
                self._enemy_velocity_confidence.pop(key, None)
                continue
            dist = (key[0] - rounded_key[0]) ** 2 + (key[1] - rounded_key[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_key = key
        if best_key is None:
            self._enemy_track[rounded_key] = {"pos": enemy_coords, "time": current_time}
            self._enemy_velocity_confidence[rounded_key] = 0
            self.enemy_velocity_confidence = 0.0
            return 0.0, 0.0

        previous = self._enemy_track.pop(best_key)
        previous_smooth = self._enemy_velocity_smooth.pop(best_key, None)
        previous_confidence = self._enemy_velocity_confidence.pop(best_key, 0)
        dt = max(0.001, current_time - previous["time"])
        raw_vx = max(-1200.0, min(1200.0, (enemy_coords[0] - previous["pos"][0]) / dt))
        raw_vy = max(-1200.0, min(1200.0, (enemy_coords[1] - previous["pos"][1]) / dt))
        alpha = self.velocity_ema_alpha
        if previous_smooth is None:
            smooth_vx, smooth_vy = raw_vx, raw_vy
        else:
            smooth_vx = alpha * raw_vx + (1.0 - alpha) * previous_smooth[0]
            smooth_vy = alpha * raw_vy + (1.0 - alpha) * previous_smooth[1]

        new_confidence = min(previous_confidence + 1, 8)
        self.enemy_velocity_confidence = min(1.0, new_confidence / 4.0)
        self._enemy_track[rounded_key] = {"pos": enemy_coords, "time": current_time}
        self._enemy_velocity_smooth[rounded_key] = (smooth_vx, smooth_vy)
        self._enemy_velocity_confidence[rounded_key] = new_confidence
        return smooth_vx, smooth_vy

    def lead_shot_angle(self, player_pos, enemy_coords, enemy_velocity, projectile_speed_px_s=None, confidence=1.0):
        projectile_speed = projectile_speed_px_s or self.projectile_speed_px_s
        dx = enemy_coords[0] - player_pos[0]
        dy = enemy_coords[1] - player_pos[1]
        direct_angle = self.angle_from_direction(dx, dy)
        if math.hypot(dx, dy) < 1 or projectile_speed <= 1:
            return direct_angle

        vx, vy = enemy_velocity
        if math.hypot(vx, vy) < 15:
            return direct_angle

        a = vx * vx + vy * vy - projectile_speed * projectile_speed
        b = 2 * (dx * vx + dy * vy)
        c = dx * dx + dy * dy
        if abs(a) < 1e-6:
            if abs(b) < 1e-6:
                return direct_angle
            t = -c / b
        else:
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                return direct_angle
            root = math.sqrt(discriminant)
            candidates = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
            positive = [value for value in candidates if value > 0]
            if not positive:
                return direct_angle
            t = min(positive)
        if t <= 0 or t > 1.5:
            return direct_angle

        led_angle = self.angle_from_direction(dx + vx * t, dy + vy * t)
        if confidence < 1.0:
            led_angle = self.blend_angles(direct_angle, led_angle, confidence)
        return led_angle

    def get_closest_teammate(self, player_data, teammate_data):
        player_pos = self.get_player_pos(player_data)
        closest_teammate = None
        closest_distance = float('inf')
        for tm in teammate_data or []:
            tm_pos = self.get_enemy_pos(tm)
            dist = self.get_distance(tm_pos, player_pos)
            if dist < closest_distance:
                closest_distance = dist
                closest_teammate = tm_pos
        return closest_teammate, closest_distance

    def choose_locked_teammate(self, player_pos, teammate_data, walls=None):
        """Keep following the same teammate as long as the detector can track them."""
        closest_teammate, closest_distance = self.find_closest_teammate(teammate_data, player_pos, walls)
        if closest_teammate is None:
            if self.teammate_lock_lost_since <= 0:
                self.teammate_lock_lost_since = time.time()
            if time.time() - self.teammate_lock_lost_since > 1.5:
                self.locked_teammate = None
                self.locked_teammate_distance = float('inf')
            return self.locked_teammate, self.locked_teammate_distance

        self.teammate_lock_lost_since = 0.0
        if self.locked_teammate is None:
            self.locked_teammate = closest_teammate
            self.locked_teammate_distance = closest_distance
            return self.locked_teammate, self.locked_teammate_distance

        candidates = []
        for teammate in teammate_data or []:
            teammate_pos = self.get_enemy_pos(teammate)
            dist_to_lock = self.get_distance(teammate_pos, self.locked_teammate)
            dist_to_player = self.get_distance(teammate_pos, player_pos)
            candidates.append((dist_to_lock, dist_to_player, teammate_pos))
        candidates.sort(key=lambda item: item[0])

        tracked_lock = None
        tracked_distance = float('inf')
        if candidates and candidates[0][0] <= self.teammate_lock_max_jump:
            tracked_lock = candidates[0][2]
            tracked_distance = candidates[0][1]

        if tracked_lock is None:
            self.locked_teammate = closest_teammate
            self.locked_teammate_distance = closest_distance
            return self.locked_teammate, self.locked_teammate_distance

        switch_distance = tracked_distance * (1.0 - self.teammate_hysteresis)
        if closest_distance < switch_distance:
            vlog(
                "follow teammate: switching lock "
                f"tracked={int(tracked_distance)}px closest={int(closest_distance)}px"
            )
            self.locked_teammate = closest_teammate
            self.locked_teammate_distance = closest_distance
        else:
            self.locked_teammate = tracked_lock
            self.locked_teammate_distance = tracked_distance
        return self.locked_teammate, self.locked_teammate_distance

    def showdown_follow_teammate(self, player_data, teammate_data, walls):
        """Official Pyla follower behavior adapted to angle movement.

        The newest Pyla follower does not orbit or keep spacing. When no enemy
        is reachable it simply moves toward the closest teammate, trying the
        direct diagonal vector first, then the horizontal and vertical parts.
        """
        player_pos = self.get_player_pos(player_data)
        closest_teammate, closest_distance = self.choose_locked_teammate(player_pos, teammate_data, walls)

        if closest_teammate is None:
            self.locked_teammate = None
            self.locked_teammate_distance = float('inf')
            marker_angle = self.teammate_marker_follow_angle(player_pos)
            if marker_angle is not None:
                return marker_angle
            vlog("follow teammate: no teammate detected -> roam")
            return self.showdown_roam(player_data, walls)

        direction_x = closest_teammate[0] - player_pos[0]
        direction_y = closest_teammate[1] - player_pos[1]
        direct_angle = self.angle_from_direction(direction_x, direction_y)

        if self.teammate_follow_force_direct and closest_distance > self.teammate_follow_step_distance:
            vlog(f"follow teammate: force direct -> angle={direct_angle:.1f}° (dist={int(closest_distance)}px)")
            return direct_angle

        movement_vectors = [(direction_x, direction_y), (direction_x, 0), (0, direction_y)]
        fallback_angle = direct_angle

        for dx, dy in movement_vectors:
            if math.hypot(dx, dy) < 1:
                continue
            angle = self.angle_from_direction(dx, dy)
            if fallback_angle is None:
                fallback_angle = angle
            if not self.is_path_blocked_angle(player_pos, angle, walls):
                vlog(f"follow teammate -> angle={angle:.1f}° (dist={int(closest_distance)}px)")
                return angle

        for angle in (270.0, 180.0, 90.0, 0.0):
            if not self.is_path_blocked_angle(player_pos, angle, walls):
                vlog(f"follow teammate: preferred blocked -> fallback angle={angle:.1f}°")
                return angle

        angle = fallback_angle if fallback_angle is not None else self.showdown_roam(player_data, walls)
        vlog(f"follow teammate: all paths blocked -> forcing angle={float(angle):.1f}°")
        return angle

    def get_showdown_movement(self, player_data, enemy_data, teammate_data, walls, brawler, jump_pads=None, projectile_data=None):
        """Showdown movement using analog joystick angles.

        Always returns a float angle in degrees (0–360).
        0° = right, 90° = down, 180° = left, 270° = up.
        """
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Brawler '{brawler}' not found in brawlers info.")

        must_brawler_hold_attack = self.must_brawler_hold_attack(brawler, self.brawlers_info)
        if must_brawler_hold_attack and self.time_since_holding_attack is not None and \
                time.time() - self.time_since_holding_attack >= brawler_info['hold_attack'] + self.seconds_to_hold_attack_after_reaching_max:
            self.attack(touch_up=True, touch_down=False)
            self.time_since_holding_attack = None

        safe_range, attack_range, super_range = self.get_brawler_range(brawler)
        player_pos = self.get_player_pos(player_data)
        raw_enemy_data = enemy_data or []
        attack_excluded_boxes = self.build_attack_excluded_boxes(player_data, teammate_data)
        enemy_data, friendly_excluded_targets = self.sanitize_enemy_targets(raw_enemy_data, attack_excluded_boxes)
        if friendly_excluded_targets and (getattr(self, "attack_decision_debug", False) or getattr(self, "combat_brain_debug", False)):
            self._aimlog(f"friendly_fire_guard excluded={friendly_excluded_targets}")

        enemy_coords = None
        enemy_distance = None
        follow_teammates = self.showdown_playstyle_mode in ("follow", "follower", "team", "teammate", "teammates")

        # Fog override is applied uniformly at the end so it works for all
        # movement sources. In teammate-follow mode this check is deliberately
        # unthrottled; following a teammate toward smoke is worse than spending
        # a few extra milliseconds checking the screen.
        self._fog_check_counter += 1
        if follow_teammates or self._fog_check_counter >= self.fog_check_every_n_frames:
            self._fog_threat_cached = self.detect_fog_threat(self.current_frame, player_pos)
            self._fog_direction_escape_cached = self.detect_fog_direction_escape(self.current_frame, player_pos)
            self._fog_check_counter = 0
        fog_flee_angle = self._fog_direction_escape_cached or self._fog_threat_cached
        flicker_active, flicker_confidence = self.detect_player_bar_flicker(
            self.current_frame,
            player_data,
            time.time(),
        )
        health_ratio = self.estimate_player_health_ratio(self.current_frame, player_data)
        heal_active, heal_reason = self.update_heal_state(
            health_ratio,
            flicker_active,
            flicker_confidence,
            time.time(),
        )

        # --- No enemy in sight: follow teammate or roam ---
        if not self.is_there_enemy(enemy_data):
            if follow_teammates and (teammate_data or self.current_frame is not None):
                if teammate_data:
                    vlog(f"no enemy → follow teammate ({len(teammate_data)} visible)")
                else:
                    vlog("no enemy -> follow teammate marker if visible")
                angle = self.showdown_follow_teammate(player_data, teammate_data, walls)
            else:
                vlog("no enemy → hide/roam")
                angle = self.showdown_roam(player_data, walls)
        else:
            enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
            if enemy_coords is None:
                if follow_teammates and (teammate_data or self.current_frame is not None):
                    if teammate_data:
                        vlog("enemy detected but unreachable → follow teammate")
                    else:
                        vlog("enemy detected but unreachable -> follow teammate marker if visible")
                    angle = self.showdown_follow_teammate(player_data, teammate_data, walls)
                else:
                    vlog("enemy detected but unreachable → hide/roam")
                    angle = self.showdown_roam(player_data, walls)
            else:
                # --- Compute exact angle toward/away from enemy, then wall-avoid ---
                direction_x = enemy_coords[0] - player_pos[0]
                direction_y = enemy_coords[1] - player_pos[1]
                toward_angle = self.angle_from_direction(direction_x, direction_y)
                now_t = time.time()
                if self.lead_shots_enabled:
                    self.enemy_velocity = self.track_enemy_velocity(enemy_coords, now_t)
                else:
                    self.enemy_velocity = (0.0, 0.0)
                    self.enemy_velocity_confidence = 0.0

                if enemy_distance > safe_range:
                    desired = toward_angle
                    vlog(f"enemy detected → approach desired={desired:.1f}° (dist={int(enemy_distance)}px, safe={safe_range}px)")
                    if self.approach_flank_blend > 0 and enemy_distance > safe_range * 1.2:
                        flank_angle = (toward_angle + 90 * self._strafe_side) % 360
                        desired = self.blend_angles(desired, flank_angle, self.approach_flank_blend)
                        vlog(f"approach flank blend -> desired={desired:.1f}°")
                else:
                    desired = self.angle_opposite(toward_angle)
                    vlog(f"enemy too close → retreat desired={desired:.1f}° (dist={int(enemy_distance)}px, safe={safe_range}px)")
                    if self.multi_enemy_flee_weight > 0 and enemy_data and len(enemy_data) > 1:
                        mass_x = sum(self.get_enemy_pos(enemy)[0] for enemy in enemy_data) / len(enemy_data)
                        mass_y = sum(self.get_enemy_pos(enemy)[1] for enemy in enemy_data) / len(enemy_data)
                        mass_dx = mass_x - player_pos[0]
                        mass_dy = mass_y - player_pos[1]
                        if math.hypot(mass_dx, mass_dy) > 10:
                            mass_flee = self.angle_opposite(self.angle_from_direction(mass_dx, mass_dy))
                            desired = self.blend_angles(desired, mass_flee, self.multi_enemy_flee_weight)
                            vlog(f"multi-enemy flee blend -> desired={desired:.1f}°")

                if (
                        self.strafe_enabled
                        and fog_flee_angle is None
                        and safe_range < enemy_distance <= attack_range
                ):
                    strafe_angle = self.get_strafe_angle(toward_angle, now_t, enemy_distance, safe_range)
                    desired = self.blend_angles(desired, strafe_angle, self.strafe_blend)
                    vlog(f"strafe blend → desired={desired:.1f}°")
                elif (
                        self.strafe_enabled
                        and fog_flee_angle is None
                        and enemy_distance <= safe_range
                        and self.retreat_strafe_fraction > 0
                ):
                    strafe_angle = self.get_strafe_angle(toward_angle, now_t, enemy_distance, safe_range)
                    desired = self.blend_angles(
                        desired,
                        strafe_angle,
                        self.strafe_blend * self.retreat_strafe_fraction,
                    )
                    vlog(f"retreat strafe blend -> desired={desired:.1f}°")

                if self.strafe_enabled and fog_flee_angle is None and enemy_distance <= attack_range:
                    desired = self.apply_combat_dodge(desired, toward_angle, now_t, enemy_distance, safe_range)
                    vlog(f"combat dodge blend -> desired={desired:.1f}°")

                if (follow_teammates and teammate_data and enemy_distance > attack_range):
                    closest_teammate, teammate_distance = self.get_closest_teammate(player_data, teammate_data)
                    if closest_teammate is not None and teammate_distance > self.teammate_follow_step_distance:
                        team_angle = self.angle_from_direction(
                            closest_teammate[0] - player_pos[0],
                            closest_teammate[1] - player_pos[1],
                        )
                        team_weight = self.teammate_combat_bias
                        if self.trio_grouping_enabled and teammate_distance > self.teammate_combat_regroup_distance:
                            team_weight = max(team_weight, 0.85)
                        desired = self.blend_angles(desired, team_angle, team_weight)
                        vlog(f"combat teammate pull -> desired={desired:.1f}° (team dist={int(teammate_distance)}px, weight={team_weight:.2f})")

                angle = self.find_best_angle(player_pos, desired, walls)
                vlog(f"showdown: movement angle={angle:.1f}° (desired={desired:.1f}°)")

        if heal_active and fog_flee_angle is None:
            angle, heal_angle_reason = self.choose_heal_retreat_angle(player_pos, enemy_data, teammate_data, walls, angle)
            self._manslog(
                f"heal_retreat angle={angle:.1f} reason={heal_reason} angle_reason={heal_angle_reason} "
                f"health_ratio={health_ratio}"
            )

        if fog_flee_angle is None:
            if flicker_active:
                retreat_angle, retreat_reason = self.choose_flicker_retreat_angle(
                    player_pos,
                    enemy_data,
                    teammate_data,
                    walls,
                    safe_range,
                    attack_range,
                    angle,
                    flicker_confidence,
                    fog_flee_angle=fog_flee_angle,
                )
                if retreat_angle is not None:
                    angle = retreat_angle
                    attack_suppression = 0.0
                    if not (enemy_distance is not None and enemy_distance <= attack_range * 0.72):
                        attack_suppression = min(0.25, self.flicker_retreat_hold_seconds)
                    self._suppress_attack_until = max(
                        getattr(self, "_suppress_attack_until", 0.0),
                        time.time() + attack_suppression,
                    )
                    vlog(
                        f"flicker retreat -> angle={angle:.1f} "
                        f"confidence={flicker_confidence:.2f}"
                    )
                else:
                    self._manslog(
                        f"flicker retreat_allowed=False reason={retreat_reason} "
                        f"flicker_confidence={flicker_confidence:.2f}"
                    )

            angle, dodge_info = self.choose_tactical_dodge_angle(
                angle,
                player_pos,
                enemy_data,
                teammate_data,
                walls,
                safe_range,
                attack_range,
                time.time(),
                flicker_active=flicker_active,
                projectile_data=projectile_data,
            )
        else:
            dodge_info = {"mode": "no_dodge", "threat": 0.0}

        if (
                follow_teammates
                and fog_flee_angle is None
                and self.angle_points_into_fog(self.current_frame, player_pos, angle)
        ):
            fog_flee_angle = self.angle_opposite(angle)
            vlog(f"showdown: follow path points into fog -> fallback escape={fog_flee_angle:.1f}°")

        # --- Fog proximity override ---
        # If trusted fog is close, replace movement with a flee angle. Attack
        # block below still fires independently based on enemy_distance.
        jump_pad_flee_angle = fog_flee_angle
        if (
                jump_pad_flee_angle is None
                and jump_pads
                and self.jump_pad_detection_enabled
                and (not self.jump_pad_escape_requires_edge or self.is_player_near_map_edge(player_pos))
                and not self.has_close_teammate_for_jump_escape(player_pos, teammate_data)
        ):
            jump_pad_flee_angle = self.detect_jump_pad_smoke_escape(self.current_frame, player_pos)

        if jump_pad_flee_angle is not None:
            jump_pad_angle = self.find_jump_pad_escape_angle(
                player_pos,
                jump_pads or [],
                walls,
                jump_pad_flee_angle,
                teammate_data=teammate_data,
            )
            if jump_pad_angle is not None:
                angle = jump_pad_angle
                vlog(f"showdown: fog override -> jump pad angle={angle:.1f}°")
            elif fog_flee_angle is not None:
                angle = self.find_best_angle(player_pos, fog_flee_angle, walls)
                vlog(f"showdown: fog override → angle={angle:.1f}°")

        angle, intent_attack_allowed, movement_intent = self.apply_movement_intent(
            base_angle=angle,
            player_pos=player_pos,
            enemy_coords=enemy_coords,
            enemy_distance=enemy_distance,
            enemy_data=enemy_data,
            teammate_data=teammate_data,
            walls=walls,
            safe_range=safe_range,
            attack_range=attack_range,
            fog_flee_angle=jump_pad_flee_angle,
            health_ratio=health_ratio,
            heal_active=heal_active,
            projectile_data=projectile_data,
            current_time=time.time(),
        )

        combat_brain_active = bool(getattr(self, "combat_brain_enabled", False))
        health_state = self.build_health_state(
            health_ratio,
            heal_active,
            flicker_active=flicker_active,
            flicker_confidence=flicker_confidence,
        )
        projectile_threat_active = False
        projectile_threat = None
        if getattr(self, "projectile_dodge_enabled", False):
            projectile_threat = self.select_incoming_projectile_threat(
                projectile_data or [],
                player_pos,
                time.time(),
            )
            projectile_threat_active = projectile_threat is not None
        safety_result = SafetyResult(angle=angle, safe=True, status="not_checked")
        if combat_brain_active and getattr(self, "wall_angle_fail_escape_enabled", True):
            safe_angle, wall_safe, wall_status = self.find_best_angle_status(
                player_pos,
                angle,
                walls,
            )
            safety_result = SafetyResult(
                angle=safe_angle,
                safe=bool(wall_safe),
                status=wall_status,
                reasons=[] if wall_safe else ["no_safe_movement_angle"],
            )
            angle = safe_angle
            if not wall_safe:
                intent_attack_allowed = False
                self._combatlog(
                    f"wall_escape_triggered status={wall_status} angle={angle:.1f} "
                    f"health={health_ratio} enemy_distance={None if enemy_distance is None else int(enemy_distance)}"
                )
                self.save_combat_snapshot(
                    "wall_no_safe_angle",
                    extra={
                        "angle": angle,
                        "wall_status": wall_status,
                        "health_ratio": health_ratio,
                        "enemy_distance": enemy_distance,
                    },
                    brawler=brawler,
                )

        target_score = self.choose_combat_target_score(
            player_pos,
            enemy_data,
            walls,
            brawler,
            safe_range,
            attack_range,
            attack_decision=None,
        )
        closest_teammate, teammate_distance = self.get_closest_teammate(
            (player_pos[0], player_pos[1], player_pos[0], player_pos[1]),
            teammate_data,
        )
        teammate_near = closest_teammate is not None and teammate_distance <= getattr(self, "teammate_combat_regroup_distance", 650)
        teammate_angle = None
        if closest_teammate is not None:
            teammate_angle = self.angle_from_direction(closest_teammate[0] - player_pos[0], closest_teammate[1] - player_pos[1])
        base_intent_mode = movement_intent.mode if movement_intent is not None else None
        profile = self.get_combat_profile(brawler)
        preferred_distance = float(profile.get(
            "preferred_distance_px",
            safe_range * float(profile.get("preferred_distance_multiplier", 1.0)),
        ))
        commit_distance = float(profile.get("commit_distance_px", min(attack_range, safe_range * 1.35)))
        adaptation = self.current_tactical_adaptation(time.time())
        try:
            attack_ignores_walls = self.can_attack_through_walls(brawler, "attack", self.brawlers_info)
        except (KeyError, TypeError):
            attack_ignores_walls = False
        projectile_dodge_angle = None
        if dodge_info and dodge_info.get("mode") == "projectile_dodge":
            projectile_dodge_angle = angle
        combat_intent = choose_combat_intent(
            frame=CombatFrame(
                player_pos=player_pos,
                enemy_data=enemy_data or [],
                teammate_data=teammate_data or [],
                walls=walls or [],
                projectiles=projectile_data or [],
                health=health_state,
                desired_angle=angle,
                current_mode=base_intent_mode,
                safe_range=safe_range,
                attack_range=attack_range,
                fog_danger=jump_pad_flee_angle is not None,
                projectile_incoming=projectile_threat_active,
                wall_trap=not safety_result.safe,
                teammate_near=teammate_near,
                suppress_active=time.time() < getattr(self, "_suppress_attack_until", 0.0),
                projectile_dodge_angle=projectile_dodge_angle,
                teammate_angle=teammate_angle,
                preferred_distance=preferred_distance,
                commit_distance=commit_distance,
                aggression_penalty=adaptation["aggression_penalty"],
                fire_threshold_delta=adaptation["fire_threshold_delta"],
            ),
            target=target_score,
            safety=safety_result,
            defensive_gate_enabled=bool(combat_brain_active and getattr(self, "defensive_attack_gate_enabled", True)),
            panic_shot_range=float(self.get_combat_profile(brawler).get("panic_shot_range", getattr(self, "panic_shot_range", 150))),
            tactical_planner_enabled=bool(combat_brain_active and getattr(self, "tactical_planner_enabled", False)),
            angle_samples=getattr(self, "tactical_angle_samples", 16),
            is_angle_blocked=lambda candidate: self.is_path_blocked_angle(player_pos, candidate, walls),
            points_into_fog=lambda candidate: self.angle_points_into_fog(self.current_frame, player_pos, candidate),
            projectile_danger_for_angle=lambda candidate: max(
                0.0,
                1.0 - min(1.0, abs(score_projectile_dodge_angle(candidate, projectile_threat)[0]) / 4.0),
            ) if projectile_threat else 0.0,
            line_of_sight_after_move=lambda candidate: (
                bool(target_score and target_score.center)
                and (
                    attack_ignores_walls
                    or not self.walls_block_line_of_sight(
                        (
                            player_pos[0] + math.cos(math.radians(candidate)) * 120.0,
                            player_pos[1] + math.sin(math.radians(candidate)) * 120.0,
                        ),
                        target_score.center,
                        walls,
                    )
                )
            ),
            survival_score_min_to_commit=getattr(self, "survival_score_min_to_commit", 0.62),
            kill_confirm_score_threshold=getattr(self, "kill_confirm_score_threshold", 0.68),
        )
        angle = combat_intent.movement_angle if combat_intent.movement_angle is not None else angle
        if combat_brain_active:
            intent_attack_allowed = bool(intent_attack_allowed and combat_intent.attack_allowed)
            self._combatlog(
                f"intent mode={combat_intent.mode} attack_allowed={combat_intent.attack_allowed} "
                f"attack_denied={combat_intent.attack_denied_reason or 'none'} threat={combat_intent.threat.score:.2f} "
                f"reasons={','.join(combat_intent.reasons) if combat_intent.reasons else 'none'}"
            )
            self.record_combat_decision({
                "mode": combat_intent.mode,
                "angle": angle,
                "health_ratio": health_ratio,
                "health_source": health_state.source,
                "threat_score": combat_intent.threat.score,
                "threat_reasons": combat_intent.threat.reasons,
                "target": None if target_score is None else {
                    "bbox": target_score.bbox,
                    "distance": target_score.distance,
                    "score": target_score.score,
                    "line_of_sight": target_score.line_of_sight,
                    "in_attack_range": target_score.in_attack_range,
                    "close_threat": target_score.close_threat,
                    "stale": target_score.stale,
                },
                "attack_allowed": combat_intent.attack_allowed,
                "attack_denied": combat_intent.attack_denied_reason,
                "safety_status": safety_result.status,
                "projectile_incoming": projectile_threat_active,
                "fog_danger": jump_pad_flee_angle is not None,
                "tactical_objective": combat_intent.tactical_plan.objective,
                "tactical_survival": combat_intent.tactical_plan.survival_score,
                "tactical_engagement": combat_intent.tactical_plan.engagement_score,
                "kill_confirm_score": combat_intent.tactical_plan.kill_confirm_score,
                "rejected_angles": combat_intent.tactical_plan.rejected_angles,
                "friendly_excluded_targets": friendly_excluded_targets,
            })

        # --- Skills (only when an attackable enemy was found) ---
        if enemy_coords is None:
            return angle

        vlog(f"showdown movement → angle={angle:.1f}°")

        attack_decision = self.choose_attack_decision(
            brawler,
            player_pos,
            enemy_data,
            walls,
            attack_range=attack_range,
            current_time=time.time(),
            excluded_boxes=attack_excluded_boxes,
        )
        if attack_decision.in_range:
            vlog(
                f"enemy in attack window (dist={int(attack_decision.distance or enemy_distance)}px, range={attack_range}px), "
                f"line_of_sight={attack_decision.los_status} close_threat={attack_decision.close_threat} "
                f"close_threshold={int(self.close_attack_threat_threshold(attack_range))}"
            )
        else:
            vlog(
                f"enemy out of attack window (dist={int(attack_decision.distance or enemy_distance)}px, "
                f"range={attack_range}px)"
            )

        intent_mode = combat_intent.mode if combat_brain_active else (movement_intent.mode if movement_intent is not None else None)
        if intent_mode is None:
            if heal_active:
                intent_mode = "retreat_heal"
            elif jump_pad_flee_angle is not None:
                intent_mode = "escape_fog"
            else:
                intent_mode = "strafe_attack_lane" if attack_decision.should_fire else "approach"
        attack_target_score = self.choose_combat_target_score(
            player_pos,
            enemy_data,
            walls,
            brawler,
            safe_range,
            attack_range,
            attack_decision=attack_decision,
        )
        if attack_target_score is not None:
            target_score = attack_target_score
        ability_plan = None
        ability_brain_active = bool(combat_brain_active and getattr(self, "ability_brain_enabled", False))
        if ability_brain_active:
            ability_plan = self.choose_combat_ability_plan(
                brawler,
                brawler_info,
                player_pos,
                enemy_data,
                teammate_data,
                walls,
                safe_range,
                attack_range,
                target_score,
                health_state,
                intent_mode,
                fog_flee_angle=jump_pad_flee_angle,
                projectile_incoming=projectile_threat_active or intent_mode == "dodge_projectile",
                tactical_plan=combat_intent.tactical_plan,
            )
            combat_intent.ability_plan = ability_plan
        else:
            self.try_use_super_on_enemy(brawler, brawler_info, player_pos, enemy_coords, enemy_distance, walls)

        ability_used = self.execute_ability_plan(ability_plan)
        attack_denied_by = None
        if not attack_decision.should_fire:
            attack_denied_by = attack_decision.denied_by or attack_decision.reason
        else:
            suppress_active = time.time() < getattr(self, "_suppress_attack_until", 0.0)
            gate_mode = intent_mode
            if not intent_attack_allowed and gate_mode not in {"escape_fog", "wall_escape", "unstuck", "retreat_heal", "dodge_projectile"}:
                gate_mode = "dodge_projectile"
            allowed, gate_reason = choose_attack_gate(
                mode=gate_mode,
                target=target_score,
                health=health_state,
                defensive_gate_enabled=bool(combat_brain_active and getattr(self, "defensive_attack_gate_enabled", True)),
                panic_shot_range=float(self.get_combat_profile(brawler).get("panic_shot_range", getattr(self, "panic_shot_range", 150))),
                suppress_active=suppress_active,
            )
            if combat_brain_active and combat_intent.attack_denied_reason and not combat_intent.attack_allowed:
                allowed = False
                gate_reason = combat_intent.attack_denied_reason
            if not allowed:
                attack_denied_by = gate_reason or "movement_intent_blocks_attack"
            elif gate_reason == "panic_shot" and getattr(self, "attack_decision_debug", False):
                self._aimlog(
                    "attack_defensive_gate_override "
                    f"reason=panic_shot heal_active={heal_active} intent_attack_allowed={intent_attack_allowed} "
                    f"selected_distance={int(attack_decision.distance or enemy_distance)} attack_range={int(attack_range)}"
                )

        self.update_tactical_adaptation(
            combat_intent,
            flicker_active=flicker_active,
            attack_denied_by=attack_denied_by,
            current_time=time.time(),
        )
        if combat_brain_active:
            self.record_combat_decision({
                "mode": combat_intent.mode,
                "tactical_objective": combat_intent.tactical_plan.objective,
                "final_angle": angle,
                "attack_denied": attack_denied_by,
                "ability_used": ability_used,
                "ability_denies": [] if not ability_plan else ability_plan.denies,
                "ability_super": None if not ability_plan else ability_plan.super_reason,
                "ability_gadget": None if not ability_plan else ability_plan.gadget_reason,
                "ability_hypercharge": None if not ability_plan else ability_plan.hypercharge_reason,
                "rejected_angles": combat_intent.tactical_plan.rejected_angles,
                "final_tactical_reasons": combat_intent.tactical_plan.reasons,
                "raw_target_count": len(raw_enemy_data or []),
                "sanitized_target_count": len(enemy_data or []),
                "friendly_excluded_targets": friendly_excluded_targets,
            })

        if attack_denied_by:
            self.log_attack_decision(
                attack_decision,
                attack_range,
                input_busy=attack_denied_by in {"defensive_retreat_active", "movement_intent_blocks_attack"},
                denied_by=attack_denied_by,
            )
        else:
            if (not ability_brain_active) and self.should_use_gadget_on_enemy(brawler, player_data, enemy_data, walls):
                if self.use_gadget():
                    self.time_since_gadget_checked = time.time()
                    self.clear_ability_ready("gadget")

            skip_basic_attack = bool(ability_used and ability_plan and ability_plan.use_super)
            if skip_basic_attack:
                self._aimlog("attack_skipped reason=super_ability_plan_executed")
            elif not must_brawler_hold_attack:
                self.auto_aim_attack(
                    brawler,
                    player_pos,
                    enemy_data,
                    walls,
                    attack_range=attack_range,
                    decision=attack_decision,
                    excluded_boxes=attack_excluded_boxes,
                )
            else:
                if self.time_since_holding_attack is None:
                    self.time_since_holding_attack = time.time()
                    self.attack(touch_up=False, touch_down=True)
                elif time.time() - self.time_since_holding_attack >= self.brawlers_info[brawler]['hold_attack']:
                    self.attack(touch_up=True, touch_down=False)
                    self.time_since_holding_attack = None

        return angle

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        if self.walls_block_line_of_sight(player_pos, enemy_pos, walls):
            return False
        return True

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        player_pos_x, player_pos_y = player_coords
        closest_hittable_distance = float('inf')
        closest_unhittable_distance = float('inf')
        closest_hittable = None
        closest_unhittable = None
        for enemy in enemy_data:
            enemy_pos = self.get_enemy_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            if self.is_enemy_hittable((player_pos_x, player_pos_y), enemy_pos, walls, skill_type):
                if distance < closest_hittable_distance:
                    closest_hittable_distance = distance
                    closest_hittable = [enemy_pos, distance]
            else:
                if distance < closest_unhittable_distance:
                    closest_unhittable_distance = distance
                    closest_unhittable = [enemy_pos, distance]
        if closest_hittable:
            return closest_hittable
        elif closest_unhittable:
            return closest_unhittable

        return None, None

    @staticmethod
    def _count_mask_pixels(hsv_roi, lower, upper):
        if hsv_roi.size == 0:
            return 0
        mask = cv2.inRange(hsv_roi, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        return int(cv2.countNonZero(mask))

    def _entity_team_color_scores(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, self.normalize_box(box))
        pad_x = max(18, int((x2 - x1) * 0.45))
        pad_y = max(24, int((y2 - y1) * 0.75))
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return 0, 0
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        # Friendly self/teammate overlays are bright green; enemy HP/name UI is red/orange.
        green = self._count_mask_pixels(hsv, (35, 80, 80), (85, 255, 255))
        red = (
            self._count_mask_pixels(hsv, (0, 80, 80), (14, 255, 255))
            + self._count_mask_pixels(hsv, (170, 80, 80), (179, 255, 255))
        )
        return green, red

    def _entity_marker_color_scores(self, frame, box):
        frame_id = id(frame)
        cache_key = tuple(map(int, self.normalize_box(box)))
        if getattr(self, "_entity_marker_cache_frame_id", None) == frame_id:
            cache = getattr(self, "_entity_marker_score_cache", {})
            if cache_key in cache:
                self._perf_entity_marker_cache_hits = getattr(self, "_perf_entity_marker_cache_hits", 0) + 1
                return cache[cache_key]
        else:
            self._entity_marker_cache_frame_id = frame_id
            self._entity_marker_score_cache = {}

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = cache_key
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        empty = {"enemy": 0.0, "teammate": 0.0, "self": 0.0, "trusted": False}
        if bw < 12 or bh < 16:
            self._entity_marker_score_cache[cache_key] = empty
            return empty

        pad_x = max(10, int(bw * 0.42))
        rx1 = max(0, x1 - pad_x)
        rx2 = min(w, x2 + pad_x)
        ry1 = max(0, int(y1 + bh * 0.48))
        below_ratio = max(0.0, min(0.75, getattr(self, "entity_marker_below_box_ratio", 0.22)))
        ry2 = min(h, int(y2 + bh * below_ratio) + max(8, int(bh * 0.18)))
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            self._entity_marker_score_cache[cache_key] = empty
            return empty

        self._perf_entity_marker_scores = getattr(self, "_perf_entity_marker_scores", 0) + 1
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        area = max(1, roi.shape[0] * roi.shape[1])
        red = (
            self._count_mask_pixels(hsv, (0, 75, 70), (10, 255, 255))
            + self._count_mask_pixels(hsv, (170, 75, 70), (179, 255, 255))
        )
        orange = self._count_mask_pixels(hsv, (8, 70, 80), (26, 255, 255))
        yellow = self._count_mask_pixels(hsv, (24, 60, 95), (45, 255, 255))
        blue = (
            self._count_mask_pixels(hsv, (88, 60, 70), (112, 255, 255))
            + self._count_mask_pixels(hsv, (106, 55, 70), (136, 255, 255))
        )
        green = self._count_mask_pixels(hsv, (36, 70, 70), (88, 255, 255))

        enemy_pixels = red + orange + yellow
        teammate_pixels = blue
        enemy = enemy_pixels / area
        teammate = teammate_pixels / area
        self_score = green / area
        strongest = max(enemy, teammate, self_score)
        trusted = (
            strongest >= getattr(self, "entity_marker_min_ratio", 0.012)
            and max(enemy_pixels, teammate_pixels, green) >= getattr(self, "entity_marker_min_pixels", 12)
            and (
                enemy >= getattr(self, "entity_marker_enemy_min_ratio", 0.012)
                or teammate >= getattr(self, "entity_marker_blue_min_ratio", 0.012)
                or self_score >= getattr(self, "entity_marker_min_ratio", 0.012)
            )
        )
        result = {
            "enemy": enemy,
            "teammate": teammate,
            "self": self_score,
            "trusted": trusted,
        }
        self._entity_marker_score_cache[cache_key] = result
        return result

    def _marker_role(self, frame, box):
        scores = self._entity_marker_color_scores(frame, box)
        if not scores["trusted"]:
            return None
        ordered = sorted(
            (("enemy", scores["enemy"]), ("teammate", scores["teammate"]), ("self", scores["self"])),
            key=lambda item: item[1],
            reverse=True,
        )
        if ordered[0][1] < ordered[1][1] * getattr(self, "entity_marker_decision_margin", 1.25):
            return None
        return ordered[0][0]

    def select_own_player_box(self, frame, player_boxes):
        if not player_boxes:
            return None, []
        h, w = frame.shape[:2]
        screen_center = (w * 0.5, h * 0.54)
        radius = max(1.0, self.player_center_bias_radius * self.window_controller.scale_factor)
        scored = []
        for box in player_boxes:
            cx, cy = self.get_player_pos(box)
            center_dist = math.hypot(cx - screen_center[0], cy - screen_center[1])
            center_score = max(0.0, 1.0 - center_dist / radius)
            green, red = self._entity_team_color_scores(frame, box)
            color_score = green * self.player_green_pixel_weight - red * self.player_red_pixel_penalty
            scored.append((center_score + color_score, center_dist, box))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        own_box = scored[0][2]
        rejected = [item[2] for item in scored[1:]]
        if visual_debug and rejected:
            print(f"[DBG] own player selected: {own_box}; reclassified {len(rejected)} player boxes as enemy")
        return own_box, rejected

    def stabilize_entity_roles(self, frame, data):
        players = data.get("player") or []
        own_box, rejected_players = self.select_own_player_box(frame, players)
        if own_box is not None:
            data["player"] = [own_box]
        if rejected_players:
            for box in rejected_players:
                role = self._marker_role(frame, box)
                if role == "teammate":
                    data.setdefault("teammate", [])
                    data["teammate"].append(box)
                elif role == "self":
                    data.setdefault("player", [])
                    data["player"].append(box)
                else:
                    data.setdefault("enemy", [])
                    data["enemy"].append(box)

        for role in ("enemy", "teammate"):
            refined = {"enemy": [], "teammate": [], "player": []}
            for box in data.get(role) or []:
                marker_role = self._marker_role(frame, box)
                target_role = marker_role if marker_role in refined else role
                refined[target_role].append(box)
            data[role] = refined[role]
            if refined["enemy"] and role != "enemy":
                data.setdefault("enemy", [])
                data["enemy"].extend(refined["enemy"])
            if refined["teammate"] and role != "teammate":
                data.setdefault("teammate", [])
                data["teammate"].extend(refined["teammate"])
            if refined["player"]:
                data.setdefault("player", [])
                data["player"].extend(refined["player"])

        if len(data.get("player") or []) > 1:
            own_box, extra_players = self.select_own_player_box(frame, data["player"])
            data["player"] = [own_box] if own_box is not None else []
            if extra_players:
                data.setdefault("enemy", [])
                data["enemy"].extend(extra_players)
        return data

    @staticmethod
    def _box_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
        inter_x1 = max(min(ax1, ax2), min(bx1, bx2))
        inter_y1 = max(min(ay1, ay2), min(by1, by2))
        inter_x2 = min(max(ax1, ax2), max(bx1, bx2))
        inter_y2 = min(max(ay1, ay2), max(by1, by2))
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        area_a = max(0.0, abs(ax2 - ax1) * abs(ay2 - ay1))
        area_b = max(0.0, abs(bx2 - bx1) * abs(by2 - by1))
        union = area_a + area_b - intersection
        return 0.0 if union <= 0 else intersection / union

    def merge_retry_entity_data(self, data, retry_data):
        merged = {key: [list(box) for box in boxes] for key, boxes in (data or {}).items() if boxes}
        existing_boxes = [
            box
            for boxes in merged.values()
            for box in boxes
        ]
        added = 0
        for role in ("player", "enemy", "teammate"):
            for box in retry_data.get(role) or []:
                if any(self._box_iou(box, existing) >= 0.55 for existing in existing_boxes):
                    continue
                merged.setdefault(role, []).append(box)
                existing_boxes.append(box)
                added += 1
        if visual_debug and added:
            print(f"[DBG] entity retry merged {added} low-confidence boxes")
        return merged

    def get_main_data(self, frame):
        data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_confidence)
        should_retry = (
            self.entity_detection_retry_confidence < self.entity_detection_confidence
            and (
                not data.get("player")
                or (getattr(self, "entity_retry_when_enemy_missing", True) and not data.get("enemy"))
            )
        )
        if should_retry:
            self._perf_entity_retry_count = getattr(self, "_perf_entity_retry_count", 0) + 1
            retry_data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_retry_confidence)
            if retry_data.get("player") or retry_data.get("enemy") or retry_data.get("teammate"):
                if visual_debug:
                    print(
                        "[DBG] entities recovered with lower entity threshold "
                        f"{self.entity_detection_retry_confidence:.2f}"
                    )
                data = self.merge_retry_entity_data(data, retry_data)
        return self.stabilize_entity_roles(frame, data)

    def is_path_blocked(self, player_pos, move_direction, walls, distance=None):  # Increased distance
        if distance is None:
            distance = self.TILE_SIZE*self.window_controller.scale_factor
        dx, dy = 0, 0
        if 'w' in move_direction.lower():
            dy -= distance
        if 's' in move_direction.lower():
            dy += distance
        if 'a' in move_direction.lower():
            dx -= distance
        if 'd' in move_direction.lower():
            dx += distance
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        return self.walls_block_line_of_sight(player_pos, new_pos, walls, padding=self.wall_path_padding)

    def is_path_blocked_angle(self, player_pos, angle_degrees, walls, distance=None):
        """Check if the path in the given angle direction is blocked by walls.

        Uses two probe distances (half-tile and full-tile) so that walls that
        start very close to the player are also detected.
        """
        if distance is None:
            distance = self.TILE_SIZE * self.window_controller.scale_factor
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        max_probe = max(1.0, self.wall_path_probe_tiles)
        probes = (distance * 0.5, distance, distance * max_probe)
        for d in probes:
            new_pos = (player_pos[0] + cos_a * d, player_pos[1] + sin_a * d)
            if self.walls_block_line_of_sight(player_pos, new_pos, walls, padding=self.wall_path_padding):
                return True
        return False

    def find_best_angle(self, player_pos, desired_angle, walls, sweep_range=160, step=10):
        """Find the closest unblocked angle to desired_angle within ±sweep_range degrees.

        Sweeps outward from the desired angle in alternating left/right steps so
        the first hit is always the least deviation from the goal direction.
        Returns desired_angle unchanged if no walls (or no clear path found).
        """
        angle, _safe, _reason = self.find_best_angle_status(player_pos, desired_angle, walls, sweep_range=sweep_range, step=step)
        return angle

    def find_best_angle_status(self, player_pos, desired_angle, walls, sweep_range=160, step=10):
        """Return (angle, safe, reason) for wall-aware movement selection."""
        if not self.is_path_blocked_angle(player_pos, desired_angle, walls):
            return desired_angle, True, "clear"

        for offset in range(step, sweep_range + 1, step):
            for sign in (1, -1):
                candidate = (desired_angle + sign * offset) % 360
                if not self.is_path_blocked_angle(player_pos, candidate, walls):
                    return candidate, True, "adjusted"

        return desired_angle, False, "no_safe_angle"

    @staticmethod
    def validate_game_data(data):
        incomplete = False
        if not data.get("player"):
            incomplete = True  # This is required so track_no_detections can also keep track if enemy is missing

        if "enemy" not in data.keys():
            data['enemy'] = None

        if "teammate" not in data.keys():
            data['teammate'] = None

        if 'wall' not in data.keys() or not data['wall']:
            data['wall'] = []

        if 'jump_pad' not in data.keys() or not data['jump_pad']:
            data['jump_pad'] = []
        if 'projectile' not in data.keys() or not data['projectile']:
            data['projectile'] = []
        if 'bullet' in data and data['bullet']:
            data['projectile'].extend(data['bullet'])

        return False if incomplete else data

    def track_no_detections(self, data):
        if not data:
            data = {
                "enemy": None,
                "player": None,
                "teammate": None,
            }
        for key in self.time_since_detections:
            if key in data and data[key]:
                self.time_since_detections[key] = time.time()

    def do_movement(self, movement):
        if isinstance(movement, float):
            if not self.enable_joystick_movement or self.movement_input_mode == "wasd":
                movement = [
                    "d", "sd", "s", "sa", "a", "wa", "w", "wd"
                ][int((float(movement) % 360 + 22.5) / 45) % 8]
            else:
                # Analog joystick path: movement is an angle in degrees
                try:
                    self.window_controller.move_joystick_angle(movement)
                    self.keys_hold = []
                    return
                except Exception as exc:
                    self._movement_fallback_to_wasd = True
                    print(f"[MOVE] fallback_to_wasd reason={exc}")
                    movement = [
                        "d", "sd", "s", "sa", "a", "wa", "w", "wd"
                    ][int((float(movement) % 360 + 22.5) / 45) % 8]

        movement = str(movement or "").lower()
        angle = movement_keys_to_angle(movement)
        use_joystick = (
            self.enable_joystick_movement
            and not self._movement_fallback_to_wasd
            and self.movement_input_mode in ("auto", "joystick")
            and angle is not None
        )
        if use_joystick:
            try:
                self.window_controller.move_joystick_angle(angle)
                self.keys_hold = []
                if visual_debug:
                    print(
                        f"[MOVE] movement_input_mode={self.movement_input_mode} "
                        f"movement_vector={movement} joystick_angle={angle:.1f}"
                    )
                return
            except Exception as exc:
                self._movement_fallback_to_wasd = True
                print(f"[MOVE] fallback_to_wasd reason={exc}")
        # Legacy WASD path
        keys_to_keyDown = []
        keys_to_keyUp = []
        for key in ['w', 'a', 's', 'd']:
            if key in movement:
                keys_to_keyDown.append(key)
            else:
                keys_to_keyUp.append(key)

        if keys_to_keyDown:
            self.window_controller.keys_down(keys_to_keyDown)

        self.window_controller.keys_up(keys_to_keyUp)
        self.keys_hold = keys_to_keyDown

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        safe_range, attack_range, super_range = self.brawler_ranges[brawler]
        multiplier = max(0.75, min(1.35, float(getattr(self, "adaptive_safe_range_multiplier", 1.0))))
        return int(safe_range * multiplier), attack_range, super_range

    def _debounce_angle(self, angle: float, threshold_deg: float = 10.0) -> float:
        """Suppress small angle changes and smooth accepted turns.

        Only adopts the new angle if it differs by more than threshold_deg
        from the last committed angle, OR if no angle was committed yet.
        """
        if self.last_movement is None or not isinstance(self.last_movement, float):
            self.last_movement = angle
            self.last_movement_time = time.time()
            return angle

        diff = abs((angle - self.last_movement + 180) % 360 - 180)
        if diff > threshold_deg:
            if self.angle_smooth_factor > 0:
                self.last_movement = self.blend_angles(angle, self.last_movement, self.angle_smooth_factor)
            else:
                self.last_movement = angle
            self.last_movement_time = time.time()

        return self.last_movement

    def loop(self, brawler, data, current_time):
        if self.is_showdown:
            movement = self.get_showdown_movement(
                player_data=data['player'][0],
                enemy_data=data['enemy'],
                teammate_data=data['teammate'],
                walls=data['wall'],
                brawler=brawler,
                jump_pads=data.get('jump_pad') or [],
                projectile_data=data.get('projectile') or [],
            )
            strict_following = (
                self.showdown_playstyle_mode in ("follow", "follower", "team", "teammate", "teammates")
                and bool(data.get('teammate'))
            )
            if strict_following:
                self.last_movement = movement
                self.last_movement_time = time.time()
            else:
                # Debounce small angle jitter before sending to joystick.
                movement = self._debounce_angle(movement)
        else:
            movement = self.get_movement(player_data=data['player'][0], enemy_data=data['enemy'], walls=data['wall'], brawler=brawler)

        movement = self.enemy_pressure_movement_fallback(movement, data, brawler, current_time)

        current_time = time.time()
        if current_time - self.time_since_movement > self.minimum_movement_delay:
            if isinstance(movement, float):
                # 1. If a semicircle escape is already running, just advance it.
                escape_angle = self.semicircle_escape_step(current_time)
                if escape_angle is not None:
                    movement = escape_angle
                else:
                    # 2. Wall-based stuck detector triggers the semicircle escape.
                    player_pos = self.get_player_pos(data['player'][0]) if data.get('player') else None
                    walls = data.get('wall') or []
                    is_trying = isinstance(movement, float)
                    if self.detect_wall_stuck(walls, player_pos, is_trying, current_time):
                        self.capture_vision_frame("wall_stuck", self.current_frame, data, brawler)
                        self.start_semicircle_escape(movement, current_time)
                        self._reset_wall_stuck_state(current_time)
                        movement = self.semicircle_escape_step(current_time) or movement
            else:
                movement = self.unstuck_movement_if_needed(movement, current_time)
            self.do_movement(movement)
            self.time_since_movement = time.time()
        else:
            if getattr(self.window_controller, "joystick_debug", False):
                skipped_ms = int((self.minimum_movement_delay - (current_time - self.time_since_movement)) * 1000)
                print(
                    "[MOVE] movement_update_skipped_reason=minimum_movement_delay "
                    f"active_movement_intent={bool(movement)} wait_remaining_ms={max(0, skipped_ms)}"
                )
        if getattr(self.window_controller, "are_we_moving", False):
            last_cmd = getattr(self.window_controller, "last_joystick_command_time", 0.0)
            age = time.time() - last_cmd if last_cmd else 0.0
            warning_after = getattr(self.window_controller, "movement_stall_warning_seconds", 0.35)
            last_warn = getattr(self.window_controller, "_last_movement_stall_warning", 0.0)
            if last_cmd and age > warning_after and time.time() - last_warn > 1.0:
                self.window_controller._last_movement_stall_warning = time.time()
                print(
                    "[MOVE] movement_stall_detected "
                    f"last_movement_command_age_ms={int(age * 1000)} "
                    f"active_movement_intent={bool(movement)} "
                    "blocked_by_sleep_or_io=unknown"
                )
        return movement

    def enemy_pressure_movement_fallback(self, movement, data, brawler, current_time):
        if isinstance(movement, float):
            return movement
        if isinstance(movement, str) and movement.strip():
            return movement
        if not data or not data.get("player") or not data.get("enemy"):
            return movement

        player_pos = self.get_player_pos(data["player"][0])
        walls = data.get("wall") or []
        enemy_coords, enemy_distance = self.find_closest_enemy(data["enemy"], player_pos, walls, "attack")
        if enemy_coords is None or enemy_distance is None:
            return movement

        safe_range, attack_range, _ = self.get_brawler_range(brawler)
        pressure_range = max(safe_range, attack_range) * self.enemy_pressure_move_range_multiplier
        if enemy_distance > pressure_range:
            return movement

        toward_angle = self.angle_from_direction(
            enemy_coords[0] - player_pos[0],
            enemy_coords[1] - player_pos[1],
        )
        if enemy_distance <= safe_range:
            desired = self.blend_angles(
                self.angle_opposite(toward_angle),
                self.get_strafe_angle(toward_angle, current_time, enemy_distance, safe_range),
                0.35,
            )
        else:
            desired = self.get_strafe_angle(toward_angle, current_time, enemy_distance, safe_range)

        return self.find_best_angle(player_pos, desired, walls)

    def release_held_attack_for_super(self):
        if self.time_since_holding_attack is None:
            return
        try:
            self.window_controller.press_key("M", touch_up=True, touch_down=False)
        except Exception as exc:
            print(f"Could not release held attack before super: {exc}")
        self.time_since_holding_attack = None

    def try_use_super_on_enemy(self, brawler, brawler_info, player_pos, enemy_coords, enemy_distance, walls):
        if not self.is_super_ready:
            self._aimlog("super_decision use_super=False reason=not_ready ready=False cooldown_remaining_ms=0")
            return False
        super_type = brawler_info['super_type']
        _, attack_range, super_range = self.get_brawler_range(brawler)
        enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "super")
        near_range = min(max(super_range, attack_range * 0.75), attack_range)
        retry_multiplier = (
            max(0.05, min(1.0, getattr(self, "super_retry_cooldown_multiplier", 0.5)))
            if getattr(self, "last_super_time", 0.0) > 0.0
            else 1.0
        )
        cooldown_remaining_ms = self.super_cooldown_remaining_ms(retry_multiplier)
        if not self.should_use_super_on_enemy(
                brawler, super_type, enemy_distance, attack_range, super_range, enemy_hittable
        ):
            self._aimlog(
                "super_decision "
                f"use_super=False reason=low_value ready={self.is_super_ready} super_type={super_type} "
                f"enemy_distance={int(enemy_distance)} attack_range={int(attack_range)} "
                f"super_range={int(super_range)} near_range={int(near_range)} "
                f"enemy_hittable={enemy_hittable} cooldown_remaining_ms={cooldown_remaining_ms}"
            )
            return False

        self._aimlog(
            "super_decision "
            f"use_super=True reason=valuable_{super_type}_opportunity ready={self.is_super_ready} "
            f"enemy_distance={int(enemy_distance)} attack_range={int(attack_range)} "
            f"super_range={int(super_range)} near_range={int(near_range)} "
            f"enemy_hittable={enemy_hittable} cooldown_remaining_ms={cooldown_remaining_ms}"
        )
        self.release_held_attack_for_super()
        if self.is_hypercharge_ready:
            self.use_hypercharge()
            self.time_since_hypercharge_checked = time.time()
            self.clear_ability_ready("hypercharge")
        if self.use_super(cooldown_multiplier=retry_multiplier):
            self.time_since_super_checked = time.time()
            self.clear_ability_ready("super")
            return True
        self._aimlog(
            "super_decision "
            f"use_super=False reason=super_on_cooldown ready={self.is_super_ready} super_type={super_type} "
            f"enemy_distance={int(enemy_distance)} attack_range={int(attack_range)} "
            f"super_range={int(super_range)} near_range={int(near_range)} "
            f"enemy_hittable={enemy_hittable} cooldown_remaining_ms={self.super_cooldown_remaining_ms(retry_multiplier)}"
        )
        return False

    def should_use_gadget_on_enemy(self, brawler, player_data, enemy_data, walls):
        if not self.should_use_gadget or not self.is_gadget_ready or self.time_since_holding_attack is not None:
            return False
        if not enemy_data:
            return False

        player_pos = self.get_player_pos(player_data)
        enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
        if enemy_coords is None:
            return False

        _, attack_range, _ = self.get_brawler_range(brawler)
        enemies_in_range = sum(
            1
            for enemy in (enemy_data or [])
            if self.get_distance(self.get_enemy_pos(enemy), player_pos) <= attack_range
        )
        gadget_threshold = attack_range if enemies_in_range >= 2 else attack_range * 0.7
        if enemy_distance > gadget_threshold:
            return False
        return self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")

    def remember_ability_ready(self, ability_name, detected_ready, current_time):
        seen_attr = f"_{ability_name}_ready_seen_at"
        if detected_ready:
            setattr(self, seen_attr, current_time)
            return True
        return False

    def clear_ability_ready(self, ability_name):
        setattr(self, f"_{ability_name}_ready_seen_at", 0.0)
        setattr(self, f"is_{ability_name}_ready", False)

    def try_use_ready_abilities_when_enemy_visible(
            self,
            enemy_data,
            brawler=None,
            brawler_info=None,
            player_data=None,
            walls=None,
            teammate_data=None,
            safe_range=None,
            attack_range=None,
            intent_mode="strafe_attack_lane",
            health_state=None):
        """Use ready abilities only when enough combat context is available.

        The old visible-enemy fallback was intentionally inert to avoid spam.
        Keep that safety: no player/context means no ability press.
        """
        if not (getattr(self, "combat_brain_enabled", False) and getattr(self, "ability_brain_enabled", False)):
            return False
        if not enemy_data or player_data is None or not brawler or not brawler_info:
            return False
        walls = walls or []
        teammate_data = teammate_data or []
        player_pos = self.get_player_pos(player_data)
        if safe_range is None or attack_range is None:
            safe_range, attack_range, _ = self.get_brawler_range(brawler)
        target_score = self.choose_combat_target_score(
            player_pos,
            enemy_data,
            walls,
            brawler,
            safe_range,
            attack_range,
        )
        plan = self.choose_combat_ability_plan(
            brawler,
            brawler_info,
            player_pos,
            enemy_data,
            teammate_data,
            walls,
            safe_range,
            attack_range,
            target_score,
            health_state or HealthState(),
            intent_mode,
        )
        return self.execute_ability_plan(plan)

    def refresh_ready_abilities(self, frame, current_time):
        if current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold:
            detected = self.check_if_hypercharge_ready(frame)
            self.is_hypercharge_ready = bool(detected)
            self.time_since_hypercharge_checked = current_time
        if current_time - self.time_since_gadget_checked > self.gadget_treshold:
            detected = self.check_if_gadget_ready(frame)
            self.is_gadget_ready = bool(detected)
            self.time_since_gadget_checked = current_time
        if current_time - self.time_since_super_checked > self.super_treshold:
            detected = self.check_if_super_ready(frame)
            mem = float(getattr(self, "ability_ready_memory_seconds", 1.25) or 0.0)
            last_seen = float(getattr(self, "_super_ready_seen_at", 0.0) or 0.0)
            if detected:
                self._super_ready_seen_at = current_time
                self.is_super_ready = True
            elif last_seen > 0.0 and mem > 0.0 and (current_time - last_seen) <= mem:
                # HUD color detection flickers between frames; keep latched until memory expires
                # or clear_ability_ready() zeros _super_ready_seen_at after a super press.
                self.is_super_ready = True
            else:
                self.is_super_ready = False
            self.time_since_super_checked = current_time

    @staticmethod
    def _scaled_pixel_threshold(base_threshold, screenshot, crop_area):
        reference_area = max(1, abs(crop_area[2] - crop_area[0]) * abs(crop_area[3] - crop_area[1]))
        actual_area = max(1, screenshot.shape[0] * screenshot.shape[1])
        return max(1.0, float(base_threshold) * (actual_area / reference_area))

    def check_if_hypercharge_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(self.hypercharge_crop_area[0] * wr), int(self.hypercharge_crop_area[1] * hr)
        x2, y2 = int(self.hypercharge_crop_area[2] * wr), int(self.hypercharge_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        purple_pixels = count_hsv_pixels(screenshot, (137, 158, 159), (179, 255, 255))
        threshold = self._scaled_pixel_threshold(self.hypercharge_pixels_minimum, screenshot, self.hypercharge_crop_area)
        if debug:
            print("hypercharge purple pixels:", purple_pixels, "(if > ", threshold, " then hypercharge is ready)")
            cv2.imwrite(f"debug_frames/hypercharge_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        if purple_pixels > threshold:
            return True
        return False

    def check_if_gadget_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(self.gadget_crop_area[0] * wr), int(self.gadget_crop_area[1] * hr)
        x2, y2 = int(self.gadget_crop_area[2] * wr), int(self.gadget_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        green_pixels = count_hsv_pixels(screenshot, (57, 219, 165), (62, 255, 255))
        threshold = self._scaled_pixel_threshold(self.gadget_pixels_minimum, screenshot, self.gadget_crop_area)
        if debug:
            print(
                "gadget green pixels:",
                green_pixels,
                "(if > ",
                threshold,
                " then gadget is ready)"
            )
            cv2.imwrite(f"debug_frames/gadget_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        if green_pixels > threshold:
            return True
        return False

    def check_if_super_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(self.super_crop_area[0] * wr), int(self.super_crop_area[1] * hr)
        x2, y2 = int(self.super_crop_area[2] * wr), int(self.super_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        yellow_pixels = count_hsv_pixels(screenshot, (17, 170, 200), (27, 255, 255))
        orange_pixels = count_hsv_pixels(screenshot, (8, 120, 150), (38, 255, 255))
        threshold = self._scaled_pixel_threshold(self.super_pixels_minimum, screenshot, self.super_crop_area) * 1.5
        if debug:
            print(
                "super pixels:",
                f"yellow={yellow_pixels}",
                f"orange={orange_pixels}",
                f"threshold={threshold}",
                "(if above threshold, super is ready)",
            )
            cv2.imwrite(f"debug_frames/super_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        if yellow_pixels > threshold:
            return True
        if orange_pixels > threshold * 1.25:
            return True
        return False

    def get_tile_data(self, frame):
        tile_data = self.Detect_tile_detector.detect_objects(frame, conf_tresh=self.wall_detection_confidence)
        return tile_data

    @staticmethod
    def normalize_box(box):
        x1, y1, x2, y2 = box[:4]
        return [int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))]

    @staticmethod
    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def box_center_distance(a, b):
        acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
        bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
        return math.hypot(acx - bcx, acy - bcy)

    def merge_wall_boxes(self, boxes, min_hits=1):
        clusters = []
        for raw_box in boxes:
            box = self.normalize_box(raw_box)
            width = box[2] - box[0]
            height = box[3] - box[1]
            if width < self.wall_box_min_size or height < self.wall_box_min_size:
                continue

            matched = None
            for cluster in clusters:
                if (
                        self.box_iou(cluster["box"], box) >= self.wall_box_merge_iou
                        or self.box_center_distance(cluster["box"], box) <= self.wall_box_merge_center_distance
                ):
                    matched = cluster
                    break

            if matched is None:
                clusters.append({"box": box, "hits": 1})
                continue

            old = matched["box"]
            hits = matched["hits"]
            matched["box"] = [
                int((old[0] * hits + box[0]) / (hits + 1)),
                int((old[1] * hits + box[1]) / (hits + 1)),
                int((old[2] * hits + box[2]) / (hits + 1)),
                int((old[3] * hits + box[3]) / (hits + 1)),
            ]
            matched["hits"] = hits + 1

        return [cluster["box"] for cluster in clusters if cluster["hits"] >= min_hits]

    def process_tile_data(self, tile_data):
        walls = []
        for class_name, boxes in tile_data.items():
            if class_name != 'bush':
                walls.extend(boxes)
        walls = self.merge_wall_boxes(walls)

        # Add walls to history
        self.wall_history.append(walls)
        if len(self.wall_history) > self.wall_history_length:
            self.wall_history.pop(0)
        # Combine walls from history
        combined_walls = self.combine_walls_from_history()

        return combined_walls

    @staticmethod
    def box_center(box):
        return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5

    def is_player_near_map_edge(self, player_pos):
        frame = getattr(self, "current_frame", None)
        if frame is None or player_pos is None:
            return False
        h, w = frame.shape[:2]
        margin = max(0.05, min(0.45, float(self.jump_pad_escape_edge_margin)))
        edge_x = w * margin
        edge_y = h * margin
        x, y = player_pos
        return x <= edge_x or x >= w - edge_x or y <= edge_y or y >= h - edge_y

    def has_close_teammate_for_jump_escape(self, player_pos, teammate_data):
        if player_pos is None:
            return True
        for teammate in teammate_data or []:
            teammate_pos = self.get_enemy_pos(teammate)
            if self.get_distance(teammate_pos, player_pos) <= self.jump_pad_escape_teammate_safe_distance:
                return True
        return False

    def detect_jump_pads(self, frame):
        """Detect jump pads from their yellow arrow inside a dark gray tile.

        The wall model does not include jump pads, so this uses strict color and
        shape anchors from the game art. It returns pad boxes in frame coords.
        """
        if not self.jump_pad_detection_enabled or frame is None:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        yellow = cv2.inRange(
            hsv,
            np.array((18, 90, 120), dtype=np.uint8),
            np.array((38, 255, 255), dtype=np.uint8),
        )
        yellow = cv2.morphologyEx(
            yellow,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        scale = max(0.4, min(1.2, w / brawl_stars_width))
        min_area = max(80, int(450 * scale * scale))
        max_area = max(min_area + 1, int(9000 * scale * scale))
        pads = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 12 * scale or bh < 12 * scale:
                continue
            if bw > 160 * scale or bh > 160 * scale:
                continue

            pad_size = int(max(bw, bh) * 2.4)
            cx, cy = x + bw * 0.5, y + bh * 0.5
            rx1 = max(0, int(cx - pad_size * 0.5))
            ry1 = max(0, int(cy - pad_size * 0.5))
            rx2 = min(w, int(cx + pad_size * 0.5))
            ry2 = min(h, int(cy + pad_size * 0.5))
            roi = hsv[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            yellow_ratio = self._count_mask_pixels(roi, (18, 90, 120), (38, 255, 255)) / max(1, roi.shape[0] * roi.shape[1])
            gray_ratio = self._count_mask_pixels(roi, (0, 0, 40), (179, 95, 190)) / max(1, roi.shape[0] * roi.shape[1])
            dark_ratio = self._count_mask_pixels(roi, (0, 0, 0), (179, 255, 95)) / max(1, roi.shape[0] * roi.shape[1])
            if not (0.035 <= yellow_ratio <= 0.42 and gray_ratio > 0.16 and dark_ratio > 0.10):
                continue

            pads.append([rx1, ry1, rx2, ry2])

        return self.merge_wall_boxes(pads)

    def find_jump_pad_escape_angle(self, player_pos, jump_pads, walls, fog_flee_angle=None, teammate_data=None):
        if not jump_pads or player_pos is None:
            return None
        if self.jump_pad_escape_requires_edge and not self.is_player_near_map_edge(player_pos):
            vlog("jump pad escape skipped: player is not near map edge")
            return None
        if self.has_close_teammate_for_jump_escape(player_pos, teammate_data):
            vlog("jump pad escape skipped: teammate is close")
            return None

        candidates = []
        for pad in jump_pads:
            pad_pos = self.box_center(self.normalize_box(pad))
            distance = self.get_distance(pad_pos, player_pos)
            if distance < self.jump_pad_escape_min_distance or distance > self.jump_pad_escape_distance:
                continue
            angle = self.angle_from_direction(pad_pos[0] - player_pos[0], pad_pos[1] - player_pos[1])
            if self.is_path_blocked_angle(player_pos, angle, walls, distance=max(40, min(distance, self.TILE_SIZE * 2))):
                continue
            fog_alignment = 0.0
            if fog_flee_angle is not None:
                fog_alignment = abs((angle - fog_flee_angle + 180) % 360 - 180)
            candidates.append((fog_alignment, distance, angle, pad_pos))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, distance, angle, pad_pos = candidates[0]
        vlog(f"jump pad escape -> angle={angle:.1f}° dist={int(distance)}px pad={tuple(map(int, pad_pos))}")
        return angle

    def combine_walls_from_history(self):
        if not self.wall_history:
            return []
        current_walls = self.wall_history[-1]
        historical_walls = [wall for walls in self.wall_history for wall in walls]
        stable_history = self.merge_wall_boxes(historical_walls, min_hits=max(1, self.wall_history_min_hits))
        return self.merge_wall_boxes(current_walls + stable_history)

    def get_movement(self, player_data, enemy_data, walls, brawler):
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Brawler '{brawler}' not found in brawlers info.")
        playstyle_movement = self.run_playstyle(player_data, enemy_data, walls, brawler)
        if playstyle_movement is not None:
            return playstyle_movement
        must_brawler_hold_attack = self.must_brawler_hold_attack(brawler, self.brawlers_info)
        # if a brawler has been holding an attack for its max duration + the bot setting, then we release
        if must_brawler_hold_attack and self.time_since_holding_attack is not None and time.time() - self.time_since_holding_attack >= brawler_info['hold_attack'] + self.seconds_to_hold_attack_after_reaching_max:
            self.attack(touch_up=True, touch_down=False)
            self.time_since_holding_attack = None

        safe_range, attack_range, super_range = self.get_brawler_range(brawler)
        player_pos = self.get_player_pos(player_data)
        if debug: print("found player pos:", player_pos)
        if not self.is_there_enemy(enemy_data):
            return self.no_enemy_movement(player_data, walls)
        enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
        if enemy_coords is None:
            return self.no_enemy_movement(player_data, walls)
        if debug: print("found enemy pos:", enemy_coords)
        direction_x = enemy_coords[0] - player_pos[0]
        direction_y = enemy_coords[1] - player_pos[1]

        # Determine initial movement direction
        if enemy_distance > safe_range:  # Move towards the enemy
            move_horizontal = self.get_horizontal_move_key(direction_x)
            move_vertical = self.get_vertical_move_key(direction_y)
        else:  # Move away from the enemy
            move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
            move_vertical = self.get_vertical_move_key(direction_y, opposite=True)

        movement_options = [move_horizontal + move_vertical]
        if self.game_mode == 3:
            movement_options += [move_vertical, move_horizontal]
        elif self.game_mode == 5:
            movement_options += [move_horizontal, move_vertical]
        else:
            raise ValueError("Gamemode type is invalid")

        # Check for walls and adjust movement
        for move in movement_options:
            if not self.is_path_blocked(player_pos, move, walls):
                movement = move
                break
        else:
            print("default paths are blocked")
            # If all preferred directions are blocked, try other directions
            alternative_moves = ['W', 'A', 'S', 'D']
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_pos, move, walls):
                    movement = move
                    break
            else:
                # if no movement is available, we still try to go in the best direction
                # because it's better than doing nothing
                movement = move_horizontal + move_vertical

        current_time = time.time()
        if movement != self.last_movement:
            if current_time - self.last_movement_time >= self.minimum_movement_delay:
                self.last_movement = movement
                self.last_movement_time = current_time
            else:
                movement = self.last_movement  # Continue previous movement
        else:
            self.last_movement_time = current_time  # Reset timer if movement didn't change

        self.try_use_super_on_enemy(brawler, brawler_info, player_pos, enemy_coords, enemy_distance, walls)

        attack_decision = self.choose_attack_decision(
            brawler,
            player_pos,
            enemy_data,
            walls,
            attack_range=attack_range,
            current_time=current_time,
        )
        if attack_decision.should_fire:
            if self.strafe_enabled:
                toward_angle = self.angle_from_direction(direction_x, direction_y)
                desired = self.apply_combat_dodge(
                    self.angle_from_direction(*self.movement_to_vector(movement)),
                    toward_angle,
                    current_time,
                    enemy_distance,
                    safe_range,
                )
                movement = self.find_best_angle(player_pos, desired, walls)
            if self.should_use_gadget_on_enemy(brawler, player_data, enemy_data, walls):
                if self.use_gadget():
                    self.time_since_gadget_checked = time.time()
                    self.clear_ability_ready("gadget")

            if not must_brawler_hold_attack:
                self.auto_aim_attack(
                    brawler,
                    player_pos,
                    enemy_data,
                    walls,
                    attack_range=attack_range,
                    decision=attack_decision,
                )
            else:
                if self.time_since_holding_attack is None:
                    self.time_since_holding_attack = time.time()
                    self.attack(touch_up=False, touch_down=True)
                elif time.time() - self.time_since_holding_attack >= self.brawlers_info[brawler]['hold_attack']:
                    self.attack(touch_up=True, touch_down=False)
                    self.time_since_holding_attack = None
        elif attack_decision.visible_enemy_count:
            self.log_attack_decision(
                attack_decision,
                attack_range,
                denied_by=attack_decision.denied_by or attack_decision.reason,
            )


        return movement

    def main(self, frame, brawler, main):
        current_time = time.time()
        raw_data = self.get_main_data(frame)
        data = raw_data
        if self.should_detect_walls and current_time - self.time_since_walls_checked > self.walls_treshold:

            tile_data = self.get_tile_data(frame)

            walls = self.process_tile_data(tile_data)
            jump_pads = self.detect_jump_pads(frame)

            self.time_since_walls_checked = current_time
            self.last_walls_data = walls
            self.last_jump_pad_data = jump_pads
            data['wall'] = walls
            data['jump_pad'] = jump_pads
        elif self.keep_walls_in_memory:
            data['wall'] = self.last_walls_data
            data['jump_pad'] = self.last_jump_pad_data

        data = self.validate_game_data(data)
        self.track_no_detections(data)
        if data:
            self.time_since_player_last_found = time.time()
            if main.state != "match":
                main.state = get_state(frame)
                if main.state != "match":
                    self.reset_tactical_adaptation()
                    data = None
        if not data:
            if current_time - self.time_since_player_last_found > 1.0:
                self.capture_vision_frame(
                    "player_lost",
                    frame,
                    {"raw_detection": raw_data},
                    brawler,
                    {"state": getattr(main, "state", None)},
                )
                self.save_combat_snapshot(
                    "player_lost",
                    extra={
                        "raw_detection_keys": list(raw_data.keys()) if isinstance(raw_data, dict) else None,
                        "state": getattr(main, "state", None),
                    },
                    brawler=brawler,
                )
                self.window_controller.keys_up(list("wasd"))
            self.time_since_different_movement = time.time()
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                current_state = get_state(frame)
                if current_state != "match":
                    self.reset_tactical_adaptation()
                    self.time_since_last_proceeding = current_time
                else:
                    if current_time - self.time_since_last_no_detection_q >= self.no_detection_q_press_interval:
                        print("No detection fallback: pressing Q.")
                        self.window_controller.press_key("Q")
                        self.time_since_last_no_detection_q = current_time
                    self.time_since_last_proceeding = time.time()
            return
        self.time_since_last_proceeding = time.time()
        self.refresh_ready_abilities(frame, current_time)

        self.current_frame = frame
        self.last_playstyle_teammate_data = data.get("teammate")
        movement = self.loop(brawler, data, current_time)

        if visual_debug:
            self.queue_visual_debug(frame, data, brawler)

        # if data:
        #     # Record scene data
        #     self.scene_data.append({
        #         'frame_number': len(self.scene_data),
        #         'player': data.get('player', []),
        #         'enemy': data.get('enemy', []),
        #         'wall': data.get('wall', []),
        #         'movement': movement,
        #     })

    def _copy_visual_debug_data(self, data):
        copied = {}
        for key, value in (data or {}).items():
            if isinstance(value, list):
                copied[key] = [
                    list(item) if isinstance(item, (list, tuple, np.ndarray)) else item
                    for item in value
                ]
            else:
                copied[key] = value
        return copied

    def _ensure_visual_debug_thread(self):
        if self._visual_debug_thread and self._visual_debug_thread.is_alive():
            return
        self._visual_debug_stop = False
        self._visual_debug_thread = threading.Thread(
            target=self._visual_debug_loop,
            name="PylaVisualDebug",
            daemon=True,
        )
        self._visual_debug_thread.start()

    def queue_visual_debug(self, frame, data, brawler=None):
        now = time.time()
        frame_delay = 1.0 / self.visual_debug_max_fps
        if now < self._visual_debug_next_enqueue_at:
            return
        self._visual_debug_next_enqueue_at = now + frame_delay
        self._ensure_visual_debug_thread()
        payload = (
            frame.copy() if isinstance(frame, np.ndarray) else np.array(frame),
            self._copy_visual_debug_data(data),
            brawler,
        )
        with self._visual_debug_lock:
            self._visual_debug_payload = payload

    def _visual_debug_loop(self):
        frame_delay = 1.0 / self.visual_debug_max_fps
        while not self._visual_debug_stop:
            loop_started = time.time()
            with self._visual_debug_lock:
                payload = self._visual_debug_payload
                self._visual_debug_payload = None
            if payload is not None:
                try:
                    self.show_visual_debug(*payload, respect_throttle=False)
                except Exception as exc:
                    print(f"Visual debug renderer error: {exc}")
            sleep_for = frame_delay - (time.time() - loop_started)
            if sleep_for > 0:
                time.sleep(min(sleep_for, frame_delay))

    def show_visual_debug(self, frame, data, brawler=None, respect_throttle=True):
        import numpy as np
        now = time.time()
        if respect_throttle and now < self._visual_debug_next_frame_at:
            return
        if respect_throttle:
            self._visual_debug_next_frame_at = now + (1.0 / self.visual_debug_max_fps)

        scale = self.visual_debug_scale
        if scale < 0.999:
            img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            img = frame.copy() if isinstance(frame, np.ndarray) else np.array(frame)

        def s(value):
            return int(value * scale)

        def sp(point):
            return s(point[0]), s(point[1])

        # --- Fog overlay ---
        # Only draw the fog tint + centroid arrow when a fog threat is strong
        # enough to trigger evasion (same thresholds as detect_fog_threat):
        # trusted mask inside flee-radius must contain >= fog_min_pixels_in_radius.
        if data.get("player"):
            px, py = self.get_player_pos(data["player"][0])
            r = self.fog_flee_distance
            built = self._build_trusted_fog_mask(frame, roi_center=(px, py), roi_radius=r)
            if built is not None:
                mask, (ox, oy) = built
                ys, xs = np.nonzero(mask)
                if xs.size > 0:
                    dx_all = (xs + ox) - px
                    dy_all = (ys + oy) - py
                    dist_sq = dx_all * dx_all + dy_all * dy_all
                    inside = dist_sq <= r * r
                    if int(inside.sum()) >= self.fog_min_pixels_in_radius:
                        # Paint only the fog ROI instead of allocating a full-frame mask/tint.
                        roi_mask = np.zeros_like(mask)
                        roi_mask[ys[inside], xs[inside]] = 255
                        if scale < 0.999:
                            roi_mask = cv2.resize(roi_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                        x0, y0 = s(ox), s(oy)
                        x1 = min(img.shape[1], x0 + roi_mask.shape[1])
                        y1 = min(img.shape[0], y0 + roi_mask.shape[0])
                        roi_mask = roi_mask[:max(0, y1 - y0), :max(0, x1 - x0)]
                        if roi_mask.size:
                            roi = img[y0:y1, x0:x1]
                            magenta = np.empty_like(roi)
                            magenta[:, :] = (255, 0, 255)
                            blended = cv2.addWeighted(roi, 0.55, magenta, 0.45, 0)
                            roi[roi_mask > 0] = blended[roi_mask > 0]
                        fog_cx = int(dx_all[inside].mean() + px)
                        fog_cy = int(dy_all[inside].mean() + py)
                        cv2.circle(img, sp((fog_cx, fog_cy)), max(3, s(8)), (255, 0, 255), -1)
                        cv2.putText(img, "fog", sp((fog_cx + 10, fog_cy)),
                                    cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.6 * scale), (255, 0, 255), 2)
                        cv2.arrowedLine(img, sp((px, py)), sp((fog_cx, fog_cy)),
                                        (255, 0, 255), 2, tipLength=0.15)

        # Colors in RGB (frame is kept in RGB; converted to BGR only for imshow).
        colors = {
            "player":   (0, 255, 0),    # green
            "teammate": (0, 0, 255),    # blue
            "enemy":    (255, 0, 0),    # red
            "wall":     (128, 128, 128),  # gray
        }
        boxes_drawn = 0
        for key, color in colors.items():
            boxes = data.get(key)
            if not boxes:
                continue
            for box in boxes:
                if boxes_drawn >= self.visual_debug_max_boxes:
                    break
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, sp((x1, y1)), sp((x2, y2)), color, max(1, s(2)))
                if key != "wall":
                    cv2.putText(img, key, sp((x1, max(y1 - 6, 0))),
                                cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.5 * scale), color, 1)
                boxes_drawn += 1

        # Draw attack/super ranges around the player based on brawlers_info.json.
        if brawler and data.get("player"):
            info = self.brawlers_info.get(brawler)
            if info:
                px, py = self.get_player_pos(data["player"][0])
                center = sp((px, py))
                attack_range = s(int(info.get("attack_range", 0)))
                super_range = s(int(info.get("super_range", 0)))
                if attack_range > 0:
                    cv2.circle(img, center, attack_range, (160, 32, 240), 2)  # purple
                if super_range > 0:
                    cv2.circle(img, center, super_range, (255, 255, 0), 2)  # yellow

        cv2.imshow("Pyla 143 Visual Debug", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    @staticmethod
    def movement_to_direction(movement):
        mapping = {
            'w': 'up',
            'a': 'left',
            's': 'down',
            'd': 'right',
            'wa': 'up-left',
            'aw': 'up-left',
            'wd': 'up-right',
            'dw': 'up-right',
            'sa': 'down-left',
            'as': 'down-left',
            'sd': 'down-right',
            'ds': 'down-right',
        }
        movement = movement.lower()
        movement = ''.join(sorted(movement))
        return mapping.get(movement, 'idle' if movement == '' else movement)
