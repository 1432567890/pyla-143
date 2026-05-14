import math
from dataclasses import dataclass


@dataclass
class AttackDecision:
    should_fire: bool
    aim_angle: float = None
    target: tuple = None
    predicted: tuple = None
    distance: float = None
    attack_range: float = None
    in_range: bool = False
    line_of_sight: bool = False
    confidence: float = 0.0
    reason: str = ""
    denied_by: str = ""
    use_tap: bool = False
    velocity: tuple = (0.0, 0.0)
    threshold: float = 0.0
    close_threat: bool = False
    close_range_override: bool = False
    los_status: str = "unknown"
    target_bbox: tuple = None
    cooldown_remaining_ms: int = 0
    visible_enemy_count: int = 0
    closest_enemy_distance: float = None
    aim_fallback_reason: str = ""


AutoAimDecision = AttackDecision


def _center(box):
    return (float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5)


def _angle_from_direction(dx, dy):
    return math.degrees(math.atan2(dy, dx)) % 360


def _angle_delta(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _predict_point(player_pos, target_pos, velocity, projectile_speed, velocity_confidence, attack_range):
    direct_distance = math.hypot(target_pos[0] - player_pos[0], target_pos[1] - player_pos[1])
    if projectile_speed <= 1 or direct_distance < 1 or velocity_confidence <= 0:
        return target_pos, 0.0

    vx, vy = velocity
    if math.hypot(vx, vy) < 12:
        return target_pos, 0.0

    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]
    a = vx * vx + vy * vy - projectile_speed * projectile_speed
    b = 2.0 * (dx * vx + dy * vy)
    c = dx * dx + dy * dy
    t = direct_distance / projectile_speed

    if abs(a) < 1e-6:
        if abs(b) > 1e-6:
            candidate = -c / b
            if candidate > 0:
                t = candidate
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0:
            root = math.sqrt(discriminant)
            candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
            positive = [value for value in candidates if value > 0]
            if positive:
                t = min(positive)

    t = max(0.0, min(1.2, t))
    lead_scale = max(0.0, min(1.0, velocity_confidence))
    lead_x = vx * t * lead_scale
    lead_y = vy * t * lead_scale
    max_lead = max(35.0, attack_range * 0.32)
    lead_len = math.hypot(lead_x, lead_y)
    if lead_len > max_lead:
        scale = max_lead / lead_len
        lead_x *= scale
        lead_y *= scale

    return (target_pos[0] + lead_x, target_pos[1] + lead_y), math.hypot(lead_x, lead_y)


def _range_score(distance, attack_range):
    if attack_range <= 0:
        return 0.0
    ratio = distance / attack_range
    if ratio <= 0.82:
        return 1.0
    if ratio >= 1.0:
        return 0.35
    return max(0.35, 1.0 - (ratio - 0.82) / 0.18 * 0.65)


def _target_box_score(box):
    width = abs(float(box[2]) - float(box[0]))
    height = abs(float(box[3]) - float(box[1]))
    if width < 12 or height < 16:
        return 0.25
    area = width * height
    if area < 500:
        return 0.55
    return 1.0


def _target_box_radius(box):
    width = abs(float(box[2]) - float(box[0]))
    height = abs(float(box[3]) - float(box[1]))
    return max(18.0, math.hypot(width, height) * 0.5)


def _bbox_hit_points(box):
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    inset_x = max(2.0, abs(x2 - x1) * 0.28)
    inset_y = max(2.0, abs(y2 - y1) * 0.28)
    return [
        (cx, cy),
        (max(x1, cx - inset_x), cy),
        (min(x2, cx + inset_x), cy),
        (cx, max(y1, cy - inset_y)),
        (cx, min(y2, cy + inset_y)),
    ]


def _clear_hit_point(player_pos, target_box, walls, can_ignore_walls, walls_block_line_of_sight):
    for point in _bbox_hit_points(target_box):
        if _line_of_sight_clear(player_pos, point, walls, can_ignore_walls, walls_block_line_of_sight):
            return point
    return None


def _clamp_prediction_to_lane(player_pos, target, predicted, lead_distance, attack_range, target_box, close_override):
    if lead_distance <= 0:
        return predicted, lead_distance, ""
    if close_override:
        return target, 0.0, "close_snap_to_target"

    target_distance = math.hypot(target[0] - player_pos[0], target[1] - player_pos[1])
    lane_radius = _target_box_radius(target_box)
    target_angle = _angle_from_direction(target[0] - player_pos[0], target[1] - player_pos[1])
    predicted_angle = _angle_from_direction(predicted[0] - player_pos[0], predicted[1] - player_pos[1])
    angle_mismatch = _angle_delta(target_angle, predicted_angle)

    if target_distance <= attack_range * 0.65:
        max_lead = max(lane_radius * 1.5, min(75.0, attack_range * 0.14))
        if lead_distance > max_lead:
            scale = max_lead / lead_distance
            return (
                (target[0] + (predicted[0] - target[0]) * scale, target[1] + (predicted[1] - target[1]) * scale),
                max_lead,
                "lead_clamped",
            )

    if target_distance <= attack_range * 0.75 and lead_distance > lane_radius * 2.5 and angle_mismatch > 18.0:
        return target, 0.0, "lead_left_hit_lane"

    return predicted, lead_distance, ""


def _line_of_sight_clear(player_pos, target_pos, walls, can_ignore_walls, walls_block_line_of_sight):
    if can_ignore_walls:
        return True
    return not walls_block_line_of_sight(player_pos, target_pos, walls)


def _decision_sort_key(decision, dangerous_close_range):
    distance = decision.distance if decision.distance is not None else float("inf")
    if decision.should_fire:
        return (0, 0 if decision.close_threat else 1, distance, -(decision.confidence or 0.0))
    diagnostic_close = distance <= dangerous_close_range
    return (1, 0 if diagnostic_close else 1, distance, -(decision.confidence or 0.0))


def choose_auto_aim(
    *,
    player_pos,
    enemy_data,
    walls,
    attack_range,
    can_ignore_walls,
    walls_block_line_of_sight,
    track_enemy_velocity,
    velocity_confidence,
    projectile_speed,
    current_time,
    aim_line_angle=None,
    min_confidence=0.62,
    close_tap_range=None,
    close_range_override=True,
    dangerous_close_range=None,
    close_los_override_range=None,
):
    if not player_pos:
        return AttackDecision(False, reason="target_invalid", denied_by="target_invalid")
    visible_enemy_count = len(enemy_data or [])
    if not enemy_data:
        return AttackDecision(False, reason="no_enemy", denied_by="no_enemy", visible_enemy_count=0)
    if attack_range <= 0:
        return AttackDecision(False, reason="target_invalid", denied_by="target_invalid", attack_range=attack_range)

    close_tap_range = close_tap_range if close_tap_range is not None else min(120.0, attack_range * 0.28)
    dangerous_close_range = dangerous_close_range if dangerous_close_range is not None else max(close_tap_range, min(150.0, attack_range * 0.40))
    close_los_override_range = (
        close_los_override_range
        if close_los_override_range is not None
        else min(dangerous_close_range, max(close_tap_range, attack_range * 0.45))
    )
    attack_window = float(attack_range) * 1.035
    best = None

    closest_enemy_distance = None
    for enemy in enemy_data:
        target = _center(enemy)
        distance = math.hypot(target[0] - player_pos[0], target[1] - player_pos[1])
        if closest_enemy_distance is None or distance < closest_enemy_distance:
            closest_enemy_distance = distance
        in_range = distance <= attack_window
        close_override = bool(close_range_override and distance <= dangerous_close_range)
        use_tap = distance <= close_tap_range
        if not in_range:
            decision = AttackDecision(
                False,
                target=target,
                distance=distance,
                attack_range=attack_range,
                in_range=False,
                line_of_sight=False,
                reason="enemy_out_of_range",
                denied_by="out_of_range",
                use_tap=use_tap,
                threshold=min_confidence,
                close_threat=close_override,
                close_range_override=close_override,
                los_status="not_checked",
                target_bbox=tuple(enemy),
                visible_enemy_count=visible_enemy_count,
                closest_enemy_distance=closest_enemy_distance,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range) < _decision_sort_key(best, dangerous_close_range):
                best = decision
            continue

        clear_target = _clear_hit_point(player_pos, enemy, walls, can_ignore_walls, walls_block_line_of_sight)
        target_los_clear = clear_target is not None
        if not target_los_clear:
            decision = AttackDecision(
                False,
                target=target,
                distance=distance,
                attack_range=attack_range,
                in_range=True,
                line_of_sight=False,
                reason="los_blocked",
                denied_by="los_blocked",
                use_tap=use_tap,
                threshold=min_confidence,
                close_threat=close_override,
                close_range_override=close_override,
                los_status="blocked",
                target_bbox=tuple(enemy),
                visible_enemy_count=visible_enemy_count,
                closest_enemy_distance=closest_enemy_distance,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range) < _decision_sort_key(best, dangerous_close_range):
                best = decision
            continue
        if clear_target != target:
            target = clear_target

        velocity = track_enemy_velocity(target, current_time)
        current_velocity_confidence = (
            float(velocity_confidence())
            if callable(velocity_confidence)
            else float(velocity_confidence or 0.0)
        )
        predicted, lead_distance = _predict_point(
            player_pos,
            target,
            velocity,
            projectile_speed,
            current_velocity_confidence,
            attack_range,
        )
        predicted, lead_distance, aim_fallback_reason = _clamp_prediction_to_lane(
            player_pos,
            target,
            predicted,
            lead_distance,
            attack_range,
            enemy,
            close_override,
        )
        predicted_distance = math.hypot(predicted[0] - player_pos[0], predicted[1] - player_pos[1])
        # Lead can push the aim point past attack_range*1.04 even when the enemy
        # bbox center is well inside range (common in melee). Snap to the live
        # target instead of rejecting the shot.
        snap_melee = max(
            float(dangerous_close_range),
            float(close_tap_range) * 1.3,
            float(attack_range) * 0.42,
        )
        if predicted_distance > attack_range * 1.04 and distance <= snap_melee:
            predicted = target
            predicted_distance = distance
            lead_distance = 0.0
            aim_fallback_reason = "melee_prediction_snap"
        if predicted_distance > attack_range * 1.04:
            decision = AttackDecision(
                False,
                target=target,
                predicted=predicted,
                distance=predicted_distance,
                attack_range=attack_range,
                in_range=True,
                line_of_sight=True,
                velocity=velocity,
                reason="prediction_invalid",
                denied_by="prediction_invalid",
                use_tap=use_tap,
                threshold=min_confidence,
                close_threat=close_override,
                close_range_override=close_override,
                los_status="clear",
                target_bbox=tuple(enemy),
                visible_enemy_count=visible_enemy_count,
                closest_enemy_distance=closest_enemy_distance,
                aim_fallback_reason=aim_fallback_reason,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range) < _decision_sort_key(best, dangerous_close_range):
                best = decision
            continue

        predicted_los_clear = _line_of_sight_clear(player_pos, predicted, walls, can_ignore_walls, walls_block_line_of_sight)
        if not predicted_los_clear and not close_override:
            decision = AttackDecision(
                False,
                target=target,
                predicted=predicted,
                distance=predicted_distance,
                attack_range=attack_range,
                in_range=True,
                line_of_sight=False,
                velocity=velocity,
                reason="los_blocked",
                denied_by="los_blocked",
                use_tap=use_tap,
                threshold=min_confidence,
                close_threat=close_override,
                close_range_override=close_override,
                los_status="predicted_blocked",
                target_bbox=tuple(enemy),
                visible_enemy_count=visible_enemy_count,
                closest_enemy_distance=closest_enemy_distance,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range) < _decision_sort_key(best, dangerous_close_range):
                best = decision
            continue
        if not predicted_los_clear and close_override:
            predicted = target
            predicted_distance = distance
            lead_distance = 0.0
            aim_fallback_reason = aim_fallback_reason or "close_los_snap_to_target"
            predicted_los_clear = True

        angle = _angle_from_direction(predicted[0] - player_pos[0], predicted[1] - player_pos[1])
        confidence = 1.0
        confidence *= _range_score(predicted_distance, attack_range)
        confidence *= _target_box_score(enemy)
        if close_override:
            confidence = max(confidence, 0.42)
        if lead_distance > attack_range * 0.22:
            confidence *= 0.78
        elif lead_distance > 0:
            confidence *= 0.90 + 0.10 * max(0.0, min(1.0, current_velocity_confidence))

        if aim_line_angle is not None and not close_override:
            mismatch = _angle_delta(angle, aim_line_angle)
            if mismatch <= 10:
                confidence *= 1.04
            elif mismatch <= 24:
                confidence *= 0.88
            else:
                confidence *= 0.65

        should_fire = confidence >= min_confidence or (close_override and confidence >= 0.18) or use_tap
        if should_fire:
            reason = "ok"
            if close_override and confidence < min_confidence:
                reason = "close_range_override"
            elif use_tap and confidence < min_confidence:
                reason = "close_tap"
        else:
            reason = "confidence_too_low"
        decision = AttackDecision(
            should_fire,
            aim_angle=angle,
            target=target,
            predicted=predicted,
            distance=predicted_distance,
            attack_range=attack_range,
            in_range=True,
            line_of_sight=bool(target_los_clear and predicted_los_clear),
            confidence=max(0.0, min(1.0, confidence)),
            reason=reason,
            denied_by="" if should_fire else "confidence_too_low",
            use_tap=use_tap,
            velocity=velocity,
            threshold=min_confidence,
            close_threat=close_override,
            close_range_override=close_override,
            los_status="clear" if target_los_clear and predicted_los_clear else "predicted_snap",
            target_bbox=tuple(enemy),
            visible_enemy_count=visible_enemy_count,
            closest_enemy_distance=closest_enemy_distance,
            aim_fallback_reason=aim_fallback_reason,
        )

        if best is None or _decision_sort_key(decision, dangerous_close_range) < _decision_sort_key(best, dangerous_close_range):
            best = decision

    if best:
        best.visible_enemy_count = visible_enemy_count
        best.closest_enemy_distance = closest_enemy_distance
        return best
    return AttackDecision(
        False,
        reason="no_valid_target",
        denied_by="no_valid_target",
        attack_range=attack_range,
        visible_enemy_count=visible_enemy_count,
        closest_enemy_distance=closest_enemy_distance,
    )


def detect_aim_line_angle(frame, player_pos):
    if frame is None or player_pos is None:
        return None
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    h, w = frame.shape[:2]
    px, py = int(player_pos[0]), int(player_pos[1])
    radius = max(80, int(min(w, h) * 0.10))
    x1, y1 = max(0, px - radius), max(0, py - radius)
    x2, y2 = min(w, px + radius), min(h, py + radius)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    gray = cv2.inRange(hsv, np.array((0, 0, 92), dtype=np.uint8), np.array((179, 65, 235), dtype=np.uint8))
    yy, xx = np.nonzero(gray)
    if xx.size < 45:
        return None

    gx = xx + x1 - px
    gy = yy + y1 - py
    dist = np.sqrt(gx * gx + gy * gy)
    keep = (dist > radius * 0.24) & (dist < radius * 0.95)
    if int(keep.sum()) < 35:
        return None

    mean_x = float(gx[keep].mean())
    mean_y = float(gy[keep].mean())
    if math.hypot(mean_x, mean_y) < radius * 0.18:
        return None
    return _angle_from_direction(mean_x, mean_y)
