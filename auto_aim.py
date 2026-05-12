import math
from dataclasses import dataclass


@dataclass
class AutoAimDecision:
    should_fire: bool
    aim_angle: float = None
    target: tuple = None
    predicted: tuple = None
    distance: float = None
    attack_range: float = None
    confidence: float = 0.0
    reason: str = ""
    use_tap: bool = False
    velocity: tuple = (0.0, 0.0)


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


def _line_of_sight_clear(player_pos, target_pos, walls, can_ignore_walls, walls_block_line_of_sight):
    if can_ignore_walls:
        return True
    return not walls_block_line_of_sight(player_pos, target_pos, walls)


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
):
    if not player_pos:
        return AutoAimDecision(False, reason="no_player")
    if not enemy_data:
        return AutoAimDecision(False, reason="no_enemy")
    if attack_range <= 0:
        return AutoAimDecision(False, reason="invalid_range", attack_range=attack_range)

    close_tap_range = close_tap_range if close_tap_range is not None else min(120.0, attack_range * 0.28)
    best = None

    for enemy in enemy_data:
        target = _center(enemy)
        distance = math.hypot(target[0] - player_pos[0], target[1] - player_pos[1])
        if distance > attack_range:
            decision = AutoAimDecision(
                False,
                target=target,
                distance=distance,
                attack_range=attack_range,
                reason="out_of_range",
            )
            if best is None or distance < (best.distance or float("inf")):
                best = decision
            continue

        if not _line_of_sight_clear(player_pos, target, walls, can_ignore_walls, walls_block_line_of_sight):
            decision = AutoAimDecision(
                False,
                target=target,
                distance=distance,
                attack_range=attack_range,
                reason="wall_blocked",
            )
            if best is None or distance < (best.distance or float("inf")):
                best = decision
            continue

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
        predicted_distance = math.hypot(predicted[0] - player_pos[0], predicted[1] - player_pos[1])
        if predicted_distance > attack_range * 1.04:
            decision = AutoAimDecision(
                False,
                target=target,
                predicted=predicted,
                distance=predicted_distance,
                attack_range=attack_range,
                velocity=velocity,
                reason="predicted_out_of_range",
            )
            if best is None or predicted_distance < (best.distance or float("inf")):
                best = decision
            continue

        if not _line_of_sight_clear(player_pos, predicted, walls, can_ignore_walls, walls_block_line_of_sight):
            decision = AutoAimDecision(
                False,
                target=target,
                predicted=predicted,
                distance=predicted_distance,
                attack_range=attack_range,
                velocity=velocity,
                reason="predicted_wall_blocked",
            )
            if best is None or predicted_distance < (best.distance or float("inf")):
                best = decision
            continue

        angle = _angle_from_direction(predicted[0] - player_pos[0], predicted[1] - player_pos[1])
        confidence = 1.0
        confidence *= _range_score(predicted_distance, attack_range)
        confidence *= _target_box_score(enemy)
        if lead_distance > attack_range * 0.22:
            confidence *= 0.78
        elif lead_distance > 0:
            confidence *= 0.90 + 0.10 * max(0.0, min(1.0, current_velocity_confidence))

        if aim_line_angle is not None:
            mismatch = _angle_delta(angle, aim_line_angle)
            if mismatch <= 10:
                confidence *= 1.04
            elif mismatch <= 24:
                confidence *= 0.88
            else:
                confidence *= 0.65

        use_tap = distance <= close_tap_range and confidence >= 0.25
        decision = AutoAimDecision(
            confidence >= min_confidence or use_tap,
            aim_angle=angle,
            target=target,
            predicted=predicted,
            distance=predicted_distance,
            attack_range=attack_range,
            confidence=max(0.0, min(1.0, confidence)),
            reason="ok" if confidence >= min_confidence else ("close_tap" if use_tap else "low_confidence"),
            use_tap=use_tap,
            velocity=velocity,
        )

        if best is None:
            best = decision
            continue
        best_key = (0 if best.should_fire else 1, -(best.confidence or 0), best.distance or float("inf"))
        decision_key = (0 if decision.should_fire else 1, -(decision.confidence or 0), decision.distance or float("inf"))
        if decision_key < best_key:
            best = decision

    return best or AutoAimDecision(False, reason="no_valid_target", attack_range=attack_range)


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
