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
    range_distance: float = None
    friendly_lane_status: str = ""


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


def _closest_point_on_box(player_pos, box):
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    cx, cy = (left + right) * 0.5, (top + bottom) * 0.5
    px = min(max(float(player_pos[0]), left), right)
    py = min(max(float(player_pos[1]), top), bottom)
    # Nudge the edge point inside the bbox so borderline shots still aim at
    # the target body, not at empty pixels just outside the detector box.
    inset = min(max(2.0, (right - left) * 0.12), max(2.0, (bottom - top) * 0.12), 10.0)
    if px < cx:
        px = min(cx, px + inset)
    elif px > cx:
        px = max(cx, px - inset)
    if py < cy:
        py = min(cy, py + inset)
    elif py > cy:
        py = max(cy, py - inset)
    return (px, py)


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


def _dedupe_points(points):
    seen = set()
    result = []
    for point in points:
        key = (round(float(point[0]), 3), round(float(point[1]), 3))
        if key in seen:
            continue
        seen.add(key)
        result.append(point)
    return result


def _clear_hit_point(
    player_pos,
    target_box,
    walls,
    can_ignore_walls,
    walls_block_line_of_sight,
    max_distance=None,
    point_clear=None,
):
    points = _dedupe_points([_closest_point_on_box(player_pos, target_box)] + _bbox_hit_points(target_box))
    candidates = []
    for index, point in enumerate(points):
        if _line_of_sight_clear(player_pos, point, walls, can_ignore_walls, walls_block_line_of_sight):
            if point_clear is not None and point_clear(point):
                continue
            distance = math.hypot(point[0] - player_pos[0], point[1] - player_pos[1])
            in_window = max_distance is None or distance <= max_distance
            candidates.append((0 if in_window else 1, 0 if index == 1 else 1, distance, point))
    if candidates:
        candidates.sort(key=lambda item: item[:3])
        return candidates[0][3]
    return None


def _clamp_prediction_to_lane(player_pos, target, predicted, lead_distance, attack_range, target_box, close_snap):
    if lead_distance <= 0:
        return predicted, lead_distance, ""
    if close_snap:
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


def _normalized_box(box):
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _friendly_lane_padding(box, padding_ratio, min_padding, max_padding):
    left, top, right, bottom = _normalized_box(box)
    size = max(1.0, min(right - left, bottom - top))
    return max(float(min_padding), min(float(max_padding), size * float(padding_ratio)))


def _expand_box(box, padding):
    left, top, right, bottom = _normalized_box(box)
    padding = float(padding)
    return left - padding, top - padding, right + padding, bottom + padding


def _segment_intersects_box(p1, p2, box):
    left, top, right, bottom = _normalized_box(box)
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    t_min = 0.0
    t_max = 1.0

    for p, q in (
        (-dx, x1 - left),
        (dx, right - x1),
        (-dy, y1 - top),
        (dy, bottom - y1),
    ):
        if abs(p) < 1e-9:
            if q < 0:
                return False
            continue
        ratio = q / p
        if p < 0:
            if ratio > t_max:
                return False
            t_min = max(t_min, ratio)
        else:
            if ratio < t_min:
                return False
            t_max = min(t_max, ratio)
    return True


def _friendly_lane_block_reason(
    player_pos,
    aim_point,
    excluded_boxes,
    enabled=True,
    padding_ratio=0.25,
    min_padding=8.0,
    max_padding=28.0,
):
    if not enabled or not player_pos or not aim_point:
        return ""
    best_reason = ""
    best_distance = float("inf")
    for excluded in excluded_boxes or []:
        excluded_box, kind = _excluded_box_and_kind(excluded)
        if excluded_box is None or kind == "player":
            continue
        padding = _friendly_lane_padding(excluded_box, padding_ratio, min_padding, max_padding)
        expanded = _expand_box(excluded_box, padding)
        if not _segment_intersects_box(player_pos, aim_point, expanded):
            continue
        center = _center(excluded_box)
        distance_to_friend = math.hypot(center[0] - float(player_pos[0]), center[1] - float(player_pos[1]))
        if distance_to_friend < best_distance:
            best_distance = distance_to_friend
            best_reason = f"{kind}_lane:{padding:.1f}"
    return best_reason


def friendly_lane_block_reason(
    player_pos,
    aim_point,
    excluded_boxes,
    enabled=True,
    padding_ratio=0.25,
    min_padding=8.0,
    max_padding=28.0,
):
    return _friendly_lane_block_reason(
        player_pos,
        aim_point,
        excluded_boxes,
        enabled=enabled,
        padding_ratio=padding_ratio,
        min_padding=min_padding,
        max_padding=max_padding,
    )


def _box_iou(box_a, box_b):
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


def _same_bbox_center(a, b, max_center_jump=40.0):
    if not a or not b:
        return False
    return math.hypot(_center(a)[0] - _center(b)[0], _center(a)[1] - _center(b)[1]) <= max_center_jump


def _excluded_box_and_kind(excluded):
    if isinstance(excluded, dict):
        box = excluded.get("box")
        if box is None:
            box = excluded.get("bbox")
        return box, excluded.get("kind", "friendly")
    return excluded, "friendly"


def _center_exclusion_limit(excluded_box, kind, center_distance_px):
    limit = float(center_distance_px or 0.0)
    if kind != "player" or excluded_box is None:
        return limit
    width = abs(float(excluded_box[2]) - float(excluded_box[0]))
    height = abs(float(excluded_box[3]) - float(excluded_box[1]))
    return min(limit, max(12.0, min(width, height) * 0.25))


def _friendly_exclusion_reason(enemy_box, excluded_boxes, iou_threshold, center_distance_px):
    best_reason = ""
    best_score = 0.0
    for excluded in excluded_boxes or []:
        excluded_box, kind = _excluded_box_and_kind(excluded)
        if excluded_box is None:
            continue
        iou = _box_iou(enemy_box, excluded_box)
        center_dist = math.hypot(_center(enemy_box)[0] - _center(excluded_box)[0], _center(enemy_box)[1] - _center(excluded_box)[1])
        if kind != "player" and iou >= float(iou_threshold or 0.0):
            score = iou
            if score > best_score:
                best_score = score
                best_reason = f"{kind}_iou:{iou:.2f}"
        center_limit = _center_exclusion_limit(excluded_box, kind, center_distance_px)
        if center_dist <= center_limit:
            score = 1.0 - center_dist / max(1.0, center_limit)
            if score > best_score:
                best_score = score
                best_reason = f"{kind}_center:{center_dist:.1f}"
    return best_reason


def _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox=None):
    distance = decision.distance if decision.distance is not None else float("inf")
    preferred_rank = 0 if preferred_target_bbox and _same_bbox_center(decision.target_bbox, preferred_target_bbox) else 1
    box_score = _target_box_score(decision.target_bbox) if decision.target_bbox else 0.0
    critical_close = distance <= max(75.0, float(dangerous_close_range or 0.0) * 0.55)
    if decision.should_fire:
        close_rank = 0 if critical_close else 1 if decision.close_threat else 2
        return (0, close_rank, preferred_rank, -box_score, -(decision.confidence or 0.0), distance)
    diagnostic_close = distance <= dangerous_close_range
    close_rank = 0 if critical_close else 1 if diagnostic_close else 2
    return (1, close_rank, preferred_rank, -box_score, -(decision.confidence or 0.0), distance)


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
    preferred_target_bbox=None,
    excluded_boxes=None,
    teammate_boxes=None,
    friendly_iou_threshold=0.18,
    friendly_center_distance_px=70,
    close_attack_requires_clear_hit_point=True,
    attack_wall_guard_enabled=True,
    friendly_lane_guard_enabled=True,
    friendly_lane_padding_ratio=0.25,
    friendly_lane_min_padding=8.0,
    friendly_lane_max_padding=28.0,
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
    excluded_best = None
    all_excluded_boxes = list(excluded_boxes or []) + list(teammate_boxes or [])

    closest_enemy_distance = None
    for enemy in enemy_data:
        center = _center(enemy)
        center_distance = math.hypot(center[0] - player_pos[0], center[1] - player_pos[1])
        if closest_enemy_distance is None or center_distance < closest_enemy_distance:
            closest_enemy_distance = center_distance
        excluded_reason = _friendly_exclusion_reason(
            enemy,
            all_excluded_boxes,
            friendly_iou_threshold,
            friendly_center_distance_px,
        )
        if excluded_reason:
            decision = AttackDecision(
                False,
                target=center,
                distance=center_distance,
                attack_range=attack_range,
                in_range=center_distance <= attack_window,
                line_of_sight=False,
                reason="friendly_excluded",
                denied_by="friendly_excluded",
                threshold=min_confidence,
                los_status=excluded_reason,
                target_bbox=tuple(enemy),
                visible_enemy_count=visible_enemy_count,
                closest_enemy_distance=closest_enemy_distance,
            )
            if excluded_best is None or center_distance < (excluded_best.distance or float("inf")):
                excluded_best = decision
            continue
        closest_box_point = _closest_point_on_box(player_pos, enemy)
        closest_box_distance = math.hypot(closest_box_point[0] - player_pos[0], closest_box_point[1] - player_pos[1])
        center_los_clear = _line_of_sight_clear(player_pos, center, walls, can_ignore_walls, walls_block_line_of_sight)
        clear_target_wall_only = _clear_hit_point(
            player_pos,
            enemy,
            walls,
            can_ignore_walls,
            walls_block_line_of_sight,
            max_distance=attack_window,
        )
        friendly_lane_status = ""
        clear_target = None
        if clear_target_wall_only is not None:
            clear_target = _clear_hit_point(
                player_pos,
                enemy,
                walls,
                can_ignore_walls,
                walls_block_line_of_sight,
                max_distance=attack_window,
                point_clear=lambda point: _friendly_lane_block_reason(
                    player_pos,
                    point,
                    all_excluded_boxes,
                    enabled=friendly_lane_guard_enabled,
                    padding_ratio=friendly_lane_padding_ratio,
                    min_padding=friendly_lane_min_padding,
                    max_padding=friendly_lane_max_padding,
                ),
            )
            if clear_target is None:
                friendly_lane_status = _friendly_lane_block_reason(
                    player_pos,
                    clear_target_wall_only,
                    all_excluded_boxes,
                    enabled=friendly_lane_guard_enabled,
                    padding_ratio=friendly_lane_padding_ratio,
                    min_padding=friendly_lane_min_padding,
                    max_padding=friendly_lane_max_padding,
                ) or "friendly_lane"
        target = clear_target if clear_target is not None else center
        distance = math.hypot(target[0] - player_pos[0], target[1] - player_pos[1])
        range_distance = distance if clear_target_wall_only is not None else closest_box_distance
        threat_distance = min(center_distance, range_distance)
        in_range = range_distance <= attack_window
        close_override = bool(close_range_override and threat_distance <= dangerous_close_range)
        use_tap = threat_distance <= close_tap_range
        close_los_override_active = bool(close_override and threat_distance <= close_los_override_range)
        if not in_range:
            decision = AttackDecision(
                False,
                target=center,
                distance=range_distance,
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
                range_distance=range_distance,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox) < _decision_sort_key(best, dangerous_close_range, preferred_target_bbox):
                best = decision
            continue

        target_los_clear = clear_target is not None
        if not target_los_clear and friendly_lane_status:
            decision = AttackDecision(
                False,
                target=clear_target_wall_only or center,
                distance=range_distance,
                attack_range=attack_range,
                in_range=True,
                line_of_sight=False,
                reason="friendly_lane_blocked",
                denied_by="friendly_lane_blocked",
                use_tap=use_tap,
                threshold=min_confidence,
                close_threat=close_override,
                close_range_override=close_override,
                los_status=friendly_lane_status,
                target_bbox=tuple(enemy),
                visible_enemy_count=visible_enemy_count,
                closest_enemy_distance=closest_enemy_distance,
                range_distance=range_distance,
                friendly_lane_status=friendly_lane_status,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox) < _decision_sort_key(best, dangerous_close_range, preferred_target_bbox):
                best = decision
            continue
        if not target_los_clear:
            decision = AttackDecision(
                False,
                target=center,
                distance=range_distance,
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
                range_distance=range_distance,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox) < _decision_sort_key(best, dangerous_close_range, preferred_target_bbox):
                best = decision
            continue

        velocity = track_enemy_velocity(center, current_time)
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
            close_los_override_active,
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
                range_distance=range_distance,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox) < _decision_sort_key(best, dangerous_close_range, preferred_target_bbox):
                best = decision
            continue

        predicted_los_clear = _line_of_sight_clear(player_pos, predicted, walls, can_ignore_walls, walls_block_line_of_sight)
        if not predicted_los_clear and not close_los_override_active:
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
                range_distance=range_distance,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox) < _decision_sort_key(best, dangerous_close_range, preferred_target_bbox):
                best = decision
            continue
        if not predicted_los_clear and close_los_override_active:
            predicted = target
            predicted_distance = distance
            lead_distance = 0.0
            aim_fallback_reason = aim_fallback_reason or "close_los_snap_to_target"
            predicted_los_clear = _line_of_sight_clear(
                player_pos,
                predicted,
                walls,
                can_ignore_walls,
                walls_block_line_of_sight,
            )
            if attack_wall_guard_enabled and close_attack_requires_clear_hit_point and not predicted_los_clear:
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
                    denied_by="wall_blocked_final_hitpoint",
                    use_tap=use_tap,
                    threshold=min_confidence,
                    close_threat=close_override,
                    close_range_override=close_override,
                    los_status="blocked",
                    target_bbox=tuple(enemy),
                    visible_enemy_count=visible_enemy_count,
                    closest_enemy_distance=closest_enemy_distance,
                    aim_fallback_reason=aim_fallback_reason,
                    range_distance=range_distance,
                )
                if best is None or _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox) < _decision_sort_key(best, dangerous_close_range, preferred_target_bbox):
                    best = decision
                continue

        friendly_lane_status = _friendly_lane_block_reason(
            player_pos,
            predicted,
            all_excluded_boxes,
            enabled=friendly_lane_guard_enabled,
            padding_ratio=friendly_lane_padding_ratio,
            min_padding=friendly_lane_min_padding,
            max_padding=friendly_lane_max_padding,
        )
        if friendly_lane_status and predicted != target:
            target_lane_status = _friendly_lane_block_reason(
                player_pos,
                target,
                all_excluded_boxes,
                enabled=friendly_lane_guard_enabled,
                padding_ratio=friendly_lane_padding_ratio,
                min_padding=friendly_lane_min_padding,
                max_padding=friendly_lane_max_padding,
            )
            if not target_lane_status:
                predicted = target
                predicted_distance = distance
                lead_distance = 0.0
                aim_fallback_reason = aim_fallback_reason or "friendly_lane_snap_to_target"
                friendly_lane_status = ""
        if friendly_lane_status:
            decision = AttackDecision(
                False,
                target=target,
                predicted=predicted,
                distance=predicted_distance,
                attack_range=attack_range,
                in_range=True,
                line_of_sight=False,
                velocity=velocity,
                reason="friendly_lane_blocked",
                denied_by="friendly_lane_blocked",
                use_tap=use_tap,
                threshold=min_confidence,
                close_threat=close_override,
                close_range_override=close_override,
                los_status=friendly_lane_status,
                target_bbox=tuple(enemy),
                visible_enemy_count=visible_enemy_count,
                closest_enemy_distance=closest_enemy_distance,
                aim_fallback_reason=aim_fallback_reason,
                range_distance=range_distance,
                friendly_lane_status=friendly_lane_status,
            )
            if best is None or _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox) < _decision_sort_key(best, dangerous_close_range, preferred_target_bbox):
                best = decision
            continue

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
            los_status=(
                "blocked_center_clear_edge"
                if target_los_clear and predicted_los_clear and not center_los_clear
                else "clear" if target_los_clear and predicted_los_clear else "predicted_snap"
            ),
            target_bbox=tuple(enemy),
            visible_enemy_count=visible_enemy_count,
            closest_enemy_distance=closest_enemy_distance,
            aim_fallback_reason=aim_fallback_reason,
            range_distance=range_distance,
            friendly_lane_status=friendly_lane_status,
        )

        if best is None or _decision_sort_key(decision, dangerous_close_range, preferred_target_bbox) < _decision_sort_key(best, dangerous_close_range, preferred_target_bbox):
            best = decision

    if best:
        best.visible_enemy_count = visible_enemy_count
        best.closest_enemy_distance = closest_enemy_distance
        return best
    if excluded_best:
        excluded_best.visible_enemy_count = visible_enemy_count
        excluded_best.closest_enemy_distance = closest_enemy_distance
        return excluded_best
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
