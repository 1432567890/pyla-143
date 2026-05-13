import math


def angle_from_vector(dx, dy):
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360


def vector_from_angle(angle_degrees):
    rad = math.radians(float(angle_degrees) % 360)
    return math.cos(rad), math.sin(rad)


def angle_delta(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def movement_keys_to_angle(movement):
    dx = 0.0
    dy = 0.0
    movement = str(movement or "").lower()
    if "a" in movement:
        dx -= 1.0
    if "d" in movement:
        dx += 1.0
    if "w" in movement:
        dy -= 1.0
    if "s" in movement:
        dy += 1.0
    return angle_from_vector(dx, dy)


def threat_level_from_distance(distance, attack_range, safe_range):
    if distance is None or attack_range <= 0:
        return 0.0
    if distance <= max(80.0, safe_range * 0.55):
        return 1.0
    if distance <= safe_range:
        return 0.82
    if distance <= attack_range:
        return 0.55
    if distance <= attack_range * 1.18:
        return 0.28
    return 0.0


def classify_dodge_mode(threat_level, closest_distance, safe_range, flicker_active=False):
    if threat_level <= 0.0 and not flicker_active:
        return "no_dodge"
    if closest_distance is not None and closest_distance <= max(75.0, safe_range * 0.52):
        return "kite"
    if flicker_active and threat_level < 0.55:
        return "cover_shift"
    if threat_level >= 0.78:
        return "hard_dodge"
    return "soft_strafe"


def score_dodge_angle(
    candidate_angle,
    *,
    base_angle,
    threat_angle,
    closest_enemy_distance,
    safe_range,
    attack_range,
    is_blocked,
    points_into_fog,
    current_angle=None,
    teammate_angle=None,
):
    score = 0.0
    reasons = []

    if is_blocked(candidate_angle):
        return -999.0, ["blocked_by_wall"]
    if points_into_fog(candidate_angle):
        return -950.0, ["blocked_by_poison"]

    if threat_angle is not None:
        away_angle = (threat_angle + 180.0) % 360.0
        away_alignment = 180.0 - angle_delta(candidate_angle, away_angle)
        lateral_alignment = 90.0 - abs(angle_delta(candidate_angle, threat_angle) - 90.0)
        score += max(0.0, away_alignment) * 0.030
        score += max(0.0, lateral_alignment) * 0.045
        reasons.append("threat_lateral")

    if closest_enemy_distance is not None and closest_enemy_distance <= safe_range:
        away_angle = (threat_angle + 180.0) % 360.0 if threat_angle is not None else base_angle
        score += max(0.0, 180.0 - angle_delta(candidate_angle, away_angle)) * 0.035
        reasons.append("kite_distance")
    elif attack_range and closest_enemy_distance is not None and closest_enemy_distance <= attack_range:
        score += max(0.0, 110.0 - angle_delta(candidate_angle, base_angle)) * 0.010
        reasons.append("keeps_attack_lane")

    if teammate_angle is not None and closest_enemy_distance is not None and closest_enemy_distance > safe_range:
        score += max(0.0, 120.0 - angle_delta(candidate_angle, teammate_angle)) * 0.010
        reasons.append("team_bias")

    if current_angle is not None:
        score += max(0.0, 80.0 - angle_delta(candidate_angle, current_angle)) * 0.012
        reasons.append("hysteresis")

    score += max(0.0, 90.0 - angle_delta(candidate_angle, base_angle)) * 0.006
    return score, reasons


def candidate_dodge_angles(base_angle, threat_angle):
    if base_angle is None and threat_angle is None:
        return []
    anchor = threat_angle if threat_angle is not None else base_angle
    away = (anchor + 180.0) % 360.0
    candidates = [
        (anchor + 90.0) % 360.0,
        (anchor - 90.0) % 360.0,
        (away + 35.0) % 360.0,
        (away - 35.0) % 360.0,
        (anchor + 55.0) % 360.0,
        (anchor - 55.0) % 360.0,
        away,
    ]
    if base_angle is not None:
        candidates.extend([(base_angle + 30.0) % 360.0, (base_angle - 30.0) % 360.0, base_angle])
    deduped = []
    for angle in candidates:
        if all(angle_delta(angle, existing) > 8.0 for existing in deduped):
            deduped.append(angle)
    return deduped


def projectile_threat(projectile_pos, projectile_velocity, player_pos, player_radius=34.0, horizon_seconds=0.75):
    """Estimate whether a moving projectile will cross the player's hit area."""
    if projectile_pos is None or projectile_velocity is None or player_pos is None:
        return None
    px, py = float(projectile_pos[0]), float(projectile_pos[1])
    vx, vy = float(projectile_velocity[0]), float(projectile_velocity[1])
    speed = math.hypot(vx, vy)
    if speed < 30.0:
        return None

    tx, ty = float(player_pos[0]), float(player_pos[1])
    rel_x, rel_y = tx - px, ty - py
    t = (rel_x * vx + rel_y * vy) / (speed * speed)
    if t < 0.0 or t > horizon_seconds:
        return None

    closest_x = px + vx * t
    closest_y = py + vy * t
    miss_distance = math.hypot(tx - closest_x, ty - closest_y)
    if miss_distance > player_radius:
        return None

    travel_angle = angle_from_vector(vx, vy)
    if travel_angle is None:
        return None
    danger = max(0.0, min(1.0, 1.0 - miss_distance / max(1.0, player_radius)))
    return {
        "time_to_impact": t,
        "miss_distance": miss_distance,
        "danger": danger,
        "travel_angle": travel_angle,
        "escape_angles": ((travel_angle + 90.0) % 360.0, (travel_angle - 90.0) % 360.0),
    }


def score_projectile_dodge_angle(candidate_angle, threat):
    if not threat:
        return 0.0, []
    lateral_1, lateral_2 = threat["escape_angles"]
    lateral_alignment = max(
        0.0,
        120.0 - min(angle_delta(candidate_angle, lateral_1), angle_delta(candidate_angle, lateral_2)),
    )
    urgency = 1.0 + max(0.0, 0.55 - float(threat["time_to_impact"])) * 1.8
    return lateral_alignment * 0.075 * urgency * (0.65 + threat["danger"] * 0.7), ["projectile_lateral"]


def should_seek_healing(health_ratio, recent_damage=False, active_until=0.0, now=0.0, low_threshold=0.42):
    if active_until and now < active_until:
        return True
    if health_ratio is None:
        return bool(recent_damage)
    return health_ratio <= low_threshold or (recent_damage and health_ratio <= low_threshold + 0.18)
