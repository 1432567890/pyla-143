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


def should_seek_healing(health_ratio, recent_damage=False, active_until=0.0, now=0.0, low_threshold=0.42):
    if active_until and now < active_until:
        return True
    if health_ratio is None:
        return bool(recent_damage)
    return health_ratio <= low_threshold or (recent_damage and health_ratio <= low_threshold + 0.18)
