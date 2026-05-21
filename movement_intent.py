import math
from dataclasses import dataclass, field


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def angle_delta(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def angle_to_vector(angle_degrees, weight=1.0):
    rad = math.radians(float(angle_degrees) % 360.0)
    return math.cos(rad) * weight, math.sin(rad) * weight


def vector_to_angle(dx, dy, fallback=None):
    if math.hypot(dx, dy) < 1e-6:
        return fallback
    return math.degrees(math.atan2(dy, dx)) % 360.0


def blend_angles(primary_angle, secondary_angle, secondary_weight):
    if primary_angle is None:
        return secondary_angle
    if secondary_angle is None:
        return primary_angle
    secondary_weight = clamp(secondary_weight)
    ax, ay = angle_to_vector(primary_angle, 1.0 - secondary_weight)
    bx, by = angle_to_vector(secondary_angle, secondary_weight)
    return vector_to_angle(ax + bx, ay + by, primary_angle)


@dataclass
class ThreatState:
    total_score: float = 0.0
    closest_enemy_distance: float | None = None
    enemy_close: bool = False
    enemy_approaching: bool = False
    enemy_has_line: bool = False
    projectile_incoming: bool = False
    fog_danger: bool = False
    outnumbered: bool = False
    low_hp: bool = False
    teammate_near: bool = False
    attack_lane_available: bool = False
    wall_pressure: bool = False
    reasons: list[str] = field(default_factory=list)
    projectile: dict | None = None


@dataclass
class MovementVector:
    name: str
    angle: float
    weight: float
    reason: str


@dataclass
class MovementIntent:
    mode: str
    angle: float
    score: float
    reasons: list[str]
    attack_allowed: bool
    super_allowed: bool = False
    hold_ms: int = 400


@dataclass
class MovementIntentMemory:
    intent: MovementIntent | None = None
    started_at: float = 0.0


MODE_PRIORITY = {
    "hold_position": 0,
    "regroup": 1,
    "approach": 2,
    "strafe": 3,
    "kite": 4,
    "retreat_heal": 5,
    "dodge_projectile": 6,
    "unstuck": 7,
    "escape_fog": 8,
}


def build_threat_state(
    *,
    closest_enemy_distance=None,
    safe_range=0.0,
    attack_range=0.0,
    enemy_velocity=(0.0, 0.0),
    vector_to_enemy=(0.0, 0.0),
    enemy_has_line=False,
    projectile=None,
    fog_danger=False,
    nearby_enemy_count=0,
    health_ratio=None,
    teammate_distance=None,
    teammate_near_range=360.0,
    wall_pressure=False,
    attack_lane_available=False,
    low_health_threshold=0.42,
):
    reasons = []
    score = 0.0
    safe_range = max(1.0, float(safe_range or 1.0))
    attack_range = max(safe_range, float(attack_range or safe_range))

    enemy_close = closest_enemy_distance is not None and closest_enemy_distance <= safe_range
    if closest_enemy_distance is not None:
        if enemy_close:
            score += 0.32
            reasons.append("enemy_too_close")
        elif closest_enemy_distance <= attack_range:
            score += 0.16
            reasons.append("enemy_in_attack_range")

    vx, vy = enemy_velocity or (0.0, 0.0)
    ex, ey = vector_to_enemy or (0.0, 0.0)
    enemy_approaching = False
    if math.hypot(vx, vy) > 35.0 and math.hypot(ex, ey) > 1.0:
        # Enemy position vector is player->enemy. Approaching means enemy
        # velocity points back toward the player.
        dot = vx * (-ex) + vy * (-ey)
        enemy_approaching = dot > 0
        if enemy_approaching:
            score += 0.12
            reasons.append("enemy_moving_toward_us")

    if enemy_has_line:
        score += 0.12
        reasons.append("enemy_attack_lane_open")

    projectile_incoming = bool(projectile)
    if projectile_incoming:
        danger = clamp(projectile.get("danger", 0.75))
        score += 0.34 + danger * 0.20
        reasons.append("projectile_crossing_path")

    if fog_danger:
        score += 0.55
        reasons.append("fog_danger")

    outnumbered = int(nearby_enemy_count or 0) >= 2
    if outnumbered:
        score += 0.15
        reasons.append("outnumbered")

    low_hp = health_ratio is not None and health_ratio <= float(low_health_threshold or 0.42)
    if low_hp:
        score += 0.24
        reasons.append("low_hp")

    teammate_near = teammate_distance is not None and teammate_distance <= teammate_near_range
    if teammate_near:
        score = max(0.0, score - 0.08)
        reasons.append("teammate_near")

    if attack_lane_available:
        reasons.append("attack_lane_available")

    if wall_pressure:
        score += 0.10
        reasons.append("wall_pressure")

    return ThreatState(
        total_score=clamp(score),
        closest_enemy_distance=closest_enemy_distance,
        enemy_close=enemy_close,
        enemy_approaching=enemy_approaching,
        enemy_has_line=bool(enemy_has_line),
        projectile_incoming=projectile_incoming,
        fog_danger=bool(fog_danger),
        outnumbered=outnumbered,
        low_hp=low_hp,
        teammate_near=teammate_near,
        attack_lane_available=bool(attack_lane_available),
        wall_pressure=bool(wall_pressure),
        reasons=reasons,
        projectile=projectile,
    )


def mix_movement_vectors(vectors, fallback_angle):
    dx = 0.0
    dy = 0.0
    reasons = []
    total_weight = 0.0
    for vector in vectors:
        if vector is None or vector.angle is None or vector.weight <= 0:
            continue
        vx, vy = angle_to_vector(vector.angle, vector.weight)
        dx += vx
        dy += vy
        total_weight += vector.weight
        reasons.append(vector.reason)
    angle = vector_to_angle(dx, dy, fallback_angle)
    confidence = clamp(total_weight / 1.8)
    return angle, confidence, reasons


def choose_intent_mode(threat, *, enemy_visible, enemy_distance=None, safe_range=0.0, attack_range=0.0, base_mode="approach"):
    if threat.fog_danger:
        return "escape_fog"
    if threat.projectile_incoming:
        return "dodge_projectile"
    if threat.low_hp and threat.total_score >= 0.35:
        return "retreat_heal"
    if threat.enemy_close:
        return "kite"
    if enemy_visible and enemy_distance is not None and enemy_distance <= attack_range and threat.attack_lane_available:
        return "strafe"
    if enemy_visible:
        return "approach"
    return base_mode


def build_movement_intent(
    *,
    threat,
    base_angle,
    enemy_visible=False,
    enemy_distance=None,
    safe_range=0.0,
    attack_range=0.0,
    toward_enemy_angle=None,
    away_enemy_angle=None,
    strafe_angle=None,
    projectile_escape_angle=None,
    fog_escape_angle=None,
    teammate_angle=None,
    heal_retreat_angle=None,
    wall_escape_angle=None,
    heal_attack_range=None,
):
    mode = choose_intent_mode(
        threat,
        enemy_visible=enemy_visible,
        enemy_distance=enemy_distance,
        safe_range=safe_range,
        attack_range=attack_range,
        base_mode="regroup" if teammate_angle is not None else "hold_position",
    )

    vectors = [MovementVector("base", base_angle, 0.30, "base_movement")]
    hold_ms = 420
    attack_allowed = bool(threat.attack_lane_available and not threat.fog_danger)

    if mode == "escape_fog":
        vectors.append(MovementVector("fog", fog_escape_angle, 1.25, "escape_fog"))
        attack_allowed = False if enemy_distance is None or enemy_distance > attack_range * 0.72 else attack_allowed
        hold_ms = 520
    elif mode == "dodge_projectile":
        vectors.append(MovementVector("projectile", projectile_escape_angle, 1.05, "projectile_lateral"))
        if away_enemy_angle is not None and threat.enemy_close:
            vectors.append(MovementVector("kite", away_enemy_angle, 0.35, "keep_safe_distance"))
        if strafe_angle is not None and threat.attack_lane_available:
            vectors.append(MovementVector("attack_lane", strafe_angle, 0.25, "keeps_attack_lane"))
        hold_ms = 480
    elif mode == "retreat_heal":
        vectors.append(MovementVector("heal", heal_retreat_angle or away_enemy_angle, 0.95, "retreat_heal"))
        if teammate_angle is not None:
            vectors.append(MovementVector("team", teammate_angle, 0.25, "regroup_with_teammate"))
        retreat_attack_range = max(
            0.0,
            float(heal_attack_range if heal_attack_range is not None else max(120.0, attack_range * 0.30)),
        )
        attack_allowed = bool(enemy_distance is not None and enemy_distance <= retreat_attack_range)
        hold_ms = 580
    elif mode == "kite":
        vectors.append(MovementVector("kite", away_enemy_angle, 0.80, "enemy_too_close"))
        if strafe_angle is not None:
            vectors.append(MovementVector("strafe", strafe_angle, 0.35, "keeps_attack_lane"))
        hold_ms = 460
    elif mode == "strafe":
        vectors.append(MovementVector("strafe", strafe_angle, 0.75, "strafe_attack_lane"))
        vectors.append(MovementVector("toward", toward_enemy_angle, 0.20, "maintain_pressure"))
        hold_ms = 430
    elif mode == "approach":
        vectors.append(MovementVector("toward", toward_enemy_angle, 0.70, "approach_enemy"))
        if strafe_angle is not None:
            vectors.append(MovementVector("flank", strafe_angle, 0.18, "approach_flank"))
        hold_ms = 390
    elif mode == "regroup":
        vectors.append(MovementVector("team", teammate_angle, 0.70, "regroup_with_teammate"))
        hold_ms = 500
    elif mode == "unstuck":
        vectors.append(MovementVector("wall", wall_escape_angle, 1.0, "unstuck"))
        attack_allowed = False
        hold_ms = 600

    angle, confidence, reasons = mix_movement_vectors(vectors, base_angle)
    score = clamp(max(threat.total_score, confidence))
    merged_reasons = []
    for reason in list(threat.reasons) + reasons:
        if reason and reason not in merged_reasons:
            merged_reasons.append(reason)
    return MovementIntent(
        mode=mode,
        angle=angle,
        score=score,
        reasons=merged_reasons,
        attack_allowed=attack_allowed,
        super_allowed=attack_allowed and mode in {"kite", "strafe", "approach", "dodge_projectile"},
        hold_ms=hold_ms,
    )


def smooth_intent(
    previous,
    new_intent,
    *,
    now,
    min_hold_ms=350,
    max_hold_ms=650,
    switch_score_threshold=0.18,
    angle_smoothing=0.35,
):
    if previous is None or previous.intent is None:
        return MovementIntentMemory(new_intent, now), new_intent, "new_intent"

    old = previous.intent
    held_ms = (now - previous.started_at) * 1000.0
    old_priority = MODE_PRIORITY.get(old.mode, 0)
    new_priority = MODE_PRIORITY.get(new_intent.mode, 0)
    serious_upgrade = new_priority >= old_priority + 2
    old_expired = held_ms >= min(max_hold_ms, max(min_hold_ms, old.hold_ms))
    score_upgrade = new_intent.score >= old.score + switch_score_threshold

    if held_ms < min_hold_ms and not serious_upgrade:
        return previous, old, "min_hold_active"

    if not old_expired and not serious_upgrade and not score_upgrade and new_priority <= old_priority:
        return previous, old, "score_delta_too_small"

    smoothed = new_intent
    if old.angle is not None and new_intent.angle is not None and angle_smoothing > 0:
        smoothed = MovementIntent(
            mode=new_intent.mode,
            angle=blend_angles(new_intent.angle, old.angle, angle_smoothing),
            score=new_intent.score,
            reasons=new_intent.reasons,
            attack_allowed=new_intent.attack_allowed,
            super_allowed=new_intent.super_allowed,
            hold_ms=new_intent.hold_ms,
        )
    return MovementIntentMemory(smoothed, now), smoothed, "switched"
