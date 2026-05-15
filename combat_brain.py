import math
from dataclasses import dataclass, field


DEFENSIVE_MODES = {"escape_fog", "wall_escape", "unstuck", "retreat_heal", "dodge_projectile"}


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def box_center(box):
    return (float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5)


def distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


@dataclass
class FrameFacts:
    player_pos: tuple | None = None
    enemy_data: list = field(default_factory=list)
    teammate_data: list = field(default_factory=list)
    walls: list = field(default_factory=list)
    projectiles: list = field(default_factory=list)
    is_super_ready: bool = False
    is_gadget_ready: bool = False
    is_hypercharge_ready: bool = False


@dataclass
class EntityTrack:
    track_id: int
    bbox: tuple
    center: tuple
    velocity: tuple = (0.0, 0.0)
    confidence: float = 1.0
    last_seen: float = 0.0
    role: str = "enemy"


@dataclass
class HealthState:
    ratio: float | None = None
    confidence: float = 0.0
    recent_damage: bool = False
    heal_active: bool = False
    source: str = "unknown"

    @property
    def low(self):
        return bool(self.heal_active or (self.ratio is not None and self.ratio <= 0.42))


@dataclass
class TargetScore:
    bbox: tuple | None = None
    center: tuple | None = None
    distance: float | None = None
    score: float = 0.0
    line_of_sight: bool = False
    in_attack_range: bool = False
    close_threat: bool = False
    stale: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class ThreatModel:
    score: float = 0.0
    mode: str = "roam"
    reasons: list[str] = field(default_factory=list)
    enemy_close: bool = False
    outnumbered: bool = False
    low_hp: bool = False
    fog_danger: bool = False
    projectile_incoming: bool = False
    wall_trap: bool = False
    teammate_near: bool = False


@dataclass
class AbilityPlan:
    use_hypercharge: bool = False
    use_super: bool = False
    use_gadget: bool = False
    hypercharge_reason: str = "not_ready"
    super_reason: str = "not_ready"
    gadget_reason: str = "not_ready"
    hypercharge_value: float = 0.0
    super_value: float = 0.0
    gadget_value: float = 0.0
    denies: list[str] = field(default_factory=list)


@dataclass
class CombatIntent:
    mode: str = "roam"
    movement_angle: float | None = None
    target: TargetScore | None = None
    threat: ThreatModel = field(default_factory=ThreatModel)
    ability_plan: AbilityPlan = field(default_factory=AbilityPlan)
    attack_allowed: bool = False
    attack_denied_reason: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass
class SafetyResult:
    angle: float | None = None
    safe: bool = True
    status: str = "not_checked"
    reasons: list[str] = field(default_factory=list)


@dataclass
class CombatFrame:
    player_pos: tuple | None = None
    enemy_data: list = field(default_factory=list)
    teammate_data: list = field(default_factory=list)
    walls: list = field(default_factory=list)
    projectiles: list = field(default_factory=list)
    health: HealthState = field(default_factory=HealthState)
    desired_angle: float | None = None
    current_mode: str | None = None
    safe_range: float = 0.0
    attack_range: float = 0.0
    fog_danger: bool = False
    projectile_incoming: bool = False
    wall_trap: bool = False
    teammate_near: bool = False
    suppress_active: bool = False


class TargetMemory:
    """Small target lock with stale memory for smoother aim decisions."""

    def __init__(self):
        self.locked_target: TargetScore | None = None
        self.locked_until = 0.0

    @staticmethod
    def _same_target(a, b, max_center_jump=135.0):
        if not a or not b or a.center is None or b.center is None:
            return False
        return distance(a.center, b.center) <= float(max_center_jump)

    def reset(self):
        self.locked_target = None
        self.locked_until = 0.0

    def choose(
            self,
            *,
            now,
            memory_seconds=0.75,
            switch_margin=0.18,
            **score_kwargs):
        scores = score_targets(**score_kwargs)
        if not scores:
            if self.locked_target and now <= self.locked_until:
                old = self.locked_target
                return TargetScore(
                    bbox=old.bbox,
                    center=old.center,
                    distance=old.distance,
                    score=clamp(old.score * 0.35),
                    line_of_sight=False,
                    in_attack_range=False,
                    close_threat=False,
                    stale=True,
                    reasons=[*old.reasons, "stale_memory"],
                )
            self.reset()
            return None

        best = scores[0]
        locked_current = None
        if self.locked_target is not None:
            closest_locked_distance = float("inf")
            for candidate in scores:
                if self._same_target(self.locked_target, candidate):
                    locked_distance = distance(self.locked_target.center, candidate.center)
                    if locked_distance >= closest_locked_distance:
                        continue
                    closest_locked_distance = locked_distance
                    locked_current = candidate

        chosen = best
        if locked_current is not None and best is not locked_current:
            close_override = bool(best.close_threat and not locked_current.close_threat)
            meaningful_upgrade = best.score >= locked_current.score + float(switch_margin or 0.0)
            if not close_override and not meaningful_upgrade:
                chosen = locked_current

        self.locked_target = chosen
        self.locked_until = now + max(0.0, float(memory_seconds or 0.0))
        return chosen


def score_targets(
    *,
    player_pos,
    enemy_data,
    safe_range,
    attack_range,
    walls,
    can_attack_through_walls,
    walls_block_line_of_sight,
    dangerous_close_range,
):
    if player_pos is None or not enemy_data:
        return []

    scores = []
    attack_window = float(attack_range or 0.0) * 1.035
    for enemy in enemy_data:
        center = box_center(enemy)
        dist = distance(player_pos, center)
        in_attack_range = dist <= attack_window
        los = bool(can_attack_through_walls) or not walls_block_line_of_sight(player_pos, center, walls)
        close = dist <= max(float(dangerous_close_range or 0.0), float(safe_range or 0.0) * 0.7)
        score = 0.0
        reasons = []
        if close:
            score += 0.42
            reasons.append("close_threat")
        if los:
            score += 0.26
            reasons.append("line_of_sight")
        if in_attack_range:
            score += 0.20
            reasons.append("in_attack_range")
        if dist <= safe_range:
            score += 0.10
            reasons.append("inside_safe_range")
        score += max(0.0, 1.0 - dist / max(1.0, attack_window)) * 0.18
        scores.append(
            TargetScore(
                bbox=tuple(enemy),
                center=center,
                distance=dist,
                score=clamp(score),
                line_of_sight=los,
                in_attack_range=in_attack_range,
                close_threat=close,
                reasons=reasons,
            )
        )

    scores.sort(key=lambda item: (item.close_threat, item.score, -(item.distance or 99999.0)), reverse=True)
    return scores


def choose_target(**kwargs):
    scores = score_targets(**kwargs)
    return scores[0] if scores else None


def choose_combat_intent(
    *,
    frame,
    target=None,
    safety=None,
    ability_plan=None,
    defensive_gate_enabled=True,
    panic_shot_range=150.0,
):
    safety = safety or SafetyResult(angle=frame.desired_angle, safe=True, status="not_checked")
    target = target if target is not None else None
    enemy_count_in_range = 0
    if frame.player_pos is not None:
        for enemy in frame.enemy_data or []:
            try:
                if distance(frame.player_pos, box_center(enemy)) <= float(frame.attack_range or 0.0):
                    enemy_count_in_range += 1
            except (TypeError, IndexError):
                continue
    threat = build_threat_model(
        target=target,
        enemy_count_in_range=enemy_count_in_range,
        health=frame.health,
        fog_danger=frame.fog_danger,
        projectile_incoming=frame.projectile_incoming,
        wall_trap=frame.wall_trap or not safety.safe,
        teammate_near=frame.teammate_near,
        safe_range=frame.safe_range,
    )

    mode = frame.current_mode or threat.mode
    if frame.fog_danger:
        mode = "escape_fog"
    elif frame.wall_trap or not safety.safe:
        mode = "wall_escape"
    elif frame.projectile_incoming:
        mode = "dodge_projectile"
    elif frame.health.low and mode not in {"escape_fog", "wall_escape", "dodge_projectile"}:
        mode = "retreat_heal"
    elif target and target.close_threat and not frame.health.low:
        mode = "kite_close_enemy"
    elif target and target.in_attack_range and target.line_of_sight:
        mode = "strafe_attack_lane"

    threat.mode = mode
    attack_allowed, denied_reason = choose_attack_gate(
        mode=mode,
        target=target,
        health=frame.health,
        defensive_gate_enabled=defensive_gate_enabled,
        panic_shot_range=panic_shot_range,
        suppress_active=frame.suppress_active,
    )
    if target and target.stale:
        attack_allowed = False
        denied_reason = "stale_target"

    reasons = list(threat.reasons)
    if safety.status != "not_checked":
        reasons.append(f"safety:{safety.status}")
    if frame.projectile_incoming and not frame.enemy_data:
        reasons.append("projectile_without_enemy")

    return CombatIntent(
        mode=mode,
        movement_angle=safety.angle if safety.angle is not None else frame.desired_angle,
        target=target,
        threat=threat,
        ability_plan=ability_plan or AbilityPlan(),
        attack_allowed=attack_allowed,
        attack_denied_reason=denied_reason,
        reasons=reasons,
    )


def build_threat_model(
    *,
    target=None,
    enemy_count_in_range=0,
    health=None,
    fog_danger=False,
    projectile_incoming=False,
    wall_trap=False,
    teammate_near=False,
    safe_range=0.0,
):
    reasons = []
    score = 0.0
    enemy_close = bool(target and target.distance is not None and target.distance <= max(80.0, float(safe_range or 0.0)))
    if enemy_close:
        score += 0.28
        reasons.append("enemy_close")
    elif target and target.in_attack_range:
        score += 0.12
        reasons.append("enemy_in_attack_range")

    outnumbered = int(enemy_count_in_range or 0) >= 2
    if outnumbered:
        score += 0.18
        reasons.append("outnumbered")

    low_hp = bool(health and health.low)
    if low_hp:
        score += 0.26
        reasons.append("low_hp")
    if health and health.recent_damage:
        score += 0.10
        reasons.append("recent_damage")

    if projectile_incoming:
        score += 0.32
        reasons.append("projectile_incoming")
    if fog_danger:
        score += 0.45
        reasons.append("fog_danger")
    if wall_trap:
        score += 0.20
        reasons.append("wall_trap")
    if teammate_near:
        score = max(0.0, score - 0.08)
        reasons.append("teammate_near")

    if fog_danger:
        mode = "escape_fog"
    elif wall_trap:
        mode = "wall_escape"
    elif projectile_incoming and score >= 0.45:
        mode = "dodge_projectile"
    elif low_hp and score >= 0.35:
        mode = "retreat_heal"
    elif enemy_close:
        mode = "kite_close_enemy"
    elif target and target.in_attack_range and target.line_of_sight:
        mode = "strafe_attack_lane"
    elif teammate_near:
        mode = "regroup_teammate"
    elif target:
        mode = "approach"
    else:
        mode = "roam"

    return ThreatModel(
        score=clamp(score),
        mode=mode,
        reasons=reasons,
        enemy_close=enemy_close,
        outnumbered=outnumbered,
        low_hp=low_hp,
        fog_danger=bool(fog_danger),
        projectile_incoming=bool(projectile_incoming),
        wall_trap=bool(wall_trap),
        teammate_near=bool(teammate_near),
    )


def choose_attack_gate(
    *,
    mode,
    target=None,
    health=None,
    defensive_gate_enabled=True,
    panic_shot_range=150.0,
    suppress_active=False,
):
    if not defensive_gate_enabled:
        return True, ""
    defensive = mode in DEFENSIVE_MODES or bool(suppress_active)
    if not defensive:
        return True, ""

    panic = bool(
        target
        and target.close_threat
        and target.line_of_sight
        and target.distance is not None
        and target.distance <= float(panic_shot_range or 0.0)
    )
    if panic:
        return True, "panic_shot"
    if suppress_active:
        return False, "attack_suppressed"
    if health and health.low:
        return False, "retreat_heal"
    return False, f"{mode}_blocks_attack"


def choose_ability_plan(
    *,
    target=None,
    threat=None,
    health=None,
    super_type="damage",
    super_ready=False,
    gadget_ready=False,
    hypercharge_ready=False,
    gadget_enabled=True,
    holding_attack=False,
    super_hittable=False,
    attack_hittable=False,
    enemy_count_in_range=0,
    teammate_near=False,
    super_cooldown_remaining_ms=0,
    gadget_cooldown_remaining_ms=0,
    super_min_value_score=0.55,
    gadget_min_value_score=0.50,
    hypercharge_min_value_score=0.70,
    panic_super_range=180.0,
    charge_path_safe=True,
):
    plan = AbilityPlan()
    denies = plan.denies
    threat = threat or ThreatModel()
    health = health or HealthState()

    has_target = bool(target and not target.stale and target.distance is not None)
    target_clear = bool(has_target and target.line_of_sight)
    target_in_range = bool(has_target and target.in_attack_range)
    panic = bool(has_target and target.close_threat and target.distance <= float(panic_super_range or 0.0))
    defensive_mode = threat.mode in DEFENSIVE_MODES
    committed_fight = bool(target_clear and (target_in_range or panic or threat.outnumbered or teammate_near))

    if not super_ready:
        plan.super_reason = "not_ready"
    elif super_cooldown_remaining_ms > 0:
        plan.super_reason = "cooldown"
        denies.append("super:cooldown")
    elif not has_target and super_type != "charge":
        plan.super_reason = "no_target"
        denies.append("super:no_target")
    elif defensive_mode and not panic and super_type != "charge":
        plan.super_reason = threat.mode
        denies.append(f"super:{threat.mode}")
    else:
        value = 0.0
        if panic:
            value += 0.42
        if target_clear or super_type == "charge":
            value += 0.22
        if target_in_range:
            value += 0.16
        if threat.outnumbered:
            value += 0.16
        if teammate_near:
            value += 0.08
        if threat.low_hp and not panic and super_type != "charge":
            value -= 0.20

        if super_type == "damage":
            if not super_hittable:
                plan.super_reason = "blocked_los"
                denies.append("super:blocked_los")
            else:
                plan.super_value = clamp(value + 0.12)
        elif super_type == "charge":
            if not charge_path_safe:
                plan.super_reason = "unsafe_charge_path"
                denies.append("super:unsafe_charge_path")
            elif defensive_mode and (threat.low_hp or threat.fog_danger or threat.projectile_incoming):
                plan.super_value = clamp(value + 0.28)
            elif committed_fight and not threat.low_hp:
                plan.super_value = clamp(value + 0.10)
            else:
                plan.super_value = clamp(value - 0.10)
        elif super_type in {"spawnable", "other"}:
            if committed_fight and not (threat.low_hp and not panic):
                plan.super_value = clamp(value + 0.10)
            else:
                plan.super_value = clamp(value - 0.15)
        else:
            if target_clear and committed_fight:
                plan.super_value = clamp(value + 0.06)
            else:
                plan.super_value = clamp(value - 0.15)

        if plan.super_value >= float(super_min_value_score):
            plan.use_super = True
            plan.super_reason = f"valuable_{super_type}"
        elif plan.super_reason == "not_ready":
            plan.super_reason = "low_value"
            denies.append("super:low_value")

    if not gadget_enabled:
        plan.gadget_reason = "disabled"
    elif not gadget_ready:
        plan.gadget_reason = "not_ready"
    elif holding_attack:
        plan.gadget_reason = "holding_attack"
        denies.append("gadget:holding_attack")
    elif gadget_cooldown_remaining_ms > 0:
        plan.gadget_reason = "cooldown"
        denies.append("gadget:cooldown")
    elif not has_target:
        plan.gadget_reason = "no_target"
        denies.append("gadget:no_target")
    elif defensive_mode and not panic:
        plan.gadget_reason = threat.mode
        denies.append(f"gadget:{threat.mode}")
    elif not attack_hittable:
        plan.gadget_reason = "blocked_los"
        denies.append("gadget:blocked_los")
    else:
        value = 0.0
        if panic:
            value += 0.36
        if target_in_range:
            value += 0.18
        if enemy_count_in_range >= 2:
            value += 0.18
        if plan.use_super:
            value += 0.10
        if teammate_near:
            value += 0.06
        plan.gadget_value = clamp(value)
        if plan.gadget_value >= float(gadget_min_value_score):
            plan.use_gadget = True
            plan.gadget_reason = "valuable_gadget"
        else:
            plan.gadget_reason = "low_value"
            denies.append("gadget:low_value")

    if not hypercharge_ready:
        plan.hypercharge_reason = "not_ready"
    else:
        value = 0.0
        if plan.use_super:
            value += 0.48
        if committed_fight:
            value += 0.18
        if threat.outnumbered:
            value += 0.16
        value += threat.score * 0.45
        if has_target and target.close_threat:
            value += 0.12
        if panic and not threat.low_hp:
            value += 0.12
        if defensive_mode and not panic:
            value -= 0.35
        if health.low and not panic:
            value -= 0.20
        plan.hypercharge_value = clamp(value)
        if plan.hypercharge_value >= float(hypercharge_min_value_score):
            plan.use_hypercharge = True
            plan.hypercharge_reason = "combo_super" if plan.use_super else "committed_fight"
        else:
            plan.hypercharge_reason = "low_value"
            denies.append("hypercharge:low_value")

    if plan.use_hypercharge and plan.use_super and plan.use_gadget:
        plan.use_gadget = False
        plan.gadget_reason = "combo_limit"
        denies.append("gadget:combo_limit")

    return plan
