import math
from dataclasses import dataclass, field


DEFENSIVE_MODES = {
    "escape_fog",
    "wall_escape",
    "unstuck",
    "retreat_heal",
    "dodge_projectile",
    "survive",
    "disengage",
    "hold_cover",
}


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
class KillOpportunity:
    score: float = 0.0
    low_enemy_proxy: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class TacticalAngleScore:
    angle: float | None = None
    score: float = 0.0
    survival_score: float = 0.0
    engagement_score: float = 0.0
    fire_window: bool = False
    rejected: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class TacticalPlan:
    objective: str = "hold_cover"
    position_goal: float | None = None
    engagement_score: float = 0.0
    survival_score: float = 0.0
    kill_confirm_score: float = 0.0
    fire_window: bool = False
    retreat_vector: float | None = None
    commit_allowed: bool = False
    selected_angle: float | None = None
    rejected_angles: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    kill_opportunity: KillOpportunity = field(default_factory=KillOpportunity)


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
    tactical_plan: TacticalPlan = field(default_factory=TacticalPlan)


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
    projectile_dodge_angle: float | None = None
    teammate_angle: float | None = None
    preferred_distance: float | None = None
    commit_distance: float | None = None
    aggression_penalty: float = 0.0
    fire_threshold_delta: float = 0.0


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


def angle_delta(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def angle_from_points(a, b):
    return math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))) % 360.0


def project_point(origin, angle, distance_px):
    radians = math.radians(float(angle))
    return (
        float(origin[0]) + math.cos(radians) * float(distance_px),
        float(origin[1]) + math.sin(radians) * float(distance_px),
    )


def generate_angle_candidates(base_angle, samples=16, extra_angles=None):
    samples = max(4, int(samples or 16))
    base_angle = 0.0 if base_angle is None else float(base_angle) % 360.0
    candidates = []
    for offset in (0, 35, -35, 70, -70, 110, -110, 180):
        candidates.append((base_angle + offset) % 360.0)
    step = 360.0 / samples
    for index in range(samples):
        candidates.append((index * step) % 360.0)
    for angle in extra_angles or []:
        if angle is not None:
            candidates.append(float(angle) % 360.0)

    result = []
    seen = set()
    for angle in candidates:
        key = round(angle, 1)
        if key in seen:
            continue
        seen.add(key)
        result.append(angle)
    return result


def score_kill_opportunity(
        *,
        target=None,
        health=None,
        threat=None,
        teammate_near=False,
        kill_confirm_score_threshold=0.68):
    if not target or target.stale or target.distance is None:
        return KillOpportunity(score=0.0, reasons=["no_fresh_target"])
    reasons = []
    score = 0.0
    low_enemy_proxy = clamp(target.score * 0.45)
    if target.line_of_sight:
        score += 0.24
        reasons.append("los")
    if target.in_attack_range:
        score += 0.20
        reasons.append("in_range")
    if target.close_threat:
        score += 0.18
        reasons.append("close")
    if teammate_near:
        score += 0.10
        reasons.append("teammate_pressure")
    if threat and threat.outnumbered:
        score += 0.08
        reasons.append("multi_enemy_value")
    if target.distance <= 120:
        score += 0.10
        reasons.append("finish_distance")
    if health and health.low and not target.close_threat:
        score -= 0.18
        reasons.append("low_hp_penalty")
    score += low_enemy_proxy
    if score >= float(kill_confirm_score_threshold or 0.68):
        reasons.append("kill_confirm")
    return KillOpportunity(score=clamp(score), low_enemy_proxy=low_enemy_proxy, reasons=reasons)


def choose_tactical_plan(
    *,
    frame,
    target=None,
    safety=None,
    angle_samples=16,
    is_angle_blocked=None,
    points_into_fog=None,
    projectile_danger_for_angle=None,
    line_of_sight_after_move=None,
    survival_score_min_to_commit=0.62,
    kill_confirm_score_threshold=0.68,
):
    safety = safety or SafetyResult(angle=frame.desired_angle, safe=True, status="not_checked")
    is_angle_blocked = is_angle_blocked or (lambda _angle: False)
    points_into_fog = points_into_fog or (lambda _angle: False)
    projectile_danger_for_angle = projectile_danger_for_angle or (lambda _angle: 0.0)
    line_of_sight_after_move = line_of_sight_after_move or (lambda _angle: bool(target and target.line_of_sight))
    threat = build_threat_model(
        target=target,
        enemy_count_in_range=sum(
            1 for enemy in (frame.enemy_data or [])
            if frame.player_pos is not None and distance(frame.player_pos, box_center(enemy)) <= float(frame.attack_range or 0.0)
        ),
        health=frame.health,
        fog_danger=frame.fog_danger,
        projectile_incoming=frame.projectile_incoming,
        wall_trap=frame.wall_trap or not safety.safe,
        teammate_near=frame.teammate_near,
        safe_range=frame.safe_range,
    )
    kill = score_kill_opportunity(
        target=target,
        health=frame.health,
        threat=threat,
        teammate_near=frame.teammate_near,
        kill_confirm_score_threshold=kill_confirm_score_threshold,
    )
    if frame.player_pos is None:
        return TacticalPlan(objective="disengage", reasons=["player_missing"], kill_opportunity=kill)

    preferred_distance = float(frame.preferred_distance or frame.safe_range or 180.0)
    commit_distance = float(frame.commit_distance or min(float(frame.attack_range or 0.0), preferred_distance * 1.35))
    target_angle = angle_from_points(frame.player_pos, target.center) if target and target.center else None
    retreat_angle = (target_angle + 180.0) % 360.0 if target_angle is not None else None
    extra_angles = [retreat_angle, frame.projectile_dodge_angle, frame.teammate_angle]
    candidates = []
    rejected = []
    step_distance = max(80.0, min(160.0, preferred_distance * 0.65))
    for angle in generate_angle_candidates(frame.desired_angle, angle_samples, extra_angles):
        reasons = []
        blocked = bool(is_angle_blocked(angle))
        fog = bool(points_into_fog(angle))
        projectile_danger = clamp(projectile_danger_for_angle(angle))
        candidate_pos = project_point(frame.player_pos, angle, step_distance)
        survival = 0.82
        if blocked:
            survival -= 0.70
            reasons.append("blocked")
        if fog:
            survival -= 0.75
            reasons.append("fog")
        if frame.health.low:
            survival -= 0.12
            reasons.append("low_hp")
        if frame.projectile_incoming:
            survival -= projectile_danger * 0.45
            if projectile_danger > 0.35:
                reasons.append("projectile_lane")
        if frame.teammate_angle is not None and angle_delta(angle, frame.teammate_angle) <= 45:
            survival += 0.10
            reasons.append("teammate_path")
        engagement = 0.0
        fire_window = False
        if target and target.center and not target.stale:
            after_dist = distance(candidate_pos, target.center)
            current_dist = target.distance if target.distance is not None else after_dist
            if after_dist >= preferred_distance * 0.70:
                survival += 0.08
                reasons.append("keeps_distance")
            elif target.close_threat:
                survival -= 0.20
                reasons.append("too_close")
            distance_score = 1.0 - min(1.0, abs(after_dist - preferred_distance) / max(1.0, preferred_distance))
            engagement += distance_score * 0.26
            if after_dist <= float(frame.attack_range or 0.0) * 1.02:
                engagement += 0.18
                reasons.append("attack_distance")
            if line_of_sight_after_move(angle):
                engagement += 0.30
                fire_window = after_dist <= float(frame.attack_range or 0.0) * 1.02
                reasons.append("fire_window" if fire_window else "future_los")
            if target.close_threat and after_dist > current_dist:
                engagement += 0.12
                reasons.append("kites_close")
            if kill.score >= kill_confirm_score_threshold and after_dist <= commit_distance:
                engagement += 0.14
                reasons.append("kill_commit_lane")
        survival = clamp(survival - float(frame.aggression_penalty or 0.0))
        engagement = clamp(engagement)
        rejected_candidate = blocked or fog or survival < 0.18
        total = survival * 0.62 + engagement * 0.38
        if frame.projectile_incoming and frame.projectile_dodge_angle is not None:
            if angle_delta(angle, frame.projectile_dodge_angle) <= 28:
                total += 0.22
                reasons.append("projectile_dodge_angle")
        candidate = TacticalAngleScore(
            angle=angle,
            score=clamp(total),
            survival_score=survival,
            engagement_score=engagement,
            fire_window=bool(fire_window and survival >= 0.40),
            rejected=rejected_candidate,
            reasons=reasons,
        )
        if rejected_candidate:
            rejected.append({"angle": round(angle, 1), "reasons": reasons[:3], "survival": round(survival, 2)})
        candidates.append(candidate)

    usable = [candidate for candidate in candidates if not candidate.rejected]
    if not usable:
        return TacticalPlan(
            objective="disengage",
            position_goal=safety.angle,
            survival_score=0.0,
            engagement_score=0.0,
            kill_confirm_score=kill.score,
            fire_window=False,
            retreat_vector=retreat_angle,
            commit_allowed=False,
            selected_angle=safety.angle,
            rejected_angles=rejected[:8],
            reasons=["no_safe_angle"],
            kill_opportunity=kill,
        )
    usable.sort(key=lambda item: (item.score, item.fire_window, item.survival_score), reverse=True)
    best = usable[0]
    commit_threshold = float(survival_score_min_to_commit or 0.62)
    commit_allowed = bool(
        target
        and not target.stale
        and best.survival_score >= commit_threshold
        and kill.score >= float(kill_confirm_score_threshold or 0.68) + float(frame.fire_threshold_delta or 0.0)
        and not (frame.health.low and not target.close_threat)
    )
    if frame.fog_danger or frame.projectile_incoming:
        objective = "survive"
    elif frame.health.low and not (target and target.close_threat and best.fire_window):
        objective = "regroup" if frame.teammate_angle is not None else "hold_cover"
    elif not safety.safe or frame.wall_trap:
        objective = "disengage"
    elif commit_allowed and best.fire_window:
        objective = "finish_kill"
    elif target and target.close_threat:
        objective = "kite"
    elif target and best.fire_window and best.survival_score >= 0.50:
        objective = "pressure"
    elif frame.teammate_near:
        objective = "regroup"
    else:
        objective = "hold_cover"
    reasons = list(best.reasons)
    if commit_allowed:
        reasons.append("commit_allowed")
    if kill.score >= kill_confirm_score_threshold:
        reasons.append("kill_window")
    return TacticalPlan(
        objective=objective,
        position_goal=best.angle,
        engagement_score=best.engagement_score,
        survival_score=best.survival_score,
        kill_confirm_score=kill.score,
        fire_window=best.fire_window,
        retreat_vector=retreat_angle,
        commit_allowed=commit_allowed,
        selected_angle=best.angle,
        rejected_angles=rejected[:8],
        reasons=reasons,
        kill_opportunity=kill,
    )


def choose_combat_intent(
    *,
    frame,
    target=None,
    safety=None,
    ability_plan=None,
    tactical_plan=None,
    defensive_gate_enabled=True,
    panic_shot_range=150.0,
    tactical_planner_enabled=False,
    angle_samples=16,
    is_angle_blocked=None,
    points_into_fog=None,
    projectile_danger_for_angle=None,
    line_of_sight_after_move=None,
    survival_score_min_to_commit=0.62,
    kill_confirm_score_threshold=0.68,
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

    if tactical_planner_enabled:
        tactical_plan = tactical_plan or choose_tactical_plan(
            frame=frame,
            target=target,
            safety=safety,
            angle_samples=angle_samples,
            is_angle_blocked=is_angle_blocked,
            points_into_fog=points_into_fog,
            projectile_danger_for_angle=projectile_danger_for_angle,
            line_of_sight_after_move=line_of_sight_after_move,
            survival_score_min_to_commit=survival_score_min_to_commit,
            kill_confirm_score_threshold=kill_confirm_score_threshold,
        )
        mode = tactical_plan.objective
    else:
        tactical_plan = tactical_plan or TacticalPlan(
            objective=frame.current_mode or threat.mode,
            position_goal=safety.angle if safety.angle is not None else frame.desired_angle,
            survival_score=max(0.0, 1.0 - threat.score),
            engagement_score=target.score if target else 0.0,
            fire_window=bool(target and target.line_of_sight and target.in_attack_range and not target.stale),
            selected_angle=safety.angle if safety.angle is not None else frame.desired_angle,
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
    if tactical_planner_enabled:
        panic = bool(
            target
            and target.close_threat
            and target.line_of_sight
            and target.distance is not None
            and target.distance <= float(panic_shot_range or 0.0)
        )
        if mode in {"survive", "disengage", "hold_cover"} and not panic:
            attack_allowed = False
            denied_reason = f"{mode}_blocks_attack"
        elif not tactical_plan.fire_window and not panic:
            attack_allowed = False
            denied_reason = "no_fire_window"

    reasons = list(threat.reasons)
    reasons.extend(tactical_plan.reasons)
    if safety.status != "not_checked":
        reasons.append(f"safety:{safety.status}")
    if frame.projectile_incoming and not frame.enemy_data:
        reasons.append("projectile_without_enemy")

    return CombatIntent(
        mode=mode,
        movement_angle=(
            tactical_plan.selected_angle
            if tactical_planner_enabled and tactical_plan.selected_angle is not None
            else safety.angle if safety.angle is not None else frame.desired_angle
        ),
        target=target,
        threat=threat,
        ability_plan=ability_plan or AbilityPlan(),
        attack_allowed=attack_allowed,
        attack_denied_reason=denied_reason,
        reasons=reasons,
        tactical_plan=tactical_plan,
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
    tactical_plan=None,
    gadget_mode="generic",
    finisher_super=False,
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
    tactical_commit = bool(tactical_plan and tactical_plan.commit_allowed)
    tactical_finish = bool(tactical_plan and tactical_plan.objective == "finish_kill")
    tactical_fire_window = bool(tactical_plan and tactical_plan.fire_window)
    committed_fight = bool(
        target_clear
        and (target_in_range or panic or threat.outnumbered or teammate_near or tactical_commit or tactical_finish)
        and (not tactical_plan or tactical_plan.objective not in {"survive", "disengage", "hold_cover"})
    )

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
        if tactical_commit:
            value += 0.14
        if tactical_finish or finisher_super:
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
            elif committed_fight and not threat.low_hp and (not tactical_plan or tactical_plan.survival_score >= 0.62):
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
    elif defensive_mode and not panic and gadget_mode not in {"defensive"}:
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
        if gadget_mode == "defensive" and panic:
            value += 0.10
        elif gadget_mode == "engage" and committed_fight:
            value += 0.08
        elif gadget_mode == "combo" and plan.use_super:
            value += 0.10
        elif gadget_mode == "finisher" and tactical_finish:
            value += 0.12
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
        if tactical_commit:
            value += 0.12
        if threat.outnumbered:
            value += 0.16
        value += threat.score * 0.45
        if has_target and target.close_threat:
            value += 0.12
        if panic and not threat.low_hp:
            value += 0.12
        if tactical_plan and tactical_plan.objective in {"survive", "disengage", "hold_cover"} and not panic:
            value -= 0.45
        elif defensive_mode and not panic:
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
