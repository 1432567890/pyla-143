from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

import utils


SAFE_KEYS = {
    "cfg/telegram_config.toml": {
        "enabled",
        "admin_ids",
        "heartbeat_enabled",
        "heartbeat_interval_sec",
        "notify_on_start",
        "notify_on_stop",
        "notify_on_error",
        "notify_on_brawler_change",
        "notify_on_config_reload",
        "notify_on_trophy_update",
        "remote_control_enabled",
        "poll_timeout_seconds",
    },
    "cfg/bot_config.toml": {
        "movement_input_mode",
        "showdown_playstyle_mode",
        "post_match_action",
        "auto_aim_debug",
        "joystick_debug",
    },
}

RESTART_KEYS = {
    "cfg/general_config.toml": {
        "cpu_or_gpu",
        "directml_device_id",
        "current_emulator",
        "emulator_port",
        "max_ips",
        "scrcpy_max_fps",
        "scrcpy_max_width",
        "model_path",
        "device_id",
    },
    "cfg/bot_config.toml": {
        "current_playstyle",
        "enable_joystick_movement",
        "enable_flicker_retreat",
        "enable_combat_mans",
    },
    "cfg/telegram_config.toml": {
        "bot_token",
    },
}


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    out = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        out.update(_flatten(child_prefix, child))
    return out


def reload_config_safe(paths: list[str] | None = None) -> dict[str, list[str]]:
    paths = paths or ["cfg/telegram_config.toml", "cfg/bot_config.toml", "cfg/general_config.toml"]
    report = {"applied": [], "requires_restart": [], "errors": []}
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        old = dict(utils.load_toml_as_dict(path))
        utils.clear_toml_cache(path)
        try:
            new = toml.load(path)
        except Exception as exc:
            report["errors"].append(f"{path}: {exc}")
            continue
        old_flat = _flatten("", old)
        new_flat = _flatten("", new)
        changed = sorted(key for key, value in new_flat.items() if old_flat.get(key) != value)
        safe_keys = SAFE_KEYS.get(path, set())
        restart_keys = RESTART_KEYS.get(path, set())
        for key in changed:
            if key in safe_keys:
                report["applied"].append(f"{Path(path).stem}.{key}")
            elif key in restart_keys:
                report["requires_restart"].append(f"{Path(path).stem}.{key}")
            else:
                report["requires_restart"].append(f"{Path(path).stem}.{key}")
    utils.clear_toml_cache()
    return report
