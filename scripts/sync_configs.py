from __future__ import annotations

import argparse
import copy
import shutil
import time
from pathlib import Path
from typing import Any

import toml


SECRET_KEYS = {
    "player_tag",
    "email",
    "developer_email",
    "developer_password",
    "password",
    "api_token",
    "bot_token",
    "discord_bot_token",
    "admin_ids",
    "notification_chat_ids",
    "webhook",
    "webhook_url",
    "device_id",
    "discord_id",
    "discord_control_user_id",
    "discord_control_channel_id",
    "discord_control_guild_id",
    "emulator_launch_command",
    "mumu_manager_path",
    "ldplayer_console_path",
    "key",
    "last_public_ip",
}

SECRET_FRAGMENTS = ("token", "password", "secret", "credential", "webhook")


def is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in SECRET_KEYS or any(fragment in lowered for fragment in SECRET_FRAGMENTS)


def merge_config(example: dict[str, Any], current: dict[str, Any], prefix: str = "") -> tuple[dict[str, Any], list[str], list[str]]:
    merged = copy.deepcopy(current)
    added: list[str] = []
    kept_secret: list[str] = []

    for key, example_value in example.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in current:
            merged[key] = copy.deepcopy(example_value)
            added.append(path)
            continue

        current_value = current[key]
        if isinstance(example_value, dict) and isinstance(current_value, dict):
            nested, nested_added, nested_kept = merge_config(example_value, current_value, path)
            merged[key] = nested
            added.extend(nested_added)
            kept_secret.extend(nested_kept)
        elif is_secret_key(key):
            merged[key] = current_value
            kept_secret.append(path)

    return merged, added, kept_secret


def sync_one(example_path: Path, write: bool = True) -> dict[str, Any]:
    real_path = example_path.with_suffix("")
    example = toml.load(example_path)
    current = toml.load(real_path) if real_path.exists() else {}
    merged, added, kept_secret = merge_config(example, current)

    backup_path = None
    changed = merged != current
    if write and changed:
        if real_path.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup_path = real_path.with_name(f"{real_path.name}.bak-{stamp}")
            shutil.copy2(real_path, backup_path)
        real_path.write_text(toml.dumps(merged), encoding="utf-8")

    obsolete = sorted(set(current.keys()) - set(example.keys()))
    return {
        "example": str(example_path),
        "real": str(real_path),
        "changed": changed,
        "backup": str(backup_path) if backup_path else "",
        "added": added,
        "kept_secret": kept_secret,
        "obsolete_top_level": obsolete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely merge cfg/*.toml.example structure into real cfg/*.toml files.")
    parser.add_argument("--dry-run", action="store_true", help="Show report without writing files.")
    parser.add_argument("examples", nargs="*", help="Optional example files to sync. Defaults to cfg/*.toml.example.")
    args = parser.parse_args()

    examples = [Path(item) for item in args.examples] or sorted(Path("cfg").glob("*.toml.example"))
    for example_path in examples:
        report = sync_one(example_path, write=not args.dry_run)
        print(f"{report['real']}: changed={report['changed']}")
        if report["backup"]:
            print(f"  backup: {report['backup']}")
        if report["added"]:
            print("  added: " + ", ".join(report["added"]))
        if report["kept_secret"]:
            print("  preserved secrets: " + ", ".join(sorted(set(report["kept_secret"]))))
        if report["obsolete_top_level"]:
            print("  obsolete top-level keys kept: " + ", ".join(report["obsolete_top_level"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
