from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any

from utils import load_brawl_stars_api_config


STATS_PATH = Path("cfg/pyla_stats.json")
_LOCK = threading.RLock()


def _today() -> str:
    return date.today().isoformat()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_stats() -> dict[str, Any]:
    return {
        "player_tag": "",
        "total_trophies_gained": 0,
        "today_trophies_gained": 0,
        "session_trophies_gained": 0,
        "current_brawler": "",
        "daily_date": _today(),
        "brawlers": {},
        "sessions": {
            "last_start": None,
            "last_stop": None,
            "uptime_sec": 0,
            "current_start": None,
        },
        "last_activity": None,
        "last_error": "",
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_stats(path: str | Path = STATS_PATH) -> dict[str, Any]:
    path = Path(path)
    with _LOCK:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = default_stats()
        except Exception:
            data = default_stats()
        return reset_daily_if_needed(data)


def save_stats(data: dict[str, Any], path: str | Path = STATS_PATH) -> None:
    with _LOCK:
        _atomic_write_json(Path(path), data)


def reset_daily_if_needed(data: dict[str, Any], today: str | None = None) -> dict[str, Any]:
    today = today or _today()
    data = deepcopy(data or default_stats())
    data.setdefault("brawlers", {})
    data.setdefault("sessions", {})
    if data.get("daily_date") == today:
        return data
    data["daily_date"] = today
    data["today_trophies_gained"] = 0
    for row in data["brawlers"].values():
        if isinstance(row, dict):
            row["gained_today"] = 0
    return data


def start_session(current_brawler: str = "", player_tag: str = "", path: str | Path = STATS_PATH) -> dict[str, Any]:
    data = load_stats(path)
    data["session_trophies_gained"] = 0
    data["current_brawler"] = current_brawler or data.get("current_brawler", "")
    data["player_tag"] = player_tag or data.get("player_tag", "")
    data["sessions"]["current_start"] = _now()
    data["sessions"]["last_start"] = data["sessions"]["current_start"]
    data["last_activity"] = data["sessions"]["current_start"]
    save_stats(data, path)
    return data


def stop_session(path: str | Path = STATS_PATH) -> dict[str, Any]:
    data = load_stats(path)
    data["sessions"]["last_stop"] = _now()
    data["last_activity"] = data["sessions"]["last_stop"]
    save_stats(data, path)
    return data


def record_brawler(current_brawler: str, trophies: int | None = None, path: str | Path = STATS_PATH) -> dict[str, Any]:
    data = load_stats(path)
    data["current_brawler"] = current_brawler or data.get("current_brawler", "")
    if current_brawler:
        row = data["brawlers"].setdefault(current_brawler, {})
        if trophies is not None:
            row.setdefault("baseline_trophies", int(trophies))
            row["current_trophies"] = int(trophies)
        row["last_seen"] = _now()
    data["last_activity"] = _now()
    save_stats(data, path)
    return data


def record_trophy_update(
        brawler: str,
        old_trophies: int,
        new_trophies: int,
        path: str | Path = STATS_PATH,
) -> dict[str, Any]:
    data = load_stats(path)
    delta = int(new_trophies) - int(old_trophies)
    positive_delta = max(0, delta)
    data["current_brawler"] = brawler
    data["total_trophies_gained"] = int(data.get("total_trophies_gained", 0)) + positive_delta
    data["today_trophies_gained"] = int(data.get("today_trophies_gained", 0)) + positive_delta
    data["session_trophies_gained"] = int(data.get("session_trophies_gained", 0)) + positive_delta
    row = data["brawlers"].setdefault(brawler, {})
    row.setdefault("baseline_trophies", int(old_trophies))
    row["current_trophies"] = int(new_trophies)
    row["gained_today"] = int(row.get("gained_today", 0)) + positive_delta
    row["gained_total"] = int(row.get("gained_total", 0)) + positive_delta
    row["last_seen"] = _now()
    data["last_activity"] = row["last_seen"]
    save_stats(data, path)
    return data


def record_error(message: str, path: str | Path = STATS_PATH) -> dict[str, Any]:
    data = load_stats(path)
    data["last_error"] = str(message or "")[:500]
    data["last_activity"] = _now()
    save_stats(data, path)
    return data


def get_player_tag() -> str:
    try:
        return str(load_brawl_stars_api_config("cfg/brawl_stars_api.toml").get("player_tag", "")).strip()
    except Exception:
        return ""


def uptime_seconds(data: dict[str, Any] | None = None) -> int:
    data = data or load_stats()
    sessions = data.get("sessions", {})
    start = sessions.get("current_start") or sessions.get("last_start")
    if not start:
        return int(sessions.get("uptime_sec", 0) or 0)
    try:
        started = datetime.fromisoformat(start)
    except ValueError:
        return int(sessions.get("uptime_sec", 0) or 0)
    return max(0, int(time.time() - started.timestamp()))


def format_duration(seconds: int | float) -> str:
    seconds = max(0, int(seconds or 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"
