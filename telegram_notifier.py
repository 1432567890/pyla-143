from __future__ import annotations

import io
import html
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from PIL import Image

from localization import get_config_language, normalize_language, tr
from utils import _config_bool, load_toml_as_dict, save_dict_as_toml


TELEGRAM_CONFIG_PATH = "cfg/telegram_config.toml"
LOCAL_TELEGRAM_CONFIG_PATH = "cfg/telegram_config.local.toml"
TELEGRAM_CHATS_PATH = "cfg/telegram_chats.toml"


def _clean_chat_id(value: Any) -> str:
    return str(value or "").strip()


def _as_chat_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_clean_chat_id(item) for item in value if _clean_chat_id(item)]
    text = _clean_chat_id(value)
    if not text:
        return []
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def _as_admin_ids(value: Any) -> list[str]:
    return _as_chat_ids(value)


def load_telegram_settings() -> dict[str, Any]:
    settings = {}
    if Path(TELEGRAM_CONFIG_PATH).exists():
        settings.update(load_toml_as_dict(TELEGRAM_CONFIG_PATH))
    if Path(LOCAL_TELEGRAM_CONFIG_PATH).exists():
        settings.update(load_toml_as_dict(LOCAL_TELEGRAM_CONFIG_PATH))
    settings.setdefault("enabled", False)
    settings["bot_token"] = str(settings.get("bot_token", "")).strip()
    settings["notification_chat_ids"] = _as_chat_ids(settings.get("notification_chat_ids"))
    settings["admin_ids"] = _as_admin_ids(settings.get("admin_ids"))
    settings.setdefault("send_match_summary", True)
    settings.setdefault("include_screenshot", True)
    settings.setdefault("attach_screenshot_on_game_finished", settings.get("include_screenshot", True))
    settings.setdefault("remote_control_enabled", True)
    settings.setdefault("poll_timeout_seconds", 25)
    settings.setdefault("heartbeat_enabled", False)
    settings.setdefault("heartbeat_interval_sec", 300)
    settings.setdefault("notify_on_start", True)
    settings.setdefault("notify_on_stop", True)
    settings.setdefault("notify_on_error", True)
    settings.setdefault("notify_on_game_finished", settings.get("send_match_summary", True))
    settings.setdefault("notify_on_brawler_change", True)
    settings.setdefault("notify_on_goal_confirmed", True)
    settings.setdefault("notify_on_config_reload", True)
    settings.setdefault("notify_on_trophy_update", True)
    settings.setdefault("notification_buttons_mode", "minimal")
    settings.setdefault("business_enabled", False)
    settings.setdefault("business_change_name_enabled", False)
    settings.setdefault("business_name_template", "{trophies}")
    settings.setdefault("business_change_bio_enabled", False)
    settings.setdefault("business_bio_template", "{trophies}")
    settings["language"] = normalize_language(settings.get("language") or get_config_language())
    return settings


def load_known_chat_ids() -> list[str]:
    if not Path(TELEGRAM_CHATS_PATH).exists():
        return []
    chats = load_toml_as_dict(TELEGRAM_CHATS_PATH)
    return _as_chat_ids(chats.get("chat_ids"))


def remember_chat_id(chat_id: int | str | None) -> bool:
    chat_id_text = _clean_chat_id(chat_id)
    if not chat_id_text:
        return False
    chats = load_toml_as_dict(TELEGRAM_CHATS_PATH) if Path(TELEGRAM_CHATS_PATH).exists() else {}
    chat_ids = _as_chat_ids(chats.get("chat_ids"))
    if chat_id_text in chat_ids:
        return False
    chat_ids.append(chat_id_text)
    chats["chat_ids"] = chat_ids
    save_dict_as_toml(chats, TELEGRAM_CHATS_PATH)
    return True


def load_business_connection() -> dict[str, Any]:
    if not Path(TELEGRAM_CHATS_PATH).exists():
        return {}
    chats = load_toml_as_dict(TELEGRAM_CHATS_PATH)
    connection = chats.get("business_connection")
    return connection if isinstance(connection, dict) else {}


def remember_business_connection(connection: dict[str, Any]) -> bool:
    connection_id = str(connection.get("id") or "").strip()
    if not connection_id:
        return False
    chats = load_toml_as_dict(TELEGRAM_CHATS_PATH) if Path(TELEGRAM_CHATS_PATH).exists() else {}
    previous = chats.get("business_connection") if isinstance(chats.get("business_connection"), dict) else {}
    rights = connection.get("rights") or {}
    cleaned = {
        "id": connection_id,
        "is_enabled": bool(connection.get("is_enabled", True)),
        "user_chat_id": str(connection.get("user_chat_id") or ""),
    }
    if "can_change_name" in rights:
        cleaned["can_change_name"] = bool(rights.get("can_change_name"))
    if "can_change_bio" in rights:
        cleaned["can_change_bio"] = bool(rights.get("can_change_bio"))
    if previous == cleaned:
        return False
    chats["business_connection"] = cleaned
    save_dict_as_toml(chats, TELEGRAM_CHATS_PATH)
    return True


def notification_chat_ids(settings: dict[str, Any] | None = None) -> list[str]:
    settings = settings or load_telegram_settings()
    ordered = []
    seen = set()
    for chat_id in _as_chat_ids(settings.get("notification_chat_ids")) + load_known_chat_ids():
        if chat_id in seen:
            continue
        seen.add(chat_id)
        ordered.append(chat_id)
    return ordered


def _format_title(event_type: str, details: dict[str, Any], language: str | None = None) -> str:
    title = tr(f"telegram.title.{event_type}", language, "Pyla 143 update")
    if event_type == "match":
        result = str(details.get("result") or "finished")
        brawler = str(details.get("brawler") or "").title()
        if brawler:
            return f"{title}: {result} with {brawler}"
        return f"{title}: {result}"
    return title


def _field_label(key: str, language: str | None = None) -> str:
    return tr(f"field.{key}", language, key.replace("_", " ").title())


def _format_message(event_type: str, details: dict[str, Any], language: str | None = None) -> str:
    language = normalize_language(language or details.get("language") or get_config_language())
    lines = [f"<b>{html.escape(_format_title(event_type, details, language))}</b>", "────────────────"]
    message = str(details.get("message") or details.get("reason") or "").strip()
    if message:
        lines.append(html.escape(message))

    hidden = {"message", "reason", "event_type", "language"}
    ordered = [
        "brawler",
        "result",
        "before",
        "after",
        "delta",
        "started_trophies",
        "trophies",
        "target",
        "wins",
        "win_streak",
        "brawlers_left",
        "ips",
        "state",
        "emulator",
        "adb_device",
        "runtime",
    ]
    for key in ordered + [key for key in details if key not in ordered]:
        if key in hidden or key not in details:
            continue
        value = details.get(key)
        if value is None or value == "":
            continue
        text = str(value)
        if len(text) > 180:
            text = text[:177] + "..."
        lines.append(f"{_field_label(key, language)}: {html.escape(text)}")
    return "\n".join(lines)


def main_keyboard(paused=False, heartbeat_enabled=True, language: str | None = None) -> dict[str, Any]:
    language = normalize_language(language or get_config_language())
    pause_text = tr("telegram.button.resume" if paused else "telegram.button.pause", language)
    heartbeat_text = tr("telegram.button.heartbeat_off" if heartbeat_enabled else "telegram.button.heartbeat_on", language)
    return {
        "inline_keyboard": [
            [
                {"text": tr("telegram.button.status", language), "callback_data": "status"},
                {"text": pause_text, "callback_data": "resume" if paused else "pause"},
                {"text": tr("telegram.button.stop", language), "callback_data": "stop"},
            ],
            [
                {"text": tr("telegram.button.reload_config", language), "callback_data": "reload_config"},
                {"text": tr("telegram.button.change_brawler", language), "callback_data": "brawler:0"},
            ],
            [
                {"text": tr("telegram.button.stats", language), "callback_data": "stats"},
                {"text": heartbeat_text, "callback_data": "heartbeat"},
            ],
        ]
    }


def notification_keyboard(mode: str = "minimal", language: str | None = None) -> dict[str, Any] | None:
    mode = str(mode or "minimal").strip().lower()
    if mode in {"none", "off", "disabled"}:
        return None
    language = normalize_language(language or get_config_language())
    if mode == "full":
        return main_keyboard(language=language)
    if mode == "minimal":
        return {"inline_keyboard": [[{"text": tr("telegram.button.status", language), "callback_data": "status"}]]}
    return None


def _image_to_png_bytes(screenshot: Any) -> bytes | None:
    if screenshot is None:
        return None
    if isinstance(screenshot, np.ndarray):
        image = Image.fromarray(screenshot)
    elif isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def async_send_message(
        chat_id: int | str,
        text: str,
        token: str | None = None,
        reply_markup: dict[str, Any] | None = None,
) -> bool:
    settings = load_telegram_settings()
    token = token or settings.get("bot_token", "")
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": str(chat_id),
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=15) as response:
                if response.status != 200:
                    body = await response.text()
                    print(f"telegram_message_send_failed status={response.status} body={body[:180]}")
                    if payload.get("parse_mode"):
                        payload.pop("parse_mode", None)
                        async with session.post(url, json=payload, timeout=15) as retry:
                            return retry.status == 200
                    return False
                return True
    except Exception as exc:
        print(f"Telegram message failed: {exc}")
        return False


async def async_edit_message(
        chat_id: int | str,
        message_id: int | str,
        text: str,
        token: str | None = None,
        reply_markup: dict[str, Any] | None = None,
) -> bool:
    settings = load_telegram_settings()
    token = token or settings.get("bot_token", "")
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/editMessageText"
    payload = {
        "chat_id": str(chat_id),
        "message_id": int(message_id),
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=15) as response:
                if response.status != 200:
                    body = await response.text()
                    print(f"telegram_message_send_failed status={response.status} body={body[:180]}")
                    if payload.get("parse_mode"):
                        payload.pop("parse_mode", None)
                        async with session.post(url, json=payload, timeout=15) as retry:
                            return retry.status == 200
                    return False
                return True
    except Exception as exc:
        print(f"Telegram edit failed: {exc}")
        return False


async def async_answer_callback(callback_query_id: str, text: str = "", token: str | None = None) -> bool:
    settings = load_telegram_settings()
    token = token or settings.get("bot_token", "")
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    payload = {"callback_query_id": callback_query_id, "text": text[:200]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as response:
                return response.status == 200
    except Exception as exc:
        print(f"Telegram callback answer failed: {exc}")
        return False


async def async_answer_inline_query(
        inline_query_id: str,
        results: list[dict[str, Any]],
        token: str | None = None,
        cache_time: int = 5,
) -> bool:
    settings = load_telegram_settings()
    token = token or settings.get("bot_token", "")
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/answerInlineQuery"
    payload = {
        "inline_query_id": inline_query_id,
        "results": results,
        "cache_time": cache_time,
        "is_personal": True,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as response:
                return response.status == 200
    except Exception as exc:
        print(f"Telegram inline answer failed: {exc}")
        return False


async def async_send_photo(chat_id: int | str, screenshot: Any, caption: str = "", token: str | None = None) -> bool:
    settings = load_telegram_settings()
    token = token or settings.get("bot_token", "")
    if not token:
        return False
    png_bytes = _image_to_png_bytes(screenshot)
    if not png_bytes:
        return await async_send_message(chat_id, caption or "No screenshot available.", token=token)
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    data = aiohttp.FormData()
    data.add_field("chat_id", str(chat_id))
    if caption:
        data.add_field("caption", caption[:1024])
        data.add_field("parse_mode", "HTML")
    data.add_field("photo", png_bytes, filename="pyla_screenshot.png", content_type="image/png")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, timeout=30) as response:
                return response.status == 200
    except Exception as exc:
        print(f"Telegram photo failed: {exc}")
        return False


async def async_notify_user(
    event_type: str | None = None,
    screenshot: Any = None,
    details: dict[str, Any] | None = None,
) -> bool:
    settings = load_telegram_settings()
    if not _config_bool(settings.get("enabled"), False):
        return False
    token = settings.get("bot_token", "")
    if not token:
        print("Telegram skipped: no bot token configured.")
        return False
    chat_ids = notification_chat_ids(settings)
    if not chat_ids:
        print("Telegram skipped: no known chats yet. Send /start or /help to the Telegram bot once.")
        return False

    event_type = event_type or "update"
    details = dict(details or {})
    notify_flag = {
        "start": "notify_on_start",
        "stop": "notify_on_stop",
        "error": "notify_on_error",
        "match": "notify_on_game_finished",
        "brawler_changed": "notify_on_brawler_change",
        "config_reload": "notify_on_config_reload",
        "trophy_update": "notify_on_trophy_update",
    }.get(event_type)
    if notify_flag and not _config_bool(settings.get(notify_flag), True):
        return False
    if event_type == "match" and not _config_bool(settings.get("send_match_summary"), False):
        return False

    language = settings.get("language")
    text = _format_message(event_type, details, language=language)
    include_screenshot = _config_bool(settings.get("include_screenshot"), True)
    if event_type == "match":
        include_screenshot = _config_bool(settings.get("attach_screenshot_on_game_finished"), include_screenshot)
    reply_markup = notification_keyboard(settings.get("notification_buttons_mode"), language)
    sent_any = False
    for chat_id in chat_ids:
        if include_screenshot and screenshot is not None:
            sent = await async_send_photo(chat_id, screenshot, caption=text, token=token)
        else:
            sent = await async_send_message(chat_id, text, token=token, reply_markup=reply_markup)
        sent_any = sent_any or sent
    if sent_any:
        print(f"Telegram notification sent: {event_type}")
    return sent_any


async def async_send_test_notification() -> bool:
    return await async_notify_user(
        "test",
        details={
            "state": "configured",
            "message": "Telegram is connected correctly.",
        },
    )
