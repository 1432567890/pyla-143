from __future__ import annotations

import asyncio
import html
import inspect
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import aiohttp

from brawler_selection import filter_brawler_cards, build_brawler_cards
from config_reload import reload_config_safe
from pyla_stats import format_duration, load_stats, record_brawler, uptime_seconds
from runtime_control import PAUSED, RUNNING, STOPPED, read_state, write_state
from telegram_notifier import (
    async_answer_callback,
    async_answer_inline_query,
    async_edit_message,
    async_send_message,
    async_send_photo,
    load_business_connection,
    load_telegram_settings,
    main_keyboard,
    remember_business_connection,
    remember_chat_id,
)
from utils import (
    _config_bool,
    fetch_brawl_stars_player,
    get_brawler_list,
    load_brawl_stars_api_config,
    load_saved_brawler_data,
    save_brawler_data,
)

BUSINESS_TROPHIES_PLACEHOLDER = "{trophies}"
BUSINESS_NAME_UPDATE_INTERVAL_SEC = 60


def set_runtime_state(state_path: str | Path, paused: bool) -> str:
    state = PAUSED if paused else RUNNING
    write_state(state_path, state)
    return state


def is_admin(settings: dict[str, Any], user_id: int | str | None) -> bool:
    admins = [str(item) for item in settings.get("admin_ids", []) if str(item).strip()]
    if not admins:
        return False
    return str(user_id) in admins


def _short(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return html.escape(text if text else fallback)


def format_business_trophies(trophies: int | str | float | None) -> str:
    try:
        value = max(0, int(trophies or 0))
    except (TypeError, ValueError):
        value = 0
    if value < 1000:
        return str(value)
    truncated_tenths = value // 100
    whole = truncated_tenths // 10
    decimal = truncated_tenths % 10
    if decimal == 0:
        return f"{whole}k"
    return f"{whole},{decimal}k"


def format_business_name(template: str | None, trophies_text: str) -> str:
    name_template = str(template or "").strip()
    if not name_template:
        return trophies_text
    if BUSINESS_TROPHIES_PLACEHOLDER in name_template:
        return name_template.replace(BUSINESS_TROPHIES_PLACEHOLDER, trophies_text).strip()
    return f"{name_template} {trophies_text}".strip()


def _status_from_details(state_path: str | Path, details: dict[str, Any], stats: dict[str, Any]) -> str:
    runtime = read_state(state_path)
    current_state = "paused" if runtime == PAUSED else ("stopped" if runtime == STOPPED else "running")
    brawler = details.get("brawler") or stats.get("current_brawler") or "unknown"
    lines = [
        "<b>Pyla 143 status</b>",
        "────────────────",
        f"Status: {_short(current_state)}",
        f"Brawler: {_short(brawler)}",
        f"Playstyle: {_short(details.get('playstyle'))}",
        f"Trophies: {_short(details.get('trophies') or _current_brawler_trophies(stats, brawler), 'Not scanned')}",
        f"Today: +{int(stats.get('today_trophies_gained', 0) or 0)}",
        f"Total: +{int(stats.get('total_trophies_gained', 0) or 0)}",
        f"Session: +{int(stats.get('session_trophies_gained', 0) or 0)}",
        f"Uptime: {format_duration(uptime_seconds(stats))}",
    ]
    api_confirmation = details.get("api_confirmation") if isinstance(details, dict) else None
    if isinstance(api_confirmation, dict) and api_confirmation.get("status"):
        api_trophies = api_confirmation.get("api_trophies")
        lines.extend([
            f"API status: {_short(api_confirmation.get('status'))}",
            f"API trophies: {_short(api_trophies, 'unavailable')}",
            f"API target: {_short(api_confirmation.get('target'), 'unknown')}",
        ])
    last_error = str(stats.get("last_error") or "").strip()
    if last_error:
        lines.append(f"Last error: {html.escape(last_error[:180])}")
    return "\n".join(lines)


def _current_brawler_trophies(stats: dict[str, Any], brawler: str) -> Any:
    row = stats.get("brawlers", {}).get(brawler, {})
    return row.get("current_trophies") if isinstance(row, dict) else None


def _stats_text(stats: dict[str, Any], title: str = "Pyla 143 progress") -> str:
    return "\n".join([
        f"<b>{html.escape(title)}</b>",
        "────────────────",
        f"Player: {_short(stats.get('player_tag'), 'unknown')}",
        f"Today: +{int(stats.get('today_trophies_gained', 0) or 0)} trophies",
        f"Total: +{int(stats.get('total_trophies_gained', 0) or 0)} trophies",
        f"Current brawler: {_short(stats.get('current_brawler'), 'unknown')}",
        f"Session: +{int(stats.get('session_trophies_gained', 0) or 0)} trophies",
        f"Uptime: {format_duration(uptime_seconds(stats))}",
    ])


def _brawler_keyboard(page: int = 0, page_size: int = 12) -> dict[str, Any]:
    saved = load_saved_brawler_data()
    stats = load_stats()
    trophies = {
        name: row.get("current_trophies")
        for name, row in stats.get("brawlers", {}).items()
        if isinstance(row, dict) and row.get("current_trophies") is not None
    }
    cards = filter_brawler_cards(
        build_brawler_cards(get_brawler_list(), trophies, [row.get("brawler", "") for row in saved]),
        sort_mode="trophies_desc",
    )
    total_pages = max(1, (len(cards) + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    rows = []
    for card in cards[page * page_size:(page + 1) * page_size]:
        trophies_text = "?" if card.trophies is None else str(card.trophies)
        rows.append([{"text": f"{card.name.title()} [{trophies_text}]", "callback_data": f"set_brawler:{card.name}"}])
    nav = []
    if page > 0:
        nav.append({"text": "Prev", "callback_data": f"brawler:{page - 1}"})
    if page < total_pages - 1:
        nav.append({"text": "Next", "callback_data": f"brawler:{page + 1}"})
    if nav:
        rows.append(nav)
    rows.append([{"text": "Back", "callback_data": "status"}])
    return {"inline_keyboard": rows}


class TelegramControlServer:
    def __init__(
            self,
            state_path: str | Path,
            settings_loader=load_telegram_settings,
            screenshot_provider: Callable[[], Any] | None = None,
            restart_game_callback: Callable[[], Any] | None = None,
            status_provider: Callable[[], dict[str, Any]] | None = None,
            reload_config_callback: Callable[[], Any] | None = None,
            brawler_change_callback: Callable[[str], Any] | None = None,
            stop_callback: Callable[[], Any] | None = None,
    ):
        self.state_path = Path(state_path)
        self.settings_loader = settings_loader
        self.screenshot_provider = screenshot_provider
        self.restart_game_callback = restart_game_callback
        self.status_provider = status_provider
        self.reload_config_callback = reload_config_callback
        self.brawler_change_callback = brawler_change_callback
        self.stop_callback = stop_callback
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: asyncio.Event | None = None
        self._offset = 0
        self._heartbeat_enabled = True
        self._last_heartbeat = 0.0
        self._last_business_name_update = 0.0
        self._last_business_name = ""
        self._last_business_bio = ""

    def start(self) -> bool:
        settings = self.settings_loader()
        self._heartbeat_enabled = _config_bool(settings.get("heartbeat_enabled"), True)
        business_enabled = _config_bool(settings.get("business_enabled"), False)
        remote_control_enabled = _config_bool(settings.get("remote_control_enabled"), True)
        print(
            "telegram_start_requested",
            f"telegram_enabled={_config_bool(settings.get('enabled'), False)}",
            f"telegram_token_present={bool(str(settings.get('bot_token') or '').strip())}",
            f"telegram_admin_ids_count={len(settings.get('admin_ids') or [])}",
            f"telegram_business_enabled={business_enabled}",
        )
        if not _config_bool(settings.get("enabled"), False):
            return False
        if not remote_control_enabled and not business_enabled:
            return False
        token = str(settings.get("bot_token") or "").strip()
        if not token:
            print("Telegram control skipped: fill bot_token in cfg/telegram_config.toml first.")
            return False
        if not settings.get("admin_ids"):
            print("Telegram control warning: admin_ids is empty; control commands will be denied.")
        if self.thread and self.thread.is_alive():
            return True

        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        return True

    def close(self) -> None:
        loop = self.loop
        stop_event = self.stop_event
        if loop is not None and stop_event is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_event.set)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            print(f"Telegram control stopped: {exc}")

    async def _run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        settings = self.settings_loader()
        token = str(settings.get("bot_token") or "").strip()
        if not await self._validate_token(token):
            print("telegram_polling_error invalid_token")
            return
        print("Telegram control started: /start /status /pause /resume /stop /heartbeat /reload_config /brawler")
        print("telegram_polling_started")
        while not self.stop_event.is_set():
            settings = self.settings_loader()
            token = str(settings.get("bot_token") or "").strip()
            if not token:
                await asyncio.sleep(5)
                continue
            timeout_seconds = max(5, int(settings.get("poll_timeout_seconds", 25) or 25))
            try:
                await self._maybe_send_heartbeat(token, settings)
                await self._maybe_update_business_name(token, settings)
                updates = await self._get_updates(token, timeout_seconds)
                for update in updates:
                    self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                    await self._handle_update(token, update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"Telegram control polling error: {exc}")
                print(f"telegram_polling_error {str(exc)[:180]}")
                await asyncio.sleep(5)

    async def _validate_token(self, token: str) -> bool:
        if not token:
            return False
        url = f"https://api.telegram.org/bot{token}/getMe"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    data = await response.json()
            return bool(data.get("ok"))
        except Exception as exc:
            print(f"telegram_polling_error token_validation_failed {str(exc)[:160]}")
            return False

    async def _get_updates(self, token: str, timeout_seconds: int) -> list[dict[str, Any]]:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params = {
            "timeout": timeout_seconds,
            "offset": self._offset,
            "allowed_updates": json.dumps(["message", "callback_query", "inline_query", "business_connection"]),
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=timeout_seconds + 10) as response:
                data = await response.json()
        if not data.get("ok"):
            raise RuntimeError(str(data))
        return list(data.get("result") or [])

    async def _handle_update(self, token: str, update: dict[str, Any]) -> None:
        if update.get("business_connection"):
            self._handle_business_connection(update["business_connection"])
            return
        if update.get("callback_query"):
            await self._handle_callback(token, update["callback_query"])
            return
        if update.get("inline_query"):
            await self._handle_inline_query(token, update["inline_query"])
            return
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        user = message.get("from") or {}
        chat_id = chat.get("id")
        text = str(message.get("text") or "").strip()
        if not text or chat_id is None:
            return

        settings = self.settings_loader()
        if not _config_bool(settings.get("remote_control_enabled"), True):
            return
        command = text.split()[0].split("@", 1)[0].lower()
        print(f"telegram_command_received command={command} chat_id={chat_id}")
        remember_chat_id(chat_id)
        if not is_admin(settings, user.get("id")):
            print(f"telegram_access_denied user_id={user.get('id')}")
            await async_send_message(chat_id, "Access denied", token=token)
            return

        if command in {"/help", "/start"}:
            await async_send_message(chat_id, self._welcome_text(), token=token, reply_markup=self._keyboard())
        elif command in {"/pause"}:
            await self._pause(chat_id, token)
        elif command in {"/resume"}:
            await self._resume(chat_id, token)
        elif command == "/stop":
            await self._stop(chat_id, token)
        elif command == "/status":
            await async_send_message(chat_id, self._status_text(), token=token, reply_markup=self._keyboard())
        elif command == "/heartbeat":
            await self._toggle_heartbeat(chat_id, token)
        elif command == "/reload_config":
            await self._reload_config(chat_id, token)
        elif command == "/brawler":
            await async_send_message(chat_id, "<b>Choose brawler</b>\n────────────────", token=token, reply_markup=_brawler_keyboard(0))
        elif command == "/stats":
            await async_send_message(chat_id, _stats_text(load_stats()), token=token, reply_markup=self._keyboard())
        elif command == "/screenshot":
            await self._send_screenshot(chat_id, token)
        elif command in {"/restart_game", "/restart"}:
            await self._restart_game(chat_id, token)
        else:
            await async_send_message(chat_id, "Unknown command. Send /help.", token=token)

    async def _handle_callback(self, token: str, callback: dict[str, Any]) -> None:
        settings = self.settings_loader()
        if not _config_bool(settings.get("remote_control_enabled"), True):
            return
        user = callback.get("from") or {}
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        message_id = message.get("message_id")
        data = str(callback.get("data") or "")
        print(f"telegram_callback_received data={data} chat_id={chat_id}")
        await async_answer_callback(str(callback.get("id")), token=token)
        if chat_id is None or not is_admin(settings, user.get("id")):
            if chat_id is not None:
                print(f"telegram_access_denied user_id={user.get('id')}")
                await async_send_message(chat_id, "Access denied", token=token)
            return
        if data == "status":
            await async_edit_message(chat_id, message_id, self._status_text(), token=token, reply_markup=self._keyboard())
        elif data == "pause":
            await self._pause(chat_id, token)
        elif data == "resume":
            await self._resume(chat_id, token)
        elif data == "stop":
            await self._stop(chat_id, token)
        elif data == "heartbeat":
            await self._toggle_heartbeat(chat_id, token)
        elif data == "reload_config":
            await self._reload_config(chat_id, token)
        elif data == "stats":
            await async_edit_message(chat_id, message_id, _stats_text(load_stats()), token=token, reply_markup=self._keyboard())
        elif data.startswith("brawler:"):
            page = int(data.split(":", 1)[1] or 0)
            await async_edit_message(chat_id, message_id, "<b>Choose brawler</b>\n────────────────", token=token, reply_markup=_brawler_keyboard(page))
        elif data.startswith("set_brawler:"):
            await self._change_brawler(chat_id, token, data.split(":", 1)[1])

    async def _handle_inline_query(self, token: str, inline_query: dict[str, Any]) -> None:
        settings = self.settings_loader()
        if not _config_bool(settings.get("remote_control_enabled"), True):
            await async_answer_inline_query(str(inline_query.get("id")), [], token=token)
            return
        if not is_admin(settings, (inline_query.get("from") or {}).get("id")):
            await async_answer_inline_query(str(inline_query.get("id")), [], token=token)
            return
        query = str(inline_query.get("query") or "").strip().lower()
        if query and "stats" not in query:
            await async_answer_inline_query(str(inline_query.get("id")), [], token=token)
            return
        stats = load_stats()
        variants = [
            ("overall", "Overall stats", _stats_text(stats, "Pyla 143 progress")),
            ("today", "Today stats", _stats_text(stats, "Pyla 143 today")),
            ("session", "Current session", _stats_text(stats, "Pyla 143 session")),
            ("brawler", "Current brawler stats", _stats_text(stats, "Pyla 143 current brawler")),
        ]
        results = []
        for result_id, title, text in variants:
            results.append({
                "type": "article",
                "id": result_id,
                "title": title,
                "description": text.replace("<b>", "").replace("</b>", "").split("\n", 2)[-1][:120],
                "input_message_content": {
                    "message_text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            })
        await async_answer_inline_query(str(inline_query.get("id")), results, token=token)

    def _handle_business_connection(self, connection: dict[str, Any]) -> None:
        settings = self.settings_loader()
        if not _config_bool(settings.get("business_enabled"), False):
            return
        changed = remember_business_connection(connection)
        rights = connection.get("rights") or {}
        print(
            "telegram_business_connection_received",
            f"changed={changed}",
            f"is_enabled={bool(connection.get('is_enabled', True))}",
            f"can_change_name={rights.get('can_change_name', 'unknown')}",
            f"can_change_bio={rights.get('can_change_bio', 'unknown')}",
        )

    async def _maybe_update_business_name(self, token: str, settings: dict[str, Any]) -> None:
        if not _config_bool(settings.get("business_enabled"), False):
            return
        name_enabled = _config_bool(settings.get("business_change_name_enabled"), False)
        bio_enabled = _config_bool(settings.get("business_change_bio_enabled"), False)
        if not name_enabled and not bio_enabled:
            return
        if time.time() - self._last_business_name_update < BUSINESS_NAME_UPDATE_INTERVAL_SEC:
            return
        self._last_business_name_update = time.time()

        connection = load_business_connection()
        connection_id = str(connection.get("id") or "").strip()
        if not connection_id:
            print("telegram_business_name_update_skipped reason=missing_business_connection")
            return
        if connection.get("is_enabled") is False:
            print("telegram_business_name_update_skipped reason=business_connection_disabled")
            return

        try:
            trophies = await asyncio.to_thread(self._fetch_player_trophies)
            trophies_text = format_business_trophies(trophies)
            if name_enabled:
                await self._update_business_name(token, connection, connection_id, settings, trophies, trophies_text)
            if bio_enabled:
                await self._update_business_bio(token, connection, connection_id, settings, trophies, trophies_text)
        except Exception as exc:
            print(f"telegram_business_name_update_error {str(exc)[:180]}")

    async def _update_business_name(
            self,
            token: str,
            connection: dict[str, Any],
            connection_id: str,
            settings: dict[str, Any],
            trophies: int,
            trophies_text: str,
    ) -> None:
        if connection.get("can_change_name") is False:
            print("telegram_business_name_update_skipped reason=missing_can_change_name_right")
            return
        target_name = format_business_name(settings.get("business_name_template"), trophies_text)
        if not target_name:
            return
        target_name = target_name[:64]
        if target_name == self._last_business_name:
            return
        if await self._set_business_account_name(token, connection_id, target_name):
            self._last_business_name = target_name
            print("telegram_business_name_updated", f"trophies={trophies}", f"name={target_name}")

    async def _update_business_bio(
            self,
            token: str,
            connection: dict[str, Any],
            connection_id: str,
            settings: dict[str, Any],
            trophies: int,
            trophies_text: str,
    ) -> None:
        if connection.get("can_change_bio") is False:
            print("telegram_business_bio_update_skipped reason=missing_can_change_bio_right")
            return
        target_bio = format_business_name(settings.get("business_bio_template"), trophies_text)
        target_bio = target_bio[:140]
        if target_bio == self._last_business_bio:
            return
        if await self._set_business_account_bio(token, connection_id, target_bio):
            self._last_business_bio = target_bio
            print("telegram_business_bio_updated", f"trophies={trophies}", f"bio={target_bio}")

    def _fetch_player_trophies(self) -> int:
        api_config = load_brawl_stars_api_config("cfg/brawl_stars_api.toml")
        timeout = int(api_config.get("timeout_seconds") or api_config.get("timeout_sec") or 15)
        player_data = fetch_brawl_stars_player(
            api_config.get("api_token", "").strip(),
            api_config.get("player_tag", "").strip(),
            timeout,
        )
        return int(player_data.get("trophies", 0) or 0)

    async def _set_business_account_name(self, token: str, connection_id: str, first_name: str) -> bool:
        url = f"https://api.telegram.org/bot{token}/setBusinessAccountName"
        payload = {
            "business_connection_id": connection_id,
            "first_name": first_name,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    if not data.get("ok"):
                        print(f"telegram_business_name_update_failed body={str(data)[:180]}")
                    return bool(data.get("ok"))
                body = await response.text()
                print(f"telegram_business_name_update_failed status={response.status} body={body[:180]}")
                return False

    async def _set_business_account_bio(self, token: str, connection_id: str, bio: str) -> bool:
        url = f"https://api.telegram.org/bot{token}/setBusinessAccountBio"
        payload = {
            "business_connection_id": connection_id,
            "bio": bio,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    if not data.get("ok"):
                        print(f"telegram_business_bio_update_failed body={str(data)[:180]}")
                    return bool(data.get("ok"))
                body = await response.text()
                print(f"telegram_business_bio_update_failed status={response.status} body={body[:180]}")
                return False

    async def _maybe_send_heartbeat(self, token: str, settings: dict[str, Any]) -> None:
        if not self._heartbeat_enabled or not _config_bool(settings.get("heartbeat_enabled"), True):
            return
        interval = max(30, int(settings.get("heartbeat_interval_sec", 300) or 300))
        if time.time() - self._last_heartbeat < interval:
            return
        self._last_heartbeat = time.time()
        ids = settings.get("notification_chat_ids") or settings.get("admin_ids") or []
        if not ids:
            return
        await async_send_message(ids[0], self._heartbeat_text(), token=token)

    def _keyboard(self):
        settings = self.settings_loader()
        return main_keyboard(
            read_state(self.state_path) == PAUSED,
            self._heartbeat_enabled,
            language=settings.get("language"),
        )

    def _welcome_text(self) -> str:
        return self._status_text().replace("<b>Pyla 143 status</b>", "<b>Pyla 143 control</b>", 1)

    def _status_text(self) -> str:
        details = self.status_provider() if self.status_provider else {}
        return _status_from_details(self.state_path, details, load_stats())

    def _heartbeat_text(self) -> str:
        stats = load_stats()
        brawler = stats.get("current_brawler") or "unknown"
        trophies = _current_brawler_trophies(stats, brawler)
        return "\n".join([
            "<b>Heartbeat</b>",
            "────────────────",
            f"Status: {'paused' if read_state(self.state_path) == PAUSED else 'running'}",
            f"Brawler: {_short(brawler)}",
            f"Trophies: {_short(trophies, 'Not scanned')}",
            f"Today: +{int(stats.get('today_trophies_gained', 0) or 0)}",
            f"Total: +{int(stats.get('total_trophies_gained', 0) or 0)}",
            f"Uptime: {format_duration(uptime_seconds(stats))}",
        ])

    async def _pause(self, chat_id: int | str, token: str) -> None:
        set_runtime_state(self.state_path, paused=True)
        await async_send_message(chat_id, "<b>Pyla 143 paused</b>\n────────────────\nStatus: paused", token=token, reply_markup=self._keyboard())

    async def _resume(self, chat_id: int | str, token: str) -> None:
        set_runtime_state(self.state_path, paused=False)
        await async_send_message(chat_id, "<b>Pyla 143 resumed</b>\n────────────────\nStatus: running", token=token, reply_markup=self._keyboard())

    async def _stop(self, chat_id: int | str, token: str) -> None:
        write_state(self.state_path, STOPPED)
        if self.stop_callback:
            result = self.stop_callback()
            if inspect.isawaitable(result):
                await result
        await async_send_message(chat_id, "<b>Pyla 143 stopped</b>\n────────────────\nStatus: stop requested", token=token, reply_markup=self._keyboard())

    async def _toggle_heartbeat(self, chat_id: int | str, token: str) -> None:
        self._heartbeat_enabled = not self._heartbeat_enabled
        await async_send_message(
            chat_id,
            f"<b>Heartbeat</b>\n────────────────\nStatus: {'enabled' if self._heartbeat_enabled else 'disabled'}",
            token=token,
            reply_markup=self._keyboard(),
        )

    async def _reload_config(self, chat_id: int | str, token: str) -> None:
        try:
            report = self.reload_config_callback() if self.reload_config_callback else reload_config_safe()
            if inspect.isawaitable(report):
                report = await report
        except Exception as exc:
            await async_send_message(chat_id, f"<b>Config reload failed</b>\n────────────────\n{html.escape(str(exc)[:300])}", token=token)
            return
        lines = ["<b>Config reloaded</b>", "────────────────", "Applied:"]
        applied = report.get("applied") or []
        lines.extend([f"• {html.escape(item)}" for item in applied] or ["• none"])
        lines.append("Requires restart:")
        restart = report.get("requires_restart") or []
        lines.extend([f"• {html.escape(item)}" for item in restart] or ["• none"])
        if report.get("errors"):
            lines.append("Errors:")
            lines.extend([f"• {html.escape(item)}" for item in report["errors"]])
        await async_send_message(chat_id, "\n".join(lines), token=token, reply_markup=self._keyboard())

    async def _change_brawler(self, chat_id: int | str, token: str, brawler: str) -> None:
        try:
            result = self.brawler_change_callback(brawler) if self.brawler_change_callback else self._change_saved_brawler(brawler)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            await async_send_message(chat_id, f"Brawler change failed: {html.escape(str(exc)[:240])}", token=token)
            return
        stats = load_stats()
        trophies = _current_brawler_trophies(stats, brawler)
        await async_send_message(
            chat_id,
            "\n".join([
                "<b>Brawler change requested</b>",
                "────────────────",
                f"Selected: {_short(brawler)}",
                f"Trophies: {_short(trophies, 'Not scanned')}",
                f"Apply mode: {_short((result or {}).get('apply_mode'), 'after current safe point')}",
            ]),
            token=token,
            reply_markup=self._keyboard(),
        )

    def _change_saved_brawler(self, brawler: str) -> dict[str, Any]:
        data = load_saved_brawler_data()
        if not data:
            data = [{"brawler": brawler, "push_until": 1000, "trophies": 0, "wins": 0, "type": "trophies", "automatically_pick": False, "win_streak": 0}]
        else:
            data[0] = dict(data[0])
            data[0]["brawler"] = brawler
        save_brawler_data(data)
        record_brawler(brawler)
        return {"apply_mode": "config only"}

    async def _send_screenshot(self, chat_id: int | str, token: str) -> None:
        if self.screenshot_provider is None:
            await async_send_message(chat_id, "Screenshot is not available in this process.", token=token)
            return
        try:
            screenshot = self.screenshot_provider()
        except Exception as exc:
            await async_send_message(chat_id, f"Could not capture screenshot: {html.escape(str(exc))}", token=token)
            return
        sent = await async_send_photo(chat_id, screenshot, caption="<b>Current screenshot</b>", token=token)
        if not sent:
            await async_send_message(chat_id, "Could not send screenshot.", token=token)

    async def _restart_game(self, chat_id: int | str, token: str) -> None:
        if self.restart_game_callback is None:
            await async_send_message(chat_id, "Restart callback is not available.", token=token)
            return
        await async_send_message(chat_id, "Restarting Brawl Stars and scrcpy...", token=token)
        try:
            result = self.restart_game_callback()
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            await async_send_message(chat_id, f"Restart failed: {html.escape(str(exc))}", token=token)
            return
        await async_send_message(
            chat_id,
            "Restart finished." if result else "Restart command ran, but recovery reported a problem.",
            token=token,
        )
