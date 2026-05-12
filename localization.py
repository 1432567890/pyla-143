from __future__ import annotations

from typing import Any

from utils import load_toml_as_dict


SUPPORTED_LANGUAGES = {"en", "ru"}
DEFAULT_LANGUAGE = "ru"


TRANSLATIONS = {
    "en": {
        "status.running": "Running",
        "status.paused": "Paused",
        "telegram.game_finished.title": "Match finished",
        "telegram.button.status": "Status",
        "telegram.button.menu": "Menu",
        "telegram.button.pause": "Pause",
        "telegram.button.resume": "Resume",
        "telegram.button.stop": "Stop",
        "telegram.button.reload_config": "Reload config",
        "telegram.button.change_brawler": "Change brawler",
        "telegram.button.stats": "Stats",
        "telegram.button.heartbeat_off": "Heartbeat off",
        "telegram.button.heartbeat_on": "Heartbeat on",
        "telegram.title.match": "Match finished",
        "telegram.title.brawler_complete": "Brawler target reached",
        "telegram.title.completed": "All targets complete",
        "telegram.title.bot_is_stuck": "Bot needs attention",
        "telegram.title.test": "Telegram test",
        "telegram.title.start": "Pyla 143 started",
        "telegram.title.stop": "Pyla 143 stopped",
        "telegram.title.pause": "Pyla 143 paused",
        "telegram.title.resume": "Pyla 143 resumed",
        "telegram.title.error": "Pyla 143 error",
        "telegram.title.brawler_changed": "Brawler changed",
        "telegram.title.config_reload": "Config reloaded",
        "telegram.title.trophy_update": "Trophy update",
        "telegram.title.heartbeat": "Heartbeat",
        "field.brawler": "Brawler",
        "field.result": "Result",
        "field.started_trophies": "Started trophies",
        "field.trophies": "Trophies",
        "field.target": "Target",
        "field.wins": "Wins",
        "field.win_streak": "Win streak",
        "field.brawlers_left": "Brawlers left",
        "field.ips": "IPS",
        "field.state": "State",
        "field.emulator": "Emulator",
        "field.adb_device": "ADB device",
        "field.runtime": "Runtime",
        "field.before": "Before",
        "field.after": "After",
        "field.delta": "Delta",
        "field.playstyle": "Playstyle",
        "field.mode": "Mode",
        "field.auto_aim": "Auto-aim",
        "field.status": "Status",
        "field.api_confirmation": "API confirmation",
        "api.confirmed": "confirmed",
        "api.pending": "pending",
        "api.unavailable": "unavailable",
        "gui.start": "Start",
        "gui.stop": "Stop",
        "gui.pause": "Pause",
        "gui.selected_brawler": "Selected brawler",
        "gui.telegram_settings": "Telegram settings",
        "gui.discord_settings": "Discord settings",
    },
    "ru": {
        "status.running": "Запущен",
        "status.paused": "Пауза",
        "telegram.game_finished.title": "Матч завершён",
        "telegram.button.status": "Статус",
        "telegram.button.menu": "Меню",
        "telegram.button.pause": "Пауза",
        "telegram.button.resume": "Продолжить",
        "telegram.button.stop": "Стоп",
        "telegram.button.reload_config": "Обновить конфиг",
        "telegram.button.change_brawler": "Сменить бойца",
        "telegram.button.stats": "Статистика",
        "telegram.button.heartbeat_off": "Пульс выкл.",
        "telegram.button.heartbeat_on": "Пульс вкл.",
        "telegram.title.match": "Матч завершён",
        "telegram.title.brawler_complete": "Цель бойца достигнута",
        "telegram.title.completed": "Все цели выполнены",
        "telegram.title.bot_is_stuck": "Боту нужно внимание",
        "telegram.title.test": "Тест Telegram",
        "telegram.title.start": "Pyla 143 запущен",
        "telegram.title.stop": "Pyla 143 остановлен",
        "telegram.title.pause": "Pyla 143 на паузе",
        "telegram.title.resume": "Pyla 143 продолжен",
        "telegram.title.error": "Ошибка Pyla 143",
        "telegram.title.brawler_changed": "Боец изменён",
        "telegram.title.config_reload": "Конфиг обновлён",
        "telegram.title.trophy_update": "Обновление трофеев",
        "telegram.title.heartbeat": "Пульс",
        "field.brawler": "Боец",
        "field.result": "Результат",
        "field.started_trophies": "Трофеи до старта",
        "field.trophies": "Трофеи",
        "field.target": "Цель",
        "field.wins": "Победы",
        "field.win_streak": "Серия побед",
        "field.brawlers_left": "Бойцов осталось",
        "field.ips": "IPS",
        "field.state": "Состояние",
        "field.emulator": "Эмулятор",
        "field.adb_device": "ADB-устройство",
        "field.runtime": "Время работы",
        "field.before": "До",
        "field.after": "После",
        "field.delta": "Изменение",
        "field.playstyle": "Стиль игры",
        "field.mode": "Режим",
        "field.auto_aim": "Автоприцел",
        "field.status": "Статус",
        "field.api_confirmation": "API-подтверждение",
        "api.confirmed": "подтверждено",
        "api.pending": "ожидается",
        "api.unavailable": "недоступно",
        "gui.start": "Старт",
        "gui.stop": "Стоп",
        "gui.pause": "Пауза",
        "gui.selected_brawler": "Выбранный боец",
        "gui.telegram_settings": "Настройки Telegram",
        "gui.discord_settings": "Настройки Discord",
    },
}


def normalize_language(language: Any) -> str:
    value = str(language or "").strip().lower()
    if value in SUPPORTED_LANGUAGES:
        return value
    return DEFAULT_LANGUAGE


def get_config_language() -> str:
    for path in ("cfg/general_config.toml", "cfg/telegram_config.toml"):
        config = load_toml_as_dict(path)
        if isinstance(config.get("interface"), dict):
            language = config["interface"].get("language")
            if language:
                return normalize_language(language)
        language = config.get("language")
        if language:
            return normalize_language(language)
    return DEFAULT_LANGUAGE


def tr(key: str, language: Any = None, default: str | None = None) -> str:
    lang = normalize_language(language or get_config_language())
    return TRANSLATIONS.get(lang, {}).get(key) or TRANSLATIONS["en"].get(key) or default or key
