# AGENTS.md

## Project

Pyla 143 is a fork of PylaAI, based on Pyla XXZ.

- Pyla XXZ: https://github.com/xxz-888/PylaAi-XXZ
- PylaAI: https://github.com/PylaAI/PylaAI

Pyla 143 is an external Brawl Stars automation bot focused on Showdown trio behavior, joystick movement, auto-aim support, lobby automation, brawler selection, trophy tracking, Discord/Telegram notifications, and emulator recovery.

## Architecture

- `main.py` is the application entry point and owns the main runtime loop in `pyla_main()`.
- `play.py` contains the match play loop logic and calls playstyle code.
- `stage_manager.py` handles state transitions, lobby/end-screen flow, trophy updates, brawler queue progression, rewards, and notifications.
- `window_controller.py` owns emulator/scrcpy interaction, screenshots, clicks, keys, and recovery.
- `auto_aim.py` contains auto-aim target selection and attack helpers. Treat this as high-risk.
- `tactical_movement.py` contains movement, dodge, joystick, retreat, wall bypass, and combat movement helpers. Treat this as high-risk.
- `playstyles/` contains `.pyla` playstyle scripts. `playstyles/team_showdown.pyla` is the primary tuned trio playstyle.
- `gui/` contains the CustomTkinter GUI. `gui/hub.py` is the main settings hub, `gui/select_brawler.py` is brawler selection, and `gui/main.py` wires login, hub, selection, and bot startup.
- `lobby_automation.py` selects brawlers in the emulator and starts matches from lobby.
- `trophy_observer.py` calculates trophy deltas and match history.
- `telegram_notifier.py` and `telegram_control.py` implement Telegram notifications, polling commands, inline buttons, inline mode, and safe remote control.
- `pyla_stats.py` stores local JSON stats for Telegram/status use.
- `config_reload.py` implements safe config reload reporting.
- Config files live in `cfg/`.
- Models live in `models/`.
- Tests live in `tests/`.

## Runtime Flow

1. `main.py` starts GUI via `run_app()`.
2. `gui/main.py` opens login, hub, then `SelectBrawler`.
3. Selected brawler rows are saved to `latest_brawler_data.json`.
4. `pyla_main(data)` creates `WindowController`, `Play`, `StageManager`, `RuntimeControlWindow`, Discord control, and Telegram control.
5. The loop screenshots the emulator, detects state, lets `StageManager` handle lobby/reward/end screens, and calls `Play.main()` only in match state.
6. Pause/resume is controlled through `logs/runtime_control_<pid>.state`.

## Running

- Normal run: `python main.py`
- GUI run: `python main.py`
- Setup: `python setup.py --pyla-install`
- Tests: `python -m unittest discover`
- Quick compile check: `python -m py_compile main.py gui/select_brawler.py telegram_control.py telegram_notifier.py pyla_stats.py config_reload.py`
- Focused checks:
  - `python -m unittest tests.test_auto_aim tests.test_tactical_movement`
  - `python -m unittest tests.test_telegram_support tests.test_telegram_rendering tests.test_pyla_stats tests.test_brawler_selection_filters tests.test_config_reload`

## Config Hygiene

- Version sanitized defaults as `cfg/*.toml.example`.
- Keep real `cfg/*.toml` files local and ignored; they can contain player tags, email, passwords, API tokens, Telegram bot tokens, Discord webhooks/tokens, admin IDs, and device paths.
- Create or update local configs with `python scripts/sync_configs.py`.
- Preview config merges with `python scripts/sync_configs.py --dry-run`.
- The sync script adds new keys from examples, keeps obsolete top-level keys for compatibility, creates backups before writes, and preserves known secret keys.

## Development Rules For Agents

- Read the architecture before changing code.
- Avoid broad refactors unless the task truly requires them.
- Do not rename packages/modules just for branding.
- Do not break `auto_aim.py` or `auto_aim_attack()`.
- Do not break `playstyles/team_showdown.pyla`.
- Do not break movement, poison avoidance, teammate follow, cube pickup, wall bypass, joystick movement, or dodge logic.
- New risky behavior must be behind config flags and default to safe/off when appropriate.
- Add fallbacks for missing APIs, missing trophies, missing Telegram token, missing admin IDs, offline emulator, stale configs, and corrupt stats files.
- Add debug logs around new runtime behavior. Useful fields include loaded brawler count, selected brawler, trophy source, applied filters, config updates, and traceback summaries for GUI errors.
- Run `py_compile` on changed Python files.
- Add unit tests for pure functions when possible.
- Do not introduce magic screen coordinates without tying them to resolution ratios or config.
- Do not run GUI-blocking or network-heavy work in the Tk main thread.
- On Windows, consider path handling, Tk threading rules, subprocess behavior, and multiprocessing startup.
- Never log `bot_token`, developer passwords, or API tokens.
- Prefer atomic writes for config/state/stats files that can be updated by GUI and Telegram.

## Files To Treat Carefully

- `main.py`
- `play.py`
- `stage_manager.py`
- `auto_aim.py`
- `tactical_movement.py`
- `window_controller.py`
- `lobby_automation.py`
- `playstyles/team_showdown.pyla`
- `cfg/bot_config.toml`
- `cfg/general_config.toml`
- `cfg/brawl_stars_api.toml`
- `latest_brawler_data.json`

Small targeted edits are fine. Large rewrites need a clear reason and regression checks.
