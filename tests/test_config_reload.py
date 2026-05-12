import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config_reload


class ConfigReloadTests(unittest.TestCase):
    def test_reload_config_safe_reports_safe_and_restart_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telegram_config.toml"
            path.write_text('heartbeat_interval_sec = 300\nbot_token = ""\n', encoding="utf-8")

            def fake_load(file_path):
                return {"heartbeat_interval_sec": 300, "bot_token": ""}

            path.write_text('heartbeat_interval_sec = 120\nbot_token = "new"\n', encoding="utf-8")
            with patch.object(config_reload.utils, "load_toml_as_dict", side_effect=fake_load), \
                    patch.object(config_reload.utils, "clear_toml_cache"):
                old_safe = dict(config_reload.SAFE_KEYS)
                old_restart = dict(config_reload.RESTART_KEYS)
                try:
                    config_reload.SAFE_KEYS = {str(path): {"heartbeat_interval_sec"}}
                    config_reload.RESTART_KEYS = {str(path): {"bot_token"}}
                    report = config_reload.reload_config_safe([str(path)])
                finally:
                    config_reload.SAFE_KEYS = old_safe
                    config_reload.RESTART_KEYS = old_restart

        self.assertIn("telegram_config.heartbeat_interval_sec", report["applied"])
        self.assertIn("telegram_config.bot_token", report["requires_restart"])


if __name__ == "__main__":
    unittest.main()
