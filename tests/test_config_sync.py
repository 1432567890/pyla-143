import tempfile
import unittest
from pathlib import Path

import toml

from scripts.sync_configs import sync_one


class ConfigSyncTests(unittest.TestCase):
    def test_sync_adds_new_keys_without_overwriting_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            example = Path(tmp) / "telegram_config.toml.example"
            real = Path(tmp) / "telegram_config.toml"
            example.write_text(
                'enabled = false\nbot_token = ""\nadmin_ids = []\nnotify_on_game_finished = true\n',
                encoding="utf-8",
            )
            real.write_text(
                'enabled = true\nbot_token = "secret-token"\nadmin_ids = [123]\n',
                encoding="utf-8",
            )

            report = sync_one(example)
            merged = toml.load(real)

        self.assertTrue(report["changed"])
        self.assertEqual(merged["bot_token"], "secret-token")
        self.assertEqual(merged["admin_ids"], [123])
        self.assertTrue(merged["notify_on_game_finished"])


if __name__ == "__main__":
    unittest.main()
