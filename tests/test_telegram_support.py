import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import telegram_notifier
from runtime_control import PAUSED, RUNNING, read_state
from telegram_control import business_status_text, format_business_name, format_business_trophies, set_runtime_state


class TelegramSupportTests(unittest.TestCase):
    def test_local_config_overrides_template_without_committing_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "telegram_config.toml"
            local = Path(tmp) / "telegram_config.local.toml"
            base.write_text(
                'enabled = false\nbot_token = ""\n',
                encoding="utf-8",
            )
            local.write_text(
                'enabled = true\nbot_token = "local-token"\nnotification_chat_ids = [123]\n',
                encoding="utf-8",
            )
            with patch.object(telegram_notifier, "TELEGRAM_CONFIG_PATH", str(base)), \
                    patch.object(telegram_notifier, "LOCAL_TELEGRAM_CONFIG_PATH", str(local)):
                settings = telegram_notifier.load_telegram_settings()

        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["bot_token"], "local-token")
        self.assertEqual(settings["notification_chat_ids"], ["123"])

    def test_known_chats_are_remembered_for_notifications(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat_path = Path(tmp) / "telegram_chats.toml"
            with patch.object(telegram_notifier, "TELEGRAM_CHATS_PATH", str(chat_path)):
                self.assertTrue(telegram_notifier.remember_chat_id(123))
                self.assertFalse(telegram_notifier.remember_chat_id("123"))
                self.assertTrue(telegram_notifier.remember_chat_id(456))
                self.assertEqual(telegram_notifier.load_known_chat_ids(), ["123", "456"])

    def test_notification_chat_ids_merge_config_and_known_chats(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat_path = Path(tmp) / "telegram_chats.toml"
            with patch.object(telegram_notifier, "TELEGRAM_CHATS_PATH", str(chat_path)):
                telegram_notifier.remember_chat_id(456)
                ids = telegram_notifier.notification_chat_ids({"notification_chat_ids": ["123", "456"]})
        self.assertEqual(ids, ["123", "456"])

    def test_missing_config_defaults_are_ready_except_master_enable(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_telegram_config.toml"
            missing_local = Path(tmp) / "missing_telegram_config.local.toml"
            with patch.object(telegram_notifier, "TELEGRAM_CONFIG_PATH", str(missing)), \
                    patch.object(telegram_notifier, "LOCAL_TELEGRAM_CONFIG_PATH", str(missing_local)):
                settings = telegram_notifier.load_telegram_settings()

        self.assertFalse(settings["enabled"])
        self.assertTrue(settings["send_match_summary"])
        self.assertTrue(settings["include_screenshot"])
        self.assertTrue(settings["remote_control_enabled"])
        self.assertFalse(settings["business_enabled"])
        self.assertFalse(settings["business_change_name_enabled"])
        self.assertEqual(settings["business_name_template"], "{trophies}")
        self.assertFalse(settings["business_change_bio_enabled"])
        self.assertEqual(settings["business_bio_template"], "{trophies}")

    def test_business_trophy_format_uses_truncated_decimal_k(self):
        self.assertEqual(format_business_trophies(64834), "64,8k")
        self.assertEqual(format_business_trophies(843493), "843,4k")
        self.assertEqual(format_business_trophies(1284), "1,2k")
        self.assertEqual(format_business_trophies(1000), "1k")
        self.assertEqual(format_business_trophies(999), "999")

    def test_business_name_template_appends_when_placeholder_missing(self):
        self.assertEqual(format_business_name("segment ✦ {trophies}", "1,2k"), "segment ✦ 1,2k")
        self.assertEqual(format_business_name("segment ✦", "1,2k"), "segment ✦ 1,2k")
        self.assertEqual(format_business_name("", "1,2k"), "1,2k")

    def test_business_connection_survives_chat_id_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat_path = Path(tmp) / "telegram_chats.toml"
            with patch.object(telegram_notifier, "TELEGRAM_CHATS_PATH", str(chat_path)):
                telegram_notifier.remember_business_connection({
                    "id": "bc-1",
                    "is_enabled": True,
                    "user_chat_id": 456,
                    "rights": {"can_change_name": True, "can_change_bio": True},
                })
                telegram_notifier.remember_chat_id(123)
                connection = telegram_notifier.load_business_connection()

        self.assertEqual(connection["id"], "bc-1")
        self.assertTrue(connection["can_change_name"])
        self.assertTrue(connection["can_change_bio"])
        self.assertEqual(connection["user_chat_id"], "456")

    def test_business_status_reports_connection_and_rights(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat_path = Path(tmp) / "telegram_chats.toml"
            with patch.object(telegram_notifier, "TELEGRAM_CHATS_PATH", str(chat_path)):
                telegram_notifier.remember_business_connection({
                    "id": "bc-1",
                    "is_enabled": True,
                    "rights": {"can_change_name": False, "can_change_bio": True},
                })
                text = business_status_text({
                    "business_enabled": True,
                    "business_change_name_enabled": False,
                    "business_name_template": "{trophies}",
                    "business_change_bio_enabled": True,
                    "business_bio_template": "{trophies}",
                })

        self.assertIn("Business mode: yes", text)
        self.assertIn("Connection ID: bc-1", text)
        self.assertIn("Connection active: yes", text)
        self.assertIn("Can change name: no", text)
        self.assertIn("Can change bio: yes", text)
        self.assertIn("Bio updates: yes", text)

    def test_match_summary_hides_potentially_stale_brawler_name(self):
        text = telegram_notifier._format_message("match", {"result": "4th", "brawler": "amber"}, language="en")

        self.assertNotIn("Amber", text)
        self.assertIn("4th", text)

    def test_set_runtime_state_writes_pause_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "runtime.state"
            self.assertEqual(set_runtime_state(state_path, paused=True), PAUSED)
            self.assertEqual(read_state(state_path), PAUSED)
            self.assertEqual(set_runtime_state(state_path, paused=False), RUNNING)
            self.assertEqual(read_state(state_path), RUNNING)


if __name__ == "__main__":
    unittest.main()
