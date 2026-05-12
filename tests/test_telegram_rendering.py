import unittest

import telegram_notifier
from telegram_control import is_admin


class TelegramRenderingTests(unittest.TestCase):
    def test_html_message_has_pyla_143_title(self):
        text = telegram_notifier._format_message("heartbeat", {"brawler": "nita", "trophies": 856})

        self.assertIn("<b>Heartbeat</b>", text)
        self.assertIn("Brawler: nita", text)

    def test_html_message_escapes_values(self):
        text = telegram_notifier._format_message("heartbeat", {"brawler": "<bad&name>"})

        self.assertIn("&lt;bad&amp;name&gt;", text)
        self.assertNotIn("Brawler: <bad&name>", text)

    def test_admin_ids_are_required_for_control(self):
        self.assertTrue(is_admin({"admin_ids": ["123"]}, 123))
        self.assertFalse(is_admin({"admin_ids": []}, 123))
        self.assertFalse(is_admin({"admin_ids": ["456"]}, 123))


if __name__ == "__main__":
    unittest.main()
