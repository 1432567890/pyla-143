import unittest

import telegram_notifier
from telegram_control import is_admin


class TelegramRenderingTests(unittest.TestCase):
    def test_html_message_has_pyla_143_title(self):
        text = telegram_notifier._format_message("heartbeat", {"brawler": "nita", "trophies": 856}, language="en")

        self.assertIn("<b>Heartbeat</b>", text)
        self.assertIn("Brawler: nita", text)

    def test_html_message_escapes_values(self):
        text = telegram_notifier._format_message("heartbeat", {"brawler": "<bad&name>"}, language="en")

        self.assertIn("&lt;bad&amp;name&gt;", text)
        self.assertNotIn("Brawler: <bad&name>", text)

    def test_admin_ids_are_required_for_control(self):
        self.assertTrue(is_admin({"admin_ids": ["123"]}, 123))
        self.assertFalse(is_admin({"admin_ids": []}, 123))
        self.assertFalse(is_admin({"admin_ids": ["456"]}, 123))

    def test_minimal_notification_keyboard_has_only_status(self):
        keyboard = telegram_notifier.notification_keyboard("minimal", language="en")
        buttons = keyboard["inline_keyboard"]
        self.assertEqual(len(buttons), 1)
        self.assertEqual(len(buttons[0]), 1)
        self.assertEqual(buttons[0][0]["callback_data"], "status")

    def test_russian_game_finished_title(self):
        text = telegram_notifier._format_message("match", {"brawler": "Colt", "delta": 8}, language="ru")
        self.assertIn("<b>Матч заверш", text)
        self.assertNotIn("Боец: Colt", text)
        self.assertIn("Изменение: 8", text)

    def test_photo_fallback_without_screenshot_returns_message(self):
        async def run():
            calls = []

            async def fake_send(chat_id, text, token=None, reply_markup=None):
                calls.append((chat_id, text, token, reply_markup))
                return True

            original = telegram_notifier.async_send_message
            telegram_notifier.async_send_message = fake_send
            try:
                self.assertTrue(await telegram_notifier.async_send_photo(123, None, caption="caption", token="token"))
            finally:
                telegram_notifier.async_send_message = original
            self.assertEqual(calls[0][1], "caption")

        import asyncio
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
