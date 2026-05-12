import unittest

from localization import normalize_language, tr


class LocalizationTests(unittest.TestCase):
    def test_unknown_language_falls_back(self):
        self.assertEqual(normalize_language("zz"), "ru")

    def test_key_falls_back_to_english(self):
        self.assertEqual(tr("telegram.button.status", language="zz"), "Статус")
        self.assertEqual(tr("missing.key", language="ru", default="Default"), "Default")


if __name__ == "__main__":
    unittest.main()
