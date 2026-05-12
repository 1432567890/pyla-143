import tempfile
import unittest
from pathlib import Path

from pyla_stats import default_stats, load_stats, record_trophy_update, reset_daily_if_needed, save_stats


class PylaStatsTests(unittest.TestCase):
    def test_record_trophy_update_counts_positive_gains(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            stats = record_trophy_update("shelly", 800, 812, path)

        self.assertEqual(stats["total_trophies_gained"], 12)
        self.assertEqual(stats["today_trophies_gained"], 12)
        self.assertEqual(stats["session_trophies_gained"], 12)
        self.assertEqual(stats["brawlers"]["shelly"]["current_trophies"], 812)

    def test_record_trophy_update_ignores_losses_for_gain_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            stats = record_trophy_update("shelly", 812, 804, path)

        self.assertEqual(stats["total_trophies_gained"], 0)
        self.assertEqual(stats["brawlers"]["shelly"]["current_trophies"], 804)

    def test_daily_reset_clears_today_only(self):
        stats = default_stats()
        stats["daily_date"] = "2000-01-01"
        stats["today_trophies_gained"] = 20
        stats["total_trophies_gained"] = 100
        stats["brawlers"]["shelly"] = {"gained_today": 20, "gained_total": 100}

        reset = reset_daily_if_needed(stats, today="2000-01-02")

        self.assertEqual(reset["today_trophies_gained"], 0)
        self.assertEqual(reset["total_trophies_gained"], 100)
        self.assertEqual(reset["brawlers"]["shelly"]["gained_today"], 0)

    def test_load_stats_falls_back_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = load_stats(Path(tmp) / "missing.json")

        self.assertIn("total_trophies_gained", stats)


if __name__ == "__main__":
    unittest.main()
