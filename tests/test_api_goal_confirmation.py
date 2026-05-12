import unittest
from unittest.mock import patch

from stage_manager import StageManager


class ApiGoalConfirmationTests(unittest.TestCase):
    def make_stage(self):
        stage = StageManager.__new__(StageManager)
        stage.goal_confirmation_status = {"next_retry_at": 0.0}
        return stage

    @patch("stage_manager.load_brawl_stars_api_config")
    @patch.object(StageManager, "get_api_brawler_trophies")
    def test_confirms_goal_when_api_trophies_reach_target(self, get_trophies, load_config):
        load_config.return_value = {"confirm_goal_reached": True, "poll_interval_sec": 60, "retry_backoff_sec": 30}
        get_trophies.return_value = (1000, "#TAG")
        stage = self.make_stage()

        self.assertTrue(stage.confirm_goal_reached_via_api("shelly", 1000))
        self.assertTrue(stage.goal_confirmation_status["confirmed"])

    @patch("stage_manager.load_brawl_stars_api_config")
    @patch.object(StageManager, "get_api_brawler_trophies")
    def test_denies_goal_when_api_trophies_are_below_target(self, get_trophies, load_config):
        load_config.return_value = {"confirm_goal_reached": True, "poll_interval_sec": 60, "retry_backoff_sec": 30}
        get_trophies.return_value = (999, "#TAG")
        stage = self.make_stage()

        self.assertFalse(stage.confirm_goal_reached_via_api("shelly", 1000))
        self.assertEqual(stage.goal_confirmation_status["status"], "below_target")

    @patch("stage_manager.load_brawl_stars_api_config")
    @patch.object(StageManager, "get_api_brawler_trophies")
    def test_api_failure_keeps_confirmation_pending(self, get_trophies, load_config):
        load_config.return_value = {"confirm_goal_reached": True, "poll_interval_sec": 60, "retry_backoff_sec": 30}
        get_trophies.side_effect = RuntimeError("token missing")
        stage = self.make_stage()

        self.assertFalse(stage.confirm_goal_reached_via_api("shelly", 1000))
        self.assertEqual(stage.goal_confirmation_status["status"], "pending")

    @patch("stage_manager.load_toml_as_dict")
    @patch("stage_manager.load_brawl_stars_api_config")
    def test_config_load_failure_keeps_confirmation_pending(self, load_config, load_raw):
        load_config.side_effect = RuntimeError("developer credentials missing")
        load_raw.return_value = {"retry_backoff_sec": 30, "player_tag": "#TAG"}
        stage = self.make_stage()

        self.assertFalse(stage.confirm_goal_reached_via_api("shelly", 1000))
        self.assertEqual(stage.goal_confirmation_status["status"], "pending")


if __name__ == "__main__":
    unittest.main()
