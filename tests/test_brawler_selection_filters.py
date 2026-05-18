import unittest

from brawler_selection import (
    build_brawler_cards,
    filter_brawler_cards,
    trophy_sort_available,
    upsert_selected_brawler_row,
)


class BrawlerSelectionFilterTests(unittest.TestCase):
    def test_search_and_sort_by_trophies_desc(self):
        cards = build_brawler_cards(["shelly", "colt", "nita"], {"shelly": 200, "colt": 500, "nita": 300})

        filtered = filter_brawler_cards(cards, search="", sort_mode="trophies_desc")

        self.assertEqual([card.name for card in filtered], ["colt", "nita", "shelly"])

    def test_default_sort_by_rarity(self):
        cards = build_brawler_cards(
            ["spike", "shelly", "colt", "dynamike"],
            rarities_by_brawler={
                "spike": "Legendary",
                "shelly": "Common",
                "colt": "Rare",
                "dynamike": "Super Rare",
            },
        )

        filtered = filter_brawler_cards(cards, sort_mode="rarity")

        self.assertEqual([card.name for card in filtered], ["shelly", "colt", "dynamike", "spike"])

    def test_unknown_rarity_goes_after_known_rarities(self):
        cards = build_brawler_cards(
            ["unknown", "shelly"],
            rarities_by_brawler={"shelly": "Common"},
        )

        filtered = filter_brawler_cards(cards, sort_mode="rarity")

        self.assertEqual([card.name for card in filtered], ["shelly", "unknown"])

    def test_selected_filter(self):
        cards = build_brawler_cards(["shelly", "colt"], {}, ["colt"])

        filtered = filter_brawler_cards(cards, selected_only=True)

        self.assertEqual([card.name for card in filtered], ["colt"])

    def test_needs_push_ignores_unknown_trophies(self):
        cards = build_brawler_cards(["shelly", "colt"], {"shelly": 999}, [])

        filtered = filter_brawler_cards(cards, needs_push_only=True, target_trophies=1000)

        self.assertEqual([card.name for card in filtered], ["shelly"])

    def test_trophy_sort_availability_requires_loaded_trophy_data(self):
        self.assertFalse(trophy_sort_available({}))
        self.assertFalse(trophy_sort_available({"shelly": None}))
        self.assertTrue(trophy_sort_available({"shelly": 200}))

    def test_gui_submit_moves_user_selected_brawler_to_front(self):
        data = [
            {"brawler": "shelly", "automatically_pick": False, "trophies": 100},
            {"brawler": "colt", "automatically_pick": True, "trophies": 200},
        ]
        selected = {"brawler": "nita", "automatically_pick": True, "trophies": 50}

        reordered = upsert_selected_brawler_row(data, selected)

        self.assertEqual([row["brawler"] for row in reordered], ["nita", "shelly", "colt"])
        self.assertFalse(reordered[0]["automatically_pick"])
        self.assertTrue(reordered[1]["automatically_pick"])
        self.assertTrue(reordered[2]["automatically_pick"])

    def test_gui_submit_replaces_existing_brawler_and_moves_it_to_front(self):
        data = [
            {"brawler": "shelly", "automatically_pick": False, "trophies": 100},
            {"brawler": "colt", "automatically_pick": True, "trophies": 200},
        ]
        selected = {"brawler": "colt", "automatically_pick": True, "trophies": 222}

        reordered = upsert_selected_brawler_row(data, selected)

        self.assertEqual([row["brawler"] for row in reordered], ["colt", "shelly"])
        self.assertEqual(reordered[0]["trophies"], 222)
        self.assertFalse(reordered[0]["automatically_pick"])
        self.assertTrue(reordered[1]["automatically_pick"])


if __name__ == "__main__":
    unittest.main()
