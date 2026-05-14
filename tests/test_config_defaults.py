import importlib
import ast
from pathlib import Path
import unittest
from unittest.mock import patch


class ConfigDefaultTests(unittest.TestCase):
    def test_missing_api_base_url_defaults_to_localhost(self):
        import utils

        original_api_base_url = utils.api_base_url
        original_cfg_api_base_url = utils.cfg_api_base_url
        try:
            with patch("utils.load_toml_as_dict", return_value={}):
                importlib.reload(utils)
            self.assertEqual(utils.api_base_url, "localhost")
        finally:
            utils.api_base_url = original_api_base_url
            utils.cfg_api_base_url = original_cfg_api_base_url

    def test_hub_timer_settings_have_defaults(self):
        source = Path("gui/hub.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        timer_params = set()
        timer_defaults = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "create_timer_setting":
                for keyword in node.keywords:
                    if keyword.arg == "param_name" and isinstance(keyword.value, ast.Constant):
                        timer_params.add(keyword.value.value)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "timer_defaults":
                        if isinstance(node.value, ast.Dict):
                            for key in node.value.keys:
                                if isinstance(key, ast.Constant):
                                    timer_defaults.add(key.value)

        self.assertTrue(timer_params)
        self.assertFalse(timer_params - timer_defaults)


if __name__ == "__main__":
    unittest.main()
