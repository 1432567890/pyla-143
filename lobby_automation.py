from difflib import SequenceMatcher
import time

import cv2
import numpy as np

from state_finder import get_state
from utils import (
    extract_text_and_positions,
    extract_text_strings,
    count_hsv_pixels,
    load_toml_as_dict,
    find_template_center,
    resolve_brawler_name_alias,
)

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"
gray_pixels_treshold = load_toml_as_dict("./cfg/bot_config.toml")['idle_pixels_minimum']
class LobbyAutomation:

    def __init__(self, window_controller):
        self.coords_cfg = load_toml_as_dict("./cfg/lobby_config.toml")
        self.window_controller = window_controller

    def _read_state(self):
        try:
            screenshot = self.window_controller.screenshot()
            if screenshot is None:
                return None
            return get_state(screenshot)
        except Exception as e:
            if debug:
                print(f"Could not read state while opening brawler menu: {e}")
            return None

    def open_brawler_selection(self, attempts=None):
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        # Keep these clicks in the left-side BRAWLERS button band. Different
        # emulator scales and event layouts shift the safe center a bit; points
        # that are too low can open the pass/event panels instead.
        cfg_point = tuple(self.coords_cfg.get("lobby", {}).get("brawler_btn", (110, 490)))
        brawler_button_points = (
            (70, 500),
            (90, 500),
            (110, 490),
            (128, 500),
            (60, 535),
            (145, 505),
            cfg_point,
            (76, 420),
            (98, 420),
            (122, 420),
            (72, 455),
            (100, 455),
            (132, 455),
            (82, 385),
            (112, 385),
        )
        if attempts is None:
            attempts = len(brawler_button_points)

        state = self._read_state()
        if state == "brawler_selection":
            return True

        if state == "lobby" and self.click_visible_brawler_menu_button():
            time.sleep(0.8)
            state = self._read_state()
            if state == "brawler_selection":
                return True

        for attempt in range(attempts):
            if state == "shop":
                print("Brawler menu click opened a lobby panel; backing out and retrying Brawlers.")
                self.press_back()
                time.sleep(0.8)
                state = self._read_state()
                if state == "brawler_selection":
                    return True
                if state == "lobby" and self.click_visible_brawler_menu_button():
                    time.sleep(0.8)
                    state = self._read_state()
                    if state == "brawler_selection":
                        return True

            x, y = brawler_button_points[min(attempt, len(brawler_button_points) - 1)]
            self.window_controller.click(int(x * wr), int(y * hr))
            time.sleep(0.8)

            state = self._read_state()
            if state == "brawler_selection":
                return True
            if state == "shop":
                continue
            if state is None:
                # Some tests/controllers cannot provide a state image here. Let
                # the OCR loop continue instead of failing selection up front.
                return True

        return False

    def click_visible_brawler_menu_button(self):
        try:
            screenshot = self.window_controller.screenshot()
            if screenshot is None:
                return False
            results = extract_text_and_positions(screenshot)
        except Exception:
            return False

        for text, box in results.items():
            normalized = self.normalize_ocr_name(text)
            if normalized not in {"brawlers", "brawler"}:
                continue
            center = box.get("center")
            if not center:
                continue
            x, y = center
            if x > screenshot.shape[1] * 0.35:
                continue
            self.window_controller.click(int(x), int(y))
            return True
        return False

    def check_for_idle(self, frame):
        general_config = load_toml_as_dict("cfg/general_config.toml")
        bot_config = load_toml_as_dict("./cfg/bot_config.toml")
        debug_enabled = str(general_config.get("super_debug", "no")).lower() in ("yes", "true", "1")
        gray_pixels_threshold = bot_config.get("idle_pixels_minimum", gray_pixels_treshold)
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        # Tight ROI centered on the Idle Disconnect dialog body, so we don't
        # pick up dark gameplay pixels outside the box. V range is wide enough
        # to cover both LDPlayer (bright overlay, V~82) and MuMu (dark overlay, V~28).
        x_start, x_end = int(700 * wr), int(1220 * wr)
        y_start, y_end = int(470 * hr), int(620 * hr)
        gray_pixels = count_hsv_pixels(frame[y_start:y_end, x_start:x_end], (0, 0, 18), (10, 20, 100))
        if debug_enabled: print(f"gray pixels (if > {gray_pixels_threshold} then bot will try to unidle) :", gray_pixels)
        if gray_pixels > gray_pixels_threshold:
            self.window_controller.click(int(535 * wr), int(615 * hr))

    def select_brawler(self, brawler):
        self.window_controller.screenshot()
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        general_config = load_toml_as_dict("cfg/general_config.toml")
        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        debug_enabled = str(general_config.get("super_debug", "no")).lower() in ("yes", "true", "1")
        try:
            ocr_scale = float(general_config.get("ocr_scale_down_factor", 0.65))
        except (TypeError, ValueError):
            ocr_scale = 0.65
        ocr_scale = max(0.35, min(1.0, ocr_scale))
        ocr_attempts = max(1, min(5, int(bot_config.get("brawler_ocr_attempts", 3))))
        confidence_threshold = max(0.50, min(1.0, float(bot_config.get("brawler_ocr_confidence_threshold", 0.86))))
        target_key = self.normalize_ocr_name(brawler)

        if not self.open_brawler_selection():
            print(f"WARNING: Could not open brawler selection menu for '{brawler}'. "
                  "Continuing with the currently selected brawler instead of crashing.")
            self.press_back()
            return False
        c = 0
        found_brawler = False
        for i in range(50):
            screenshot_full = self.wait_for_stable_screen()
            full_h = screenshot_full.shape[0]
            matches, ocr_debug = self.find_brawler_ocr_matches(
                target_key,
                ocr_scale,
                attempts=ocr_attempts,
                confidence_threshold=confidence_threshold,
            )
            if debug_enabled:
                print(
                    f"target_brawler={brawler} OCR attempts={ocr_debug['attempts']} "
                    f"raw={ocr_debug['raw']} normalized={ocr_debug['normalized']} "
                    f"best={ocr_debug.get('best')} confidence={ocr_debug.get('confidence', 0):.2f}"
                )
            if matches:
                matches.sort(key=lambda item: item[0], reverse=True)
                confidence, detected_name, text_box = matches[0]
                if confidence < confidence_threshold:
                    print(
                        f"false_click_prevented target_brawler={brawler} matched={detected_name} "
                        f"confidence={confidence:.2f} threshold={confidence_threshold:.2f}"
                    )
                    time.sleep(0.35)
                    continue
                x, y = text_box['center']
                click_x = int(x / ocr_scale)
                # EasyOCR returns the text label center, not the card/icon center.
                # Tapping above the label avoids selecting the brawler in the row below.
                y_offset = int(full_h * 0.088)
                click_y = int((y / ocr_scale) - y_offset)
                click_y = max(0, min(full_h - 1, click_y))
                click_allowed = self.is_brawler_click_position_safe(click_x, click_y, screenshot_full)
                print(
                    f"Found brawler {brawler} (OCR: {detected_name}) confidence={confidence:.2f} "
                    f"click_allowed={click_allowed} clicking icon at ({click_x}, {click_y}), y_offset={y_offset}"
                )
                if not click_allowed:
                    print("false_click_prevented reason=unsafe_card_position")
                    time.sleep(0.35)
                    continue
                self.window_controller.click(click_x, click_y)
                time.sleep(1.0)

                verify_screenshot = self.window_controller.screenshot()
                verify_state = get_state(verify_screenshot)
                card_is_open = verify_state in ("brawler_selection", "shop")
                if not card_is_open:
                    try:
                        select_words = {"select", "selegt", "selec", "selct", "selert"}
                        card_is_open = any(
                            self.normalize_ocr_name(text) in select_words
                            for text in extract_text_strings(verify_screenshot)
                        )
                        if card_is_open:
                            print(f"Brawler card detected by SELECT text (state was {verify_state}).")
                    except Exception:
                        pass

                if not card_is_open:
                    print(f"Brawler card did not open after tap (state={verify_state}); retrying without scrolling.")
                    time.sleep(0.5)
                    continue

                card_crop = verify_screenshot[
                    int(full_h * 0.05):int(full_h * 0.22),
                    0:verify_screenshot.shape[1],
                ]
                try:
                    card_texts = extract_text_strings(card_crop)
                except Exception:
                    card_texts = []
                card_name_match = any(
                    self.names_match(self.normalize_ocr_name(text), target_key)
                    for text in card_texts
                ) if card_texts else True

                if not card_name_match:
                    print(f"Card OCR shows {card_texts} but expected '{brawler}'; re-tapping with adjusted offset.")
                    self.press_back()
                    time.sleep(0.5)
                    click_y = int((y / ocr_scale) - int(full_h * 0.04))
                    click_y = max(0, min(full_h - 1, click_y))
                    self.window_controller.click(click_x, click_y)
                    time.sleep(1.0)

                select_x, select_y = self.coords_cfg['lobby']['select_btn'][0], self.coords_cfg['lobby']['select_btn'][1]
                self.window_controller.click(select_x, select_y, already_include_ratio=False)
                time.sleep(0.5)
                print(f"Selected brawler {brawler}")
                found_brawler = True
                break
            if c == 0:
                wr = self.window_controller.width_ratio
                hr = self.window_controller.height_ratio
                self.window_controller.swipe(int(1700 * wr), int(900 * hr), int(1700 * wr), int(850 * hr), duration=0.8)
                print(f"scroll_count={i + 1} screen_stable=False click_allowed=False")
                self.wait_for_stable_screen()
                c += 1
                continue

            self.window_controller.swipe(int(1700 * wr), int(900 * hr), int(1700 * wr), int(650 * hr), duration=0.8)
            print(f"scroll_count={i + 1} screen_stable=False click_allowed=False")
            self.wait_for_stable_screen()
            time.sleep(1)
        if not found_brawler:
            print(f"WARNING: Brawler '{brawler}' was not found after 50 scroll attempts. "
                  f"The bot will continue with the currently selected brawler.")
            return False
        return True

    def wait_for_stable_screen(self, attempts=4, delay=0.18, diff_threshold=2.8):
        previous = None
        latest = self.window_controller.screenshot()
        stable = False
        for _ in range(attempts):
            time.sleep(delay)
            latest = self.window_controller.screenshot()
            if previous is not None:
                small_latest = cv2.resize(latest, (160, 90), interpolation=cv2.INTER_AREA)
                small_previous = cv2.resize(previous, (160, 90), interpolation=cv2.INTER_AREA)
                diff = float(np.mean(cv2.absdiff(small_latest, small_previous)))
                if diff <= diff_threshold:
                    stable = True
                    break
            previous = latest
        print(f"screen_stable={stable}")
        return latest

    def find_brawler_ocr_matches(self, target_key, ocr_scale, attempts=3, confidence_threshold=0.86):
        votes = {}
        raw_seen = []
        normalized_seen = []
        for attempt in range(attempts):
            screenshot_full = self.window_controller.screenshot()
            screenshot = cv2.resize(
                screenshot_full,
                (int(screenshot_full.shape[1] * ocr_scale), int(screenshot_full.shape[0] * ocr_scale)),
                interpolation=cv2.INTER_AREA,
            )
            results = extract_text_and_positions(screenshot)
            for raw_text, text_box in results.items():
                raw_seen.append(str(raw_text))
                detected_name = self.resolve_ocr_typos(self.normalize_ocr_name(raw_text))
                normalized_seen.append(detected_name)
                if not self.names_match(detected_name, target_key):
                    continue
                score = self.name_match_score(detected_name, target_key)
                item = votes.setdefault(detected_name, {"count": 0, "score": 0.0, "box": text_box})
                item["count"] += 1
                item["score"] += score
                item["box"] = text_box
            time.sleep(0.12 + attempt * 0.04)

        matches = []
        for detected_name, item in votes.items():
            vote_ratio = item["count"] / max(1, attempts)
            avg_score = item["score"] / max(1, item["count"])
            confidence = min(1.0, vote_ratio * 0.58 + max(0.0, min(1.0, avg_score / 2.0)) * 0.42)
            if confidence >= confidence_threshold * 0.75:
                matches.append((confidence, detected_name, item["box"]))
        matches.sort(key=lambda item: item[0], reverse=True)
        debug_payload = {
            "attempts": attempts,
            "raw": raw_seen[:20],
            "normalized": normalized_seen[:20],
            "best": matches[0][1] if matches else None,
            "confidence": matches[0][0] if matches else 0.0,
        }
        return matches, debug_payload

    def is_brawler_click_position_safe(self, click_x, click_y, screenshot):
        h, w = screenshot.shape[:2]
        if not (0 <= click_x < w and 0 <= click_y < h):
            return False
        if click_y < h * 0.12 or click_y > h * 0.88:
            return False
        if click_x < w * 0.08 or click_x > w * 0.92:
            return False
        return True

    def select_lowest_trophy_brawler(self):
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio

        def tap(x, y, wait=0.6):
            self.window_controller.click(int(x * wr), int(y * hr))
            time.sleep(wait)

        print("Selecting next brawler by sorting lowest trophies.")
        tap(128, 500, 1.4)   # left Brawlers button in lobby
        tap(1210, 45, 0.6)   # sort dropdown
        tap(1210, 426, 1.0)  # Least Trophies
        tap(422, 359, 1.0)   # first brawler card after sorting
        tap(260, 991, 1.0)   # Select
        if self.ensure_lobby_after_selection():
            return True

        print("Lowest-trophy brawler selection did not return to lobby; trying one recovery pass.")
        self.press_back()
        time.sleep(0.8)
        tap(260, 991, 1.0)   # Select again if the brawler details screen is still open
        return self.ensure_lobby_after_selection()

    def ensure_lobby_after_selection(self, timeout=6.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state = get_state(self.window_controller.screenshot())
            except Exception as e:
                print(f"Could not verify lobby after brawler selection: {e}")
                return False
            if state == "lobby":
                return True
            if state == "brawler_selection":
                # The card opened but Select may not have registered yet.
                self.window_controller.click(
                    int(260 * self.window_controller.width_ratio),
                    int(991 * self.window_controller.height_ratio),
                )
            elif state == "match":
                # Immediately after selecting a brawler, "match" usually means
                # an unrecognized brawler details/stats screen, not a real game.
                self.press_back()
            time.sleep(0.7)
        return False

    def press_back(self):
        if hasattr(self.window_controller, "android_back") and self.window_controller.android_back():
            return
        self.window_controller.click(
            int(100 * self.window_controller.width_ratio),
            int(60 * self.window_controller.height_ratio),
        )

    @staticmethod
    def resolve_ocr_typos(potential_brawler_name: str) -> str:
        """
        Matches well known 'typos' from OCR to the correct brawler's name
        or returns the original string
        """

        return resolve_brawler_name_alias(potential_brawler_name)

    @staticmethod
    def normalize_ocr_name(value: str) -> str:
        normalized = str(value).lower()
        replacements = {
            "0": "o",
            "1": "l",
            "|": "l",
            "!": "i",
            "€": "e",
            "$": "s",
        }
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)
        for symbol in [' ', '-', '.', "&", "'", "`", "_", ":", ";", ",", "’"]:
            normalized = normalized.replace(symbol, "")
        return normalized

    @staticmethod
    def bounded_edit_distance(left: str, right: str, limit: int) -> int:
        if abs(len(left) - len(right)) > limit:
            return limit + 1
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, 1):
            current = [i]
            best = current[0]
            for j, right_char in enumerate(right, 1):
                cost = 0 if left_char == right_char else 1
                value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
                current.append(value)
                best = min(best, value)
            if best > limit:
                return limit + 1
            previous = current
        return previous[-1]

    @classmethod
    def names_match(cls, detected_name: str, target_name: str) -> bool:
        if detected_name == target_name:
            return True
        if len(target_name) >= 4 and (target_name in detected_name or detected_name in target_name):
            return True
        limit = 1 if len(target_name) <= 5 else 2
        if cls.bounded_edit_distance(detected_name, target_name, limit) <= limit:
            return True
        return SequenceMatcher(None, detected_name, target_name).ratio() >= 0.84

    @classmethod
    def name_match_score(cls, detected_name: str, target_name: str) -> float:
        if detected_name == target_name:
            return 2.0
        ratio = SequenceMatcher(None, detected_name, target_name).ratio()
        distance = cls.bounded_edit_distance(detected_name, target_name, 3)
        return ratio - (distance * 0.05)
