import json
import os
import threading
import time
import traceback
import tkinter as tk
from difflib import SequenceMatcher
from math import ceil

import cv2
import customtkinter as ctk
import numpy as np
import pyautogui
from adbutils import adb
from PIL import Image
from customtkinter import CTkImage
from brawler_selection import build_brawler_cards, filter_brawler_cards, selected_names_from_rows, trophy_sort_available
from pyla_stats import load_stats
from utils import (
    extract_text_strings,
    fetch_brawl_stars_player,
    load_brawl_stars_api_config,
    load_toml_as_dict,
    normalize_brawler_name,
    resolve_brawler_name_alias,
    save_brawler_icon,
    get_dpi_scale,
    load_saved_brawler_data,
)
from tkinter import filedialog

from gui.main import install_tk_background_error_filter
from gui.theme import COLORS, font

RARITY_CONFIG_PATH = "cfg/brawler_rarities.json"

orig_screen_width, orig_screen_height = 1920, 1080
width, height = pyautogui.size()
width_ratio = width / orig_screen_width
height_ratio = height / orig_screen_height
scale_factor = min(width_ratio, height_ratio)
scale_factor *= 96/get_dpi_scale()
pyla_version = load_toml_as_dict("./cfg/general_config.toml")['pyla_version']

class SelectBrawler:

    def __init__(self, data_setter, brawlers):
        ctk.set_appearance_mode("dark")
        self.app = ctk.CTk()
        install_tk_background_error_filter(self.app)
        tk._default_root = self.app

        square_size = int(75 * scale_factor)
        self.square_size = square_size
        self.grid_pad = max(1, int(5 * scale_factor))
        self.grid_max_columns = 10
        self.scrollbar_allowance = max(18, int(28 * scale_factor))
        window_width = max(
            int(860 * scale_factor),
            self.grid_max_columns * (self.square_size + 2 * self.grid_pad) + self.scrollbar_allowance + int(18 * scale_factor),
        )
        image_frame_width = max(int(300 * scale_factor), window_width - int(12 * scale_factor))
        self.image_frame_width = image_frame_width
        self.image_frame_top = int(100 * scale_factor)
        amount_of_rows = ceil(len(brawlers)/10) + 1
        necessary_height = (int(145 * scale_factor) + amount_of_rows*square_size + (amount_of_rows-1)*self.grid_pad)
        window_height = min(necessary_height, int(820 * scale_factor))
        image_frame_height = max(int(240 * scale_factor), window_height - int(190 * scale_factor))
        self.image_frame_height = image_frame_height
        self.app.title(f"PylaAi-143 v{pyla_version}")
        self.brawlers = brawlers

        self.app.geometry(f"{window_width}x{window_height}+{str(int(600 * scale_factor))}")
        self.data_setter = data_setter
        self.colors = {
            'gray': "#7d7777",
            'red': "#cd5c5c",
            'darker_white': '#c4c4c4',
            'dark gray': '#1c1c1c',
            'cherry red': '#960a00',
            'ui box gray': '#242424',
            'chess white': '#f0d9b5',
            'chess brown': '#b58863',
            'indian red': "#cd5c5c",
            'bg': '#242424',
            'panel': '#242424',
            'panel_alt': '#1c1c1c',
            'card': '#242424',
            'card_hover': '#1c1c1c',
            'selected': '#960a00',
            'accent': '#960a00',
            'accent_2': '#960a00',
            'muted': '#c4c4c4',
            'text': 'white',
            'border': '#960a00',
            'danger': '#cd5c5c',
            'warning': '#cd5c5c',
            'success': '#7d7777',
        }

        self.app.configure(fg_color=self.colors['ui box gray'])

        self.images = []
        self.visible_image_labels = []
        self.brawlers_data = load_saved_brawler_data()
        self.farm_type = ""
        self.api_trophies_by_brawler = None
        self.api_trophies_by_normalized_brawler = None
        self.trophies_source = "stats cache"
        self.api_trophy_error_reported = False
        self._filter_after_id = None
        self._image_render_after_id = None
        self._current_filter_text = None
        self._current_sort_mode = "rarity"
        self._rendered_card_signature = None
        self._layout_columns = None
        self._resize_after_id = None
        self._api_data_available = False
        self._selected_only = False
        self._needs_push_only = False
        self._target_trophies = 1000
        self._closing = False
        self._closed = False
        self.brawler_rarities = self.load_brawler_rarities()
        self._load_cached_trophies()

        for brawler in self.brawlers:
            img_path = f"./api/assets/brawler_icons/{brawler}.png"
            try:
                img = Image.open(img_path)
            except FileNotFoundError:
                save_brawler_icon(brawler)
                img = Image.open(img_path)

            img_tk = CTkImage(img, size=(square_size, square_size))
            self.images.append((brawler, img_tk))  # Store tuple of brawler name and image

        print(
            "Brawler selector debug:",
            f"loaded_brawlers_count={len(self.brawlers)}",
            f"selected_brawler={self.brawlers_data[0].get('brawler', '') if self.brawlers_data else ''}",
            f"trophies_source={self.trophies_source}",
        )

        self.filter_var = tk.StringVar()
        self.filter_entry = ctk.CTkEntry(
            self.app, textvariable=self.filter_var,
            placeholder_text="Type brawler name...", font=("", int(20 * scale_factor)), width=int(200 * scale_factor),
            fg_color=self.colors['ui box gray'], border_color=self.colors['cherry red'], text_color="white"
        )
        header_text = "Write brawler"
        search_x = int(330 * scale_factor)
        search_width = int(220 * scale_factor)
        search_label = ctk.CTkLabel(
            self.app,
            text=header_text,
            font=("Comic sans MS", int(20 * scale_factor)),
            text_color=self.colors['cherry red'],
            width=search_width,
            anchor="center",
        )
        search_label.place(x=search_x, y=int(scale_factor * 18))
        self.filter_entry.configure(width=search_width)
        self.filter_entry.place(x=search_x, y=int(scale_factor * 52))
        self.filter_var.trace_add("write", lambda *args: self.queue_image_filter_update())

        self.sort_var = tk.StringVar(value="Rarity")
        self.sort_menu = ctk.CTkButton(
            self.app,
            text="Sort: Rarity",
            command=self.open_sort_selector,
            fg_color=self.colors["ui box gray"],
            hover_color=self.colors["dark gray"],
            text_color="white",
            font=("Comic sans MS", int(13 * scale_factor)),
            border_color=self.colors["cherry red"],
            border_width=int(1 * scale_factor),
            width=int(130 * scale_factor),
        )
        self.sort_menu.place(x=int(570 * scale_factor), y=int(52 * scale_factor))
        self.sort_status_label = ctk.CTkLabel(
            self.app,
            text="",
            font=font(int(11 * scale_factor)),
            text_color=self.colors["indian red"],
        )
        self.sort_status_label.place(x=int(570 * scale_factor), y=int(84 * scale_factor))

        self.selected_only_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.app,
            text="Current",
            variable=self.selected_only_var,
            command=self.on_filter_toggle,
            fg_color=self.colors["cherry red"],
            hover_color=self.colors["indian red"],
            text_color="white",
            border_color=self.colors["cherry red"],
        ).place(x=int(710 * scale_factor), y=int(50 * scale_factor))

        self.needs_push_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.app,
            text="Below 1000",
            variable=self.needs_push_var,
            command=self.on_filter_toggle,
            fg_color=self.colors["cherry red"],
            hover_color=self.colors["indian red"],
            text_color="white",
            border_color=self.colors["cherry red"],
        ).place(x=int(710 * scale_factor), y=int(76 * scale_factor))

        # Frame to hold the images
        self.image_frame = ctk.CTkScrollableFrame(
            self.app,
            fg_color=self.colors['ui box gray'],
            width=image_frame_width,
            height=image_frame_height,
        )
        self.image_frame.place(
            x=0,
            y=self.image_frame_top,
            relwidth=1.0,
        )
        self.app.bind("<Configure>", self.on_window_resize)

        self.update_images("")
        ctk.CTkButton(self.app, text="Start", command=self.start_bot, fg_color=self.colors['ui box gray'],
                      text_color="white",
                      font=("Comic sans MS", int(25 * scale_factor)), border_color=self.colors['cherry red'],
                      border_width=int(2 * scale_factor)).place(x=int(390 * scale_factor), y=int((window_height-60* scale_factor) ))

        ctk.CTkButton(self.app, text="Push All", command=self.open_push_all_target_window, fg_color=self.colors['ui box gray'],
                      text_color="white",
                      font=("Comic sans MS", int(25 * scale_factor)), border_color=self.colors['cherry red'],
                      border_width=int(2 * scale_factor)).place(x=int(10 * scale_factor),
                                                                y=int((window_height-60* scale_factor) ))

        self._refresh_trophies_async()
        self.app.mainloop()

    def _load_cached_trophies(self):
        trophies = {}
        try:
            stats = load_stats()
            for brawler, row in stats.get("brawlers", {}).items():
                if isinstance(row, dict) and row.get("current_trophies") is not None:
                    trophies[brawler] = int(row["current_trophies"])
        except Exception as exc:
            print(f"GUI error loading trophy cache: {exc}\n{traceback.format_exc()}")
        self.api_trophies_by_brawler = trophies
        self.api_trophies_by_normalized_brawler = {
            normalize_brawler_name(name): value for name, value in trophies.items()
        }

    def load_brawler_rarities(self):
        rarities = {}
        if not os.path.exists(RARITY_CONFIG_PATH):
            print(f"Brawler selector debug: rarity_config_missing={RARITY_CONFIG_PATH}")
            return rarities
        try:
            with open(RARITY_CONFIG_PATH, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except Exception as exc:
            print(f"GUI error loading brawler rarities: {exc}\n{traceback.format_exc()}")
            return rarities
        if not isinstance(raw, dict):
            print(f"Brawler selector debug: rarity_config_invalid={RARITY_CONFIG_PATH}")
            return rarities
        for brawler, rarity in raw.items():
            normalized = normalize_brawler_name(brawler)
            if normalized and rarity:
                rarities[normalized] = str(rarity)
        print(
            "Brawler selector debug:",
            f"rarity_config_loaded={len(rarities)}",
            f"rarity_config_path={RARITY_CONFIG_PATH}",
        )
        return rarities

    def _refresh_trophies_async(self):
        def worker():
            try:
                self.api_trophies_by_brawler = None
                trophies = self.get_api_trophies_by_brawler()
                if trophies:
                    self.trophies_source = "Brawl Stars API"
                    self._api_data_available = True
                if not self._closing:
                    self.app.after(0, lambda: self.update_images(self.filter_var.get(), force=True))
            except Exception as exc:
                print(f"GUI error refreshing trophies: {exc}\n{traceback.format_exc()}")
        threading.Thread(target=worker, daemon=True).start()

    def queue_image_filter_update(self):
        if self._closing:
            return
        if self._filter_after_id is not None:
            try:
                self.app.after_cancel(self._filter_after_id)
            except Exception:
                pass
        self._filter_after_id = self.app.after(
            120,
            lambda: self.update_images(self.filter_var.get(), force=True)
        )

    def on_window_resize(self, event):
        if self._closing or event.widget is not self.app:
            return
        if self._resize_after_id is not None:
            try:
                self.app.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.app.after(120, self._rerender_after_resize)

    def _rerender_after_resize(self):
        self._resize_after_id = None
        if self._closing:
            return
        columns = self.get_grid_columns()
        if columns != self._layout_columns:
            self.update_images(self.filter_var.get(), force=True)

    def get_grid_columns(self):
        width = self.image_frame_width
        try:
            self.image_frame.update_idletasks()
            canvas = getattr(self.image_frame, "_parent_canvas", None)
            if canvas is not None:
                canvas_width = canvas.winfo_width()
                if canvas_width > 100:
                    width = canvas_width
        except Exception:
            pass
        available_width = max(1, int(width) - self.scrollbar_allowance)
        cell_width = max(1, self.square_size + 2 * self.grid_pad)
        columns = max(1, available_width // cell_width)
        return max(1, min(self.grid_max_columns, columns))

    def on_sort_change(self, value):
        modes = {
            "Rarity": "rarity",
            "Name": "name",
            "Trophies high -> low": "trophies_desc",
            "Trophies low -> high": "trophies_asc",
        }
        requested = modes.get(value, "rarity")
        if requested.startswith("trophies") and not trophy_sort_available(self.api_trophies_by_brawler):
            self._current_sort_mode = "rarity"
            self.sort_var.set("Rarity")
            self.sort_menu.configure(text="Sort: Rarity")
            self.sort_status_label.configure(text="Trophy sorting unavailable: API data not loaded")
        else:
            self._current_sort_mode = requested
            self.sort_var.set(value)
            self.sort_menu.configure(text=f"Sort: {value}")
            self.sort_status_label.configure(text="")
        self.update_images(self.filter_var.get(), force=True)

    def open_sort_selector(self):
        top = ctk.CTkToplevel(self.app)
        top.configure(fg_color=self.colors["panel"])
        top.title("Sort brawlers")
        top.attributes("-topmost", True)
        top.geometry(f"{int(320 * scale_factor)}x{int(220 * scale_factor)}+{int(830 * scale_factor)}+{int(180 * scale_factor)}")
        ctk.CTkLabel(
            top,
            text="Sort brawlers",
            font=font(int(18 * scale_factor), "bold"),
            text_color=self.colors["text"],
        ).pack(pady=(int(14 * scale_factor), int(8 * scale_factor)))
        options = ["Rarity", "Name"]
        if trophy_sort_available(self.api_trophies_by_brawler):
            options.extend(["Trophies high -> low", "Trophies low -> high"])
        else:
            ctk.CTkLabel(
                top,
                text="Trophy sorting unavailable: API data not loaded",
                font=font(int(12 * scale_factor)),
                text_color=self.colors["warning"],
                wraplength=int(260 * scale_factor),
            ).pack(pady=(0, int(8 * scale_factor)))
        for option in options:
            ctk.CTkButton(
                top,
                text=option,
                command=lambda value=option: (self.on_sort_change(value), top.destroy()),
                fg_color=self.colors["accent"] if self.sort_var.get() == option else self.colors["panel_alt"],
                hover_color=self.colors["card_hover"],
                text_color=self.colors["text"],
                width=int(250 * scale_factor),
            ).pack(pady=int(4 * scale_factor))

    def on_filter_toggle(self):
        self._selected_only = bool(self.selected_only_var.get())
        self._needs_push_only = bool(self.needs_push_var.get())
        self.update_images(self.filter_var.get(), force=True)

    def set_farm_type(self, value):
        self.farm_type = value

    def start_bot(self):
        if self._closing:
            return
        brawlers_data = list(self.brawlers_data)
        self._closing = True
        self._cancel_queued_callbacks()
        self._hide_window()
        self.data_setter(brawlers_data)
        if brawlers_data:
            print(
                "Brawler selector debug:",
                f"selected_brawler={brawlers_data[0].get('brawler', '')}",
                "config_updated=true",
            )
        try:
            self.app.quit()
        except Exception:
            pass

    def _cancel_queued_callbacks(self):
        for after_id in (self._filter_after_id, self._image_render_after_id, self._resize_after_id):
            if after_id is None:
                continue
            try:
                self.app.after_cancel(after_id)
            except Exception:
                pass
        self._filter_after_id = None
        self._image_render_after_id = None
        self._resize_after_id = None

    def _hide_window(self):
        try:
            self.app.withdraw()
        except Exception:
            pass
        try:
            self.app.update_idletasks()
        except Exception:
            pass
        try:
            self.app.update()
        except Exception:
            pass

    def close_app(self):
        if self._closed:
            return
        self._closing = True
        self._cancel_queued_callbacks()
        self._hide_window()

        try:
            self.app.quit()
        except Exception:
            pass
        try:
            self.app.destroy()
        except Exception:
            pass
        self._closed = True

    def load_brawler_config(self):
        # open file select dialog to select a json file
        file_path = filedialog.askopenfilename(
            title="Select Brawler Config File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    brawlers_data = json.load(file)
                    try:
                        brawlers_data = [
                            bd for bd in brawlers_data
                            if not (bd["push_until"] <= bd[bd["type"]])
                        ]
                        self.brawlers_data = brawlers_data
                        print("Brawler data loaded successfully :", brawlers_data)
                    except Exception as e:
                        print("Invalid data format. Expected a list of brawler data.", e)
            except Exception as e:
                print(f"Error loading brawler data: {e}")

    def get_push_all_data(self, target_trophies=1000):
        target_trophies = int(target_trophies)
        api_config = load_brawl_stars_api_config("cfg/brawl_stars_api.toml")
        player_data = fetch_brawl_stars_player(
            api_config.get("api_token", "").strip(),
            api_config.get("player_tag", "").strip(),
            int(api_config.get("timeout_seconds", 15)),
        )
        known_by_normalized_name = {
            normalize_brawler_name(brawler): brawler
            for brawler in self.brawlers
        }
        rows = []
        for index, api_brawler in enumerate(player_data.get("brawlers", [])):
            brawler = known_by_normalized_name.get(normalize_brawler_name(api_brawler.get("name", "")))
            if not brawler:
                continue
            trophies = int(api_brawler.get("trophies", 0))
            if trophies < target_trophies:
                rows.append((trophies, index, brawler))

        rows.sort(key=lambda item: (item[0], item[1]))
        data = []
        for idx, (trophies, _, brawler) in enumerate(rows):
            data.append({
                "brawler": brawler,
                "push_until": target_trophies,
                "trophies": trophies,
                "wins": 0,
                "type": "trophies",
                "automatically_pick": idx != 0,
                "selection_method": "lowest_trophies",
                "win_streak": 0,
            })
        return data

    def get_push_all_1k_data(self):
        return self.get_push_all_data(1000)

    @staticmethod
    def _match_brawler_from_ocr_texts(texts, known_brawlers):
        best_brawler = None
        best_score = 0.0
        ambiguous = False
        known_names = [(brawler, normalize_brawler_name(brawler)) for brawler in known_brawlers]
        for raw_text in texts:
            normalized_text = resolve_brawler_name_alias(normalize_brawler_name(raw_text))
            if not normalized_text:
                continue
            for brawler, normalized_brawler in known_names:
                normalized_brawler = resolve_brawler_name_alias(normalized_brawler)
                if normalized_text == normalized_brawler:
                    return brawler
                if normalized_brawler in normalized_text or normalized_text in normalized_brawler:
                    score = min(len(normalized_text), len(normalized_brawler)) / max(
                        len(normalized_text), len(normalized_brawler)
                    )
                else:
                    score = SequenceMatcher(None, normalized_text, normalized_brawler).ratio()
                if abs(score - best_score) < 0.025 and brawler != best_brawler:
                    ambiguous = True
                if score > best_score:
                    best_score = score
                    best_brawler = brawler
                    ambiguous = False
        if ambiguous:
            print(f"Could not choose first sorted brawler confidently; OCR was ambiguous: {texts}")
            return None
        return best_brawler if best_score >= 0.78 else None

    @staticmethod
    def _move_brawler_to_front(data, selected_brawler):
        if not selected_brawler:
            return data
        selected_normalized = normalize_brawler_name(selected_brawler)
        selected_index = None
        for index, row in enumerate(data):
            if normalize_brawler_name(row.get("brawler", "")) == selected_normalized:
                selected_index = index
                break
        if selected_index is None:
            return data
        reordered = [dict(row) for row in data]
        selected_row = reordered.pop(selected_index)
        reordered.insert(0, selected_row)
        for index, row in enumerate(reordered):
            row["automatically_pick"] = index != 0
        return reordered

    def detect_first_sorted_brawler(self, device):
        last_texts = []
        for attempt in range(3):
            try:
                screenshot = device.screenshot()
                frame = np.array(screenshot)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            except Exception as e:
                print(f"Could not screenshot brawler screen for OCR: {e}")
                return None

            height, width = frame.shape[:2]
            crop = frame[
                int(height * 0.16):int(height * 0.56),
                int(width * 0.10):int(width * 0.36),
            ]
            try:
                texts = extract_text_strings(crop)
            except Exception as e:
                print(f"Could not OCR first sorted brawler card: {e}")
                return None

            last_texts = texts
            detected_brawler = self._match_brawler_from_ocr_texts(texts, self.brawlers)
            if detected_brawler:
                print(f"Detected first sorted brawler from game screen: {detected_brawler} (OCR: {texts})")
                return detected_brawler
            time.sleep(0.35 + attempt * 0.2)

        print(f"Could not match first sorted brawler from OCR: {last_texts}")
        return None

    def get_adb_device_for_quick_select(self):
        general_config = load_toml_as_dict("cfg/general_config.toml")
        configured_port = general_config.get("emulator_port", 0)
        selected_emulator = general_config.get("current_emulator", "LDPlayer")
        brawl_package = general_config.get("brawl_stars_package", "com.supercell.brawlstars").strip()
        emulator_ports = {
            "LDPlayer": [5555, 5557, 5559, 5554],
            "MuMu": [16384, 16416, 16448, 7555, 5558, 5557, 5556, 5555, 5554],
        }
        if selected_emulator not in emulator_ports:
            try:
                configured_port_int = int(configured_port)
            except (TypeError, ValueError):
                configured_port_int = 0
            selected_emulator = "MuMu" if configured_port_int in (16384, 16416, 16448, 7555) else "LDPlayer"
        try:
            configured_port = int(configured_port)
        except (TypeError, ValueError):
            configured_port = 0
        preferred_ports = []
        port_candidates = [configured_port] + emulator_ports[selected_emulator] + emulator_ports["LDPlayer"] + emulator_ports["MuMu"]
        for port in port_candidates:
            try:
                port = int(port)
            except (TypeError, ValueError):
                continue
            if port != 5037 and port not in preferred_ports:
                preferred_ports.append(port)
        configured_ports = []
        try:
            configured_ports = [int(configured_port)]
        except (TypeError, ValueError):
            pass

        def serial_port(serial):
            if serial.startswith("emulator-"):
                try:
                    return int(serial.rsplit("-", 1)[1])
                except ValueError:
                    return None
            if ":" in serial:
                try:
                    return int(serial.rsplit(":", 1)[1])
                except ValueError:
                    return None
            return None

        def online_devices():
            devices = []
            for dev in adb.device_list():
                try:
                    if dev.get_state() == "device":
                        devices.append(dev)
                except Exception:
                    pass
            return devices

        def choose_device(devices):
            best_device = None
            best_score = None
            for index, dev in enumerate(devices):
                port = serial_port(dev.serial)
                try:
                    opened_package = dev.app_current().package.strip()
                except Exception:
                    opened_package = ""
                score = (
                    opened_package == brawl_package,
                    port in configured_ports,
                    port in preferred_ports,
                    -index,
                )
                if best_score is None or score > best_score:
                    best_device = dev
                    best_score = score
            return best_device

        devices = online_devices()
        device = choose_device(devices)
        if device:
            return device

        for port in preferred_ports:
            if port == 5037:
                continue
            try:
                adb.connect(f"127.0.0.1:{port}")
            except Exception:
                pass

        devices = online_devices()
        device = choose_device(devices)
        if not device:
            raise ConnectionError("No ADB device found for Push All.")
        return device

    def quick_select_least_trophies_brawler(self):
        device = self.get_adb_device_for_quick_select()
        size = device.window_size()
        wr = size.width / 1920
        hr = size.height / 1080

        def tap(x, y, wait=0.8):
            device.shell(f"input tap {int(x * wr)} {int(y * hr)}")
            time.sleep(wait)

        print(f"Push All using ADB device: {device.serial}")
        tap(128, 500, 1.4)   # left Brawlers button in lobby
        tap(1210, 45, 0.6)   # sort dropdown
        tap(1210, 426, 1.0)  # Least Trophies
        selected_brawler = self.detect_first_sorted_brawler(device)
        tap(422, 359, 1.0)   # first brawler card
        tap(260, 991, 1.0)   # Select
        return device.serial, selected_brawler

    def open_push_all_target_window(self):
        top = ctk.CTkToplevel(self.app)
        top.configure(fg_color=self.colors['ui box gray'])
        top.title("Push All Target")
        top.attributes("-topmost", True)
        win_w = int(360 * scale_factor)
        win_h = int(300 * scale_factor)
        top.geometry(f"{win_w}x{win_h}+{str(int(950 * scale_factor))}+{str(int(260 * scale_factor))}")

        ctk.CTkLabel(
            top,
            text="Push all brawlers to:",
            font=font(int(22 * scale_factor), "bold"),
            text_color=self.colors['red'],
        ).pack(pady=(int(18 * scale_factor), int(10 * scale_factor)))

        button_frame = ctk.CTkFrame(top, fg_color=self.colors['ui box gray'])
        button_frame.pack(pady=int(8 * scale_factor))

        def choose_target(target):
            try:
                top.destroy()
            except Exception:
                pass
            self.push_all(target)

        targets = [250, 500, 750, 1000, 1250, 1500]
        for index, target in enumerate(targets):
            row = index // 2
            col = index % 2
            ctk.CTkButton(
                button_frame,
                text=str(target),
                command=lambda t=target: choose_target(t),
                fg_color=self.colors['ui box gray'],
                hover_color=self.colors['cherry red'],
                text_color=self.colors["text"],
                font=font(int(20 * scale_factor), "bold"),
                border_color=self.colors['cherry red'],
                border_width=int(2 * scale_factor),
                width=int(120 * scale_factor),
                height=int(42 * scale_factor),
            ).grid(row=row, column=col, padx=int(10 * scale_factor), pady=int(8 * scale_factor))

    def push_all(self, target_trophies=1000):
        if self._closing:
            return
        target_trophies = int(target_trophies)
        hidden_for_start = False
        try:
            self.app.withdraw()
            self.app.update_idletasks()
            self.app.update()
            hidden_for_start = True

            data = self.get_push_all_data(target_trophies)
            if not data:
                print(f"Push All: no brawlers below {target_trophies} trophies were found.")
                self.app.deiconify()
                return
            selected_serial, selected_brawler = self.quick_select_least_trophies_brawler()
            if selected_brawler:
                data = self._move_brawler_to_front(data, selected_brawler)
            print(f"Push All {target_trophies} first brawler:", data[0])
            self.brawlers_data = data
            self.start_bot()
        except Exception as e:
            print(f"Push All failed: {e}")
            print(
                "Open cfg/brawl_stars_api.toml and make sure player_tag, developer_email, "
                "developer_password, and auto_refresh_token are set correctly."
            )
            if hidden_for_start:
                try:
                    self.app.deiconify()
                except Exception:
                    pass

    def push_all_1k(self):
        self.push_all(1000)

    def get_api_trophies_by_brawler(self):
        if self.api_trophies_by_brawler is not None:
            return self.api_trophies_by_brawler

        config_path = "cfg/brawl_stars_api.toml"
        try:
            api_config = load_brawl_stars_api_config(config_path)
            if not api_config.get("api_token") or not api_config.get("player_tag"):
                self.api_trophies_by_brawler = {}
                return self.api_trophies_by_brawler
            player_data = fetch_brawl_stars_player(
                api_config.get("api_token", "").strip(),
                api_config.get("player_tag", "").strip(),
                int(api_config.get("timeout_seconds", 15)),
            )
            known_by_normalized_name = {
                normalize_brawler_name(brawler): brawler
                for brawler in self.brawlers
            }
            self.api_trophies_by_brawler = {}
            self.api_trophies_by_normalized_brawler = {}
            for api_brawler in player_data.get("brawlers", []):
                normalized_name = normalize_brawler_name(api_brawler.get("name", ""))
                brawler = known_by_normalized_name.get(normalized_name)
                if brawler:
                    trophies = int(api_brawler.get("trophies", 0))
                    self.api_trophies_by_brawler[brawler] = trophies
                    self.api_trophies_by_normalized_brawler[normalize_brawler_name(brawler)] = trophies
                    self.api_trophies_by_normalized_brawler[normalized_name] = trophies
            print(f"Loaded current trophies for {len(self.api_trophies_by_brawler)} brawlers from Brawl Stars API.")
        except Exception as e:
            self.api_trophies_by_brawler = {}
            self.api_trophies_by_normalized_brawler = {}
            if not self.api_trophy_error_reported:
                print(f"Could not auto-fill trophies. Check {config_path}: {e}")
                self.api_trophy_error_reported = True
        return self.api_trophies_by_brawler

    def get_api_trophies_for_brawler(self, brawler):
        api_trophies = self.get_api_trophies_by_brawler()
        if brawler in api_trophies:
            return api_trophies[brawler]
        if self.api_trophies_by_normalized_brawler is None:
            self.api_trophies_by_normalized_brawler = {
                normalize_brawler_name(name): trophies
                for name, trophies in api_trophies.items()
            }
        return self.api_trophies_by_normalized_brawler.get(normalize_brawler_name(brawler))

    def on_image_click(self, brawler):
        self.open_brawler_entry(brawler)

    def open_brawler_entry(self, brawler):
        top = ctk.CTkToplevel(self.app)
        top.configure(fg_color=self.colors['ui box gray'])
        win_w = int(300 * scale_factor)
        win_h = int(400 * scale_factor)
        top.geometry(
            f"{win_w}x{win_h}+{str(int(1100 * scale_factor))}+{str(int(200 * scale_factor))}")
        top.title("Enter Brawler Data")
        top.attributes("-topmost", True)

        # --- Variables ---
        push_until_var = tk.StringVar()
        trophies_var = tk.StringVar()
        wins_var = tk.StringVar()
        current_win_streak_var = tk.StringVar(value="0")
        auto_pick_var = tk.BooleanVar(value=True) if self.brawlers_data else tk.BooleanVar(value=False)
        api_trophies = self.get_api_trophies_for_brawler(brawler)
        if api_trophies is not None:
            trophies_var.set(str(api_trophies))

        # --- Fixed Y positions for placed widgets ---
        y_title = int(7 * scale_factor)
        y_buttons = int(50 * scale_factor)
        y_field1_label = int(100 * scale_factor)
        y_field1_entry = int(125 * scale_factor)
        y_field2_label = int(165 * scale_factor)
        y_field2_entry = int(190 * scale_factor)
        y_field3_label = int(230 * scale_factor)
        y_field3_entry = int(255 * scale_factor)
        y_auto_pick = int(300 * scale_factor)
        y_submit = int(350 * scale_factor)
        x_center_label = int(70 * scale_factor)
        x_center_entry = int(60 * scale_factor)
        entry_width = int(170 * scale_factor)

        # --- Title ---
        ctk.CTkLabel(top, text=f"Brawler: {brawler}", font=font(int(20 * scale_factor), "bold"),
                     text_color=self.colors['red']).place(x=x_center_label, y=y_title)

        # --- Push type buttons ---
        farm_type_button_frame = ctk.CTkFrame(top, width=int(210 * scale_factor), height=int(40 * scale_factor),
                                              fg_color=self.colors['ui box gray'])
        farm_type_button_frame.place(x=int(45 * scale_factor), y=y_buttons)

        # --- Entry widgets (created but NOT placed yet) ---
        push_until_label = ctk.CTkLabel(top, text="Target Amount", font=font(int(15 * scale_factor), "bold"),
                     text_color=self.colors['chess white'])
        push_until_entry = ctk.CTkEntry(
            top, textvariable=push_until_var, fg_color=self.colors['ui box gray'], text_color=self.colors["text"],
            border_color=self.colors['cherry red'], border_width=int(2 * scale_factor),
            height=int(28 * scale_factor), width=entry_width
        )

        trophies_label = ctk.CTkLabel(top, text="Current Trophies", font=font(int(15 * scale_factor), "bold"),
                     text_color=self.colors['chess white'])
        trophies_entry = ctk.CTkEntry(
            top, textvariable=trophies_var, fg_color=self.colors['ui box gray'], text_color=self.colors["text"],
            border_color=self.colors['cherry red'], border_width=int(2 * scale_factor),
            height=int(28 * scale_factor), width=entry_width
        )

        wins_label = ctk.CTkLabel(top, text="Current Wins", font=font(int(15 * scale_factor), "bold"),
                     text_color=self.colors['chess white'])
        wins_entry = ctk.CTkEntry(
            top, textvariable=wins_var, fg_color=self.colors['ui box gray'], text_color=self.colors["text"],
            border_color=self.colors['cherry red'], border_width=int(2 * scale_factor),
            height=int(28 * scale_factor), width=entry_width
        )

        win_streak_label = ctk.CTkLabel(top, text="Current Brawler's Win Streak", font=font(int(15 * scale_factor), "bold"),
                     text_color=self.colors['chess white'])
        current_win_streak_entry = ctk.CTkEntry(
            top, textvariable=current_win_streak_var, fg_color=self.colors['ui box gray'], text_color=self.colors["text"],
            border_color=self.colors['cherry red'], border_width=int(2 * scale_factor),
            height=int(28 * scale_factor), width=entry_width
        )

        auto_pick_checkbox = ctk.CTkCheckBox(
            top, text="Bot auto-selects brawler", variable=auto_pick_var,
            fg_color=self.colors['cherry red'], text_color=self.colors["text"], checkbox_height=int(24 * scale_factor)
        )

        def submit_data():
            push_until_raw = push_until_var.get()
            push_until_value = int(push_until_raw) if push_until_raw.isdigit() else 0
            trophies_raw = trophies_var.get()
            trophies_value = int(trophies_raw) if trophies_raw.isdigit() else 0
            wins_raw = wins_var.get()
            wins_value = int(wins_raw) if wins_raw.isdigit() else 0
            current_win_streak_raw = current_win_streak_var.get()
            current_win_streak_value = int(current_win_streak_raw) if current_win_streak_raw.isdigit() else 0
            data = {
                "brawler": brawler,
                "push_until": push_until_value,
                "trophies": trophies_value,
                "wins": wins_value,
                "type": self.farm_type,
                "automatically_pick": auto_pick_var.get(),
                "win_streak": current_win_streak_value
            }

            self.brawlers_data = [item for item in self.brawlers_data if item["brawler"] != data["brawler"]]
            self.brawlers_data.append(data)

            print("Selected Brawler Data :", self.brawlers_data)
            print(f"Brawler selector debug: selected_brawler={brawler} config_updated=pending_start")
            self.update_images(self.filter_var.get(), force=True)
            top.destroy()

        submit_button = ctk.CTkButton(
            top, text="Submit", command=submit_data, fg_color=self.colors['ui box gray'],
            border_color=self.colors['cherry red'],
            text_color=self.colors["text"], border_width=int(2 * scale_factor), width=int(80 * scale_factor)
        )

        # --- All dynamic widgets that can be shown/hidden ---
        all_dynamic_widgets = [
            push_until_label, push_until_entry,
            trophies_label, trophies_entry,
            wins_label, wins_entry,
            win_streak_label, current_win_streak_entry,
            auto_pick_checkbox, submit_button
        ]

        def hide_all_fields():
            for w in all_dynamic_widgets:
                w.place_forget()

        def check_submit_visibility():
            """Show submit only when push type is selected and required numeric fields are filled."""
            if self.farm_type == "":
                submit_button.place_forget()
                return
            target_ok = push_until_var.get().isdigit()
            if self.farm_type == "trophies":
                fields_ok = target_ok and trophies_var.get().isdigit() and current_win_streak_var.get().isdigit()
            else:  # wins
                fields_ok = target_ok and wins_var.get().isdigit()
            if fields_ok:
                submit_button.place(x=int(110 * scale_factor), y=y_submit)
            else:
                submit_button.place_forget()

        # Trace all entry vars to re-check submit visibility on every keystroke
        push_until_var.trace_add("write", lambda *a: check_submit_visibility())
        trophies_var.trace_add("write", lambda *a: check_submit_visibility())
        wins_var.trace_add("write", lambda *a: check_submit_visibility())
        current_win_streak_var.trace_add("write", lambda *a: check_submit_visibility())

        def show_trophies_fields():
            hide_all_fields()
            self.farm_type = "trophies"
            self.wins_button.configure(fg_color=self.colors['ui box gray'])
            self.trophies_button.configure(fg_color=self.colors['cherry red'])
            # Field 1: Target Amount
            push_until_label.place(x=x_center_label, y=y_field1_label)
            push_until_entry.place(x=x_center_entry, y=y_field1_entry)
            # Field 2: Current Trophies
            trophies_label.place(x=x_center_label, y=y_field2_label)
            trophies_entry.place(x=x_center_entry, y=y_field2_entry)
            # Field 3: Win Streak
            win_streak_label.place(x=int(40 * scale_factor), y=y_field3_label)
            current_win_streak_entry.place(x=x_center_entry, y=y_field3_entry)
            # Auto-pick checkbox
            auto_pick_checkbox.place(x=int(60 * scale_factor), y=y_auto_pick)
            check_submit_visibility()

        def show_wins_fields():
            hide_all_fields()
            self.farm_type = "wins"
            self.wins_button.configure(fg_color=self.colors['cherry red'])
            self.trophies_button.configure(fg_color=self.colors['ui box gray'])
            # Field 1: Target Amount
            push_until_label.place(x=x_center_label, y=y_field1_label)
            push_until_entry.place(x=x_center_entry, y=y_field1_entry)
            # Field 2: Current Wins
            wins_label.place(x=x_center_label, y=y_field2_label)
            wins_entry.place(x=x_center_entry, y=y_field2_entry)
            # Auto-pick checkbox
            auto_pick_checkbox.place(x=int(60 * scale_factor), y=y_auto_pick)
            check_submit_visibility()

        self.wins_button = ctk.CTkButton(farm_type_button_frame, text="Win Amount", width=int(90 * scale_factor),
                                            command=show_wins_fields,
                                            hover_color=self.colors['cherry red'],
                                            font=font(int(15 * scale_factor), "semibold"),
                                            fg_color=self.colors["ui box gray"],
                                            border_color=self.colors['cherry red'],
                                            border_width=int(2 * scale_factor)
                                            )
        self.trophies_button = ctk.CTkButton(farm_type_button_frame, text="Trophies", width=int(85 * scale_factor),
                                             command=show_trophies_fields,
                                             hover_color=self.colors['cherry red'],
                                             font=font(int(15 * scale_factor), "semibold"),
                                             fg_color=self.colors["ui box gray"],
                                             border_color=self.colors['cherry red'], border_width=int(2 * scale_factor)
                                             )

        self.trophies_button.place(x=int(10 * scale_factor))
        self.wins_button.place(x=int(110 * scale_factor))


    def update_images(self, filter_text, force=False):
        if self._closing:
            return
        filter_text = (filter_text or "").strip().lower()
        filter_signature = (
            filter_text,
            self._current_sort_mode,
            self._selected_only,
            self._needs_push_only,
            tuple(sorted(selected_names_from_rows(self.brawlers_data))),
            self.trophies_source,
            len(self.brawler_rarities),
        )
        if not force and filter_signature == self._current_filter_text:
            print(
                "Brawler selector debug:",
                "gui_render_reason=signature_unchanged",
                "cards_updated_count=0",
                "full_rerender_avoided=true",
            )
            return
        self._current_filter_text = filter_signature
        if self._image_render_after_id is not None:
            try:
                self.app.after_cancel(self._image_render_after_id)
            except Exception:
                pass
            self._image_render_after_id = None
        self.visible_image_labels = []

        image_by_brawler = dict(self.images)
        cards = build_brawler_cards(
            [brawler for brawler, _ in self.images],
            self.api_trophies_by_brawler or {},
            selected_names_from_rows(self.brawlers_data),
            self.brawler_rarities,
        )
        matches = filter_brawler_cards(
            cards,
            search=filter_text,
            sort_mode=self._current_sort_mode,
            selected_only=self._selected_only,
            needs_push_only=self._needs_push_only,
            target_trophies=self._target_trophies,
        )
        if self._current_sort_mode.startswith("trophies") and not trophy_sort_available(self.api_trophies_by_brawler):
            self._current_sort_mode = "rarity"
            self.sort_var.set("Rarity")
            self.sort_menu.configure(text="Sort: Rarity")
            self.sort_status_label.configure(text="Trophy sorting unavailable: API data not loaded")
            matches = filter_brawler_cards(
                cards,
                search=filter_text,
                sort_mode="rarity",
                selected_only=self._selected_only,
                needs_push_only=self._needs_push_only,
                target_trophies=self._target_trophies,
            )
        elif not trophy_sort_available(self.api_trophies_by_brawler):
            self.sort_status_label.configure(text="Trophy sorting unavailable: API data not loaded")
        else:
            self.sort_status_label.configure(text="")
        columns = self.get_grid_columns()
        card_signature = (columns, tuple((card.name, card.trophies, card.rarity, card.selected) for card in matches))
        if not force and card_signature == self._rendered_card_signature:
            print(
                "Brawler selector debug:",
                "gui_render_reason=cards_unchanged",
                "cards_updated_count=0",
                "full_rerender_avoided=true",
                f"trophy_sort_available={trophy_sort_available(self.api_trophies_by_brawler)}",
                f"api_data_available={self._api_data_available}",
            )
            return
        self._rendered_card_signature = card_signature
        old_scroll = None
        try:
            old_scroll = self.image_frame._parent_canvas.yview()
        except Exception:
            pass
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        print(
            "Brawler selector debug:",
            f"gui_render_reason={'force' if force else 'filter_changed'}",
            f"filters_applied=search:{filter_text or '*'},sort:{self._current_sort_mode},"
            f"current:{self._selected_only},below_target:{self._needs_push_only}",
            f"visible={len(matches)}",
            f"trophies_source={self.trophies_source}",
            f"trophy_sort_available={trophy_sort_available(self.api_trophies_by_brawler)}",
            f"api_data_available={self._api_data_available}",
        )

        def render_batch(start_index=0):
            if self._closing:
                return
            for index in range(start_index, min(start_index + 40, len(matches))):
                card_data = matches[index]
                brawler = card_data.name
                img_tk = image_by_brawler[brawler]
                row_num = index // columns
                col_num = index % columns
                label = ctk.CTkLabel(
                    self.image_frame,
                    image=img_tk,
                    text="",
                    fg_color=self.colors["cherry red"] if card_data.selected else self.colors["ui box gray"],
                )
                label._pyla_image_ref = img_tk
                self.visible_image_labels.append(label)
                label.bind("<Button-1>", lambda e, b=brawler: self.on_image_click(b))
                label.bind("<Enter>", lambda e, c=label, s=card_data.selected: c.configure(
                    fg_color=self.colors["cherry red"] if s else self.colors["dark gray"]
                ))
                label.bind("<Leave>", lambda e, c=label, s=card_data.selected: c.configure(
                    fg_color=self.colors["cherry red"] if s else self.colors["ui box gray"]
                ))
                label.grid(row=row_num, column=col_num, padx=self.grid_pad, pady=self.grid_pad)
            next_index = start_index + 40
            if next_index < len(matches):
                self._image_render_after_id = self.app.after(1, lambda: render_batch(next_index))
            else:
                self._image_render_after_id = None
                self._layout_columns = columns
                if old_scroll:
                    try:
                        self.image_frame._parent_canvas.yview_moveto(old_scroll[0])
                    except Exception:
                        pass
                print(
                    "Brawler selector debug:",
                    f"cards_updated_count={len(matches)}",
                    "full_rerender_avoided=false",
                    f"selected_brawler={self.brawlers_data[0].get('brawler', '') if self.brawlers_data else ''}",
                )

        render_batch()

def dummy_data_setter(data):
    print("Data set:", data)
