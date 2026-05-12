import customtkinter as ctk

FONT_FAMILY = "Nunito"

COLORS = {
    "bg": "#0B0F14",
    "surface": "#111821",
    "panel": "#151D27",
    "panel_alt": "#1B2633",
    "card": "#182231",
    "card_hover": "#223044",
    "border": "#2D3A4D",
    "border_soft": "#243144",
    "text": "#EEF4FA",
    "muted": "#9AA8B7",
    "subtle": "#697789",
    "accent": "#3B82F6",
    "accent_hover": "#60A5FA",
    "accent_strong": "#2563EB",
    "accent_2": "#22D3EE",
    "danger": "#FB7185",
    "success": "#34D399",
    "link": "#67E8F9",
    "warning": "#F59E0B",
    "violet": "#8B5CF6",
}


def apply_theme():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")


def font(size, weight="normal"):
    tk_weight = "bold" if weight in {"bold", "semibold"} else "normal"
    return (FONT_FAMILY, size, tk_weight)


def button_colors(selected=False):
    if selected:
        return {
            "fg_color": COLORS["accent_strong"],
            "hover_color": COLORS["accent"],
            "text_color": COLORS["text"],
        }
    return {
        "fg_color": COLORS["panel_alt"],
        "hover_color": COLORS["card_hover"],
        "text_color": COLORS["text"],
    }
