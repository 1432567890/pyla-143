import customtkinter as ctk

FONT_FAMILY = "Arial"

COLORS = {
    "bg": "#242424",
    "surface": "#2B2B2B",
    "panel": "#4A4A4A",
    "panel_alt": "#333333",
    "card": "#2B2B2B",
    "card_hover": "#555555",
    "border": "#555555",
    "border_soft": "#444444",
    "text": "#FFFFFF",
    "muted": "#CCCCCC",
    "subtle": "#AAAAAA",
    "accent": "#BB3A3A",
    "accent_hover": "#BB3A3A",
    "accent_strong": "#AA2A2A",
    "accent_2": "#777777",
    "danger": "#FF5555",
    "success": "#55AA55",
    "link": "#4DA3FF",
    "warning": "#D8A03D",
    "violet": "#777777",
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
