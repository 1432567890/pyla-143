import customtkinter as ctk

FONT_FAMILY = "Nunito"

COLORS = {
    "bg": "#090712",
    "surface": "#100C1D",
    "panel": "#171127",
    "panel_alt": "#1F1735",
    "card": "#211936",
    "card_hover": "#2B2146",
    "border": "#3C2E60",
    "border_soft": "#2B2243",
    "text": "#F7F2FF",
    "muted": "#B9ACD4",
    "subtle": "#81739D",
    "accent": "#8B5CF6",
    "accent_hover": "#A78BFA",
    "accent_strong": "#6D28D9",
    "accent_2": "#C084FC",
    "danger": "#FB7185",
    "success": "#34D399",
    "link": "#A78BFA",
    "warning": "#F59E0B",
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
