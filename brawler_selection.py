from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from utils import normalize_brawler_name


@dataclass(frozen=True)
class BrawlerCard:
    name: str
    trophies: int | None = None
    selected: bool = False
    available: bool = True


def _known_trophies(card: BrawlerCard) -> int:
    return int(card.trophies) if card.trophies is not None else -1


def build_brawler_cards(
        brawlers: Iterable[str],
        trophies_by_brawler: dict[str, int] | None = None,
        selected_brawlers: Iterable[str] | None = None,
) -> list[BrawlerCard]:
    trophies_by_brawler = trophies_by_brawler or {}
    normalized_trophies = {
        normalize_brawler_name(name): int(trophies)
        for name, trophies in trophies_by_brawler.items()
        if trophies is not None
    }
    selected = {
        normalize_brawler_name(name)
        for name in (selected_brawlers or [])
        if normalize_brawler_name(name)
    }
    cards = []
    for brawler in brawlers:
        normalized = normalize_brawler_name(brawler)
        cards.append(
            BrawlerCard(
                name=brawler,
                trophies=normalized_trophies.get(normalized),
                selected=normalized in selected,
            )
        )
    return cards


def filter_brawler_cards(
        cards: Iterable[BrawlerCard],
        search: str = "",
        sort_mode: str = "name",
        selected_only: bool = False,
        needs_push_only: bool = False,
        target_trophies: int | None = None,
) -> list[BrawlerCard]:
    search_key = normalize_brawler_name(search)
    filtered = []
    for card in cards:
        if search_key and search_key not in normalize_brawler_name(card.name):
            continue
        if selected_only and not card.selected:
            continue
        if needs_push_only:
            if target_trophies is None or card.trophies is None or card.trophies >= target_trophies:
                continue
        filtered.append(card)

    if sort_mode == "trophies_desc":
        filtered.sort(key=lambda card: (_known_trophies(card), card.name.lower()), reverse=True)
    elif sort_mode == "trophies_asc":
        filtered.sort(key=lambda card: (card.trophies is None, _known_trophies(card), card.name.lower()))
    else:
        filtered.sort(key=lambda card: card.name.lower())
    return filtered


def selected_names_from_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    names = []
    for row in rows or []:
        name = str(row.get("brawler", "")).strip()
        if name:
            names.append(name)
    return names
