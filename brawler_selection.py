from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from utils import normalize_brawler_name


@dataclass(frozen=True)
class BrawlerCard:
    name: str
    trophies: int | None = None
    rarity: str | None = None
    selected: bool = False
    available: bool = True
    target: int | None = None
    today: int | None = None


RARITY_ORDER = {
    "starting": 0,
    "common": 0,
    "rare": 1,
    "super rare": 2,
    "super_rare": 2,
    "epic": 3,
    "mythic": 4,
    "legendary": 5,
    "ultra legendary": 6,
    "ultra_legendary": 6,
    "chromatic": 7,
}


def _known_trophies(card: BrawlerCard) -> int:
    return int(card.trophies) if card.trophies is not None else -1


def _rarity_rank(rarity: str | None) -> int:
    if rarity is None:
        return 999
    return RARITY_ORDER.get(str(rarity).strip().lower(), 999)


def build_brawler_cards(
        brawlers: Iterable[str],
        trophies_by_brawler: dict[str, int] | None = None,
        selected_brawlers: Iterable[str] | None = None,
        rarities_by_brawler: dict[str, str] | None = None,
) -> list[BrawlerCard]:
    trophies_by_brawler = trophies_by_brawler or {}
    rarities_by_brawler = rarities_by_brawler or {}
    normalized_trophies = {
        normalize_brawler_name(name): int(trophies)
        for name, trophies in trophies_by_brawler.items()
        if trophies is not None
    }
    normalized_rarities = {
        normalize_brawler_name(name): str(rarity)
        for name, rarity in rarities_by_brawler.items()
        if rarity
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
                rarity=normalized_rarities.get(normalized),
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
    elif sort_mode == "rarity":
        filtered.sort(key=lambda card: (_rarity_rank(card.rarity), card.name.lower()))
    else:
        filtered.sort(key=lambda card: card.name.lower())
    return filtered


def trophy_sort_available(trophies_by_brawler: dict[str, int] | None) -> bool:
    return bool(trophies_by_brawler) and any(value is not None for value in trophies_by_brawler.values())


def selected_names_from_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    names = []
    for row in rows or []:
        name = str(row.get("brawler", "")).strip()
        if name:
            names.append(name)
    return names
