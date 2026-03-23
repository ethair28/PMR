from __future__ import annotations

import re

from pmr.models import Market


def build_story_family_key(market: Market) -> str:
    """Build a stable-ish story-family key for related market variants."""

    source = market.event_title or market.question or market.slug or market.market_id
    text = source.lower()
    text = text.replace("__", " <num> ")
    text = re.sub(r"\$?\d[\d,]*(?:\.\d+)?", " <num> ", text)
    text = re.sub(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        " <month> ",
        text,
    )
    text = re.sub(r"[^a-z0-9<>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or market.market_id


def build_story_family_label(market: Market) -> str:
    """Return the human-readable label used for a story family."""

    return market.event_title or market.question
