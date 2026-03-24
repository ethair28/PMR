from __future__ import annotations

import re


PUBLIC_MARKET_PROXY_ASSET_TERMS = (
    "bitcoin",
    "btc",
    "ether",
    "ethereum",
    "eth",
    "gold",
    "silver",
    "crude oil",
    "oil",
    "brent",
    "wti",
    "natural gas",
    "copper",
    "gc",
    "cl",
    "xau",
    "xag",
    "spx",
    "s&p 500",
    "nasdaq",
    "dow",
    "qqq",
    "spy",
)

PUBLIC_MARKET_PROXY_STRONG_PHRASES = (
    "official cme settlement price",
    "active month",
    "front month",
    "binance 1 minute candle",
    "btc/usdt",
    "eth/usdt",
    "final high price",
    "final low price",
    "last traded price",
    "settlement page",
)

PUBLIC_MARKET_PROXY_PATTERNS = (
    r"\bwhat price will\b",
    r"\bwill\b.*\bhit\b",
    r"\bwill\b.*\breach\b",
    r"\bwill\b.*\bdip to\b",
    r"\bequal to or above\b",
    r"\bequal to or below\b",
)

SOCIAL_ACTIVITY_TRACKER_STRONG_PHRASES = (
    "post counter",
    "xtracker.polymarket.com",
    "posts on x",
    "posts on truth social",
    "tweet count",
    "# tweets",
)

SOCIAL_ACTIVITY_TRACKER_PATTERNS = (
    r"\btweets? from\b",
    r"\bhow many tweets\b",
    r"\bhow many posts\b",
    r"\bnumber of times\b",
    r"\bfollowers?\b",
    r"\blikes?\b",
    r"\bviews?\b",
    r"\bsubscribers?\b",
    r"\breposts?\b",
    r"\bretweets?\b",
)


def classify_universe_market_exclusion(
    *,
    question: str | None,
    description: str | None,
    slug: str | None,
    event_title: str | None,
) -> str | None:
    """Return a stable exclusion reason for markets outside PMR's value proposition."""

    if _is_public_market_proxy_market(
        question=question,
        description=description,
        slug=slug,
        event_title=event_title,
    ):
        return "public_market_proxy"
    if _is_social_activity_tracker_market(
        question=question,
        description=description,
        slug=slug,
        event_title=event_title,
    ):
        return "social_activity_tracker"
    return None


def _is_public_market_proxy_market(
    *,
    question: str | None,
    description: str | None,
    slug: str | None,
    event_title: str | None,
) -> bool:
    text = " ".join(part for part in (question, description, slug, event_title) if part).lower()
    if not text:
        return False

    if any(phrase in text for phrase in PUBLIC_MARKET_PROXY_STRONG_PHRASES):
        return True

    has_asset_term = any(_contains_term(text, term) for term in PUBLIC_MARKET_PROXY_ASSET_TERMS)
    if not has_asset_term:
        return False

    return any(re.search(pattern, text) is not None for pattern in PUBLIC_MARKET_PROXY_PATTERNS)


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _is_social_activity_tracker_market(
    *,
    question: str | None,
    description: str | None,
    slug: str | None,
    event_title: str | None,
) -> bool:
    text = " ".join(part for part in (question, description, slug, event_title) if part).lower()
    if not text:
        return False

    if any(phrase in text for phrase in SOCIAL_ACTIVITY_TRACKER_STRONG_PHRASES):
        return True

    return any(re.search(pattern, text) is not None for pattern in SOCIAL_ACTIVITY_TRACKER_PATTERNS)
