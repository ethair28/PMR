from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class PolymarketClient(Protocol):
    """Protocol for the subset of Polymarket endpoints used by PMR."""

    def list_markets(self, limit: int, offset: int) -> Sequence[dict[str, Any]]:
        """Return one page of active market payloads from Gamma."""

    def get_price_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity_minutes: int,
    ) -> Sequence[dict[str, Any]]:
        """Return a token price history from the public CLOB API."""

    def get_open_interest(self, condition_id: str) -> float | None:
        """Return open interest for a condition from the public Data API."""


@dataclass(slots=True)
class HttpPolymarketClient:
    """Minimal HTTP client for Polymarket's public APIs."""

    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    data_base_url: str = "https://data-api.polymarket.com"
    timeout_seconds: float = 20.0
    user_agent: str = "PMR/0.1"

    def list_markets(self, limit: int, offset: int) -> Sequence[dict[str, Any]]:
        payload = self._get_json(
            base_url=self.gamma_base_url,
            path="/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": str(limit),
                "offset": str(offset),
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected Polymarket markets payload shape.")
        return payload

    def get_price_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity_minutes: int,
    ) -> Sequence[dict[str, Any]]:
        payload = self._get_json(
            base_url=self.clob_base_url,
            path="/prices-history",
            params={
                "market": token_id,
                "interval": "max",
                "startTs": str(start_ts),
                "endTs": str(end_ts),
                "fidelity": str(fidelity_minutes),
            },
        )
        history = payload.get("history", [])
        if not isinstance(history, list):
            raise RuntimeError("Unexpected Polymarket price-history payload shape.")
        return history

    def get_open_interest(self, condition_id: str) -> float | None:
        payload = self._get_json(
            base_url=self.data_base_url,
            path="/oi",
            params={"market": condition_id},
        )
        if not isinstance(payload, list) or not payload:
            return None
        first_item = payload[0]
        try:
            return float(first_item["value"])
        except (KeyError, TypeError, ValueError):
            return None

    def _get_json(
        self,
        *,
        base_url: str,
        path: str,
        params: dict[str, str],
    ) -> Any:
        query = urlencode(params, doseq=True)
        url = f"{base_url}{path}?{query}"
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
