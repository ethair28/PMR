from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
from urllib.error import HTTPError
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
    max_price_history_span_days: int = 14

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
        if end_ts <= start_ts:
            return ()

        span_seconds = max(self.max_price_history_span_days, 1) * 24 * 60 * 60
        collected: dict[int, dict[str, Any]] = {}
        chunk_start = start_ts

        while chunk_start < end_ts:
            chunk_end = min(end_ts, chunk_start + span_seconds)
            for item in self._get_price_history_chunk(
                token_id=token_id,
                start_ts=chunk_start,
                end_ts=chunk_end,
                fidelity_minutes=fidelity_minutes,
            ):
                try:
                    timestamp = int(float(item["t"]))
                except (KeyError, TypeError, ValueError):
                    continue
                collected[timestamp] = item
            if chunk_end >= end_ts:
                break
            chunk_start = chunk_end + 1

        return [collected[timestamp] for timestamp in sorted(collected)]

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

    def _get_price_history_chunk(
        self,
        *,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity_minutes: int,
    ) -> Sequence[dict[str, Any]]:
        try:
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
        except HTTPError as exc:
            if (
                exc.code == 400
                and self._error_body_contains(exc, "interval is too long")
                and end_ts - start_ts > 24 * 60 * 60
            ):
                midpoint = start_ts + ((end_ts - start_ts) // 2)
                left = self._get_price_history_chunk(
                    token_id=token_id,
                    start_ts=start_ts,
                    end_ts=midpoint,
                    fidelity_minutes=fidelity_minutes,
                )
                right = self._get_price_history_chunk(
                    token_id=token_id,
                    start_ts=midpoint + 1,
                    end_ts=end_ts,
                    fidelity_minutes=fidelity_minutes,
                )
                return [*left, *right]
            raise

        history = payload.get("history", [])
        if not isinstance(history, list):
            raise RuntimeError("Unexpected Polymarket price-history payload shape.")
        return history

    @staticmethod
    def _error_body_contains(exc: HTTPError, needle: str) -> bool:
        try:
            body = exc.read().decode("utf-8", errors="ignore").lower()
        except OSError:
            return False
        return needle.lower() in body
