"""Asynchronous OpenAI-compatible clients for a fleet of local vLLM servers."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import httpx


@dataclass
class LLMResult:
    text: str
    logprobs: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    latency: float
    endpoint: str
    error: str | None = None
    finish_reason: str | None = None


class AsyncLLMPool:
    """Round-robin asynchronous inference with bounded concurrency per replica."""

    def __init__(
        self,
        strong_endpoints: Iterable[str],
        weak_endpoints: Iterable[str] | None = None,
        generator_endpoints: Iterable[str] | None = None,
        *,
        strong_model: str = "qwen25-7b",
        weak_model: str = "qwen25-1_5b",
        generator_model: str | None = None,
        max_concurrency: int = 1536,
        per_replica_concurrency: int = 256,
        timeout: float = 600.0,
        retries: int = 3,
    ) -> None:
        self.endpoints = {
            "strong": [x.rstrip("/") for x in strong_endpoints if x.strip()],
            "weak": [x.rstrip("/") for x in (weak_endpoints or []) if x.strip()],
            "generator": [x.rstrip("/") for x in (generator_endpoints or []) if x.strip()],
        }
        if not self.endpoints["strong"]:
            raise ValueError("At least one strong-model endpoint is required")
        if not self.endpoints["weak"]:
            self.endpoints["weak"] = list(self.endpoints["strong"])
        if not self.endpoints["generator"]:
            self.endpoints["generator"] = list(self.endpoints["strong"])
        self.models = {"strong": strong_model, "weak": weak_model, "generator": generator_model or strong_model}
        self.timeout = timeout
        self.retries = retries
        self._clients = {
            endpoint: httpx.AsyncClient(
                base_url=endpoint,
                timeout=httpx.Timeout(timeout, connect=30.0),
                limits=httpx.Limits(max_connections=per_replica_concurrency, max_keepalive_connections=per_replica_concurrency),
                headers={"Content-Type": "application/json"},
            )
            for values in self.endpoints.values() for endpoint in values
        }
        self._semaphores = {endpoint: asyncio.Semaphore(per_replica_concurrency) for endpoint in self._clients}
        self._rr = {"strong": 0, "weak": 0, "generator": 0}
        self.max_concurrency = max_concurrency
        self._global_semaphore = asyncio.Semaphore(max_concurrency)
        self.stats: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0})

    def _endpoint(self, role: str) -> str:
        values = self.endpoints[role]
        index = self._rr[role] % len(values)
        self._rr[role] += 1
        return values[index]

    @staticmethod
    def request(system: str, user: str, *, role: str = "strong", temperature: float = 0.0,
                max_tokens: int = 512, seed: int | None = None, logprobs: bool = False,
                tag: str = "global", structured_json: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"system": system, "user": user, "role": role, "temperature": temperature,
                "max_tokens": max_tokens, "seed": seed, "logprobs": logprobs, "tag": tag,
                "structured_json": structured_json}

    async def _one(self, request: dict[str, Any]) -> LLMResult:
        role = request.get("role", "strong")
        endpoint = self._endpoint(role)
        tag = request.get("tag", "global")
        payload: dict[str, Any] = {
            "model": self.models[role],
            "messages": [{"role": "system", "content": request["system"]},
                         {"role": "user", "content": request["user"]}],
            "temperature": float(request.get("temperature", 0.0)),
            "max_tokens": int(request.get("max_tokens", 512)),
        }
        if request.get("seed") is not None:
            payload["seed"] = int(request["seed"])
        if request.get("logprobs"):
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        if request.get("structured_json") is not None:
            payload["guided_json"] = request["structured_json"]
        started = time.perf_counter()
        last_error: Exception | None = None
        async with self._global_semaphore, self._semaphores[endpoint]:
            for attempt in range(self.retries):
                try:
                    response = await self._clients[endpoint].post("/chat/completions", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
                    usage = data.get("usage", {})
                    raw_logprobs = choice.get("logprobs") or {}
                    entries = raw_logprobs.get("content", []) if isinstance(raw_logprobs, dict) else []
                    elapsed = time.perf_counter() - started
                    self.stats[tag]["calls"] += 1
                    self.stats[tag]["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                    self.stats[tag]["completion_tokens"] += int(usage.get("completion_tokens", 0))
                    self.stats[tag]["seconds"] += elapsed
                    used = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
                    self.stats[tag][f"tokens:{role}"] = self.stats[tag].get(f"tokens:{role}", 0) + used
                    self.stats[tag][f"calls:{role}"] = self.stats[tag].get(f"calls:{role}", 0) + 1
                    return LLMResult(str(content), entries, int(usage.get("prompt_tokens", 0)),
                                     int(usage.get("completion_tokens", 0)), elapsed, endpoint,
                                     finish_reason=choice.get("finish_reason"))
                except (httpx.HTTPError, ValueError, KeyError, asyncio.TimeoutError) as exc:
                    last_error = exc
                    if attempt + 1 < self.retries:
                        await asyncio.sleep(min(2 ** attempt, 4))
        self.stats[tag]["errors"] += 1
        raise RuntimeError(f"LLM request failed at {endpoint}: {last_error}")

    async def complete_many(self, requests: list[dict[str, Any]], *, chunk_size: int = 2048) -> list[LLMResult]:
        results: list[LLMResult] = []
        # Chunking bounds task creation while keeping every replica continuously fed.
        for start in range(0, len(requests), chunk_size):
            chunk = requests[start:start + chunk_size]
            results.extend(await asyncio.gather(*(self._one(request) for request in chunk)))
        return results

    def tag_snapshot(self, tag: str) -> dict[str, float]:
        return dict(self.stats.get(tag, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0}))

    def total_snapshot(self) -> dict[str, float]:
        keys = ["calls", "prompt_tokens", "completion_tokens", "seconds", "errors"]
        return {key: sum(values[key] for values in self.stats.values()) for key in keys}

    async def close(self) -> None:
        await asyncio.gather(*(client.aclose() for client in self._clients.values()), return_exceptions=True)


async def check_endpoints(endpoints: Iterable[str], timeout: float = 10.0) -> dict[str, bool]:
    async def check(endpoint: str) -> tuple[str, bool]:
        try:
            async with httpx.AsyncClient(base_url=endpoint.rstrip("/"), timeout=timeout) as client:
                response = await client.get("/models")
                return endpoint, response.is_success
        except httpx.HTTPError:
            return endpoint, False
    return dict(await asyncio.gather(*(check(endpoint) for endpoint in endpoints)))
