"""Day 6 — OpenAI-compatible routing proxy.

Exposes ``POST /v1/chat/completions`` so any OpenAI-style client can point at
TokenGuard with no code change. For each request the proxy:

  1. extracts the user query,
  2. asks the RouterService which model to use,
  3. forwards the call to that model's backend (or, in demo mode, returns a
     stubbed completion so the routing can be exercised without GPUs),
  4. logs telemetry (chosen model, cost, latency, predicted quality).

Endpoints
---------
GET  /healthz            liveness
GET  /v1/models          list routable models
POST /v1/chat/completions  OpenAI-compatible routing entrypoint
GET  /stats              telemetry summary for the dashboard
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tokenguard.proxy.router_service import RouterService
from tokenguard.proxy.telemetry import Record, TelemetryStore


def create_app(router: RouterService, telemetry: TelemetryStore,
               demo_mode: bool = True) -> FastAPI:
    """Build the FastAPI app around a router + telemetry store.

    ``demo_mode=True`` returns a stubbed completion (no backend inference) so the
    routing layer can be demonstrated end-to-end on a laptop. In production the
    forward step would call the chosen model's real endpoint.
    """
    app = FastAPI(title="TokenGuard Router Proxy", version="1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "models": router.models, "routed": telemetry.count()}

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"object": "list",
                "data": [{"id": m, "object": "model"} for m in router.models]}

    @app.get("/stats")
    def stats() -> dict:
        return telemetry.summary()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        messages = body.get("messages", [])
        query = _last_user_message(messages)
        if not query:
            return JSONResponse(status_code=400,
                                content={"error": "no user message found"})

        t0 = time.perf_counter()
        decision = router.route(query)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if demo_mode:
            content = (f"[TokenGuard routed to {decision.chosen_model} "
                       f"(predicted quality {decision.predicted_quality:.2f}, "
                       f"est. cost ${decision.est_cost_usd:.5f})]")
        else:  # pragma: no cover - requires live backends
            content = _forward_to_backend(decision.chosen_model, messages)

        telemetry.log(Record(
            request_id=request_id,
            query_chars=len(query),
            chosen_model=decision.chosen_model,
            predicted_q=decision.predicted_quality,
            cost_usd=decision.est_cost_usd,
            latency_ms=latency_ms,
        ))

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": decision.chosen_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": len(query.split()),
                      "completion_tokens": len(content.split()),
                      "total_tokens": len(query.split()) + len(content.split())},
            "tokenguard": {
                "chosen_model": decision.chosen_model,
                "predicted_quality": decision.predicted_quality,
                "est_cost_usd": decision.est_cost_usd,
                "routing_latency_ms": latency_ms,
            },
        }

    return app


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _forward_to_backend(model: str, messages: list[dict]) -> str:  # pragma: no cover
    """Placeholder for real backend forwarding (vLLM / Ollama / API)."""
    raise NotImplementedError("backend forwarding is configured per deployment")