"""WebSocket helpers for streaming multi-agent chat responses."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from parkinson_ai.agents.router import RouterAgent
from parkinson_ai.orchestration.workflow import PDMultiAgentWorkflow


async def stream_agent_chat(
    websocket: WebSocket,
    *,
    workflow: PDMultiAgentWorkflow,
    router: RouterAgent,
) -> None:
    """Stream routed agent results to a connected SPA client."""

    await websocket.accept()
    await websocket.send_json(
        {
            "event": "connected",
            "message": "ParkinsonAI chat connected.",
            "agents": [
                "router",
                "biomarker_interpreter",
                "genetic_counselor",
                "imaging_analyst",
                "literature_agent",
                "kg_explorer",
                "staging_agent",
                "risk_assessor",
                "drug_analyst",
                "sentinel",
            ],
        }
    )
    try:
        while True:
            raw_message = await websocket.receive_text()
            payload = _parse_payload(raw_message)
            message = str(payload.get("message", "")).strip()
            patient_data = payload.get("patient_data")
            if not message:
                await websocket.send_json({"event": "error", "message": "Message must not be empty."})
                continue
            routed = router.classify_task(message)
            await websocket.send_json(
                {
                    "event": "route",
                    "agent": routed.agent,
                    "confidence": routed.confidence,
                    "reason": routed.reason,
                    "suggested_agents": routed.suggested_agents,
                }
            )
            state = await asyncio.to_thread(
                workflow.invoke,
                message,
                patient_data=patient_data if isinstance(patient_data, dict) else None,
            )
            await websocket.send_json(
                {
                    "event": "tasks",
                    "tasks": [asdict(task) for task in state.tasks],
                }
            )
            for result in state.results:
                await websocket.send_json(
                    {
                        "event": "agent",
                        "agent": result.agent,
                        "content": result.content,
                        "metadata": result.metadata,
                    }
                )
            await websocket.send_json({"event": "sentinel", "report": state.sentinel_report})
            await websocket.send_json({"event": "final", "report": state.final_report})
    except WebSocketDisconnect:
        return


def _parse_payload(raw_message: str) -> dict[str, Any]:
    """Parse a JSON chat payload, falling back to plain text."""

    try:
        parsed = json.loads(raw_message)
    except json.JSONDecodeError:
        return {"message": raw_message}
    if isinstance(parsed, dict):
        return {str(key): value for key, value in parsed.items()}
    return {"message": raw_message}
