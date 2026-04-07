"""API tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_root_serves_spa(api_client: TestClient) -> None:
    """Root endpoint should serve the production SPA."""

    response = api_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ParkinsonAI" in response.text
    assert "Patient Assessment" in response.text


def test_health_endpoint(api_client: TestClient) -> None:
    """Health endpoint should return ok."""

    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_graph_network_endpoint(api_client: TestClient) -> None:
    """Graph network endpoint should return nodes and edges for the SPA."""

    response = api_client.get("/graph/network")
    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"]
    assert payload["edges"]
    assert "Parkinson disease" in payload["autocomplete"]


def test_assessment_endpoint(api_client: TestClient) -> None:
    """Assessment endpoint should return staging, risk, and a narrative report."""

    response = api_client.post(
        "/assessment",
        json={
            "saa_result": True,
            "saa_biofluid": "CSF",
            "nfl_pg_ml": 22.0,
            "genetic_variants": ["LRRK2 G2019S"],
            "datscan_sbr": 1.7,
            "updrs_part3": 28.0,
            "rbd_present": True,
            "hyposmia": True,
            "motor_signs": True,
            "functional_impairment": "mild",
            "age": 67,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["nsd_iss"]["stage"] in {"2B", "3", "4", "5", "6"}
    assert payload["synneurge"]["label"].startswith("S")
    assert payload["risk"]["score"] >= 0.0
    assert payload["risk"]["recommended_tests"]
    assert payload["criteria_panels"]["nsd_iss"]
    assert payload["report"]


def test_nsdiss_endpoint(api_client: TestClient) -> None:
    """NSD-ISS endpoint should classify patient payloads."""

    response = api_client.post(
        "/staging/nsdiss",
        json={"saa_result": True, "datscan_abnormal": True, "rbd_present": True, "hyposmia": True},
    )
    assert response.status_code == 200
    assert response.json()["stage"] == "2B"


def test_chat_websocket_streams_agent_events(api_client: TestClient) -> None:
    """WebSocket chat should stream routing and final aggregation events."""

    with api_client.websocket_connect("/ws/chat") as websocket:
        connected = websocket.receive_json()
        assert connected["event"] == "connected"
        websocket.send_text(
            json.dumps(
                {
                    "message": "Assess this patient with SAA+, NfL 22, LRRK2 G2019S, RBD, and hyposmia",
                    "patient_data": {
                        "saa_result": True,
                        "nfl_pg_ml": 22.0,
                        "genetic_variants": ["LRRK2 G2019S"],
                        "rbd_present": True,
                        "hyposmia": True,
                        "age": 67,
                    },
                }
            )
        )
        events: list[dict[str, object]] = []
        for _ in range(16):
            payload = websocket.receive_json()
            events.append(payload)
            if payload["event"] == "final":
                break

    event_names = [str(event["event"]) for event in events]
    assert "route" in event_names
    assert "tasks" in event_names
    assert "agent" in event_names
    assert "sentinel" in event_names
    assert event_names[-1] == "final"


def test_chat_websocket_answers_simple_biomarker_query(api_client: TestClient) -> None:
    """Simple chat questions should be answered by the specialist agent."""

    with api_client.websocket_connect("/ws/chat") as websocket:
        _ = websocket.receive_json()
        websocket.send_text(json.dumps({"message": "Interpret NfL level 28 pg/mL in a 65-year-old"}))
        events: list[dict[str, object]] = []
        for _ in range(12):
            payload = websocket.receive_json()
            events.append(payload)
            if payload["event"] == "final":
                break

    agent_events = [event for event in events if event["event"] == "agent"]
    assert agent_events
    assert agent_events[0]["agent"] == "biomarker_interpreter"
    assert "NfL" in str(agent_events[0]["content"])


def test_chat_websocket_answers_general_staging_query(api_client: TestClient) -> None:
    """Staging comparison prompts should return a staging-agent answer without patient data."""

    with api_client.websocket_connect("/ws/chat") as websocket:
        _ = websocket.receive_json()
        websocket.send_text(json.dumps({"message": "Compare NSD-ISS vs SynNeurGe staging"}))
        events: list[dict[str, object]] = []
        for _ in range(12):
            payload = websocket.receive_json()
            events.append(payload)
            if payload["event"] == "final":
                break

    agent_events = [event for event in events if event["event"] == "agent"]
    assert agent_events
    assert agent_events[0]["agent"] == "staging_agent"
    assert "NSD-ISS" in str(agent_events[0]["content"])


def test_chat_websocket_answers_symptom_query(api_client: TestClient) -> None:
    """Generic symptom questions should return a knowledge-graph answer."""

    with api_client.websocket_connect("/ws/chat") as websocket:
        _ = websocket.receive_json()
        websocket.send_text(json.dumps({"message": "First symptoms of PD?"}))
        events: list[dict[str, object]] = []
        for _ in range(12):
            payload = websocket.receive_json()
            events.append(payload)
            if payload["event"] == "final":
                break

    agent_events = [event for event in events if event["event"] == "agent"]
    assert agent_events
    assert agent_events[0]["agent"] == "kg_explorer"
