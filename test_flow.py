import os

from fastapi.testclient import TestClient

from main import app


def test_flow() -> None:
    with TestClient(app) as client:
        response = client.post("/api/session", json={"players": ["Astra", "Brio"]})
        assert response.status_code == 200, response.text
        body = response.json()
        session_id = body["session_id"]
        assert body["players"] == ["Astra", "Brio"]

        for _ in range(7):
            response = client.post(f"/api/session/{session_id}/choice", json={"choice_index": 0})
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["scene"]["choices"]

        response = client.post(f"/api/session/{session_id}/answer", json={"answer": "wrong"})
        assert response.status_code == 200, response.text
        assert response.json()["game_over"] is False

        treasure = os.getenv("TREASURE_ANSWER", "").strip()
        if treasure:
            response = client.post(f"/api/session/{session_id}/answer", json={"answer": treasure})
            assert response.status_code == 200, response.text
            assert response.json()["treasure_found"] is True

        for _ in range(2):
            response = client.post(f"/api/session/{session_id}/answer", json={"answer": "wrong"})
            assert response.status_code == 200, response.text

        response = client.post(f"/api/session/{session_id}/answer", json={"answer": "wrong"})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["game_over"] is True

        secret = os.getenv("TREASURE_ANSWER", "")
        assert secret not in response.text
        assert secret not in str(response.json())
