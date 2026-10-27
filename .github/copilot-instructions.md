import asyncio
import base64
import json
import os
import re
import time
import uuid
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()

MOCK_AI = os.getenv("MOCK_AI", "1").strip().lower() not in {"0", "false", "no"}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-3-27b-it").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-dev").strip()
TREASURE_ANSWER = os.getenv("TREASURE_ANSWER", "").strip()
GEMINI_RESPONSE_MIMETYPE_FAILED = False
SESSIONS: dict[str, dict[str, Any]] = {}


class Scene(BaseModel):
    scene_id: int = 0
    scene_index: int = 0
    title: str = ""
    text: str = ""
    background_prompt: str = ""
    background_url: Optional[str] = None
    choices: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow", frozen=True)


app = FastAPI(title="MoodVerse.exe")


def make_mock_scene(scene_index: int, players: list[str]) -> Scene:
    scene_templates = [
        {
            "title": "The Hollow Gate",
            "text": f"{players[0]} and {players[1]} step into a moonlit chamber where the ancient gate hums with memory.",
            "background_prompt": "A moody futuristic gate in a moonlit cyberpunk ruin, cinematic lighting, wide shot",
            "choices": ["Open the gate", "Listen to the whispers"],
        },
        {
            "title": "The Shifting Hall",
            "text": "The hall folds and re-forms around you as if the building itself is deciding your fate.",
            "background_prompt": "A surreal hallway that bends in impossible geometry, neon reflections, moody sci-fi",
            "choices": ["Run toward the light", "Study the walls"],
        },
        {
            "title": "The Echoing Archive",
            "text": "Voices from forgotten stories drift through the air, and each one carries a warning.",
            "background_prompt": "A glowing archive chamber with floating holographic pages, dramatic lighting",
            "choices": ["Read the oldest page", "Follow the warning voice"],
        },
        {
            "title": "The Glass Forest",
            "text": "Crystal trees sway in a silent breeze while the ground itself reflects your decisions.",
            "background_prompt": "A glass forest under a purple twilight sky, surreal and serene",
            "choices": ["Touch the crystal trunk", "Follow the silver path"],
        },
        {
            "title": "The Iron Sky",
            "text": "A broken horizon hangs above the city, and the air feels like a held breath.",
            "background_prompt": "A broken iron sky above a futuristic city, dramatic storm clouds, cinematic realism",
            "choices": ["Climb the tower", "Search the rooftops"],
        },
        {
            "title": "The Last Lock",
            "text": "At the end of the path sits a sealed vault guarded by a question older than memory.",
            "background_prompt": "A mysterious vault door in a dark temple, glowing runes, atmospheric lighting",
            "choices": ["Inspect the runes", "Speak the hidden name"],
        },
        {
            "title": "The Treasure Chamber",
            "text": "The chamber opens with a breath of starlight, and the final prize waits just behind a whispering lock.",
            "background_prompt": "A luminous treasure chamber filled with forgotten artifacts and soft golden light",
            "choices": ["Claim the relic", "Step back into the dark"],
        },
    ]
    template = scene_templates[min(scene_index, len(scene_templates) - 1)]
    return Scene(
        scene_id=scene_index,
        scene_index=scene_index,
        title=template["title"],
        text=template["text"],
        background_prompt=template["background_prompt"],
        choices=template["choices"],
        background_url=None,
    )


def fallback_scene(scene_index: int, players: list[str]) -> Scene:
    return make_mock_scene(scene_index, players)


def build_scene_prompt(players: list[str], scene_index: int) -> str:
    player_names = ", ".join(players)
    return (
        "Return valid JSON only. "
        "Create a narrative adventure scene for a two-player game. "
        f"Players: {player_names}. "
        f"Scene number: {scene_index + 1}. "
        "Use this format exactly: "
        '{"scene_id": 0, "scene_index": 0, "title": "...", "text": "...", "background_prompt": "...", "choices": ["...","..."]}' "
        "Keep the tone vivid, cinematic, and concise."
    )


async def call_gemini(prompt: str, *, fallback: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    start = time.perf_counter()
    fallback = fallback or {}

    if MOCK_AI:
        return fallback

    if not GEMINI_API_KEY:
        print("[Gemini] missing GEMINI_API_KEY; using fallback response.")
        return fallback

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    request_payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7},
        "responseMimeType": "application/json",
    }

    for attempt in range(2):
        payload = dict(request_payload)
        if attempt == 1 and "responseMimeType" in payload:
            payload.pop("responseMimeType", None)

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(url, json=payload)

            latency = time.perf_counter() - start
            print(f"[Gemini] latency={latency:.2f}s status={response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError, ValueError):
                    return fallback

                cleaned = raw_text.strip()
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    print(f"[Gemini] failed to parse JSON payload: {cleaned[:500]}")
                return fallback

            if response.status_code == 400 and "responseMimeType" in request_payload and attempt == 0:
                global GEMINI_RESPONSE_MIMETYPE_FAILED
                GEMINI_RESPONSE_MIMETYPE_FAILED = True
                print("[Gemini] HTTP 400 on responseMimeType; retrying without it.")
                continue

            print(f"[Gemini] failed call: status={response.status_code} body={response.text[:1000]}")
            return fallback
        except Exception as exc:  # pragma: no cover - defensive path
            latency = time.perf_counter() - start
            print(f"[Gemini] latency={latency:.2f}s failed with exception: {exc}")
            return fallback

    return fallback


async def generate_image(prompt: str) -> Optional[str]:
    if MOCK_AI:
        return None

    if not HF_TOKEN or not IMAGE_MODEL:
        print("[Image] missing HF_TOKEN or IMAGE_MODEL; returning None.")
        return None

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt},
            )

        if response.status_code != 200:
            print(f"[Image] failed: status={response.status_code} body={response.text[:1000]}")
            return None

        return f"data:image/png;base64,{base64.b64encode(response.content).decode('ascii')}"
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"[Image] exception: {exc}")
        return None


async def _load_scene_background(session_id: str, scene: Scene) -> None:
    if not scene.background_prompt or MOCK_AI:
        return

    background_url = await generate_image(scene.background_prompt)
    if not background_url:
        return

    session = SESSIONS.get(session_id)
    if not session:
        return
    current_scene = session.get("scene")
    if current_scene is None or current_scene.scene_index != scene.scene_index:
        return
    session["scene"] = current_scene.model_copy(update={"background_url": background_url})


async def generate_scene(session_id: str, scene_index: int, players: list[str]) -> Scene:
    if MOCK_AI:
        return make_mock_scene(scene_index, players)

    prompt = build_scene_prompt(players, scene_index)
    fallback_data = {
        "scene_id": scene_index,
        "scene_index": scene_index,
        "title": "The Fading Memory",
        "text": "The world shimmers and the next scene refuses to settle into focus.",
        "background_prompt": "A dreamlike empty chamber with faint light and distant horizons",
        "choices": ["Move forward", "Pause and listen"],
    }
    payload = await call_gemini(prompt, fallback=fallback_data)

    try:
        scene = Scene.model_validate(payload)
    except ValidationError:
        scene = fallback_scene(scene_index, players)

    scene = scene.model_copy(update={"scene_index": scene_index, "scene_id": scene_index})
    if not scene.choices:
        scene = scene.model_copy(update={"choices": ["Move forward", "Pause and listen"]})
    if not scene.background_prompt:
        scene = scene.model_copy(update={"background_prompt": "A cinematic moody scene with atmospheric lighting"})

    asyncio.create_task(_load_scene_background(session_id, scene))
    return scene


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session")
@app.post("/api/start_session")
async def start_session(payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    request = payload or {}
    players = request.get("players") or request.get("player_names") or ["Player 1", "Player 2"]
    if isinstance(players, str):
        players = [players]
    players = [str(player).strip() for player in players if str(player).strip()]
    if len(players) < 2:
        players = players + ["Player 2"] if len(players) == 1 else ["Player 1", "Player 2"]

    session_id = uuid.uuid4().hex
    scene = await generate_scene(session_id, 0, players)
    SESSIONS[session_id] = {
        "players": players,
        "scene_index": 0,
        "scene": scene,
        "game_over": False,
        "wrong_attempts": 0,
        "treasure_found": False,
    }

    return {
        "session_id": session_id,
        "players": players,
        "scene": scene.model_dump(),
        "scene_ready": True,
        "game_over": False,
    }


@app.get("/api/session/{session_id}")
@app.get("/api/scene/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    scene = session["scene"]
    return {
        "session_id": session_id,
        "players": session["players"],
        "scene": scene.model_dump(),
        "scene_ready": True,
        "game_over": session["game_over"],
        "wrong_attempts": session["wrong_attempts"],
        "treasure_found": session["treasure_found"],
    }


@app.post("/api/session/{session_id}/choice")
@app.post("/api/session/{session_id}/play")
async def choose_scene(session_id: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["game_over"]:
        return {
            "session_id": session_id,
            "scene": session["scene"].model_dump(),
            "scene_ready": True,
            "game_over": True,
        }

    request = payload or {}
    choice_index = request.get("choice", request.get("choice_index", 0))
    try:
        choice_index = int(choice_index)
    except (TypeError, ValueError):
        choice_index = 0

    next_index = min(session["scene_index"] + 1, 6)
    scene = await generate_scene(session_id, next_index, session["players"])
    session["scene_index"] = next_index
    session["scene"] = scene

    return {
        "session_id": session_id,
        "choice_index": choice_index,
        "scene": scene.model_dump(),
        "scene_ready": True,
        "game_over": False,
        "wrong_attempts": session["wrong_attempts"],
    }


@app.post("/api/session/{session_id}/answer")
async def submit_answer(session_id: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    request = payload or {}
    answer = str(request.get("answer", "")).strip()
    scene = session["scene"]

    if session["game_over"]:
        return {
            "session_id": session_id,
            "scene": scene.model_dump(),
            "scene_ready": True,
            "game_over": True,
        }

    if answer == TREASURE_ANSWER:
        session["treasure_found"] = True
        return {
            "session_id": session_id,
            "scene": scene.model_dump(),
            "scene_ready": True,
            "game_over": False,
            "treasure_found": True,
            "message": "The chamber yields to your answer and the relic is yours.",
        }

    session["wrong_attempts"] += 1
    if session["wrong_attempts"] >= 3:
        session["game_over"] = True
        return {
            "session_id": session_id,
            "scene": scene.model_dump(),
            "scene_ready": True,
            "game_over": True,
            "wrong_attempts": session["wrong_attempts"],
            "message": "The vault seals itself. You have exhausted your attempts.",
        }

    return {
        "session_id": session_id,
        "scene": scene.model_dump(),
        "scene_ready": True,
        "game_over": False,
        "wrong_attempts": session["wrong_attempts"],
        "message": "The lock rejects that answer, but the chamber remains open.",
    }


@app.get("/api/background/{session_id}")
async def get_background(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    scene = session["scene"]
    return {"background_url": scene.background_url}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# --- Additional repo files ---

# .github/copilot-instructions.md
# MoodVerse.exe backend rules
# - Stack: FastAPI, pydantic v2, httpx, async only. No database, state lives in the in-memory SESSIONS dict.
# - The Scene pydantic model in main.py is frozen. Never rename or remove fields.
# - The TREASURE_ANSWER env var must NEVER appear in any prompt sent to an AI model, any API response, or any log.
# - Every AI call must have: a timeout, one retry, pydantic validation, and a fallback scene or image. A failed AI call must never crash an endpoint.
# - Keep MOCK_AI=1 working at all times.
# - Keep code simple and readable. No new frameworks.

# .gitignore
# __pycache__/
# .venv/
# .env
# .pytest_cache/

# .env.example
# GEMINI_API_KEY=
# GEMINI_MODEL=gemma-3-27b-it
# HF_TOKEN=
# IMAGE_MODEL=black-forest-labs/FLUX.1-dev
# MOCK_AI=1
# TREASURE_ANSWER=
# PORT=8000

# Dockerfile
# FROM python:3.12-slim
# WORKDIR /app
# COPY . .
# RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic python-dotenv
# EXPOSE 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker-compose.yml
# version: '3.9'
# services:
#   app:
#     build: .
#     env_file:
#       - .env
#     ports:
#       - "8000:8000"

# test_flow.py
# import os
# from fastapi.testclient import TestClient
# from main import app
#
# def test_flow() -> None:
#     with TestClient(app) as client:
#         response = client.post("/api/session", json={"players": ["Astra", "Brio"]})
#         assert response.status_code == 200, response.text
#         payload = response.json()
#         session_id = payload["session_id"]
#         assert payload["players"] == ["Astra", "Brio"]
#
#         for _ in range(7):
#             response = client.post(f"/api/session/{session_id}/choice", json={"choice_index": 0})
#             assert response.status_code == 200, response.text
#             body = response.json()
#             assert body["scene"]["choices"]
# 
#         wrong = client.post(f"/api/session/{session_id}/answer", json={"answer": "wrong"})
#         assert wrong.status_code == 200, wrong.text
#         assert wrong.json()["game_over"] is False
#
#         treasure = os.getenv("TREASURE_ANSWER", "").strip()
#         if treasure:
#             right = client.post(f"/api/session/{session_id}/answer", json={"answer": treasure})
#             assert right.status_code == 200, right.text
#             assert right.json()["treasure_found"] is True
# 
#         for _ in range(2):
#             response = client.post(f"/api/session/{session_id}/answer", json={"answer": "wrong"})
#             assert response.status_code == 200, response.text
# 
#         response = client.post(f"/api/session/{session_id}/answer", json={"answer": "wrong"})
#         assert response.status_code == 200, response.text
#         payload = response.json()
#         assert payload["game_over"] is True
#         assert treasure not in str(response.text)

# README.md
# # QuestDuo
# QuestDuo is a lightweight game backend for MoodVerse.exe, a moody, choice-driven narrative experience ...

# LICENSE
# MIT License
# Copyright (c) 2026 Srijan Jain
# ...
