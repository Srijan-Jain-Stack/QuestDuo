
import base64
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

TREASURE_ANSWER = os.getenv("TREASURE_ANSWER", "CHRONO_FLORA_42")

app = FastAPI(
    title="MoodVerse.exe Backend",
    version="1.0.0",
    description="Two-player pixel-art adventure backend",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory storage: sessions reset when the server restarts.
sessions: dict[str, dict[str, Any]] = {}


# --------------------------------------------------
# Request models
# --------------------------------------------------

class SessionCreateRequest(BaseModel):
    player1: str = "Player 1"
    player2: str = "Player 2"


class ChoiceRequest(BaseModel):
    session_id: str
    choice: str
    player_name: str = "Player"


class TreasureAnswerRequest(BaseModel):
    session_id: str
    answer: str


# --------------------------------------------------
# Home and health
# --------------------------------------------------

@app.get("/")
async def home():
    return {
        "app": "MoodVerse.exe",
        "status": "running",
        "avatar_page": "/static/avatar.html",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# --------------------------------------------------
# Create a two-player adventure session
# --------------------------------------------------

@app.post("/api/session")
async def create_session(request: SessionCreateRequest):
    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "session_id": session_id,
        "players": [
            {"name": request.player1, "choices": []},
            {"name": request.player2, "choices": []},
        ],
        "story": [],
        "treasure_unlocked": False,
        "completed": False,
    }

    return {
        "session_id": session_id,
        "players": sessions[session_id]["players"],
        "message": "Your adventure begins!",
    }


# --------------------------------------------------
# Make an adventure choice
# --------------------------------------------------

@app.post("/api/choice")
async def make_choice(request: ChoiceRequest):
    session = sessions.get(request.session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    choice = request.choice.strip()

    if not choice:
        raise HTTPException(status_code=400, detail="Choice cannot be empty")

    event = {
        "player_name": request.player_name,
        "choice": choice,
    }

    session["story"].append(event)

    for player in session["players"]:
        if player["name"].lower() == request.player_name.lower():
            player["choices"].append(choice)
            break

    return {
        "session_id": request.session_id,
        "choice": choice,
        "story": session["story"],
        "message": f"{request.player_name} chose: {choice}",
    }


# --------------------------------------------------
# Submit the treasure answer
# --------------------------------------------------

@app.post("/api/answer")
async def submit_treasure_answer(request: TreasureAnswerRequest):
    session = sessions.get(request.session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    correct = request.answer.strip().casefold() == TREASURE_ANSWER.strip().casefold()

    if correct:
        session["treasure_unlocked"] = True
        session["completed"] = True

        return {
            "correct": True,
            "treasure_unlocked": True,
            "completed": True,
            "message": "Correct! You unlocked the treasure!",
        }

    return {
        "correct": False,
        "treasure_unlocked": session["treasure_unlocked"],
        "message": "Not quite! Try another answer.",
    }


# --------------------------------------------------
# Free local pixel-art avatar generator
# --------------------------------------------------

@app.post("/api/avatar")
async def generate_avatar(
    player_name: str = Form(...),
    mood: str = Form("adventurous"),
    image: UploadFile | None = File(default=None),
):
    """Create a pixel-art SVG avatar without paid AI services."""

    player_name = player_name.strip()
    mood = mood.strip().lower()

    if not player_name:
        raise HTTPException(status_code=400, detail="Player name is required")

    # Keep optional upload validation compatible with the existing form.
    # Uploaded photos are not used to generate this local avatar.
    if image is not None and image.filename:
        allowed_types = {"image/jpeg", "image/png", "image/webp"}

        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Upload a JPG, PNG, or WebP image",
            )

        contents = await image.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image must be 5 MB or smaller",
            )

    palettes = {
        "adventurous": ("#211b45", "#a78bfa", "#fbbf24"),
        "happy": ("#422006", "#facc15", "#fb7185"),
        "calm": ("#082f49", "#67e8f9", "#a7f3d0"),
        "mysterious": ("#18181b", "#c084fc", "#64748b"),
        "brave": ("#450a0a", "#f87171", "#fbbf24"),
        "creative": ("#500724", "#f472b6", "#a78bfa"),
        "curious": ("#172554", "#60a5fa", "#fde047"),
    }

    background, cape, accent = palettes.get(
        mood, palettes["adventurous"]
    )

    # Use fixed SVG artwork so user input cannot inject SVG markup.
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
      width="256" height="256" viewBox="0 0 256 256"
      shape-rendering="crispEdges">
      <rect width="256" height="256" fill="{background}"/>
      <path d="M24 32h8v8h-8z M208 48h8v8h-8z
               M40 176h8v8h-8z M200 200h8v8h-8z"
            fill="{accent}"/>
      <path d="M120 24h16v8h-16z M128 16h8v8h-8z"
            fill="#ffffff"/>
      <path d="M72 104h24V80h16V64h40v16h16v24h16v64h-16v24H88v-24H72z"
            fill="{cape}"/>
      <path d="M96 88h64v16h16v40h-16v16H96v-16H80v-40h16z"
            fill="#f3c7a5"/>
      <path d="M88 88V64h16V48h48v16h16v24h-16V72h-48v16z"
            fill="#39233b"/>
      <path d="M104 112h16v16h-16z M144 112h16v16h-16z"
            fill="#25213d"/>
      <path d="M120 144h24v8h-24z" fill="#a84c65"/>
      <path d="M104 160h48v16h16v24h-80v-24h16z"
            fill="{accent}"/>
      <path d="M72 200h112v16H72z" fill="#17152e"/>
      <path d="M32 224h192v8H32z" fill="{accent}"/>
      <path d="M48 64h8v24h-8z M192 96h8v24h-8z"
            fill="{accent}"/>
      <text x="128" y="248" font-size="12" font-family="monospace"
            fill="#ffffff" text-anchor="middle">PIXEL HERO</text>
    </svg>"""

    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return {
        "player_name": player_name,
        "mood": mood,
        "avatar_url": f"data:image/svg+xml;base64,{encoded}",
        "message": "Your free local pixel-art avatar is ready!",
    }
