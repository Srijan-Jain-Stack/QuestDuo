# QuestDuo

QuestDuo is a lightweight game backend for MoodVerse.exe, a moody, choice-driven narrative experience where two players move through a series of atmospheric scenes and solve a final treasure puzzle. The service is designed to stay fully in-memory, keep API contracts stable, and support both mocked local development and a real AI-backed path for text generation and background imagery.

## Architecture overview

- FastAPI app in `main.py`
- In-memory `SESSIONS` dictionary holds the current state for each game session
- Pydantic `Scene` models define the request/response contract for scenes
- Gemini-based text generation is handled via `call_gemini()` and validated before use
- Background art is loaded asynchronously through `generate_image()` using the Hugging Face Inference API
- The game state tracks current scene, wrong answers, treasure success, and game-over status without a database

## Open-source models used

- Text and scene generation: Google Gemma 3 / Gemma 4 class models delivered through the Gemini API (`GEMINI_MODEL`)
- Image generation: Hugging Face image model such as `black-forest-labs/FLUX.1-dev` (`IMAGE_MODEL`)

## Local setup

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn httpx pydantic python-dotenv
   ```
3. Copy `.env.example` to `.env` and fill in the environment variables.
4. Start the app:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Visit `http://localhost:8000/docs` for the FastAPI Swagger UI.

## Docker setup

1. Copy `.env.example` to `.env` and set the variables.
2. Build and run with Docker Compose:
   ```bash
   docker compose up --build
   ```
3. The service will be exposed on `http://localhost:8000`.

## Self-hosting on a DigitalOcean GPU Droplet

1. Create a new Ubuntu GPU droplet with Python and Docker installed.
2. Clone the repo onto the server.
3. Copy `.env.example` to `.env` and populate `GEMINI_API_KEY`, `HF_TOKEN`, `TREASURE_ANSWER`, and the model names.
4. Run the app as a Docker service:
   ```bash
   docker compose up -d --build
   ```
5. Keep the container behind a reverse proxy or cloud firewall if exposing the port publicly.
6. Store your secret values in environment variables or a secret manager, never in source control.

## Notes

- `MOCK_AI=1` keeps the game fully functional without external AI calls during local development.
- The app intentionally never exposes the `TREASURE_ANSWER` value in logs, prompts, or API payloads.
- The backend is written to fail gracefully: AI generation errors return safe fallbacks, and endpoints continue running without crashing.
