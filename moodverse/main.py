"""MoodVerse.exe backend.  Run:  uvicorn main:app --port 8000
Env vars: GEMINI_API_KEY, GEMMA_MODEL (default gemma-4-26b-a4b-it), MOCK_AI=1 to force mock,
TREASURE_ANSWER (optional: fixed secret; if unset, Gemma invents one per game), PRELOAD_BOTH=0/1.
"""
import asyncio, json, os, random, re, time, uuid
from contextlib import nullcontext
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it").strip()
MOCK = os.getenv("MOCK_AI") == "1" or not API_KEY
ENV_SECRET = os.getenv("TREASURE_ANSWER", "").strip()
PRELOAD_BOTH = os.getenv("PRELOAD_BOTH", "1") == "1"
MAX_SCENES = 7
MAX_ATTEMPTS = 3
AI_SEM = asyncio.Semaphore(2)
HTTP = httpx.AsyncClient(timeout=45)   # one shared client: no new TLS handshake per call, 45s cap instead of 90s
THINK_OFF = True                        # ask model to skip "thinking" (big latency win); auto-disabled if API rejects it
# Default 7-step story path, used only if the AI does not return its own beats
ARC = ["Step through the glowing door into a strange hall", "Cross a risky bridge or path to reach a helper",
       "A helper asks for a small favor and gives a clue", "A locked gate or guard blocks the way",
       "Find the hidden tool that fits the vault", "The lights flicker as time runs short in the last corridor",
       "Stand before the vault and answer its riddle"]
WORDS = ["ECLIPSE", "LANTERN", "COMPASS", "ORBIT", "EMBER", "CIPHER", "BLOSSOM", "MERIDIAN"]

print(f"[MoodVerse] model={MODEL} mock={MOCK} fixed_secret={'yes' if ENV_SECRET else 'no'}")

app = FastAPI(title="MoodVerse.exe")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------- Schema (frozen; frontend depends on it) ----------------
class Dialogue(BaseModel):
    speaker: str
    text: str

class Choice(BaseModel):
    choice_id: str
    description: str
    uses_ability_of: str = "player_1"

class Clue(BaseModel):
    clue_id: str
    name: str
    description: str

class StateUpdate(BaseModel):
    health: int = 100
    inventory: list[str] = []
    game_over: bool = False
    victory: bool = False

class Scene(BaseModel):
    scene_number: int
    max_scenes: int = MAX_SCENES
    scene_title: str
    story_text: str
    background_prompt: str = ""
    background_fallback_tag: str = "bg_generic"
    background_url: Optional[str] = None
    character_dialogue: list[Dialogue] = []
    choices: list[Choice] = []
    clue_found: Optional[Clue] = None
    game_state_update: StateUpdate = StateUpdate()

class PlayerIn(BaseModel):
    name: str
    interests: list[str]
    photo_b64: str = ""
    look: dict = {}

class StartReq(BaseModel):
    players: list[PlayerIn]

class ChoiceReq(BaseModel):
    session_id: str
    choice_id: str

class TreasureReq(BaseModel):
    session_id: str
    answer: str


# ---------------- Helpers ----------------
def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def log_err(s: dict, msg: str) -> None:
    print("ERROR:", msg[:500])
    s.setdefault("errors", []).append(msg[:300])

def redact(text: str, secret: str) -> str:
    if not secret or not text:
        return text
    return re.sub(re.escape(secret), "????", str(text), flags=re.I)

def pname(s: dict, i: int) -> str:
    return s["players"][i]["name"]

def pint(s: dict, i: int) -> str:
    ints = s["players"][i]["interests"]
    return ints[0] if ints else "mystery"


# ---------------- Gemma call ----------------
async def call_gemini(parts: list[dict]) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    global THINK_OFF
    body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.9}}
    if THINK_OFF:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
    async with AI_SEM:
        async with nullcontext(HTTP) as c:
            t0 = time.time()
            r = await c.post(url, json=body)
            if r.status_code == 400 and THINK_OFF:
                print("thinkingConfig rejected by model, disabling it")
                THINK_OFF = False
                body["generationConfig"].pop("thinkingConfig", None)
                r = await c.post(url, json=body)
            if r.status_code == 429:
                print("AI 429 rate limit, waiting 4s and retrying once")
                await asyncio.sleep(4)
                r = await c.post(url, json=body)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            print(f"MODEL USED: {MODEL} | AI call took {time.time() - t0:.1f}s")
    cand = r.json()["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in cand if not p.get("thought")).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON in reply: " + text[:200])
    print("RAW:", text[:300].replace("\n", " "))
    return json.loads(text[a:b + 1])


# ---------------- Turn 0: avatars + world + secret ----------------
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

def clean_look(l) -> dict:
    l = l if isinstance(l, dict) else {}
    out = {"hair": str(l.get("hair", "short")).lower(), "glasses": l.get("glasses") is True, "beard": l.get("beard") is True}
    if out["hair"] not in ("short", "long", "curly", "bald"):
        out["hair"] = "short"
    for k in ("hair_color", "skin", "shirt_color"):
        v = str(l.get(k, ""))
        if HEX.match(v):
            out[k] = v
    return out

def default_avatar(s: dict, i: int) -> dict:
    return {"name": pname(s, i), "class": "Wanderer", "ability_name": "Instinct",
            "ability_desc": f"Uses knowledge of {pint(s, i)}.", "appearance": "", "sprite_prompt": "", "look": clean_look(s["players"][i].get("look"))}

async def setup_world(s: dict) -> None:
    default_world = {"title": f"The {pint(s, 0).title()} and {pint(s, 1).title()} Realm",
                     "genre": "adventure",
                     "premise": f"Two friends, one who loves {pint(s, 0)} and one who loves {pint(s, 1)}, find a strange glowing door. Something precious has gone missing behind it.",
                     "quest": "Find the lost treasure in the vault before the lights go out."}
    s["world"], s["avatars"] = {**default_world, "beats": list(ARC)}, [default_avatar(s, 0), default_avatar(s, 1)]
    secret = ENV_SECRET
    if not MOCK:
        pl = [{"name": p["name"], "interests": p["interests"]} for p in s["players"]]
        prompt = (
            "You set up a SIMPLE 2-player co-op pixel-art adventure. The two player photos follow this text, in order. "
            "Look at each face and read each player's interests. Return ONLY raw JSON (no markdown), in simple everyday words, with exactly this shape:\n"
            '{"world":{"title":"short fun title","premise":"2 simple sentences: where we are and what is wrong, using BOTH players\' interests",'
            '"quest":"ONE simple goal sentence, for example: Find the Lantern Seed in the vault before dawn.",'
            '"beats":["exactly 7 one-line plot beats in order, each in a NEW place with a NEW obstacle or helper, each one step closer to the quest; beat 7 is the vault"],'
            '"secret_word":"ONE uppercase word, 5-10 letters, tied to both interests; it is the key to the final vault"},'
            '"players":[{"name":"","class":"2-3 word RPG class from their interests","ability_name":"","ability_desc":"one short sentence",'
            '"appearance":"short description of the face and clothes",'
            '"look":{"hair":"short|long|curly|bald","glasses":true,"beard":false,"hair_color":"#hex","skin":"#hex","shirt_color":"#hex"}},{...second player...}]}\n'
            f"Players in order: {json.dumps(pl)}")
        parts = [{"text": prompt}]
        for p in s["player_models"]:
            if p.photo_b64:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": p.photo_b64.split(",")[-1]}})
        for attempt in range(2):
            try:
                data = await call_gemini(parts)
                w = data.get("world") or {}
                s["world"] = {k: str(w.get(k) or default_world[k]) for k in default_world}
                bt = [str(x) for x in (w.get("beats") or []) if str(x).strip()]
                s["world"]["beats"] = bt[:MAX_SCENES] if len(bt) >= MAX_SCENES else list(ARC)
                pls = data.get("players") or []
                av = []
                for i in range(2):
                    d = pls[i] if i < len(pls) and isinstance(pls[i], dict) else {}
                    base = default_avatar(s, i)
                    a = {k: str(d.get(k) or base[k]) for k in base if k != "look"}
                    a["name"] = pname(s, i)
                    a["look"] = clean_look(s["players"][i].get("look") or d.get("look"))
                    av.append(a)
                s["avatars"] = av
                if not secret:
                    secret = norm(w.get("secret_word", ""))[:16]
                break
            except Exception as e:
                log_err(s, f"WORLD attempt {attempt + 1}: {e!r}")
    if not secret or len(secret) < 3:
        secret = random.choice(WORDS)
    s["secret"] = norm(secret)
    s["world"]["beats"] = [redact(b, s["secret"]) for b in s["world"]["beats"]]


# ---------------- Scenes ----------------
def scene_prompt(s: dict, choice_desc: Optional[str], n: int) -> str:
    players = [{"name": pname(s, i), "interests": s["players"][i]["interests"],
                "class": s["avatars"][i]["class"], "appearance": s["avatars"][i]["appearance"]} for i in range(2)]
    if n == 1:
        stage = "OPENING scene: the heroes arrive and see the first sign of the problem. They start toward the quest goal."
    elif n < MAX_SCENES:
        stage = ("The heroes take ONE clear step closer to the quest goal, shaped by the choice they just made. "
                 "Something new and simple appears (a place, a helper, an obstacle).")
    else:
        stage = ("FINAL scene: the heroes reach the vault that holds the goal. End story_text with a short riddle whose "
                 "one-word answer is the SECRET WORD. choices MUST be an empty list.")
    choice_line = f'The heroes just chose to: "{choice_desc}"' if choice_desc else "No choice yet."
    prev = s["current"].story_text if s.get("current") else "(none, this is the opening)"
    beat = (s["world"].get("beats") or ARC)[min(n, MAX_SCENES) - 1]
    used = [h["title"] for h in s["history"]] + s["inventory"]
    return (
        "You write a SIMPLE, warm, easy-to-follow 8-bit adventure for 2 friends. Return ONLY raw JSON, no markdown.\n"
        f"WORLD: {s['world']['title']}. {s['world']['premise']}\nQUEST (never changes): {s['world'].get('quest', '')}\n"
        f"HEROES: {json.dumps(players)}\n"
        f"STORY SO FAR (one line per scene): {json.dumps([h['title'] + ': ' + h['happened'] for h in s['history']])}\n"
        f"PREVIOUS SCENE (continue straight from its last moment, never retell it): {prev}\n"
        f"PLOT BEAT FOR THIS SCENE (this scene MUST show it): {beat}\n"
        f"ALREADY USED, never reuse these places, titles, objects or phrases: {json.dumps(used)}\n"
        f"INVENTORY: {json.dumps(s['inventory'])}\n{choice_line}\n"
        f"Write scene {n} of {MAX_SCENES}. {stage}\n"
        "Style rules: the first sentence shows the result of the heroes' last choice; every scene is a NEW place and moves the plot forward, never a repeat; simple everyday words; story_text is exactly 3 short sentences (max 14 words each) with clear cause and effect; "
        "show both heroes' interests as simple objects or ideas (a star map, a seed, a telescope, a vine), never jargon; "
        "character_dialogue has exactly 2 lines, one per hero, max 10 words each, friendly; "
        f"clue_found is a small object hinting at the SECRET WORD \"{s['secret']}\" through meaning or theme, and you must NEVER write that word; "
        "non-final scenes need exactly 2 choices: two different plain actions the heroes could take next, each starting with a verb, "
        "max 9 words, never mentioning 'player', 'ability' or names. Choice 1 suits hero 1's interest, choice 2 suits hero 2's. "
        "Both choices must move toward the quest.\n"
        'JSON keys exactly: {"scene_title":"","story_text":"","background_prompt":"16-bit pixel art scene description",'
        '"background_fallback_tag":"short_snake_case_tag","character_dialogue":[{"speaker":"name","text":""},{"speaker":"name","text":""}],'
        '"choices":[{"description":""},{"description":""}],"clue_found":{"name":"","description":""}}'
    )

def coerce_scene(raw: dict, n: int, s: dict) -> Scene:
    sec = s["secret"]
    title = redact(str(raw.get("scene_title") or f"Scene {n}"), sec)
    story = redact(str(raw.get("story_text") or raw.get("story") or raw.get("narrative") or ""), sec).strip()
    if len(story) < 10:
        raise ValueError("empty story_text")
    dl = raw.get("character_dialogue") or []
    if isinstance(dl, dict):
        dl = [{"speaker": k, "text": v} for k, v in dl.items()]
    dialogue = []
    for i, d in enumerate(dl[:2]):
        if isinstance(d, str):
            d = {"speaker": pname(s, i % 2), "text": d}
        dialogue.append({"speaker": str(d.get("speaker") or pname(s, i % 2)), "text": redact(str(d.get("text") or "..."), sec)})
    choices = []
    if n < MAX_SCENES:
        for i, c in enumerate((raw.get("choices") or [])[:2]):
            desc = c if isinstance(c, str) else (c.get("description") or c.get("text") or "")
            choices.append({"choice_id": f"c{n}_{i}", "description": redact(str(desc), sec),
                            "uses_ability_of": "player_1" if i == 0 else "player_2"})
        while len(choices) < 2:
            i = len(choices)
            choices.append({"choice_id": f"c{n}_{i}", "description": ("Look around carefully", "Move ahead quietly")[i],
                            "uses_ability_of": "player_1" if i == 0 else "player_2"})
    cl = raw.get("clue_found")
    clue = None
    if isinstance(cl, str) and cl:
        cl = {"name": f"Clue {n}", "description": cl}
    if isinstance(cl, dict):
        clue = {"clue_id": f"clue_{n}", "name": redact(str(cl.get("name") or f"Clue {n}"), sec),
                "description": redact(str(cl.get("description") or ""), sec)}
    return Scene(scene_number=n, scene_title=title, story_text=story,
                 background_prompt=str(raw.get("background_prompt") or ""),
                 background_fallback_tag=norm(str(raw.get("background_fallback_tag") or title)).lower() or "bg_generic",
                 character_dialogue=dialogue, choices=choices, clue_found=clue)

FB = [
    ("The Glowing Door", "{p0} and {p1} step through the door. A hall full of {a} and {b} signs opens up. Something far ahead is humming."),
    ("The Whispering Bridge", "A rope bridge sways over a deep gap. A tiny lantern on the far side points the way. The two friends cross it slowly."),
    ("The Helper's Hut", "A small hut sits by the path, and a kind keeper waves them in. She needs help sorting her lost tools. She gives a clue in thanks."),
    ("The Locked Gate", "A stone gate blocks the road, with a pattern carved on it. The friends match the pattern together. The gate slides open."),
    ("The Mirror Garden", "Bright mirrors line a quiet garden. One mirror shows a hidden stair. The friends follow the stair down."),
    ("The Last Corridor", "The lights start to flicker as time runs short. A long hall ends at a huge round door. The vault is right there."),
    ("The Vault", "{p0} and {p1} stand before the vault. Its door waits for one word."),
]
CH = [("Follow the glowing {a} trail", "Ask the {b} for a hint"), ("Test the {a} rope first", "Jump across with a {b} plan"),
      ("Help the keeper with {a}", "Trade a {b} for her clue"), ("Study the {a} pattern", "Push the gate with {b} tricks"),
      ("Step into the {a} mirror", "Follow the {b} reflection"), ("Run for the round door", "Light the way with {a}")]

def fallback_scene(s: dict, n: int) -> Scene:
    """Used only when the AI fails. Different per scene, so it never looks like a loop."""
    a, b, p0, p1 = pint(s, 0), pint(s, 1), pname(s, 0), pname(s, 1)
    final = n >= MAX_SCENES
    title, story = FB[min(n, MAX_SCENES) - 1]
    story = story.format(a=a, b=b, p0=p0, p1=p1)
    if final:
        sec = s["secret"]
        story += f" The riddle: a word of {len(sec)} letters, starting with {sec[0]}."
    ch = [c.format(a=a, b=b) for c in CH[min(n, len(CH)) - 1]]
    return Scene(scene_number=n, scene_title=title, story_text=story,
                 character_dialogue=[Dialogue(speaker=p0, text=f"My {a} skills can help here."),
                                     Dialogue(speaker=p1, text=f"And my {b} skills too!")],
                 choices=[] if final else [Choice(choice_id=f"c{n}_0", description=ch[0], uses_ability_of="player_1"),
                                           Choice(choice_id=f"c{n}_1", description=ch[1], uses_ability_of="player_2")],
                 clue_found=None if final else Clue(clue_id=f"clue_{n}", name=f"Shard {n}", description="A faint hint glows nearby."))

def too_similar(sc: Scene, s: dict) -> bool:
    w = lambda t: set(re.findall(r"[a-z]{4,}", t.lower()))
    new = w(sc.story_text)
    if sc.scene_title.strip().lower() in {h["title"].strip().lower() for h in s["history"]}:
        return True
    return any((o := w(h["happened"])) and len(new & o) / max(1, len(new | o)) > 0.45 for h in s["history"])

async def generate_scene(s: dict, choice_desc: Optional[str], n: int) -> Scene:
    if MOCK:
        await asyncio.sleep(0.3)
        sc = fallback_scene(s, n)
        sc.scene_title = f"Mock Scene {n}"
        return sc
    for attempt in range(2):
        try:
            sc = coerce_scene(await call_gemini([{"text": scene_prompt(s, choice_desc, n)}]), n, s)
            if too_similar(sc, s):
                raise ValueError("scene repeats an earlier one")
            return sc
        except Exception as e:
            log_err(s, f"SCENE {n} attempt {attempt + 1}: {e!r}")
    return fallback_scene(s, n)


# ---------------- Session logic ----------------
SESSIONS: dict[str, dict] = {}

def preload_next(s: dict, scene: Scene) -> None:
    for t in s["cache"].values():
        t.cancel()
    s["cache"] = {}
    if scene.scene_number < MAX_SCENES:
        for i, c in enumerate(scene.choices):
            if i == 0 or PRELOAD_BOTH:
                s["cache"][c.choice_id] = asyncio.create_task(generate_scene(s, c.description, scene.scene_number + 1))

def apply_scene(s: dict, scene: Scene, choice_desc: Optional[str]) -> None:
    if scene.clue_found:
        s["inventory"].append(scene.clue_found.name)
    scene.game_state_update = StateUpdate(health=s["health"], inventory=list(s["inventory"]))
    s["history"].append({"scene": scene.scene_number, "title": scene.scene_title,
                         "happened": scene.story_text[:400], "clue": scene.clue_found.name if scene.clue_found else None,
                         "choice_taken": choice_desc})
    s["scene_number"], s["current"] = scene.scene_number, scene
    preload_next(s, scene)

async def init_session(sid: str) -> None:
    s = SESSIONS[sid]
    try:
        await setup_world(s)
        apply_scene(s, await generate_scene(s, None, 1), None)
    except Exception as e:
        log_err(s, f"INIT: {e!r}")
        s.setdefault("secret", random.choice(WORDS))
        s["avatars"] = s["avatars"] or [default_avatar(s, 0), default_avatar(s, 1)]
        apply_scene(s, fallback_scene(s, 1), None)


# ---------------- Endpoints ----------------
def get_session(sid: str) -> dict:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "No such session")
    return s

@app.post("/api/start")
async def start(req: StartReq):
    if len(req.players) != 2:
        raise HTTPException(400, "Exactly 2 players required")
    sid = uuid.uuid4().hex[:8]
    SESSIONS[sid] = {"players": [{"name": p.name, "interests": p.interests, "look": p.look} for p in req.players],
                     "player_models": req.players, "avatars": [], "world": {}, "secret": "", "history": [],
                     "inventory": [], "health": 100, "scene_number": 0, "current": None, "cache": {},
                     "attempts": 0, "errors": []}
    asyncio.create_task(init_session(sid))
    return {"session_id": sid}

@app.get("/api/status/{sid}")
async def status(sid: str):
    s = get_session(sid)
    return {"avatars_ready": bool(s["avatars"]), "scene_ready": s["current"] is not None,
            "avatars": s["avatars"], "world": {k: v for k, v in s["world"].items() if k != "beats"}, "world_title": s["world"].get("title", ""), "debug": s["errors"][-3:]}

@app.get("/api/scene/{sid}")
async def get_scene(sid: str):
    s = get_session(sid)
    if not s["current"]:
        raise HTTPException(404, "Scene not ready")
    return s["current"]

@app.post("/api/choice")
async def choose(req: ChoiceReq):
    s = get_session(req.session_id)
    cur = s["current"]
    if not cur:
        raise HTTPException(404, "No active scene")
    chosen = next((c for c in cur.choices if c.choice_id == req.choice_id), None)
    if not chosen:
        raise HTTPException(400, "Invalid choice")
    n = s["scene_number"] + 1
    task = s["cache"].get(chosen.choice_id) or asyncio.create_task(generate_scene(s, chosen.description, n))
    try:
        scene = await asyncio.wait_for(asyncio.shield(task), 60)
    except Exception as e:
        log_err(s, f"CHOICE wait: {e!r}")
        scene = fallback_scene(s, n)
    apply_scene(s, scene, chosen.description)
    return scene

@app.post("/api/validate-final-treasure")
async def validate(req: TreasureReq):
    s = get_session(req.session_id)
    if s["scene_number"] < MAX_SCENES:
        raise HTTPException(400, "Not at final scene")
    if s["attempts"] >= MAX_ATTEMPTS:
        return {"victory": False, "game_over": True, "attempts_left": 0}
    s["attempts"] += 1
    ok = norm(req.answer) == s["secret"]
    left = MAX_ATTEMPTS - s["attempts"]
    return {"victory": ok, "game_over": (not ok and left == 0), "attempts_left": left}


# Serve the frontend. Keep LAST, after all API routes.
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"), html=True), name="static")
