#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop-in dynamic voice service for Qwen3-TTS/vLLM-Omni."""
from __future__ import annotations
import asyncio, json, os, subprocess, threading, time
from typing import Any
import pymysql, requests
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.getenv("TTS_CONFIG_FILE", os.path.join(BASE_DIR, "tts_config.json"))
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)
MYSQL = CONFIG.get("mysql", {})
VLLM = CONFIG.get("vllm_omni", {})
TABLE = MYSQL.get("table", "tts_voice_prompt")
MODEL = CONFIG.get("model", {}).get("name", VLLM.get("model", ""))
VLLM_BASE = VLLM.get("base_url", "http://127.0.0.1:8091").rstrip("/")
TASK_TYPE = VLLM.get("task_type", "Base")
TIMEOUT = float(VLLM.get("timeout", 120))
PRECOMPUTE = VLLM.get("precompute_script", os.path.join(os.path.dirname(BASE_DIR), "vllm-omni", "examples", "online_serving", "text_to_speech", "qwen3_tts", "precompute_custom_voice.py"))
app = FastAPI(title="Dynamic Qwen3-TTS Voice Registry")
_registry_lock = threading.RLock()
_registered: dict[str, tuple[str, str]] = {}


def db():
    return pymysql.connect(host=MYSQL.get("host", "localhost"), port=int(MYSQL.get("port", 3306)), database=MYSQL.get("database", "xiaozhi"), user=MYSQL.get("user", "root"), password=MYSQL.get("password", "123456"), charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor, connect_timeout=5)


def ensure_columns():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SHOW COLUMNS FROM `{TABLE}` LIKE 'speaker_embedding'")
            if not cur.fetchone():
                cur.execute(f"ALTER TABLE `{TABLE}` ADD COLUMN speaker_embedding JSON NULL, ADD COLUMN embedding_dim INT NULL, ADD COLUMN embedding_model VARCHAR(1024) NULL")
        conn.commit()


def precompute_embedding(ref_audio: str, mode: str) -> list[float]:
    if mode != "xvec":
        raise ValueError("dynamic embedding registration supports mode=xvec only")
    out_dir = os.path.join(BASE_DIR, ".voice_precompute")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"_tmp_{threading.get_ident()}_{time.time_ns()}"
    subprocess.run(["python", PRECOMPUTE, "--model", MODEL, "--voice-name", stem, "--ref-audio", ref_audio, "--mode", "xvec", "--output-dir", out_dir], check=True)
    from safetensors.torch import load_file
    path = os.path.join(out_dir, f"{stem}.safetensors")
    try:
        return load_file(path, device="cpu")["speaker_embedding"].reshape(-1).float().tolist()
    finally:
        try: os.remove(path)
        except OSError: pass


def save_voice(spk_id: str, voice_name: str, embedding: list[float], name: str, ref_text: str):
    ensure_columns()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""INSERT INTO `{TABLE}` (spk_id,name,voice_name,model_name,ref_text,enabled,speaker_embedding,embedding_dim,embedding_model,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,1,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE name=VALUES(name),voice_name=VALUES(voice_name),model_name=VALUES(model_name),ref_text=VALUES(ref_text),enabled=1,speaker_embedding=VALUES(speaker_embedding),embedding_dim=VALUES(embedding_dim),embedding_model=VALUES(embedding_model),updated_at=VALUES(updated_at)""", (str(spk_id), name, voice_name, MODEL, ref_text, json.dumps(embedding, separators=(",", ":")), len(embedding), MODEL, now, now))
        conn.commit()


def load_voice(spk_id: str) -> dict[str, Any]:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM `{TABLE}` WHERE spk_id=%s AND enabled=1 LIMIT 1", (str(spk_id),))
            row = cur.fetchone()
    if not row: raise LookupError(f"voice not found: {spk_id}")
    embedding = row.get("speaker_embedding")
    if isinstance(embedding, str): embedding = json.loads(embedding)
    if not isinstance(embedding, list) or not embedding: raise ValueError(f"voice has no speaker_embedding: {spk_id}")
    row["speaker_embedding"] = [float(x) for x in embedding]
    return row


def register_voice(spk_id: str, force: bool = False) -> str:
    row = load_voice(spk_id)
    voice = str(row["voice_name"])
    updated = str(row.get("updated_at", ""))
    key = str(spk_id)
    with _registry_lock:
        if not force and _registered.get(key) == (voice, updated): return voice
        response = requests.post(f"{VLLM_BASE}/v1/audio/voices", files={"speaker_embedding": (None, json.dumps(row["speaker_embedding"])), "consent": (None, f"db-{key}"), "name": (None, voice)}, timeout=TIMEOUT)
        if not response.ok: raise RuntimeError(f"vLLM voice registration failed: {response.status_code} {response.text[:500]}")
        _registered[key] = (voice, updated)
    return voice


def ensure_registered(spk_id: str) -> str:
    try: return register_voice(spk_id)
    except RuntimeError:
        _registered.pop(str(spk_id), None)
        return register_voice(spk_id, True)


class TTSRequest(BaseModel):
    spk_id: str
    text: str
    language: str = "Chinese"
    max_new_tokens: int = 2048


@app.get("/")
def index():
    return {"service": "dynamic qwen tts voice registry", "voice_prompt_endpoint": "/voice-prompt", "tts_wav_endpoint": "/tts-wav", "vllm_base_url": VLLM_BASE}


@app.post("/voice-prompt")
async def create_voice_prompt(spk_id: str = Form(...), name: str = Form(""), voice_name: str = Form(""), ref_audio: UploadFile = File(...), ref_text: str = Form(...), mode: str = Form("xvec")):
    temp = os.path.join(BASE_DIR, f".upload_{spk_id}_{time.time_ns()}.wav")
    try:
        with open(temp, "wb") as f: f.write(await ref_audio.read())
        selected = voice_name.strip() or str(spk_id)
        embedding = await asyncio.to_thread(precompute_embedding, temp, mode)
        await asyncio.to_thread(save_voice, spk_id, selected, embedding, name, ref_text)
        await asyncio.to_thread(register_voice, spk_id, True)
        return {"success": True, "spk_id": spk_id, "voice_name": selected, "embedding_dim": len(embedding)}
    except Exception as exc: raise HTTPException(400, repr(exc)) from exc
    finally:
        try: os.remove(temp)
        except OSError: pass


@app.post("/tts-wav")
async def create_tts_wav(request: TTSRequest):
    try:
        voice = await asyncio.to_thread(ensure_registered, request.spk_id)
        response = await asyncio.to_thread(requests.post, f"{VLLM_BASE}/v1/audio/speech", json={"input": request.text, "voice": voice, "task_type": TASK_TYPE, "response_format": "wav", "language": request.language, "max_new_tokens": request.max_new_tokens}, timeout=TIMEOUT)
        if not response.ok: raise RuntimeError(f"vLLM-Omni error {response.status_code}: {response.text[:500]}")
        return Response(content=response.content, media_type="audio/wav", headers={"Content-Disposition": f'attachment; filename="tts_{request.spk_id}.wav"'})
    except Exception as exc: raise HTTPException(400, repr(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("TTS_REGISTER_PORT", "8095")))



