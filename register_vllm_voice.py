#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import json
import os
import subprocess
from datetime import datetime
from typing import Optional

import pymysql
import requests
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.getenv("TTS_CONFIG_FILE", os.path.join(BASE_DIR, "tts_config.json"))
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)
MODEL_CONFIG = CONFIG.get("model", {})
VLLM = CONFIG.get("vllm_omni", {})
MYSQL = CONFIG.get("mysql", {})
MODEL = MODEL_CONFIG.get("name", VLLM.get("model", ""))
VOICE_DIR = VLLM.get("custom_voice_dir", "/root/autodl-tmp/custom_voices")
PRECOMPUTE = VLLM.get("precompute_script", "examples/online_serving/text_to_speech/qwen3_tts/precompute_custom_voice.py")
VLLM_BASE = VLLM.get("base_url", "http://127.0.0.1:8091").rstrip("/")
TASK_TYPE = VLLM.get("task_type", "Base")
DEFAULT_LANGUAGE = MODEL_CONFIG.get("default_language", "Chinese")
TABLE = MYSQL.get("table", "tts_voice_prompt")

app = FastAPI(title="Qwen TTS Voice Management API")
voice_lock = asyncio.Lock()

class TTSWavRequest(BaseModel):
    spk_id: str
    text: str
    language: str = DEFAULT_LANGUAGE

def db():
    return pymysql.connect(host=MYSQL.get("host", "localhost"), port=int(MYSQL.get("port", 3306)), database=MYSQL.get("database", "xiaozhi"), user=MYSQL.get("user", "root"), password=MYSQL.get("password", "123456"), charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)

def ensure_table():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""CREATE TABLE IF NOT EXISTS `{TABLE}` (spk_id VARCHAR(64) PRIMARY KEY, name VARCHAR(255), voice_name VARCHAR(255) NOT NULL, model_name VARCHAR(1024), ref_audio_path VARCHAR(1024), ref_text TEXT, enabled TINYINT(1) NOT NULL DEFAULT 1, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute(f"SHOW COLUMNS FROM `{TABLE}` LIKE 'voice_name'")
            if not cur.fetchone():
                cur.execute(f"ALTER TABLE `{TABLE}` ADD COLUMN voice_name VARCHAR(255) NOT NULL DEFAULT '' AFTER name")
        conn.commit()

def resolve_voice(spk_id: str) -> str:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT voice_name FROM `{TABLE}` WHERE spk_id=%s AND enabled=1 LIMIT 1", (spk_id,))
            row = cur.fetchone()
    if not row or not row.get("voice_name"):
        raise ValueError(f"voice mapping not found: spk_id={spk_id}")
    return str(row["voice_name"])

def register_voice(spk_id, voice_name, name, ref_audio, ref_text, ref_audio_name, mode):
    os.makedirs(VOICE_DIR, exist_ok=True)
    subprocess.run(["python", PRECOMPUTE, "--model", MODEL, "--voice-name", voice_name, "--ref-audio", ref_audio, "--ref-text", ref_text, "--mode", mode, "--output-dir", VOICE_DIR], check=True)
    ensure_table()
    now = datetime.now()
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""INSERT INTO `{TABLE}` (spk_id,name,voice_name,model_name,ref_audio_path,ref_text,enabled,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,1,%s,%s) ON DUPLICATE KEY UPDATE name=VALUES(name),voice_name=VALUES(voice_name),model_name=VALUES(model_name),ref_audio_path=VALUES(ref_audio_path),ref_text=VALUES(ref_text),enabled=1,updated_at=VALUES(updated_at)""", (spk_id,name,voice_name,MODEL,ref_audio_name,ref_text,now,now))
        conn.commit()

def synthesize_wav(request: TTSWavRequest) -> bytes:
    if not request.spk_id.strip() or not request.text.strip():
        raise ValueError("spk_id and text are required")
    voice = resolve_voice(request.spk_id)
    payload = {"input": request.text, "voice": voice, "task_type": TASK_TYPE, "response_format": "wav", "language": request.language}
    response = requests.post(f"{VLLM_BASE}/v1/audio/speech", json=payload, timeout=int(VLLM.get("timeout", 120)))
    if not response.ok:
        raise ValueError(f"vLLM-Omni error {response.status_code}: {response.text[:500]}")
    return response.content

@app.get("/")
def index():
    return {"service": "qwen tts voice management", "voice_prompt_endpoint": "/voice-prompt", "tts_wav_endpoint": "/tts-wav", "vllm_base_url": VLLM_BASE, "custom_voice_dir": VOICE_DIR}

@app.post("/voice-prompt")
async def create_voice_prompt(spk_id: str = Form(...), name: str = Form(""), voice_name: str = Form(""), ref_audio: UploadFile = File(...), ref_text: str = Form(...), mode: str = Form("icl")):
    if mode not in {"icl", "xvec"}:
        raise HTTPException(status_code=400, detail="mode must be icl or xvec")
    temp_path = os.path.join(VOICE_DIR, f".upload_{spk_id}_{ref_audio.filename or 'audio.wav'}")
    try:
        os.makedirs(VOICE_DIR, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(await ref_audio.read())
        async with voice_lock:
            await asyncio.to_thread(register_voice, spk_id, voice_name, name, temp_path, ref_text, ref_audio.filename or "audio.wav", mode)
        return {"success": True, "message": "voice prompt saved", "spk_id": spk_id, "name": name, "voice_name": voice_name, "model_name": MODEL}
    except Exception as exc:
        return {"success": False, "message": repr(exc), "spk_id": spk_id, "voice_name": voice_name}
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

@app.post("/tts-wav")
async def create_tts_wav(request: TTSWavRequest):
    try:
        wav = await asyncio.to_thread(synthesize_wav, request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=repr(exc)) from exc
    return Response(content=wav, media_type="audio/wav", headers={"Content-Disposition": f'attachment; filename="tts_{request.spk_id}.wav"'})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="use command-line registration instead of starting the API")
    parser.add_argument("--spk-id", required=False)
    parser.add_argument("--voice-name", required=False)
    parser.add_argument("--name", default="")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--mode", default="icl", choices=["icl", "xvec"])
    args = parser.parse_args()

    if not args.cli:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("TTS_REGISTER_PORT", "8095")))
        return

    required = {"--spk-id": args.spk_id, "--voice-name": args.voice_name, "--ref-audio": args.ref_audio, "--ref-text": args.ref_text}
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("--cli requires: " + ", ".join(missing))
    register_voice(args.spk_id, args.voice_name, args.name, args.ref_audio, args.ref_text, args.ref_audio, args.mode)

if __name__ == "__main__":
    main()
