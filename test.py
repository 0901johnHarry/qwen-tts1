#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import base64
import json
import os
import time
import uuid
from typing import AsyncGenerator
from urllib.parse import urljoin, urlparse, urlunparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


CONFIG_FILE = os.getenv("TTS_CONFIG_FILE", "tts_config.json")


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


APP_CONFIG = load_config()
MODEL_CONFIG = APP_CONFIG.get("model", {})
VLLM_OMNI_CONFIG = APP_CONFIG.get("vllm_omni", {})


def as_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default

HOST = "0.0.0.0"
PORT = int(MODEL_CONFIG.get("port", 8093))

DEFAULT_LANGUAGE = MODEL_CONFIG.get("default_language", "Chinese")
SAMPLE_RATE = int(MODEL_CONFIG.get("sample_rate", 24000))
CHUNK_SIZE = int(MODEL_CONFIG.get("chunk_size", 4))
PERF_LOG_CHUNKS = bool(MODEL_CONFIG.get("perf_log_chunks", True))

VLLM_OMNI_BASE_URL = VLLM_OMNI_CONFIG.get("base_url", "").rstrip("/")
VLLM_OMNI_STREAM_WS_URL = VLLM_OMNI_CONFIG.get("streaming_ws_url", "")
VLLM_OMNI_MODEL = VLLM_OMNI_CONFIG.get("model", "")
VLLM_OMNI_TASK_TYPE = VLLM_OMNI_CONFIG.get("task_type", "Base")
VLLM_OMNI_RESPONSE_FORMAT = VLLM_OMNI_CONFIG.get("response_format", "pcm")
VLLM_OMNI_STREAM_AUDIO = as_bool(VLLM_OMNI_CONFIG.get("stream_audio"), True)
VLLM_OMNI_WORD_TIMESTAMPS = as_bool(VLLM_OMNI_CONFIG.get("word_timestamps"), False)
VLLM_OMNI_TIMEOUT = float(VLLM_OMNI_CONFIG.get("timeout", 120))
VLLM_OMNI_SPEAKER_MAP = VLLM_OMNI_CONFIG.get("speaker_map", {})

app = FastAPI()



def now() -> float:
    return time.perf_counter()






def build_vllm_omni_ws_url() -> str:
    if VLLM_OMNI_STREAM_WS_URL:
        return VLLM_OMNI_STREAM_WS_URL
    if not VLLM_OMNI_BASE_URL:
        return ""

    parsed = urlparse(VLLM_OMNI_BASE_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    base_path = parsed.path if parsed.netloc else ""
    path = urljoin(base_path.rstrip("/") + "/", "v1/audio/speech/stream")
    return urlunparse((scheme, netloc, path, "", "", ""))


def resolve_vllm_voice_name(spk_id) -> str:
    key = str(spk_id)
    mapped = VLLM_OMNI_SPEAKER_MAP.get(key)
    if mapped is not None:
        return str(mapped)
    return key


def build_vllm_omni_session_config(spk_id, language):
    payload = {
        "voice": resolve_vllm_voice_name(spk_id),
        "task_type": VLLM_OMNI_TASK_TYPE,
        "response_format": VLLM_OMNI_RESPONSE_FORMAT,
        "stream_audio": VLLM_OMNI_STREAM_AUDIO,
    }
    if VLLM_OMNI_MODEL:
        payload["model"] = VLLM_OMNI_MODEL
    if language:
        payload["language"] = language
    if VLLM_OMNI_WORD_TIMESTAMPS:
        payload["word_timestamps"] = True
    return payload


def extract_audio_bytes_from_json(payload):
    for key in ("audio", "audio_b64", "pcm", "data"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            try:
                return base64.b64decode(value)
            except Exception:
                continue
    return None


async def stream_vllm_omni_pcm(text, spk_id, language) -> AsyncGenerator[bytes, None]:
    ws_url = build_vllm_omni_ws_url()
    if not ws_url:
        raise RuntimeError(
            "vLLM-Omni is enabled but no streaming websocket URL is configured. "
            "Set vllm_omni.streaming_ws_url or vllm_omni.base_url."
        )

    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "vLLM-Omni streaming requires the 'websockets' package. "
            "Install uvicorn[standard] or websockets."
        ) from exc

    config = build_vllm_omni_session_config(spk_id, language)
    print(
        f"[vllm-omni] connect={ws_url}, voice={config.get('voice')}, "
        f"task_type={config.get('task_type')}, response_format={config.get('response_format')}"
    )

    async with websockets.connect(ws_url, open_timeout=VLLM_OMNI_TIMEOUT) as upstream:
        await upstream.send(json.dumps({"type": "session.config", **config}, ensure_ascii=False))
        await upstream.send(json.dumps({"type": "input.text", "text": text}, ensure_ascii=False))
        await upstream.send(json.dumps({"type": "input.done"}, ensure_ascii=False))

        while True:
            try:
                message = await asyncio.wait_for(upstream.recv(), timeout=VLLM_OMNI_TIMEOUT)
            except websockets.ConnectionClosedOK:
                break

            if isinstance(message, bytes):
                if message:
                    yield message
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue

            msg_type = payload.get("type")
            if msg_type == "error" or payload.get("error") is True:
                raise RuntimeError(payload.get("message") or payload.get("error") or payload)
            if msg_type == "audio.done":
                continue
            if msg_type in {"session.done", "done", "end", "completed"} or payload.get("done") is True:
                break

            audio_bytes = extract_audio_bytes_from_json(payload)
            if audio_bytes:
                yield audio_bytes















@app.on_event("startup")
def startup_load_model():
    print("[startup] backend: vllm-omni")
    print("[startup] vllm_omni_base_url:", VLLM_OMNI_BASE_URL)
    print("[startup] vllm_omni_stream_ws_url:", build_vllm_omni_ws_url())
    print("[startup] vllm_omni_model:", VLLM_OMNI_MODEL)
    print("[startup] sample_rate:", SAMPLE_RATE)

@app.get("/")
def index():
    return {
        "service": "paddlespeech compatible tts",
        "backend": "vllm-omni",
        "model": VLLM_OMNI_MODEL,
        "voice_source": "vllm-omni",
        "samplerate": "/paddlespeech/tts/streaming/samplerate",
        "websocket": "/paddlespeech/tts/streaming",
        "sample_rate": SAMPLE_RATE,
        "chunk_size": CHUNK_SIZE,
        "vllm_omni_enabled": True,
        "vllm_omni_stream_ws_url": build_vllm_omni_ws_url(),
    }


@app.get("/paddlespeech/tts/streaming/samplerate")
def get_samplerate():
    return {"sample_rate": SAMPLE_RATE}


@app.websocket("/paddlespeech/tts/streaming")
@app.websocket("/paddlespeech/tts/streaming/")
async def paddlespeech_compatible_ws(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    started = False
    conn_t0 = now()
    current_generation_task = None
    current_cancel_event = None
    send_lock = asyncio.Lock()

    print(f"[ws] connected: {session_id}")

    async def send_json(payload):
        async with send_lock:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    def cleanup_done_generation():
        nonlocal current_generation_task, current_cancel_event
        if current_generation_task is None or not current_generation_task.done():
            return
        try:
            current_generation_task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ws] generation task error: {repr(e)}")
        current_generation_task = None
        current_cancel_event = None

    async def cancel_current_generation(reason: str):
        nonlocal current_generation_task, current_cancel_event
        if current_cancel_event is not None:
            current_cancel_event.set()

        if current_generation_task is not None and not current_generation_task.done():
            print(f"[ws] cancel generation session={session_id}, reason={reason}")
            try:
                await current_generation_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[ws] cancel generation error: {repr(e)}")

        current_generation_task = None
        current_cancel_event = None
        await send_json({
            "status": 0,
            "signal": "cancel",
            "session": session_id,
            "reason": reason,
        })

    async def run_generation(text, spk_id, language, chunk_size, cancel_event):
        try:
            total_t0 = now()
            chunk_count = 0
            total_samples = 0
            total_pcm_bytes = 0
            first_chunk_time = None
            prev_chunk_t = total_t0
            total_pcm_time = 0.0
            total_b64_time = 0.0
            total_ws_send_time = 0.0

            async def send_pcm_chunk(pcm_bytes, sr, timing, next_t0, next_t1):
                nonlocal chunk_count, total_samples, total_pcm_bytes
                nonlocal first_chunk_time, prev_chunk_t, total_b64_time, total_ws_send_time

                if first_chunk_time is None:
                    first_chunk_time = next_t1 - total_t0
                    print(
                        f"[perf] session={session_id} first_chunk={first_chunk_time:.3f}s, "
                        f"sr={sr}, timing={timing}"
                    )

                b64_t0 = now()
                audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                b64_t1 = now()
                total_b64_time += b64_t1 - b64_t0

                if cancel_event.is_set():
                    raise asyncio.CancelledError()

                send_t0 = now()
                await send_json({
                    "status": 1,
                    "audio": audio_b64,
                    "session": session_id,
                })
                send_t1 = now()
                total_ws_send_time += send_t1 - send_t0

                chunk_count += 1
                samples = int(len(pcm_bytes) // 2)
                total_samples += samples
                total_pcm_bytes += len(pcm_bytes)

                if PERF_LOG_CHUNKS:
                    audio_sec = samples / sr if sr else 0
                    print(
                        f"[chunk] session={session_id} idx={chunk_count} "
                        f"samples={samples} audio={audio_sec:.3f}s "
                        f"interval={next_t1 - prev_chunk_t:.3f}s "
                        f"next={next_t1 - next_t0:.3f}s "
                        f"b64={b64_t1 - b64_t0:.4f}s "
                        f"ws_send={send_t1 - send_t0:.4f}s "
                        f"timing={timing}"
                    )

                prev_chunk_t = send_t1

            async for pcm_bytes in stream_vllm_omni_pcm(text, spk_id, language):
                if cancel_event.is_set():
                    raise asyncio.CancelledError()
                t = now()
                await send_pcm_chunk(pcm_bytes, SAMPLE_RATE, {"backend": "vllm-omni"}, t, t)

            await send_json({"status": 2, "session": session_id})
            total_t1 = now()
            audio_seconds = total_samples / SAMPLE_RATE if SAMPLE_RATE else 0
            total_time = total_t1 - total_t0
            rtf = total_time / audio_seconds if audio_seconds > 0 else 0

            print(
                f"[summary] session={session_id} chars={len(text)} chunks={chunk_count} "
                f"audio={audio_seconds:.3f}s total={total_time:.3f}s rtf={rtf:.3f} "
                f"pcm_total={total_pcm_time:.4f}s b64_total={total_b64_time:.4f}s "
                f"ws_send_total={total_ws_send_time:.4f}s pcm_bytes={total_pcm_bytes}"
            )

        except asyncio.CancelledError:
            print(f"[ws] generation cancelled: session={session_id}")
            try:
                await send_json({
                    "status": 3,
                    "signal": "cancelled",
                    "session": session_id,
                })
            except Exception:
                pass
            raise
        except Exception as e:
            print("[ws] error:", repr(e))
            await send_json({
                "status": -1,
                "message": repr(e),
                "session": session_id,
            })

    try:
        while True:
            cleanup_done_generation()

            recv_wait_t0 = now()
            msg = await websocket.receive_text()
            recv_wait_t1 = now()

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await send_json({
                    "status": -1,
                    "message": "invalid json",
                    "session": session_id,
                })
                continue

            if data.get("task") == "tts" and data.get("signal") == "start":
                started = True
                send_t0 = now()
                await send_json({"status": 0, "session": session_id})
                send_t1 = now()
                print(
                    f"[ws] start session={session_id}, "
                    f"recv_wait={recv_wait_t1 - recv_wait_t0:.4f}s, "
                    f"send_ack={send_t1 - send_t0:.4f}s"
                )
                continue

            if data.get("task") == "tts" and data.get("signal") == "cancel":
                await cancel_current_generation("client_cancel")
                continue

            if data.get("task") == "tts" and data.get("signal") == "end":
                recv_session = data.get("session") or session_id
                if current_generation_task is not None and not current_generation_task.done():
                    await cancel_current_generation("client_end")

                await send_json({
                    "status": 0,
                    "signal": "end",
                    "session": recv_session,
                })
                print(f"[ws] end session={recv_session}, conn_total={now() - conn_t0:.3f}s")
                break

            request_parse_t0 = now()
            text = data.get("text", "").strip()

            if not text:
                await send_json({
                    "status": -1,
                    "message": "text is required",
                    "session": session_id,
                })
                continue

            if not started:
                await send_json({
                    "status": -1,
                    "message": "please send start signal first",
                    "session": session_id,
                })
                continue

            if current_generation_task is not None and not current_generation_task.done():
                await send_json({
                    "status": -1,
                    "message": "tts generation is busy",
                    "session": session_id,
                })
                continue

            spk_id = data.get("spk_id", 0)
            language = data.get("language", DEFAULT_LANGUAGE)
            chunk_size = int(data.get("chunk_size", CHUNK_SIZE))
            request_parse_t1 = now()

            print(
                f"[ws] tts request session={session_id}, spk_id={spk_id}, "
                f"language={language}, chars={len(text)}, chunk_size={chunk_size}, text={text}"
            )
            print(
                f"[perf] session={session_id} request_parse={request_parse_t1 - request_parse_t0:.4f}s, "
                f"recv_wait={recv_wait_t1 - recv_wait_t0:.4f}s"
            )

            queue_t0 = now()
            current_cancel_event = asyncio.Event()

            async def generation_runner():
                await run_generation(text, spk_id, language, chunk_size, current_cancel_event)

            current_generation_task = asyncio.create_task(generation_runner())

    except WebSocketDisconnect:
        print(f"[ws] disconnected: {session_id}")
    except Exception as e:
        print(f"[ws] fatal error: {repr(e)}")
        try:
            await send_json({
                "status": -1,
                "message": repr(e),
                "session": session_id,
            })
        except Exception:
            pass
    finally:
        if current_generation_task is not None and not current_generation_task.done():
            if current_cancel_event is not None:
                current_cancel_event.set()
        print(f"[ws] finished: {session_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
