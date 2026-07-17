#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Database-backed dynamic voice registry for Qwen3-TTS and vLLM-Omni.

This module is intentionally independent from ``register_vllm_voice.py``.
It keeps ``spk_id`` as the public voice identifier, stores the pre-computed
speaker embedding in MySQL, and registers it with a running vLLM-Omni server
on demand through ``POST /v1/audio/voices``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import pymysql
import requests


class DynamicVoiceRegistry:
    """Resolve, persist, and lazily register voice embeddings."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.mysql = config.get("mysql", {})
        self.vllm = config.get("vllm_omni", {})
        self.table = self.mysql.get("table", "tts_voice_prompt")
        self.vllm_base_url = self.vllm.get("base_url", "http://127.0.0.1:8091").rstrip("/")
        self.vllm_timeout = float(self.vllm.get("timeout", 120))
        self.model_name = self.vllm.get("model", "")
        self.task_type = self.vllm.get("task_type", "Base")
        self._registered: dict[str, tuple[str, int]] = {}
        self._lock = threading.RLock()

    def _connect(self):
        return pymysql.connect(
            host=self.mysql.get("host", "localhost"),
            port=int(self.mysql.get("port", 3306)),
            database=self.mysql.get("database", "xiaozhi"),
            user=self.mysql.get("user", "root"),
            password=self.mysql.get("password", "123456"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
        )

    def ensure_columns(self) -> None:
        """Add embedding columns to the existing table when necessary."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SHOW COLUMNS FROM `{self.table}` LIKE 'speaker_embedding'")
                if not cur.fetchone():
                    cur.execute(
                        f"ALTER TABLE `{self.table}` "
                        "ADD COLUMN speaker_embedding JSON NULL, "
                        "ADD COLUMN embedding_dim INT NULL, "
                        "ADD COLUMN embedding_model VARCHAR(1024) NULL"
                    )
            conn.commit()

    def save_embedding(
        self,
        spk_id: str,
        voice_name: str,
        embedding: list[float],
        *,
        name: str = "",
        ref_text: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Persist one voice embedding; ``spk_id`` remains the public ID."""
        if not embedding:
            raise ValueError("speaker embedding cannot be empty")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO `{self.table}`
                    (spk_id, name, voice_name, model_name, ref_text, enabled,
                     speaker_embedding, embedding_dim, embedding_model,
                     created_at, updated_at)
                    VALUES (%s,%s,%s,%s,%s,1,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                    name=VALUES(name), voice_name=VALUES(voice_name),
                    model_name=VALUES(model_name), ref_text=VALUES(ref_text),
                    enabled=1, speaker_embedding=VALUES(speaker_embedding),
                    embedding_dim=VALUES(embedding_dim),
                    embedding_model=VALUES(embedding_model),
                    updated_at=VALUES(updated_at)""",
                    (
                        str(spk_id),
                        name,
                        voice_name,
                        model_name or self.model_name,
                        ref_text,
                        json.dumps(embedding, separators=(",", ":")),
                        len(embedding),
                        model_name or self.model_name,
                        now,
                        now,
                    ),
                )
            conn.commit()

    def load_voice(self, spk_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM `{self.table}` WHERE spk_id=%s AND enabled=1 LIMIT 1",
                    (str(spk_id),),
                )
                row = cur.fetchone()
        if not row:
            raise LookupError(f"voice not found or disabled: spk_id={spk_id}")
        embedding = row.get("speaker_embedding")
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"voice has no speaker embedding: spk_id={spk_id}")
        row["speaker_embedding"] = [float(value) for value in embedding]
        return row

    def register(self, spk_id: str, *, force: bool = False) -> str:
        """Register a DB voice in vLLM-Omni and return its voice name."""
        voice = self.load_voice(spk_id)
        voice_name = str(voice["voice_name"])
        signature = (voice_name, int(voice.get("updated_at").timestamp()) if hasattr(voice.get("updated_at"), "timestamp") else 0)
        with self._lock:
            if not force and self._registered.get(str(spk_id)) == signature:
                return voice_name
            response = requests.post(
                f"{self.vllm_base_url}/v1/audio/voices",
                files={
                    "speaker_embedding": (None, json.dumps(voice["speaker_embedding"])),
                    "consent": (None, f"db-{spk_id}"),
                    "name": (None, voice_name),
                },
                timeout=self.vllm_timeout,
            )
            if not response.ok:
                raise RuntimeError(f"vLLM-Omni voice registration failed: {response.status_code} {response.text[:500]}")
            self._registered[str(spk_id)] = signature
        return voice_name

    def ensure_registered(self, spk_id: str) -> str:
        """Register on first use; retry automatically after server restarts."""
        try:
            return self.register(spk_id)
        except RuntimeError:
            self._registered.pop(str(spk_id), None)
            return self.register(spk_id, force=True)


def load_registry(config_file: str | None = None) -> DynamicVoiceRegistry:
    path = config_file or os.getenv("TTS_CONFIG_FILE", "tts_config.json")
    with open(path, "r", encoding="utf-8") as file:
        return DynamicVoiceRegistry(json.load(file))

