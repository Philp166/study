"""
SaluteSpeech (Sber) — клиент для STT и TTS.
Авторизация по OAuth2 с кэшированием токена (30 мин).
"""

import os
import time
import uuid

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SaluteSpeech:
    AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    STT_URL = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    TTS_URL = "https://smartspeech.sber.ru/rest/v1/text:synthesize"

    def __init__(self):
        self.auth_key = os.getenv("SALUTE_SPEECH_AUTH", "")
        self.scope = os.getenv("SALUTE_SPEECH_SCOPE", "SALUTE_SPEECH_PERS")
        self._token: str | None = None
        self._token_expires: float = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.auth_key)

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires - 60:
            return self._token

        resp = requests.post(
            self.AUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self.auth_key}",
            },
            data={"scope": self.scope},
            verify=False,
        )
        resp.raise_for_status()

        data = resp.json()
        self._token = data["access_token"]
        self._token_expires = data["expires_at"] / 1000
        return self._token

    def recognize(self, audio_bytes: bytes, content_type: str = "audio/x-pcm;bit=16;rate=16000") -> str:
        """STT: сырые аудио-байты -> распознанный текст."""
        token = self._get_token()
        resp = requests.post(
            self.STT_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": content_type,
            },
            data=audio_bytes,
        )
        resp.raise_for_status()
        result = resp.json()
        texts = result.get("result", [])
        if not texts:
            return ""
        first = texts[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("normalized_text", first.get("text", ""))
        return str(first)

    def synthesize(self, text: str, voice: str = "Nec_24000") -> bytes:
        """TTS: текст -> WAV 16kHz аудио-байты."""
        token = self._get_token()
        resp = requests.post(
            self.TTS_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/text",
            },
            params={"voice": voice, "format": "wav16"},
            data=text.encode("utf-8"),
        )
        resp.raise_for_status()
        return resp.content
