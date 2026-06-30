"""Cliente HTTP defensivo con caché en disco, rate limiting y reintentos.

Las fuentes externas (Transfermarkt, Sofascore) NO ofrecen una API pública
estable: scrapear su HTML o sus endpoints internos es frágil y puede cambiar
sin aviso. Este cliente reduce el daño:

* **Caché en disco**: cada respuesta se guarda; reejecutar no vuelve a pedirla.
  Así el proyecto es reproducible y no depende de tener internet en la defensa.
* **Rate limiting**: espera entre peticiones para no saturar al servidor.
* **Reintentos con backoff**: tolera caídas puntuales.
* **User-Agent realista**: evita bloqueos triviales.

Aun así, el llamador SIEMPRE debe asumir que el scraping puede fallar y tener
un plan B (en este proyecto, datos sintéticos). Ver ``data/build_dataset.py``.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


class CachedSession:
    def __init__(
        self,
        cache_dir: str | Path,
        min_interval: float = 1.5,
        max_retries: int = 3,
        timeout: float = 20.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)

    def _cache_path(self, url: str, params: dict | None) -> Path:
        key = url + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.cache"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def get_text(self, url: str, params: dict | None = None, use_cache: bool = True) -> str:
        cache_path = self._cache_path(url, params)
        if use_cache and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._throttle()
                resp = self._session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                text = resp.text
                cache_path.write_text(text, encoding="utf-8")
                return text
            except Exception as exc:  # noqa: BLE001 - queremos tolerar cualquier fallo de red
                last_err = exc
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Fallo al solicitar {url}: {last_err}")

    def get_json(self, url: str, params: dict | None = None, use_cache: bool = True) -> dict:
        return json.loads(self.get_text(url, params=params, use_cache=use_cache))
