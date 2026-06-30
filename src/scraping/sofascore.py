"""Cliente de Sofascore (endpoints internos no oficiales).

Sofascore expone endpoints JSON internos (``api.sofascore.com/api/v1/...``) que
NO son una API pública soportada. Sirven para obtener partidos, alineaciones y
ratings de jugadores, pero:

ADVERTENCIA (la parte MÁS frágil del scraping — documentar en la defensa):
    * Los endpoints pueden cambiar, exigir tokens o bloquear por IP en cualquier
      momento. No hay garantía de estabilidad.
    * El uso intensivo viola sus Términos. Para un proyecto académico se usa con
      volumen mínimo, caché agresiva y SIEMPRE con fallback.
    * Mapear ratings de jugador -> rating agregado de selección requiere mucha
      limpieza. Aquí se deja el cliente base y un agregador best-effort; si
      cualquier paso falla, se devuelve ``None`` y el pipeline usa el fallback.
"""
from __future__ import annotations

from .base import CachedSession

API = "https://api.sofascore.com/api/v1"


class SofascoreClient:
    def __init__(self, cache_dir: str) -> None:
        self.session = CachedSession(cache_dir, min_interval=2.0)

    def event_lineups(self, event_id: int) -> dict | None:
        try:
            return self.session.get_json(f"{API}/event/{event_id}/lineups")
        except Exception:
            return None

    def average_team_rating(self, event_id: int, home: bool = True) -> float | None:
        """Rating medio del once de un equipo en un partido, o None.

        Best-effort: la estructura real del JSON debe verificarse contra el
        endpoint vigente. Cualquier discrepancia -> None (fallback).
        """
        data = self.event_lineups(event_id)
        if not data:
            return None
        try:
            side = "home" if home else "away"
            players = data[side]["players"]
            ratings = []
            for p in players:
                stats = p.get("statistics") or {}
                rating = stats.get("rating")
                if rating:
                    ratings.append(float(rating))
            if not ratings:
                return None
            return sum(ratings) / len(ratings)
        except Exception:
            return None
