"""Scraper de valor de mercado de selecciones desde Transfermarkt.

Transfermarkt publica el valor de mercado total de cada selección. Esta clase
extrae ese valor parseando el HTML de la página de la selección.

ADVERTENCIA (documentar en la defensa):
    * Transfermarkt no ofrece API pública; el HTML puede cambiar y romper el
      parser. Por eso el método devuelve ``None`` ante cualquier problema y el
      orquestador (build_dataset) decide el fallback.
    * Los Términos de Uso del sitio restringen el scraping masivo. En un
      proyecto real habría que pedir permiso o usar un dataset licenciado
      (p.ej. el dump público de Transfermarkt en Kaggle, actualizado
      periódicamente). Esta implementación es para fines académicos y con
      volumen mínimo + caché.
"""
from __future__ import annotations

import re

from .base import CachedSession

BASE = "https://www.transfermarkt.com"

# Slugs de ejemplo (selección -> ruta). En un proyecto real se resolvería
# dinámicamente buscando el nombre del país.
TEAM_SLUGS = {
    "Argentina": "/argentinien/startseite/verein/3437",
    "Brasil": "/brasilien/startseite/verein/3439",
    "Francia": "/frankreich/startseite/verein/3377",
    # ... el resto se completaría según las selecciones objetivo
}

_VALUE_RE = re.compile(r"([\d.,]+)\s*(bn|m|mil\.|mill\.)", re.IGNORECASE)


def _parse_value_to_millions(text: str) -> float | None:
    """Convierte '€1.05bn' / '€845.50m' a millones de euros (float)."""
    m = _VALUE_RE.search(text)
    if not m:
        return None
    num = float(m.group(1).replace(".", "").replace(",", ".")) if "," in m.group(1) else float(m.group(1))
    unit = m.group(2).lower()
    if unit == "bn":
        return num * 1000.0
    return num


class TransfermarktScraper:
    def __init__(self, cache_dir: str) -> None:
        self.session = CachedSession(cache_dir, min_interval=2.0)

    def market_value_millions(self, team: str) -> float | None:
        """Valor de mercado total de la selección, en millones de €, o None."""
        slug = TEAM_SLUGS.get(team)
        if slug is None:
            return None
        try:
            html = self.session.get_text(BASE + slug)
        except Exception:
            return None
        # Heurística: el valor total suele aparecer junto a 'Total market value'
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            node = soup.find(string=re.compile("market value", re.IGNORECASE))
            if node is None:
                return None
            return _parse_value_to_millions(node.parent.get_text(" ", strip=True))
        except Exception:
            return None
