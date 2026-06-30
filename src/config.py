"""Carga de configuración, semillas y utilidades de rutas."""
from __future__ import annotations

import random
from pathlib import Path

import yaml

try:  # numpy puede no estar disponible al importar config de forma aislada
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Lee el archivo config.yaml y devuelve un diccionario."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seeds(seed: int = 42) -> None:
    """Fija las semillas para garantizar reproducibilidad."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def path_of(rel: str) -> Path:
    """Devuelve una ruta absoluta relativa a la raíz del proyecto."""
    return PROJECT_ROOT / rel


def ensure_dirs(cfg: dict) -> None:
    """Crea los directorios de datos y reportes si no existen."""
    for key in ("cache_dir", "raw_dir", "processed_dir"):
        rel = cfg.get("data", {}).get(key)
        if rel:
            path_of(rel).mkdir(parents=True, exist_ok=True)
    path_of(cfg.get("output_dir", "reports")).mkdir(parents=True, exist_ok=True)
