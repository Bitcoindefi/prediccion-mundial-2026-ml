"""Ejecuta el pipeline completo de punta a punta.

    python run_all.py

Equivale a:
    python -m src.data.build_dataset
    python -m src.models.train
"""
from __future__ import annotations

import sys

from src.data import build_dataset
from src.models import train


def main() -> int:
    print("=" * 60)
    print("1/2  Construcción del dataset")
    print("=" * 60)
    rc = build_dataset.main()
    if rc != 0:
        return rc

    print("\n" + "=" * 60)
    print("2/2  Entrenamiento y evaluación de modelos")
    print("=" * 60)
    return train.main()


if __name__ == "__main__":
    sys.exit(main())
