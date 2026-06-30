"""Pruebas automatizadas del pipeline (ver docs/casos_de_prueba.md).

Cubren: generación de datos, esquema de features, NO fuga de información,
validez del objetivo y que el modelo supere a la línea base.

Ejecutar:
    pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import synthetic
from src.features.engineering import FeatureBuilder, feature_columns, result_label


@pytest.fixture(scope="module")
def small_data():
    matches, attrs = synthetic.generate(
        n_teams=16, start_year=2015, end_year=2018, matches_per_year=300, seed=7
    )
    return matches, attrs


@pytest.fixture(scope="module")
def features(small_data):
    matches, attrs = small_data
    fb = FeatureBuilder(attrs)
    return fb.fit_transform(matches)


# --- CP-01: generación de datos ---
def test_synthetic_shapes(small_data):
    matches, attrs = small_data
    assert len(matches) == 300 * 4
    assert set(["date", "home_team", "away_team", "home_score", "away_score", "neutral"]).issubset(matches.columns)
    assert set(["team", "market_value", "fifa_rank", "avg_rating", "injured"]).issubset(attrs.columns)
    assert (matches["home_team"] != matches["away_team"]).all()


# --- CP-02: etiqueta objetivo correcta ---
def test_result_label():
    assert result_label(2, 0) == 2  # victoria local
    assert result_label(1, 1) == 1  # empate
    assert result_label(0, 3) == 0  # derrota local


def test_target_distribution_valid(features):
    assert set(features["target"].unique()).issubset({0, 1, 2})
    # ninguna clase debería ser degenerada (>97%)
    assert features["target"].value_counts(normalize=True).max() < 0.97


# --- CP-03: esquema de features ---
def test_feature_columns_exclude_ids(features):
    cols = feature_columns(features)
    for forbidden in ("date", "home_team", "away_team", "target"):
        assert forbidden not in cols
    assert "elo_diff" in cols and "form_ppg_diff" in cols


# --- CP-04: NO fuga de información (crítico) ---
def test_no_leakage_first_match(small_data):
    matches, attrs = small_data
    fb = FeatureBuilder(attrs)
    feats = fb.fit_transform(matches)
    first = feats.sort_values("date").iloc[0]
    # El primer partido del histórico no puede tener forma/rest/h2h previos
    assert np.isnan(first["form_ppg_home"])
    assert np.isnan(first["form_ppg_away"])
    assert np.isnan(first["rest_days_home"])
    assert np.isnan(first["h2h_home_winrate"])
    # Elo arranca en el valor base
    assert first["elo_home"] == pytest.approx(1500.0)
    assert first["elo_away"] == pytest.approx(1500.0)


def test_elo_changes_after_matches(features):
    # Tras procesar muchos partidos, los Elo deben haberse separado del base
    assert features["elo_home"].std() > 0


# --- CP-05: el modelo supera a la línea base ---
def test_model_beats_baseline(features):
    from src.models.train import build_models, time_split

    df = features
    train_df, test_df = time_split(df, 0.2)
    cols = feature_columns(df)
    Xtr, ytr = train_df[cols], train_df["target"]
    Xte, yte = test_df[cols], test_df["target"]

    baseline = ytr.value_counts(normalize=True).max()  # acc de la clase mayoritaria
    pipe = build_models(42, ["random_forest"])["random_forest"]
    pipe.fit(Xtr, ytr)
    acc = (pipe.predict(Xte) == yte).mean()
    assert acc >= baseline  # el modelo no debe ser peor que predecir la mayoría
