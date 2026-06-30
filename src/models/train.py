"""Entrenamiento y comparación de modelos de clasificación.

* Split TEMPORAL (entrena con partidos antiguos, evalúa con los recientes) para
  no sobreestimar el desempeño: es la forma correcta de validar un modelo que
  en producción siempre predice el futuro.
* Compara: Regresión Logística, Árbol de Decisión, Random Forest y XGBoost
  (si está instalado), más una línea base (clase mayoritaria).
* Reporta accuracy, precision, recall, F1 (macro) y ROC AUC; valida con
  TimeSeriesSplit; guarda matrices de confusión, curvas ROC, importancia de
  variables y el mejor modelo serializado.

Uso:
    python -m src.models.train
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import load_config, path_of, set_seeds
from src.features.engineering import feature_columns
from src.models import evaluate

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # noqa: BLE001
    HAS_XGB = False


def _log(msg: str) -> None:
    print(f"[train] {msg}")


def build_models(seed: int, models: list[str]) -> dict:
    catalog: dict[str, Pipeline] = {}
    if "logistic_regression" in models:
        catalog["logistic_regression"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
    if "decision_tree" in models:
        catalog["decision_tree"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=seed)),
        ])
    if "random_forest" in models:
        catalog["random_forest"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                random_state=seed, n_jobs=-1,
            )),
        ])
    if "xgboost" in models and HAS_XGB:
        catalog["xgboost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=seed,
                eval_metric="mlogloss", tree_method="hist",
            )),
        ])
    elif "xgboost" in models and not HAS_XGB:
        _log("xgboost no está instalado: se omite (no es bloqueante).")
    return catalog


def time_split(df: pd.DataFrame, test_size: float):
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def extract_importance(name: str, pipe: Pipeline, feat_names, X_test, y_test):
    """Importancia de variables; usa la nativa del modelo o permutación."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_)
    if hasattr(clf, "coef_"):
        return np.abs(np.asarray(clf.coef_)).mean(axis=0)
    res = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=0, n_jobs=-1)
    return res.importances_mean


def main() -> int:
    cfg = load_config()
    set_seeds(cfg.get("seed", 42))

    feats_path = path_of(cfg["data"]["processed_dir"]) / "features.csv"
    if not feats_path.exists():
        _log(f"No existe {feats_path}. Ejecutá primero: python -m src.data.build_dataset")
        return 1

    df = pd.read_csv(feats_path, parse_dates=["date"])
    if cfg.get("classification") == "binary":
        df["target"] = (df["target"] == 2).astype(int)  # 1=Victoria local, 0=No victoria
        _log("Modo BINARIO: 1=Victoria local, 0=No victoria.")

    feat_names = feature_columns(df)
    train_df, test_df = time_split(df, cfg["split"]["test_size"])
    X_train, y_train = train_df[feat_names], train_df["target"]
    X_test, y_test = test_df[feat_names], test_df["target"]
    _log(f"Train={len(train_df)} | Test={len(test_df)} | nº features={len(feat_names)}")

    out = path_of(cfg.get("output_dir", "reports"))
    out.mkdir(parents=True, exist_ok=True)

    rows = []

    # Línea base: predecir siempre la clase mayoritaria
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_acc = (dummy.predict(X_test) == y_test).mean()
    rows.append({"model": "baseline (mayoría)", "accuracy": base_acc,
                 "precision_macro": np.nan, "recall_macro": np.nan,
                 "f1_macro": np.nan, "roc_auc": np.nan, "cv_acc": np.nan})
    _log(f"Línea base (clase mayoritaria) accuracy = {base_acc:.3f}")

    models = build_models(cfg.get("seed", 42), cfg["models"])
    cv = TimeSeriesSplit(n_splits=5)
    fitted: dict[str, Pipeline] = {}

    for name, pipe in models.items():
        _log(f"Entrenando: {name} ...")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        classes = pipe.classes_
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        m = evaluate.compute_metrics(y_test, y_pred, y_proba, classes)
        try:
            m["cv_acc"] = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean()
        except Exception:
            m["cv_acc"] = float("nan")
        m["model"] = name
        rows.append(m)

        evaluate.plot_confusion(y_test, y_pred, classes, name, out / f"confusion_{name}.png")
        evaluate.plot_roc(y_test, y_proba, classes, name, out / f"roc_{name}.png")
        _log(f"  acc={m['accuracy']:.3f} f1={m['f1_macro']:.3f} "
             f"roc_auc={m['roc_auc']:.3f} cv_acc={m['cv_acc']:.3f}")

    metrics_df = pd.DataFrame(rows)[
        ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "cv_acc"]
    ]
    metrics_df.to_csv(out / "metrics.csv", index=False, encoding="utf-8")
    evaluate.plot_model_comparison(metrics_df.dropna(subset=["f1_macro"]), out / "model_comparison.png")

    # Mejor modelo por F1 macro
    scored = [r for r in rows if r["model"] in fitted]
    best = max(scored, key=lambda r: (r["f1_macro"] if not np.isnan(r["f1_macro"]) else -1))
    best_name = best["model"]
    best_pipe = fitted[best_name]
    _log(f"Mejor modelo: {best_name} (F1 macro={best['f1_macro']:.3f})")

    importances = extract_importance(best_name, best_pipe, feat_names, X_test, y_test)
    evaluate.plot_feature_importance(importances, feat_names, best_name, out / "feature_importance.png")

    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df.sort_values("importance", ascending=False).to_csv(
        out / "feature_importance.csv", index=False, encoding="utf-8"
    )

    joblib.dump(best_pipe, out / "best_model.joblib")
    (out / "model_meta.json").write_text(
        json.dumps({
            "best_model": best_name,
            "classification": cfg.get("classification"),
            "feature_columns": feat_names,
            "classes": [int(c) for c in best_pipe.classes_],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== COMPARACIÓN DE MODELOS =====")
    print(metrics_df.round(3).to_string(index=False))
    _log(f"Artefactos guardados en: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
