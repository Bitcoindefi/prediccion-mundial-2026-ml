"""Métricas y gráficos de evaluación reutilizables.

Genera: matriz de confusión, curva(s) ROC, importancia de variables y
comparación de modelos. Todas las figuras se guardan en disco (backend Agg,
sin ventana), por lo que funciona en servidores/headless.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize  # noqa: E402


def compute_metrics(y_true, y_pred, y_proba, classes) -> dict:
    """Calcula accuracy, precision/recall/F1 (macro) y ROC AUC (OvR macro)."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        if len(classes) == 2:
            # probabilidad de la clase positiva
            pos_idx = list(classes).index(classes[-1])
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, pos_idx])
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=list(classes)
            )
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_confusion(y_true, y_pred, classes, title: str, out_path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión - {title}")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(y_true, y_proba, classes, title: str, out_path: str | Path) -> None:
    from sklearn.metrics import auc, roc_curve

    classes = list(classes)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f"Clase {c} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curva ROC - {title}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(importances, names, title: str, out_path: str | Path, top: int = 15) -> None:
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order][::-1], [importances[i] for i in order][::-1], color="#2a6f97")
    ax.set_title(f"Importancia de variables - {title}")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_model_comparison(metrics_df, out_path: str | Path) -> None:
    cols = [c for c in ["accuracy", "f1_macro", "roc_auc"] if c in metrics_df.columns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics_df))
    width = 0.8 / max(1, len(cols))
    for k, col in enumerate(cols):
        ax.bar(x + k * width, metrics_df[col].values, width, label=col)
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(metrics_df["model"].values, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Puntaje")
    ax.set_title("Comparación de modelos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
