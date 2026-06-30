"""Interfaz web (Streamlit) para predecir el resultado de un partido.

Reconstruye el estado del histórico (Elo, forma, etc.) a partir de los CSV
crudos y usa el mejor modelo entrenado para estimar las probabilidades de
Victoria local / Empate / Victoria visitante.

Uso:
    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Streamlit ejecuta el script con su carpeta (src/app) como sys.path[0], por lo
# que el paquete raíz 'src' no sería importable. Lo añadimos manualmente.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.config import load_config, path_of
from src.features.engineering import FeatureBuilder

st.set_page_config(page_title="Predicción de partidos - Mundial", page_icon="⚽", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo e histórico...")
def load_everything():
    cfg = load_config()
    proc = path_of(cfg["data"]["processed_dir"])
    out = path_of(cfg.get("output_dir", "reports"))

    matches = pd.read_csv(proc / "matches.csv", parse_dates=["date"])
    attrs = pd.read_csv(proc / "attrs.csv", parse_dates=["date"])

    fb = FeatureBuilder(
        attrs,
        form_window=cfg["features"]["form_window"],
        h2h_window=cfg["features"]["h2h_window"],
        elo_k=cfg["elo"]["k"],
        elo_home_adv=cfg["elo"]["home_advantage"],
        elo_base=cfg["elo"]["base"],
    )
    fb.fit_transform(matches)  # reconstruye el estado final

    model = joblib.load(out / "best_model.joblib")
    meta = json.loads((out / "model_meta.json").read_text(encoding="utf-8"))
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return cfg, fb, model, meta, teams


def label_for(cls: int, binary: bool) -> str:
    if binary:
        return {1: "Victoria local", 0: "No victoria"}.get(int(cls), str(cls))
    return {2: "Victoria local", 1: "Empate", 0: "Victoria visitante"}.get(int(cls), str(cls))


def main() -> None:
    st.title("⚽ Predicción de resultados de partidos")
    st.caption("Modelo de Machine Learning entrenado con desempeño reciente de selecciones.")

    if not (path_of("reports") / "best_model.joblib").exists():
        st.error("No hay modelo entrenado. Ejecutá primero:\n\n"
                 "`python -m src.data.build_dataset` y luego `python -m src.models.train`")
        return

    cfg, fb, model, meta, teams = load_everything()
    binary = meta.get("classification") == "binary"

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Equipo local", teams, index=0)
    with col2:
        away = st.selectbox("Equipo visitante", teams, index=1)
    neutral = st.checkbox("Cancha neutral (típico en Mundial)", value=True)

    if home == away:
        st.warning("Elegí dos selecciones distintas.")
        return

    if st.button("Predecir", type="primary"):
        feat = fb.current_features(home, away, neutral=neutral)
        X = pd.DataFrame([feat])[meta["feature_columns"]]
        proba = model.predict_proba(X)[0]
        classes = model.classes_

        st.subheader("Probabilidades estimadas")
        prob_df = pd.DataFrame({
            "Resultado": [label_for(c, binary) for c in classes],
            "Probabilidad": proba,
        }).sort_values("Probabilidad", ascending=False)
        st.bar_chart(prob_df.set_index("Resultado"))
        for _, r in prob_df.iterrows():
            st.write(f"**{r['Resultado']}**: {r['Probabilidad'] * 100:.1f}%")

        with st.expander("Ver Elo y forma usados"):
            st.json({
                "elo_local": round(feat["elo_home"], 1),
                "elo_visitante": round(feat["elo_away"], 1),
                "forma_ppg_local": feat["form_ppg_home"],
                "forma_ppg_visitante": feat["form_ppg_away"],
                "ranking_fifa_local": feat["fifa_rank_home"],
                "ranking_fifa_visitante": feat["fifa_rank_away"],
            })

    st.divider()
    st.caption(f"Mejor modelo: **{meta.get('best_model')}**. "
               "Las predicciones se basan en datos previos a la fecha del partido (sin fuga de información).")


main()
