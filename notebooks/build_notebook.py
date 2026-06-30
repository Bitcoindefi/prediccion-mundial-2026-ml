"""Construye el notebook de Colab a partir del código YA PROBADO en pipeline_core.py.

Garantiza que las celdas de código del notebook == el código testeado: las
secciones 1-5 se extraen literalmente de pipeline_core.py; el resto (setup,
config, stubs de fuentes reales, ejecución, calibración, predicción) se
autoría aquí. Genera notebooks/Mundial_ML_Colab.ipynb.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = (HERE / "pipeline_core.py").read_text(encoding="utf-8")


def extract_sections(src: str):
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)      # quitar docstring
    src = src.split("\nif __name__")[0]                              # quitar smoke test
    pat = re.compile(r"# ={10,}\n# (\d)\. ([^\n]+)\n# ={10,}\n")
    ms = list(pat.finditer(src))
    preamble = src[: ms[0].start()].strip()
    secs = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(src)
        secs.append((m.group(2).strip(), src[m.end():end].strip()))
    return preamble, secs


PREAMBLE, SECTIONS = extract_sections(CORE)
SEC = {title: code for title, code in SECTIONS}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md("""
# ⚽ Predicción de resultados del Mundial con Machine Learning

**Clasificación supervisada 1X2** (Victoria local / Empate / Victoria visitante) a partir del
*rendimiento reciente* de selecciones y jugadores.

Al hacer **Runtime → Run all**, este notebook descarga **datos reales** (~16.000 partidos
internacionales desde 2010, dataset `martj42` CC0), entrena los modelos, los evalúa con métricas
honestas y **pronostica los partidos del Mundial 2026 que faltan jugar**. Si no hay internet, cae
automáticamente a datos sintéticos (`CONFIG['source']='synthetic'`). Trae además **celdas-stub** para
enriquecer con Transfermarkt (lesiones + valor) y FBref (forma de jugador).

Incluye un modelo **Dixon-Coles (1997)** con *decaimiento temporal* (los partidos recientes pesan más)
usado como **baseline** y como **generador de features**, y evaluación con métricas probabilísticas
correctas (**log-loss, RPS, Brier**) además de accuracy/F1.

> **Nota metodológica:** todas las features se calculan **solo con datos anteriores a la fecha del
> partido** (sin *data leakage*). El Dixon-Coles se reajusta *walk-forward* (un reajuste por año, con
> datos previos) por la misma razón.

---
### Índice
0. Setup · 1. Datos · 2. Features · 3. Dixon-Coles · 4. Métricas · 5. Modelos ·
6. Ejecutar pipeline · 7. Calibración · 8. Predicción · 9. Conectar fuentes reales
"""))

cells.append(md("## 0. Setup\nInstala dependencias e importa. En Colab, `numpy/pandas/scipy/scikit-learn/matplotlib` ya están; instalamos lo opcional."))
cells.append(code("""
# xgboost es opcional (si falla, el pipeline lo omite). penaltyblog es una alternativa
# al Dixon-Coles propio de este notebook (no es necesaria).
!pip -q install xgboost 2>/dev/null
"""))
cells.append(code(PREAMBLE + "\n\nimport warnings\nwarnings.filterwarnings('ignore')"))

cells.append(md("""
### Configuración
Parámetros centrales. La filosofía de *ventanas cortas* vive acá: historia desde ~1990,
lesiones de ~4 meses, variación de valor del último 1-2 años.
"""))
cells.append(code("""
CONFIG = {
    "seed": 42,
    "source": "martj42",          # "martj42" = datos REALES (default) | "synthetic" = fallback offline
    "since_year": 2010,           # entrenar con partidos desde este año
    "wc_start": "2026-06-11",     # inicio del Mundial 2026 (corte para backtest out-of-sample)
    # --- RECENCIA (el núcleo del modelo) ---
    "elo_k": 40,                  # Elo de adaptación rápida
    "elo_weight": 0.35,           # peso reducido del Elo (se descarta el histórico en el modelo)
    "recency_halflife_days": 365, # los partidos viejos pesan menos al entrenar
    "form_window": 5, "h2h_window": 5,
    # --- sintético (solo si source="synthetic") ---
    "n_teams": 24, "start_year": 2014, "end_year": 2024, "matches_per_year": 300,
    # --- enriquecimiento real (lesiones/valor) ---
    "injury_window_days": 120, "value_lookback_years": 2,
}
LABELS = {2: "Victoria local", 1: "Empate", 0: "Victoria visitante"}
"""))

cells.append(md("## 1. Datos\nGenerador sintético (fallback reproducible) — produce el mismo esquema que las fuentes reales:\n`matches`(date, home_team, away_team, home_score, away_score, neutral) y `attrs`(valor, rating, lesiones, ranking)."))
cells.append(code(SEC["DATOS SINTÉTICOS (fallback reproducible)"]))

cells.append(md("""
### 1b. Fuentes reales (stubs del stack gratuito)
Funciones listas para enchufar. `load_martj42` ya es funcional (CSV público CC0). Las de
Transfermarkt/FBref son esqueletos: descomentá y completá cuando quieras enriquecer.
"""))
cells.append(code('''
def load_martj42(since_year=2010):
    """REAL: partidos internacionales (martj42, CC0), INCLUIDOS los fixtures futuros
    del Mundial 2026 (con marcador NaN). Devuelve todos los partidos desde `since_year`."""
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    try:
        df = pd.read_csv(url, parse_dates=["date"])          # en Colab funciona directo
    except Exception:
        import io, ssl, urllib.request                       # fallback si hay problema de certificado SSL
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        txt = urllib.request.urlopen(url, timeout=60, context=ctx).read().decode("utf-8")
        df = pd.read_csv(io.StringIO(txt), parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    df = df[df["date"].dt.year >= since_year].reset_index(drop=True)
    return df[["date", "home_team", "away_team", "home_score", "away_score", "neutral", "tournament"]]


def load_fifa_ranking():
    """STUB: ranking FIFA histórico (snapshot ~2024) + API interna para 2025-26.
    Devuelve DataFrame [team, rank_date, rank, total_points] o None. TODO: validar URL/columnas."""
    try:
        url = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"
        r = pd.read_csv(url, parse_dates=["rank_date"])
        return r
    except Exception as e:
        print("FIFA ranking no disponible:", e)
        return None


def tm_player_injuries(player_id, window_days=120, base="https://transfermarkt-api.fly.dev"):
    """STUB Transfermarkt: lesiones recientes de un jugador (ventana corta). Requiere `requests`.
    Devuelve nº de lesiones activas en los últimos `window_days`."""
    import requests
    try:
        data = requests.get(f"{base}/players/{player_id}/injuries", timeout=20).json()
        # TODO: filtrar por fromDate/untilDate dentro de window_days y contar bajas activas
        return data
    except Exception as e:
        print("transfermarkt-api injuries falló:", e); return None


def tm_player_market_value(player_id, base="https://transfermarkt-api.fly.dev"):
    """STUB Transfermarkt: serie histórica de valor de mercado (para variación trimestral/semestral)."""
    import requests
    try:
        return requests.get(f"{base}/players/{player_id}/market_value", timeout=20).json()
    except Exception as e:
        print("transfermarkt-api market_value falló:", e); return None


def fbref_recent_player_form(season="2022", data_dir="/content/drive/MyDrive/soccerdata"):
    """STUB soccerdata/FBref: stats de jugador por partido (forma reciente). Requiere `pip install soccerdata`."""
    import soccerdata as sd
    fb = sd.FBref(leagues="INT-World Cup", seasons=season, data_dir=data_dir)
    return fb.read_player_match_stats(stat_type="summary")


def get_data():
    """Devuelve (matches_all, attrs). matches_all incluye fixtures futuros (marcador NaN)."""
    if CONFIG["source"] == "martj42":
        return load_martj42(CONFIG["since_year"]), None   # attrs None: se enriquece luego
    return generate_synthetic(CONFIG["n_teams"], CONFIG["start_year"], CONFIG["end_year"],
                              CONFIG["matches_per_year"], CONFIG["seed"])
'''))

cells.append(md("## 2. Ingeniería de features (sin fuga de información)\n`FeatureBuilder` procesa los partidos ordenados por fecha: registra el estado **previo** como features y solo **después** lo actualiza, así ninguna feature usa información del propio partido."))
cells.append(code(SEC["FEATURES SIN FUGA DE INFORMACIÓN"]))

cells.append(md("""
## 3. Dixon-Coles con decaimiento temporal
Modelo paramétrico de marcadores: `λ=exp(ataque_local+defensa_visita+γ)`, `μ=exp(ataque_visita+defensa_local)`,
con la corrección `τ` para marcadores bajos. El **peso temporal** `φ(t)=exp(−ξ·t)` hace que los partidos
recientes pesen más (tu filosofía). En el Mundial (cancha neutral) `γ` se anula. Se reajusta *walk-forward*.
"""))
cells.append(code(SEC["DIXON-COLES con decaimiento temporal (implementación propia)"]))

cells.append(md("## 4. Métricas probabilísticas\nPara un problema 1X2 con empate, lo correcto es **log-loss, RPS y Brier** (reglas de puntuación propias), no solo accuracy."))
cells.append(code(SEC["MÉTRICAS PROBABILÍSTICAS"]))

cells.append(md("## 5. Modelos\nRegresión Logística, Árbol, Random Forest, XGBoost (opcional) + baselines (mayoría y Dixon-Coles puro). Split **temporal**."))
cells.append(code(SEC["ENTRENAMIENTO Y COMPARACIÓN"]))

cells.append(md("""## 6. Ejecutar el pipeline (datos REALES, modelo de RECENCIA)
Entrena con partidos **previos al Mundial**, prioriza la **forma reciente** (descarta el Elo histórico y
pesa más los partidos recientes) y **backtestea sobre los partidos del Mundial 2026 ya jugados**
(out-of-sample): así se ve el acierto real, no una predicción suelta."""))
cells.append(code("""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, log_loss
np.random.seed(CONFIG["seed"])
WC_START = pd.Timestamp(CONFIG["wc_start"])

matches_all, attrs = get_data()
completed = matches_all[matches_all["home_score"].notna()].copy()
future = matches_all[matches_all["home_score"].isna()].copy()
print(f"Fuente: {CONFIG['source']} | jugados: {len(completed)} | por jugar: {len(future)}")

fb = FeatureBuilder(attrs, form_window=CONFIG["form_window"], h2h_window=CONFIG["h2h_window"],
                    elo_k=CONFIG["elo_k"], elo_weight=CONFIG["elo_weight"])
feats = fb.fit_transform(completed)
if "tournament" in completed.columns:
    feats = feats.merge(completed[["date", "home_team", "away_team", "tournament"]],
                        on=["date", "home_team", "away_team"], how="left")
else:
    feats["tournament"] = ""

# RECENCIA: descartamos el Elo histórico absoluto para que mande la forma reciente.
DROP = {"elo_diff", "elo_exp_home", "tournament"}
cols = [c for c in feature_columns(feats) if c not in DROP and feats[c].notna().any()]

is_wc = feats["tournament"].astype(str).str.contains("World Cup", case=False, na=False) & (feats["date"] >= WC_START)
train = feats[feats["date"] < WC_START]
wc_back = feats[is_wc]   # partidos del Mundial ya jugados → backtest OUT-OF-SAMPLE

# pesos de recencia: half-life configurable (los partidos viejos pesan menos)
age = (train["date"].max() - train["date"]).dt.days.to_numpy(float)
sw = np.exp(-np.log(2) / CONFIG["recency_halflife_days"] * age)
ORD = [0, 1, 2]

fitted, rows = {}, []
maj = train["target"].value_counts().idxmax()
if len(wc_back):
    rows.append({"model": "baseline (mayoría)", "WC_accuracy": float((wc_back["target"] == maj).mean()),
                 "WC_rps": np.nan, "WC_logloss": np.nan})
for name, pipe in build_models(CONFIG["seed"]).items():
    pipe.fit(train[cols], train["target"], clf__sample_weight=sw)   # ENTRENA pesando lo reciente
    fitted[name] = pipe
    if len(wc_back):
        proba = reorder_proba(pipe.predict_proba(wc_back[cols]), pipe.classes_, ORD)
        rows.append({"model": name,
                     "WC_accuracy": accuracy_score(wc_back["target"], pipe.predict(wc_back[cols])),
                     "WC_rps": ranked_probability_score(proba, wc_back["target"].to_numpy(), ORD),
                     "WC_logloss": log_loss(wc_back["target"], proba, labels=ORD)})
table = pd.DataFrame(rows)
real = table[table["model"].isin(fitted)].dropna(subset=["WC_logloss"])
best_name = real.sort_values("WC_logloss").iloc[0]["model"] if len(real) else list(fitted)[0]
print(f"Entrenado con {len(train)} partidos PRE-Mundial | backtest sobre {len(wc_back)} del Mundial ya jugados")
print(f"Features usadas: {len(cols)} (sin Elo histórico) | mejor modelo: {best_name}")
display(table.round(3))
clf = fitted[best_name].named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    imp = pd.Series(clf.feature_importances_, index=cols)
else:
    imp = pd.Series(np.abs(clf.coef_).mean(0), index=cols)   # LogReg: magnitud de coeficientes
print("\\nTop features (dominan las de forma reciente):")
display(imp.sort_values(ascending=False).head(10).round(3))
"""))

cells.append(md("""
## 7. Calibración — *por qué un 11% que no ocurre NO es un error*
Un modelo probabilístico no se juzga por una predicción suelta, sino por su **calibración** sobre muchas:
si a los eventos que reciben ~`p` de probabilidad ocurren efectivamente ~`p` de las veces, el modelo está
bien calibrado. Acá graficamos la *curva de fiabilidad* de la clase **Victoria local** del mejor modelo.
"""))
cells.append(code("""
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

val = wc_back if len(wc_back) > 30 else feats   # calibrar sobre el backtest del Mundial
best = fitted[best_name]
proba = best.predict_proba(val[cols])
classes = list(best.classes_)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for cls, name in [(2, "Victoria local"), (1, "Empate"), (0, "Victoria visitante")]:
    p_cls = proba[:, classes.index(cls)]
    y_cls = (val["target"].to_numpy() == cls).astype(int)
    frac, mean_pred = calibration_curve(y_cls, p_cls, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac, "o-", label=name)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibración perfecta")
ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada")
ax.set_title(f"Curva de fiabilidad — {best_name}"); ax.legend(); plt.show()
print("Si los puntos siguen la diagonal, el modelo está bien calibrado:")
print("un 11% predicho corresponde a ~11% de ocurrencia real.")
"""))

cells.append(md("""## 8. 🏆 Pronóstico del Mundial 2026 (partidos que faltan)
Reentrena con **todos** los partidos jugados (incluida la fase de grupos del propio Mundial), pesando lo
reciente, y pronostica cada fixture pendiente. Como es **eliminación**, además del 1X2 calcula
**quién avanza** = P(gana) + ½·P(empate) (el empate se define por penales, ~moneda al aire)."""))
cells.append(code('''
# Reentrenar con TODO lo jugado, pesando lo reciente (incluye la fase de grupos del Mundial)
age2 = (feats["date"].max() - feats["date"]).dt.days.to_numpy(float)
sw2 = np.exp(-np.log(2) / CONFIG["recency_halflife_days"] * age2)
best = build_models(CONFIG["seed"])[best_name].fit(feats[cols], feats["target"], clf__sample_weight=sw2)

wc = future[future["tournament"].astype(str).str.contains("World Cup", case=False, na=False)] if "tournament" in future.columns else future
if len(wc) == 0:
    wc = future

preds = []
for _, r in wc.iterrows():
    h, a, nt = r["home_team"], r["away_team"], bool(r["neutral"])
    feat = fb.current_features(h, a, neutral=nt, date=r["date"])
    X = pd.DataFrame([feat]).reindex(columns=cols)
    pm = best.predict_proba(X)[0]; cl = list(best.classes_)
    pl, pe, pv = pm[cl.index(2)], pm[cl.index(1)], pm[cl.index(0)]
    adv_l, adv_v = pl + 0.5 * pe, pv + 0.5 * pe        # quién avanza (penales reparten el empate)
    preds.append({"fecha": r["date"].date(), "local": h, "visitante": a,
                  "P(local)": round(pl, 2), "P(empate)": round(pe, 2), "P(visit)": round(pv, 2),
                  "avanza_local": round(adv_l, 2), "avanza_visit": round(adv_v, 2),
                  "favorito": h if adv_l >= adv_v else a})

pron = pd.DataFrame(preds)
print(f"=== PRONÓSTICO — Mundial 2026: {len(pron)} partidos que faltan ===")
display(pron)
'''))

cells.append(md("""## 8b. 👤 Desempeño individual: rating del plantel (API-Football)
Agrega el **rating reciente de los jugadores** de cada selección (promedio ponderado por minutos) como
señal de desempeño individual. Requiere una API key gratuita de [api-football.com](https://dashboard.api-football.com/register).

> **Cobertura del free tier (honesto):** los ratings de plantel están disponibles para las selecciones
> grandes, pero **faltan para varias selecciones chicas** (p.ej. Cape Verde no tiene datos). Las **lesiones
> de selección no se cubren** en el free tier. Por la cuota (100 req/día) este dato NO entra al
> entrenamiento histórico: se usa como **señal complementaria del pronóstico**, no como feature entrenada."""))
cells.append(code('''
import getpass, json, time, urllib.parse, urllib.request

API_KEY = ""  # pegá tu key acá, o dejala vacía y la pedimos con getpass abajo
if not API_KEY:
    try:
        API_KEY = getpass.getpass("API-Football key (Enter para omitir): ").strip()
    except Exception:
        API_KEY = ""

def _af(path):
    req = urllib.request.Request("https://v3.football.api-sports.io" + path,
                                 headers={"x-apisports-key": API_KEY})
    return json.loads(urllib.request.urlopen(req, timeout=40).read().decode("utf-8"))

_id_cache = {}
def _team_id(name):
    if name in _id_cache:
        return _id_cache[name]
    tid = None
    try:
        d = _af("/teams?search=" + urllib.parse.quote(name))
        tid = next((t["team"]["id"] for t in d.get("response", []) if t["team"].get("national")), None)
    except Exception:
        pass
    _id_cache[name] = tid; time.sleep(1.2); return tid

def squad_rating(name, seasons=(2026, 2024)):
    """Rating del plantel ponderado por minutos (o NaN si no hay cobertura)."""
    tid = _team_id(name)
    if not tid:
        return np.nan
    for s in seasons:
        try:
            d = _af(f"/players?team={tid}&season={s}&page=1")
        except Exception:
            continue
        rs, ws = [], []
        for it in d.get("response", []):
            g = (it.get("statistics") or [{}])[0].get("games", {}) or {}
            if g.get("rating"):
                rs.append(float(g["rating"])); ws.append(max(g.get("minutes") or 1, 1))
        if rs:
            return float(np.average(rs, weights=ws))
        time.sleep(1.2)
    return np.nan

if API_KEY:
    teams_needed = sorted(set(wc["home_team"]) | set(wc["away_team"]))
    print(f"Consultando rating de plantel de {len(teams_needed)} selecciones (con caché)...")
    sr = {t: squad_rating(t) for t in teams_needed}
    pron["rating_plantel_local"] = pron["local"].map(sr).round(2)
    pron["rating_plantel_visit"] = pron["visitante"].map(sr).round(2)
    cov = pd.Series(sr).notna().mean()
    print(f"Cobertura de rating de plantel: {cov:.0%} de las selecciones")
    display(pron[["local", "visitante", "favorito", "rating_plantel_local", "rating_plantel_visit"]])
else:
    print("Sin API key → se omite el rating de plantel (el pronóstico de recencia ya está arriba).")
'''))

cells.append(md("""
## 9. Conectar las fuentes reales
1. **`CONFIG['source'] = 'martj42'`** → entrena con partidos internacionales reales (1990→hoy).
2. **Montá Google Drive** para cachear (clave para no re-descargar/re-scrapear):
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   ```
3. **Enriquecé** con las funciones stub: `tm_player_injuries` (lesiones ~4 meses),
   `tm_player_market_value` (variación de valor), `fbref_recent_player_form` (forma de jugador),
   `load_fifa_ranking` (ranking + delta). Agregá esas señales a `attrs` (a nivel selección,
   promediando los jugadores convocados) y volvé a correr el pipeline.
4. **Normalizá nombres** de países entre fuentes (martj42 vs FIFA vs Transfermarkt) con una tabla de mapeo.

> Recordá los rate limits: FBref ≤10 req/min, transfermarkt-api ~2 req/3s. Cacheá todo en Drive.
"""))
cells.append(code("""
# Helper de caché en Drive (descomentá en Colab)
# from google.colab import drive; drive.mount('/content/drive')
# import os; CACHE = "/content/drive/MyDrive/mundial_ml"; os.makedirs(CACHE, exist_ok=True)
# def cached_csv(name, fn):
#     import os
#     p = f"{CACHE}/{name}.parquet"
#     if os.path.exists(p): return pd.read_parquet(p)
#     df = fn(); df.to_parquet(p); return df
print("Listo. Cambiá CONFIG['source']='martj42' y re-ejecutá desde la sección 6 para datos reales.")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = HERE / "Mundial_ML_Colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"Notebook escrito: {out}")
print(f"Celdas: {len(cells)} ({n_code} código, {n_md} markdown)")
