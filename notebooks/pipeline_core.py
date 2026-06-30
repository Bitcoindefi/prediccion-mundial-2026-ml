"""Núcleo AUTOCONTENIDO del pipeline (para portar a celdas de Colab).

No depende de `src/`: incluye generación de datos, ingeniería de features sin
fuga de información, modelo Dixon-Coles con decaimiento temporal, métricas
probabilísticas (RPS, log-loss, Brier) y entrenamiento/comparación de modelos.

Se prueba con `python notebooks/pipeline_core.py` (smoke test al final).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import factorial, gammaln

# ============================================================
# 1. DATOS SINTÉTICOS (fallback reproducible)
# ============================================================
COUNTRIES = [
    "Argentina", "Brasil", "Francia", "Inglaterra", "España", "Alemania",
    "Portugal", "Países Bajos", "Italia", "Bélgica", "Uruguay", "Croacia",
    "Colombia", "México", "Estados Unidos", "Japón", "Corea del Sur",
    "Senegal", "Marruecos", "Suiza", "Dinamarca", "Polonia", "Serbia",
    "Ecuador", "Australia", "Ghana", "Camerún", "Túnez", "Canadá",
    "Costa Rica", "Catar", "Arabia Saudita",
]


def generate_synthetic(n_teams=24, start_year=2014, end_year=2024,
                       matches_per_year=300, seed=42):
    rng = np.random.default_rng(seed)
    teams = COUNTRIES[:n_teams]
    base = {t: rng.normal(0.0, 1.0) for t in teams}
    snap_dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")
    n_snaps = len(snap_dates)
    drift = {t: rng.normal(0.0, 0.04, size=n_snaps).cumsum() for t in teams}

    attr_rows = []
    for t in teams:
        for i, d in enumerate(snap_dates):
            s = base[t] + drift[t][i]
            mv = float(max(15.0, 350.0 / (1.0 + np.exp(-s)) + rng.normal(0, 18)))
            rating = float(np.clip(6.7 + 0.45 * s + rng.normal(0, 0.18), 5.5, 8.6))
            attr_rows.append([d, t, mv, rating, int(rng.integers(0, 5)), s])
    attrs = pd.DataFrame(attr_rows, columns=["date", "team", "market_value", "avg_rating", "injured", "_s"])
    attrs["fifa_rank"] = attrs.groupby("date")["_s"].rank(ascending=False, method="first").astype(int)
    attrs = attrs.sort_values(["team", "date"]).reset_index(drop=True)
    attrs["market_value_var"] = attrs.groupby("team")["market_value"].pct_change().fillna(0.0)
    attrs = attrs.drop(columns="_s")

    all_days = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    snap_vals = snap_dates.values
    match_rows = []
    for _ in range(matches_per_year * (end_year - start_year + 1)):
        d = all_days[int(rng.integers(0, len(all_days)))]
        a, b = rng.choice(teams, size=2, replace=False)
        a, b = str(a), str(b)
        neutral = bool(rng.random() < 0.30)
        i = max(0, min(int(np.searchsorted(snap_vals, np.datetime64(d), "right") - 1), n_snaps - 1))
        sa = base[a] + drift[a][i] + (0.0 if neutral else 0.35)
        sb = base[b] + drift[b][i]
        ga = int(rng.poisson(float(np.clip(np.exp(0.15 + 0.55 * (sa - sb)), 0.15, 6.0))))
        gb = int(rng.poisson(float(np.clip(np.exp(0.15 + 0.55 * (sb - sa)), 0.15, 6.0))))
        match_rows.append([d, a, b, ga, gb, neutral])
    matches = pd.DataFrame(match_rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"])
    return matches.sort_values("date").reset_index(drop=True), attrs


# ============================================================
# 2. FEATURES SIN FUGA DE INFORMACIÓN
# ============================================================
NON_FEATURE_COLS = ("date", "home_team", "away_team", "target")


def result_label(hs, as_):
    return 2 if hs > as_ else (1 if hs == as_ else 0)


def _points(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)


class FeatureBuilder:
    def __init__(self, attrs_df, form_window=5, h2h_window=5,
                 elo_k=40.0, elo_home_adv=65.0, elo_base=1500.0, elo_weight=0.35):
        # elo_k alto (40) = el Elo se adapta rápido a resultados recientes.
        # elo_weight (<1) = baja el peso del Elo histórico para que mande la forma reciente.
        self.form_window, self.h2h_window = form_window, h2h_window
        self.elo_k, self.elo_home_adv, self.elo_base = elo_k, elo_home_adv, elo_base
        self.elo_weight = elo_weight
        self._lookup = {}
        if attrs_df is not None and len(attrs_df):
            a = attrs_df.copy()
            a["date"] = pd.to_datetime(a["date"])
            for team, g in a.sort_values(["team", "date"]).groupby("team"):
                g = g.reset_index(drop=True)
                self._lookup[str(team)] = {"dates": g["date"].values, "rows": g}
        self._reset()

    def _reset(self):
        self.elo = defaultdict(lambda: self.elo_base)
        self.hist = defaultdict(lambda: deque(maxlen=10))        # guardamos hasta 10 partidos
        self.elo_hist = defaultdict(lambda: deque(maxlen=11))    # para el momentum de Elo
        self.streak = defaultdict(int)                           # racha de victorias consecutivas
        self.last = {}
        self.h2h = defaultdict(lambda: deque(maxlen=self.h2h_window))

    def _attrs(self, team, date):
        d = {"market_value": np.nan, "market_value_var": np.nan, "fifa_rank": np.nan,
             "avg_rating": np.nan, "injured": np.nan}
        look = self._lookup.get(str(team))
        if look is None:
            return d
        idx = int(np.searchsorted(look["dates"], np.datetime64(pd.Timestamp(date)), "right") - 1)
        if idx < 0:
            return d
        row = look["rows"].iloc[idx]
        return {c: (row[c] if c in row else np.nan) for c in d}

    @staticmethod
    def _pair(a, b):
        return tuple(sorted((str(a), str(b))))

    def _form(self, team):
        """Forma reciente en varias ventanas, con énfasis en lo más reciente."""
        h = self.hist[team]
        n = len(h)
        keys = ("ppg3", "ppg5", "ppg10", "wexp", "gd5", "gd10", "gf5", "ga5", "winrate5")
        if n == 0:
            return {k: np.nan for k in keys} | {"n": 0}
        pts = [x["p"] for x in h]; gf = [x["gf"] for x in h]; ga = [x["ga"] for x in h]
        wins = [1.0 if x["p"] == 3 else 0.0 for x in h]

        def last(a, k):
            a = a[-k:]
            return float(np.mean(a)) if a else np.nan

        # ponderación exponencial: el último partido pesa 1, los previos decaen 0.7^k
        ww = np.array([0.7 ** (n - 1 - i) for i in range(n)])
        wexp = float(np.dot(pts, ww) / ww.sum())
        return {"ppg3": last(pts, 3), "ppg5": last(pts, 5), "ppg10": last(pts, 10),
                "wexp": wexp, "gd5": last(gf, 5) - last(ga, 5), "gd10": last(gf, 10) - last(ga, 10),
                "gf5": last(gf, 5), "ga5": last(ga, 5), "winrate5": last(wins, 5), "n": n}

    def _momentum(self, team):
        eh = self.elo_hist[team]
        return float(eh[-1] - eh[0]) if len(eh) >= 2 else np.nan

    def _row(self, home, away, date, neutral):
        date = pd.Timestamp(date)
        eh, ea = self.elo[home], self.elo[away]
        eff = eh + (0.0 if neutral else self.elo_home_adv)
        fh, fa = self._form(home), self._form(away)
        mh, ma = self._momentum(home), self._momentum(away)
        sh, sa = self.streak[home], self.streak[away]
        ah, aa = self._attrs(home, date), self._attrs(away, date)
        dq = self.h2h.get(self._pair(home, away))
        h2h_wr = (sum(1 for w in dq if w == home) / len(dq)) if dq else np.nan
        rest_h = (date - pd.Timestamp(self.last[home])).days if home in self.last else np.nan
        rest_a = (date - pd.Timestamp(self.last[away])).days if away in self.last else np.nan
        return {
            # --- contexto de fuerza (Elo con peso REDUCIDO vía elo_weight) ---
            "elo_diff": (eh - ea) * self.elo_weight,
            "elo_exp_home": 1.0 / (1.0 + 10 ** (-(eff - ea) / 400.0)),
            # --- RECENCIA (núcleo): forma reciente en varias ventanas ---
            "form_ppg3_diff": fh["ppg3"] - fa["ppg3"],
            "form_ppg5_diff": fh["ppg5"] - fa["ppg5"],
            "form_ppg10_diff": fh["ppg10"] - fa["ppg10"],
            "form_wexp_home": fh["wexp"], "form_wexp_away": fa["wexp"],
            "form_wexp_diff": fh["wexp"] - fa["wexp"],
            "form_gd5_diff": fh["gd5"] - fa["gd5"],
            "form_gd10_diff": fh["gd10"] - fa["gd10"],
            "winrate5_home": fh["winrate5"], "winrate5_away": fa["winrate5"],
            "winrate5_diff": fh["winrate5"] - fa["winrate5"],
            "gf5_home": fh["gf5"], "ga5_home": fh["ga5"], "gf5_away": fa["gf5"], "ga5_away": fa["ga5"],
            # --- momentum reciente (cambio de Elo últimos partidos) y racha ---
            "elo_momentum_home": mh, "elo_momentum_away": ma,
            "elo_momentum_diff": (mh - ma) if (np.isfinite(mh) and np.isfinite(ma)) else np.nan,
            "streak_home": sh, "streak_away": sa, "streak_diff": sh - sa,
            # --- contexto adicional ---
            "fifa_rank_diff": aa["fifa_rank"] - ah["fifa_rank"],
            "mktval_diff": ah["market_value"] - aa["market_value"],
            "mktval_var_home": ah["market_value_var"], "mktval_var_away": aa["market_value_var"],
            "rating_diff": ah["avg_rating"] - aa["avg_rating"],
            "injured_home": ah["injured"], "injured_away": aa["injured"],
            "h2h_home_winrate": h2h_wr, "rest_days_home": rest_h, "rest_days_away": rest_a,
            "neutral": int(bool(neutral)),
        }

    def _update(self, home, away, hs, as_, date, neutral):
        eff = self.elo[home] + (0.0 if neutral else self.elo_home_adv)
        exp_h = 1.0 / (1.0 + 10 ** (-(eff - self.elo[away]) / 400.0))
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        self.elo[home] += self.elo_k * (s_h - exp_h)
        self.elo[away] += self.elo_k * ((1.0 - s_h) - (1.0 - exp_h))
        self.elo_hist[home].append(self.elo[home]); self.elo_hist[away].append(self.elo[away])
        self.hist[home].append({"p": _points(hs, as_), "gf": hs, "ga": as_})
        self.hist[away].append({"p": _points(as_, hs), "gf": as_, "ga": hs})
        self.streak[home] = self.streak[home] + 1 if hs > as_ else 0
        self.streak[away] = self.streak[away] + 1 if as_ > hs else 0
        self.last[home] = date; self.last[away] = date
        w = home if hs > as_ else (away if hs < as_ else "draw")
        self.h2h[self._pair(home, away)].append(w)

    def fit_transform(self, matches):
        self._reset()
        m = matches.copy(); m["date"] = pd.to_datetime(m["date"])
        m = m.sort_values("date").reset_index(drop=True)
        rows = []
        for r in m.itertuples(index=False):
            f = self._row(r.home_team, r.away_team, r.date, bool(r.neutral))
            f.update(date=r.date, home_team=r.home_team, away_team=r.away_team,
                     target=result_label(r.home_score, r.away_score))
            rows.append(f)
            self._update(r.home_team, r.away_team, r.home_score, r.away_score, r.date, bool(r.neutral))
        return pd.DataFrame(rows)

    def current_features(self, home, away, neutral=True, date=None):
        if date is None:
            date = max(self.last.values()) if self.last else pd.Timestamp("2024-01-01")
        return self._row(home, away, pd.Timestamp(date), bool(neutral))


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ============================================================
# 3. DIXON-COLES con decaimiento temporal (implementación propia)
# ============================================================
class DixonColes:
    """Modelo Dixon-Coles (1997) con ponderado temporal exp(-xi*t).

    lambda = exp(ataque_local + defensa_visitante + gamma*localia)
    mu     = exp(ataque_visitante + defensa_local)
    Corrección tau para marcadores bajos; gamma se anula en cancha neutral.
    """

    def __init__(self, max_goals=10):
        self.max_goals = max_goals
        self.teams = None
        self.idx = None
        self.params = None

    @staticmethod
    def _tau(gh, ga, lam, mu, rho):
        t = np.ones_like(lam, dtype=float)
        m = (gh == 0) & (ga == 0); t[m] = 1 - lam[m] * mu[m] * rho
        m = (gh == 0) & (ga == 1); t[m] = 1 + lam[m] * rho
        m = (gh == 1) & (ga == 0); t[m] = 1 + mu[m] * rho
        m = (gh == 1) & (ga == 1); t[m] = 1 - rho
        return t

    def _nll(self, p, ih, ia, gh, ga, hmask, w, n):
        atk, dfn = p[:n], p[n:2 * n]
        gamma, rho = p[2 * n], p[2 * n + 1]
        lam = np.exp(atk[ih] + dfn[ia] + gamma * hmask)
        mu = np.exp(atk[ia] + dfn[ih])
        ll = (gh * np.log(lam) - lam - gammaln(gh + 1)) + (ga * np.log(mu) - mu - gammaln(ga + 1))
        ll = ll + np.log(np.clip(self._tau(gh, ga, lam, mu, rho), 1e-12, None))
        return -np.sum(w * ll)

    def fit(self, df, ref_date=None, xi=0.0, ridge=1e-3, min_team_matches=0):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["home_score", "away_score"])
        if min_team_matches > 0:
            cnt = pd.concat([df["home_team"], df["away_team"]]).value_counts()
            keep = set(cnt[cnt >= min_team_matches].index)
            df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]
        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        ih = df["home_team"].map(self.idx).to_numpy()
        ia = df["away_team"].map(self.idx).to_numpy()
        gh = df["home_score"].to_numpy(float)
        ga = df["away_score"].to_numpy(float)
        hmask = (~df["neutral"].to_numpy(bool)).astype(float)
        if xi > 0:
            ref = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max()
            t_days = (ref - df["date"]).dt.days.to_numpy(float)
            w = np.exp(-xi * np.clip(t_days, 0, None))
        else:
            w = np.ones(len(df))
        # La verosimilitud es invariante a un desplazamiento ataque<->defensa, así que
        # NO hace falta la restricción dura: usamos L-BFGS-B (escala a cientos de equipos)
        # con una ligera regularización ridge que fija esa dirección plana.
        p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

        def obj(p):
            return self._nll(p, ih, ia, gh, ga, hmask, w, n) + ridge * np.sum(p[:2 * n] ** 2)

        res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": 300})
        self.params = res.x
        return self

    def predict_proba(self, home, away, neutral=True):
        """Devuelve (P_home, P_draw, P_away) ó None si falta algún equipo."""
        if self.params is None or home not in self.idx or away not in self.idx:
            return None
        n = len(self.teams)
        atk, dfn = self.params[:n], self.params[n:2 * n]
        gamma, rho = self.params[2 * n], self.params[2 * n + 1]
        h, a = self.idx[home], self.idx[away]
        lam = np.exp(atk[h] + dfn[a] + gamma * (0.0 if neutral else 1.0))
        mu = np.exp(atk[a] + dfn[h])
        mg = self.max_goals
        ks = np.arange(mg + 1)
        fac = factorial(ks)
        hp = np.exp(-lam) * lam ** ks / fac
        ap = np.exp(-mu) * mu ** ks / fac
        m = np.outer(hp, ap)  # filas = goles local, columnas = goles visitante
        m[0, 0] *= 1 - lam * mu * rho
        m[0, 1] *= 1 + lam * rho
        m[1, 0] *= 1 + mu * rho
        m[1, 1] *= 1 - rho
        m = m / m.sum()
        p_home = np.tril(m, -1).sum()
        p_draw = np.trace(m)
        p_away = np.triu(m, 1).sum()
        return float(p_home), float(p_draw), float(p_away)


def dixon_coles_walkforward(matches, xi=0.0019, min_train_years=3, min_team_matches=15):
    """Genera features DC (P1/PX/P2 desde la perspectiva LOCAL) sin fuga de info.

    Reajusta el modelo al inicio de cada año usando solo partidos anteriores
    (con marcador conocido) y predice los partidos de ese año. Devuelve un
    DataFrame alineado por índice con columnas dc_p_home, dc_p_draw, dc_p_away
    (NaN si falta historia/equipo).
    """
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date").reset_index(drop=True)
    years = sorted(m["date"].dt.year.unique())
    out = pd.DataFrame(index=m.index, columns=["dc_p_home", "dc_p_draw", "dc_p_away"], dtype=float)
    start = years[0] + min_train_years
    for y in [yy for yy in years if yy >= start]:
        train = m[(m["date"] < pd.Timestamp(f"{y}-01-01")) & m["home_score"].notna()]
        if len(train) < 50:
            continue
        model = DixonColes().fit(train, ref_date=pd.Timestamp(f"{y}-01-01"),
                                 xi=xi, min_team_matches=min_team_matches)
        cur = m[m["date"].dt.year == y]
        for i, r in cur.iterrows():
            pr = model.predict_proba(r["home_team"], r["away_team"], bool(r["neutral"]))
            if pr is not None:
                out.loc[i, ["dc_p_home", "dc_p_draw", "dc_p_away"]] = pr
    return out


# ============================================================
# 4. MÉTRICAS PROBABILÍSTICAS
# ============================================================
def ranked_probability_score(proba, y_true, classes):
    """RPS ordinal para 1X2. `classes` debe ir en orden ordinal (p.ej. [2,1,0])."""
    proba = np.asarray(proba, float)
    order = list(classes)
    onehot = np.zeros_like(proba)
    pos = {c: k for k, c in enumerate(order)}
    for i, y in enumerate(y_true):
        onehot[i, pos[y]] = 1.0
    cp, co = np.cumsum(proba, axis=1), np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / (proba.shape[1] - 1)))


def reorder_proba(proba, model_classes, target_order):
    """Reordena columnas de predict_proba (orden model_classes) a target_order."""
    idx = [list(model_classes).index(c) for c in target_order]
    return np.asarray(proba)[:, idx]


# ============================================================
# 5. ENTRENAMIENTO Y COMPARACIÓN
# ============================================================
def time_split(df, test_size=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def build_models(seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    models = {
        "logistic_regression": Pipeline([("imp", SimpleImputer(strategy="median")),
                                         ("sc", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "decision_tree": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=seed))]),
        "random_forest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", RandomForestClassifier(n_estimators=300, max_depth=12,
                                                                  min_samples_leaf=5, random_state=seed, n_jobs=-1))]),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("clf", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                                            subsample=0.9, colsample_bytree=0.9, random_state=seed,
                                                            eval_metric="mlogloss", tree_method="hist"))])
    except Exception:
        pass
    return models


def evaluate_all(features, dc_features=None, test_size=0.2, seed=42):
    from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss

    df = features.copy()
    if dc_features is not None:
        df = df.join(dc_features)
    cols = feature_columns(df)
    train, test = time_split(df, test_size)
    Xtr, ytr = train[cols], train["target"]
    Xte, yte = test[cols], test["target"]
    # Orden lexicográfico de clases: 0=visitante, 1=empate, 2=local.
    # El empate (1) queda en el medio, por lo que es un orden ORDINAL válido para el RPS.
    ORD = [0, 1, 2]

    rows = []
    # baseline mayoría
    maj = ytr.value_counts().idxmax()
    rows.append({"model": "baseline (mayoría)", "accuracy": float((yte == maj).mean()),
                 "f1_macro": np.nan, "log_loss": np.nan, "rps": np.nan, "brier": np.nan})

    # baseline Dixon-Coles puro (si hay features DC en el test)
    if dc_features is not None and {"dc_p_home", "dc_p_draw", "dc_p_away"}.issubset(test.columns):
        sub = test.dropna(subset=["dc_p_home", "dc_p_draw", "dc_p_away"])
        if len(sub) > 20:
            proba = sub[["dc_p_away", "dc_p_draw", "dc_p_home"]].to_numpy()  # columnas en orden [0,1,2]
            ytrue = sub["target"].to_numpy()
            pred = proba.argmax(1)  # 0/1/2
            rows.append({"model": "Dixon-Coles puro", "accuracy": float((pred == ytrue).mean()),
                         "f1_macro": f1_score(ytrue, pred, average="macro", labels=ORD, zero_division=0),
                         "log_loss": log_loss(ytrue, proba, labels=ORD),
                         "rps": ranked_probability_score(proba, ytrue, ORD),
                         "brier": float(np.mean(np.sum((proba - np.eye(3)[ytrue]) ** 2, axis=1)))})

    fitted = {}
    for name, pipe in build_models(seed).items():
        pipe.fit(Xtr, ytr)
        fitted[name] = pipe
        pred = pipe.predict(Xte)
        proba = reorder_proba(pipe.predict_proba(Xte), pipe.classes_, ORD)  # alinear a [0,1,2]
        y_oh = np.eye(3)[yte.to_numpy()]
        rows.append({"model": name, "accuracy": accuracy_score(yte, pred),
                     "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
                     "log_loss": log_loss(yte, proba, labels=ORD),
                     "rps": ranked_probability_score(proba, yte.to_numpy(), ORD),
                     "brier": float(np.mean(np.sum((proba - y_oh) ** 2, axis=1)))})
    return pd.DataFrame(rows), fitted, cols


# ============================================================
# SMOKE TEST
# ============================================================
if __name__ == "__main__":
    print(">> Generando datos sintéticos...")
    matches, attrs = generate_synthetic(n_teams=16, start_year=2016, end_year=2022,
                                         matches_per_year=260, seed=7)
    print(f"   partidos={len(matches)} attrs={len(attrs)}")

    print(">> Construyendo features (sin fuga de info)...")
    fb = FeatureBuilder(attrs)
    feats = fb.fit_transform(matches)
    print(f"   features shape={feats.shape}")

    print(">> Dixon-Coles walk-forward (puede tardar)...")
    dc = dixon_coles_walkforward(matches, xi=0.0019, min_train_years=3)
    cov = dc["dc_p_home"].notna().mean()
    print(f"   cobertura DC={cov:.2%}")

    # validar una predicción DC concreta
    model = DixonColes().fit(matches, xi=0.0019)
    pr = model.predict_proba("Argentina", "Brasil", neutral=True)
    print(f"   DC Argentina vs Brasil (neutral): P_local={pr[0]:.2f} P_empate={pr[1]:.2f} P_visit={pr[2]:.2f} suma={sum(pr):.3f}")

    print(">> Entrenando y comparando modelos (con features DC)...")
    table, fitted, cols = evaluate_all(feats, dc_features=dc, test_size=0.2, seed=7)
    print(table.round(3).to_string(index=False))

    assert abs(sum(pr) - 1.0) < 1e-6, "las probabilidades DC deben sumar 1"
    print("\nOK: smoke test completo.")
