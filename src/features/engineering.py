"""Ingeniería de características con prevención estricta de fuga de información.

Idea central
------------
Para predecir un partido jugado en la fecha *D*, SOLO se pueden usar datos
anteriores a *D*. Si calculáramos la "forma reciente" o el Elo incluyendo el
propio partido (o partidos posteriores), el modelo "vería el futuro" y daría
métricas infladas que no se sostienen en producción (data leakage).

Para garantizarlo, ``FeatureBuilder`` procesa los partidos ORDENADOS por fecha
en una sola pasada: para cada partido primero registra el estado PREVIO
(features) y solo DESPUÉS actualiza el estado con el resultado. Así es
imposible que una feature contenga información del propio partido.

El mismo objeto se reutiliza para inferencia (``current_features``): tras
procesar todo el histórico, expone el estado final de cada selección para
predecir un partido futuro.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Columnas que NO son features (identificadores y objetivo)
NON_FEATURE_COLS = ("date", "home_team", "away_team", "target")


def _points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def result_label(home_score: int, away_score: int) -> int:
    """Etiqueta desde la perspectiva del equipo local: 2=victoria, 1=empate, 0=derrota."""
    if home_score > away_score:
        return 2
    if home_score == away_score:
        return 1
    return 0


class FeatureBuilder:
    def __init__(
        self,
        attrs_df: pd.DataFrame | None,
        form_window: int = 5,
        h2h_window: int = 5,
        elo_k: float = 30.0,
        elo_home_adv: float = 65.0,
        elo_base: float = 1500.0,
    ) -> None:
        self.form_window = form_window
        self.h2h_window = h2h_window
        self.elo_k = elo_k
        self.elo_home_adv = elo_home_adv
        self.elo_base = elo_base
        self._attr_lookup: dict = {}
        self._prep_attrs(attrs_df)
        self._reset_state()

    # ------------------------------------------------------------------ attrs
    def _prep_attrs(self, attrs_df: pd.DataFrame | None) -> None:
        """Indexa los atributos por equipo para búsqueda rápida 'última fecha <= D'."""
        self._attr_lookup = {}
        if attrs_df is None or len(attrs_df) == 0:
            return
        a = attrs_df.copy()
        a["date"] = pd.to_datetime(a["date"])
        a = a.sort_values(["team", "date"])
        for team, g in a.groupby("team"):
            g = g.reset_index(drop=True)
            self._attr_lookup[str(team)] = {
                "dates": g["date"].values,
                "rows": g,
            }

    def _attrs_for(self, team: str, date: pd.Timestamp) -> dict:
        default = {
            "market_value": np.nan,
            "market_value_var": np.nan,
            "fifa_rank": np.nan,
            "avg_rating": np.nan,
            "injured": np.nan,
        }
        look = self._attr_lookup.get(str(team))
        if look is None:
            return default
        idx = int(np.searchsorted(look["dates"], np.datetime64(pd.Timestamp(date)), side="right") - 1)
        if idx < 0:
            return default
        row = look["rows"].iloc[idx]
        return {c: (row[c] if c in row else np.nan) for c in default}

    # ------------------------------------------------------------------ estado
    def _reset_state(self) -> None:
        self.elo: dict = defaultdict(lambda: self.elo_base)
        self.history: dict = defaultdict(lambda: deque(maxlen=self.form_window))
        self.last_date: dict = {}
        self.h2h: dict = defaultdict(lambda: deque(maxlen=self.h2h_window))

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple:
        return tuple(sorted((str(a), str(b))))

    def _team_form(self, team: str) -> dict:
        h = self.history[team]
        if len(h) == 0:
            return {"ppg": np.nan, "gf": np.nan, "ga": np.nan, "gd": np.nan, "n": 0}
        gf = float(np.mean([x["gf"] for x in h]))
        ga = float(np.mean([x["ga"] for x in h]))
        return {
            "ppg": float(np.mean([x["points"] for x in h])),
            "gf": gf,
            "ga": ga,
            "gd": gf - ga,
            "n": len(h),
        }

    def _rest_days(self, team: str, date: pd.Timestamp) -> float:
        ld = self.last_date.get(team)
        if ld is None:
            return np.nan
        return float((pd.Timestamp(date) - pd.Timestamp(ld)).days)

    def _h2h_home_winrate(self, home: str, away: str) -> float:
        dq = self.h2h.get(self._pair_key(home, away))
        if not dq:
            return np.nan
        wins = sum(1 for w in dq if w == home)
        return wins / len(dq)

    # ------------------------------------------------------------------ features
    def _features_row(self, home: str, away: str, date: pd.Timestamp, neutral: bool) -> dict:
        date = pd.Timestamp(date)
        elo_h = self.elo[home]
        elo_a = self.elo[away]
        eff_h = elo_h + (0.0 if neutral else self.elo_home_adv)
        exp_home = 1.0 / (1.0 + 10 ** (-(eff_h - elo_a) / 400.0))

        fh = self._team_form(home)
        fa = self._team_form(away)
        ah = self._attrs_for(home, date)
        aa = self._attrs_for(away, date)

        return {
            # --- Elo (predictor fuerte de fuerza acumulada) ---
            "elo_home": elo_h,
            "elo_away": elo_a,
            "elo_diff": elo_h - elo_a,
            "elo_exp_home": exp_home,
            # --- Forma reciente ---
            "form_ppg_home": fh["ppg"],
            "form_ppg_away": fa["ppg"],
            "form_ppg_diff": fh["ppg"] - fa["ppg"],
            "form_gd_home": fh["gd"],
            "form_gd_away": fa["gd"],
            "gf_home": fh["gf"],
            "ga_home": fh["ga"],
            "gf_away": fa["gf"],
            "ga_away": fa["ga"],
            # --- Ranking FIFA (rank menor = mejor) ---
            "fifa_rank_home": ah["fifa_rank"],
            "fifa_rank_away": aa["fifa_rank"],
            "fifa_rank_diff": aa["fifa_rank"] - ah["fifa_rank"],
            # --- Valor de mercado (Transfermarkt) ---
            "mktval_home": ah["market_value"],
            "mktval_away": aa["market_value"],
            "mktval_diff": ah["market_value"] - aa["market_value"],
            "mktval_var_home": ah["market_value_var"],
            "mktval_var_away": aa["market_value_var"],
            # --- Rating de jugadores (Sofascore) ---
            "rating_home": ah["avg_rating"],
            "rating_away": aa["avg_rating"],
            "rating_diff": ah["avg_rating"] - aa["avg_rating"],
            # --- Estado físico ---
            "injured_home": ah["injured"],
            "injured_away": aa["injured"],
            # --- Enfrentamientos directos ---
            "h2h_home_winrate": self._h2h_home_winrate(home, away),
            # --- Descanso ---
            "rest_days_home": self._rest_days(home, date),
            "rest_days_away": self._rest_days(away, date),
            # --- Contexto ---
            "neutral": int(bool(neutral)),
        }

    # ------------------------------------------------------------------ update
    def _update(self, home, away, hs, as_, date, neutral) -> None:
        # Actualización Elo (después de registrar las features del partido)
        eff_h = self.elo[home] + (0.0 if neutral else self.elo_home_adv)
        exp_home = 1.0 / (1.0 + 10 ** (-(eff_h - self.elo[away]) / 400.0))
        if hs > as_:
            s_home = 1.0
        elif hs == as_:
            s_home = 0.5
        else:
            s_home = 0.0
        self.elo[home] += self.elo_k * (s_home - exp_home)
        self.elo[away] += self.elo_k * ((1.0 - s_home) - (1.0 - exp_home))

        # Forma reciente
        self.history[home].append({"points": _points(hs, as_), "gf": hs, "ga": as_})
        self.history[away].append({"points": _points(as_, hs), "gf": as_, "ga": hs})
        self.last_date[home] = date
        self.last_date[away] = date

        # Enfrentamiento directo
        if hs > as_:
            winner = home
        elif hs < as_:
            winner = away
        else:
            winner = "draw"
        self.h2h[self._pair_key(home, away)].append(winner)

    # ------------------------------------------------------------------ API
    def fit_transform(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Construye la tabla de features (una fila por partido), sin fuga de info."""
        self._reset_state()
        m = matches_df.copy()
        m["date"] = pd.to_datetime(m["date"])
        m = m.sort_values("date").reset_index(drop=True)

        rows = []
        for r in m.itertuples(index=False):
            feat = self._features_row(r.home_team, r.away_team, r.date, bool(r.neutral))
            feat["date"] = r.date
            feat["home_team"] = r.home_team
            feat["away_team"] = r.away_team
            feat["target"] = result_label(r.home_score, r.away_score)
            rows.append(feat)
            # actualizar estado SOLO después de registrar las features
            self._update(r.home_team, r.away_team, r.home_score, r.away_score, r.date, bool(r.neutral))

        return pd.DataFrame(rows)

    def current_features(self, home: str, away: str, neutral: bool = True, date=None) -> dict:
        """Features para un partido futuro usando el estado final del histórico."""
        if date is None:
            date = max(self.last_date.values()) if self.last_date else pd.Timestamp("2024-01-01")
        return self._features_row(home, away, pd.Timestamp(date), bool(neutral))


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Lista de columnas que son features (excluye identificadores y objetivo)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
