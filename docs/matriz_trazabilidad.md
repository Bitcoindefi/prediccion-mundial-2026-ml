# Matriz de trazabilidad

Proyecto: Predicción de resultados de partidos del Mundial · Versión 1.0

Vincula cada **requisito** con su **épica/feature**, el **componente de código**
que lo implementa y el **caso de prueba** que lo verifica. Permite comprobar la
cobertura: ningún requisito queda sin implementación ni sin prueba.

## Requisito → Épica → Componente → Prueba

| Requisito | Épica / HU        | Componente de código (implementación)                          | Caso(s) de prueba |
| --------- | ----------------- | -------------------------------------------------------------- | ----------------- |
| RF-01     | E1 / HU-1.1       | `src/data/build_dataset.py`, `src/data/synthetic.py`           | CP-01             |
| RF-02     | E1 / HU-1.1       | `src/data/build_dataset.py` (`get_raw_tables`, fallback)       | CP-09             |
| RF-03     | E2 / HU-2.1       | `src/features/engineering.py` (`FeatureBuilder`)               | CP-04, CP-11      |
| RF-04     | E2 / HU-2.2       | `src/features/engineering.py`, `build_dataset.py`              | CP-01, CP-02, CP-03 |
| RF-05     | E3 / HU-3.1       | `src/models/train.py` (`build_models`)                         | CP-05, CP-06      |
| RF-06     | E4 / HU-4.1       | `src/models/evaluate.py` (`compute_metrics`)                   | CP-06             |
| RF-07     | E3 / HU-3.2       | `src/models/train.py` (`time_split`, `TimeSeriesSplit`)        | CP-05, CP-07      |
| RF-08     | E4 / HU-4.1       | `src/models/evaluate.py` (`plot_confusion`, `plot_roc`)        | CP-06             |
| RF-09     | E4 / HU-4.2       | `src/models/train.py` (`extract_importance`)                   | CP-06             |
| RF-10     | E4 / HU-4.1       | `src/models/train.py` (`joblib.dump`, `model_meta.json`)       | CP-06             |
| RF-11     | E5 / HU-5.1       | `src/app/streamlit_app.py`                                     | CP-10, CP-11      |
| RF-12     | E2 / HU-2.1       | `src/models/train.py` (`SimpleImputer`)                        | CP-08, CP-11      |
| RNF-01    | Transversal       | `src/config.py` (`set_seeds`)                                  | CP-12             |
| RNF-02    | E1                | `src/scraping/base.py`, `build_dataset.py` (fallback)          | CP-09             |
| RNF-03    | Transversal       | Estructura modular `src/`                                     | CP-03             |
| RNF-04    | Transversal       | `config.yaml`, `src/config.py`                                | CP-06             |
| RNF-05    | Transversal       | Esta matriz + `tests/`                                        | CP-04             |
| RNF-06    | E1                | `src/scraping/base.py` (caché, rate limit)                    | CP-09             |
| RNF-07    | Transversal       | Código portable (Python 3.10+)                                | CP-06             |
| RNF-08    | E1                | `src/scraping/*` (bajo volumen, avisos)                       | —                 |

## Cobertura

* **Requisitos funcionales cubiertos por prueba:** 12 / 12.
* **Requisitos no funcionales cubiertos por prueba:** 7 / 8 (RNF-08 es de
  cumplimiento/ético, se verifica por revisión, no por prueba automatizada).
* **Casos de prueba sin requisito asociado:** 0.

## Épica → Casos de prueba

| Épica                                    | Casos de prueba             |
| ---------------------------------------- | --------------------------- |
| E1 — Obtención y preparación de datos    | CP-01, CP-09                |
| E2 — Ingeniería de características        | CP-02, CP-03, CP-04, CP-11  |
| E3 — Entrenamiento de modelos            | CP-05, CP-06, CP-07         |
| E4 — Evaluación y explicabilidad         | CP-06                       |
| E5 — Predicción e interfaz               | CP-10, CP-11                |
| Transversal (no funcionales)             | CP-08, CP-12                |
