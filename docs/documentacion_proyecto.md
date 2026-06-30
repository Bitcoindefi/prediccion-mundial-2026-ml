# Predicción de resultados de partidos de la Copa Mundial FIFA mediante Machine Learning

**Documento de proyecto** · Versión 1.0

---

## Control de versiones

| Versión | Fecha       | Autor        | Descripción                          |
| ------- | ----------- | ------------ | ------------------------------------ |
| 0.1     | 2026-06-20  | Equipo       | Estructura inicial del documento     |
| 1.0     | 2026-06-29  | Equipo       | Versión completa para entrega        |

---

## Índice

1. Introducción
2. Objetivos
3. Alcance
4. Definición del problema de Machine Learning
5. Requisitos funcionales
6. Requisitos no funcionales
7. Épicas, *features* e historias de usuario
8. Arquitectura del sistema
9. Dataset y diccionario de variables
10. Ingeniería de características y prevención de fuga de información
11. Modelado y evaluación
12. Flujo de Machine Learning (pipeline)
13. Riesgos y mitigaciones
14. Plan de pruebas (resumen)
15. Gestión de defectos (resumen)
16. Trazabilidad (resumen)
17. Glosario

---

## 1. Introducción

Este proyecto desarrolla un sistema de **Machine Learning supervisado** capaz de
**predecir el resultado de un partido de selecciones nacionales** (típicamente en
el contexto de una Copa del Mundo) a partir de información de desempeño reciente
de los equipos y sus jugadores.

A diferencia de un análisis exploratorio de datos —que describe y visualiza la
información sin realizar predicciones— aquí existe una **variable objetivo (`y`)**
bien definida y un conjunto de **variables predictoras (`X`)**, lo que constituye
un problema de clasificación formal con métricas de desempeño verificables.

El sistema se documenta y construye con enfoque de **producto profesional**, como
si fuera un desarrollo para un departamento de análisis de rendimiento deportivo:
incluye requisitos, épicas, historias de usuario, casos de prueba, gestión de
defectos y matriz de trazabilidad.

## 2. Objetivos

### 2.1 Objetivo general

Construir, evaluar y documentar un modelo de Machine Learning que prediga el
resultado de un partido de selecciones (**Victoria local, Empate, Victoria
visitante**) usando desempeño reciente de equipos y jugadores, garantizando la
ausencia de fuga de información.

### 2.2 Objetivos específicos

* OE-1. Definir la variable objetivo y el conjunto de variables predictoras.
* OE-2. Obtener y/o generar un dataset de partidos con atributos por selección.
* OE-3. Implementar ingeniería de características temporalmente correcta (sin
  *data leakage*).
* OE-4. Entrenar y comparar al menos cuatro algoritmos de clasificación.
* OE-5. Evaluar con métricas estándar (accuracy, precision, recall, F1, ROC AUC)
  y validación temporal.
* OE-6. Explicar las decisiones del modelo mediante importancia de variables.
* OE-7. Proveer una interfaz para predecir un partido concreto.
* OE-8. Documentar el proyecto con los entregables de ingeniería solicitados.

## 3. Alcance

### 3.1 Incluye

* Predicción del resultado de un partido individual (3 clases o binario).
* Variables de desempeño reciente de selección y jugadores, valor de mercado,
  estado físico (lesiones) y ranking FIFA.
* Pipeline reproducible de datos → features → entrenamiento → evaluación.
* Interfaz de predicción (Streamlit).
* Documentación completa de ingeniería.

### 3.2 No incluye

* Predicción de marcadores exactos o de eventos dentro del partido (goleadores,
  tarjetas, etc.).
* Predicción del campeón del torneo mediante simulación de llaves (posible
  extensión futura encadenando predicciones partido a partido).
* Apuestas o uso comercial.

### 3.3 Supuestos

* La calidad del modelo depende de la disponibilidad y calidad de los datos.
* Las fuentes externas (Transfermarkt, Sofascore) no ofrecen API pública estable;
  su uso es de bajo volumen, cacheado y con fines académicos.

## 4. Definición del problema de Machine Learning

* **Tipo de problema:** clasificación supervisada.
* **Unidad de análisis:** un partido entre dos selecciones.
* **Variable objetivo (`y`):**
  * *Multiclase* (por defecto): `2` = Victoria local, `1` = Empate,
    `0` = Victoria visitante (perspectiva del equipo local).
  * *Binario* (alternativa configurable): `1` = Victoria local, `0` = No victoria.
* **Variables predictoras (`X`):** diferencias y niveles de Elo, forma reciente,
  ranking FIFA, valor de mercado, rating de jugadores, lesiones, enfrentamientos
  directos y descanso (ver §9).
* **Métrica principal:** F1 *macro* (equilibra las tres clases). Se reportan
  además accuracy, precision, recall y ROC AUC.
* **Línea base:** predecir siempre la clase mayoritaria. El modelo debe superarla.

## 5. Requisitos funcionales

| ID    | Requisito                                                                                  | Prioridad |
| ----- | ------------------------------------------------------------------------------------------ | --------- |
| RF-01 | El sistema debe obtener datos de partidos y atributos por selección.                       | Alta      |
| RF-02 | Ante fallo de una fuente externa, debe usar un mecanismo de respaldo sin abortar.          | Alta      |
| RF-03 | El sistema debe calcular variables usando solo datos previos a la fecha del partido.        | Alta      |
| RF-04 | El sistema debe generar un dataset tabular (una fila por partido) listo para entrenar.     | Alta      |
| RF-05 | El sistema debe entrenar al menos cuatro modelos de clasificación.                         | Alta      |
| RF-06 | El sistema debe evaluar con accuracy, precision, recall, F1 y ROC AUC.                      | Alta      |
| RF-07 | El sistema debe usar validación temporal (split y cross-validation temporal).              | Alta      |
| RF-08 | El sistema debe producir matriz de confusión y curva ROC por modelo.                       | Media     |
| RF-09 | El sistema debe calcular la importancia de variables del mejor modelo.                      | Media     |
| RF-10 | El sistema debe persistir el mejor modelo y sus metadatos.                                  | Media     |
| RF-11 | El sistema debe permitir predecir un partido concreto vía interfaz.                        | Media     |
| RF-12 | El sistema debe manejar valores faltantes (imputación).                                     | Alta      |

## 6. Requisitos no funcionales

| ID     | Requisito                                                                              |
| ------ | -------------------------------------------------------------------------------------- |
| RNF-01 | **Reproducibilidad:** semillas fijas; misma entrada → mismos resultados.               |
| RNF-02 | **Robustez:** el pipeline corre de punta a punta aun sin conexión a internet.          |
| RNF-03 | **Mantenibilidad:** código modular por capas (datos, features, modelos, app).          |
| RNF-04 | **Configurabilidad:** parámetros clave en `config.yaml` sin tocar el código.           |
| RNF-05 | **Trazabilidad:** cada requisito se vincula a una feature y a una prueba.               |
| RNF-06 | **Eficiencia de red:** caché en disco y *rate limiting* en el scraping.                |
| RNF-07 | **Portabilidad:** ejecutable en Windows/Linux con Python 3.10+.                        |
| RNF-08 | **Ética/legal:** uso académico, bajo volumen; respeto a términos de las fuentes.        |

## 7. Épicas, *features* e historias de usuario

> Formato de historia: *Como [rol] quiero [acción] para [beneficio]*.
> Cada historia incluye **criterios de aceptación** (CA) verificables.

### Épica 1 — Obtención y preparación de datos

**Features:** descarga/generación de datos; caché; fallback; consolidación del
dataset crudo.

* **HU-1.1** — *Como* analista de datos *quiero* obtener resultados de partidos y
  atributos por selección *para* construir el dataset base.
  * CA-1.1.1: Se generan las tablas `matches` y `attrs` con el esquema definido.
  * CA-1.1.2: Si la fuente externa falla, se usa el respaldo y se registra el aviso.
* **HU-1.2** — *Como* analista *quiero* que las respuestas externas se cacheen
  *para* no depender de la red en cada ejecución.
  * CA-1.2.1: Una segunda ejecución no repite las peticiones ya almacenadas.

### Épica 2 — Ingeniería de características

**Features:** cálculo de Elo, forma reciente, ranking, valor de mercado, rating,
lesiones, h2h y descanso; prevención de fuga de información; imputación.

* **HU-2.1** — *Como* científico de datos *quiero* variables calculadas solo con
  información pasada *para* evitar fuga de información.
  * CA-2.1.1: El primer partido del histórico no tiene forma/h2h/descanso previos.
  * CA-2.1.2: El Elo de un equipo sin partidos previos es el valor base.
* **HU-2.2** — *Como* científico de datos *quiero* una tabla tabular por partido
  *para* alimentar los modelos.
  * CA-2.2.1: La tabla excluye identificadores del conjunto de variables.

### Épica 3 — Entrenamiento de modelos

**Features:** Regresión Logística, Árbol de Decisión, Random Forest, XGBoost;
línea base; split temporal; validación cruzada temporal.

* **HU-3.1** — *Como* científico de datos *quiero* comparar varios algoritmos
  *para* elegir el de mejor desempeño.
  * CA-3.1.1: Se entrenan al menos cuatro modelos (XGBoost opcional).
  * CA-3.1.2: Cada modelo supera la línea base de clase mayoritaria.
* **HU-3.2** — *Como* evaluador *quiero* validación temporal *para* estimar el
  desempeño de forma realista.
  * CA-3.2.1: El conjunto de prueba es posterior en el tiempo al de entrenamiento.

### Épica 4 — Evaluación y explicabilidad

**Features:** métricas, matriz de confusión, ROC, importancia de variables,
comparación de modelos.

* **HU-4.1** — *Como* evaluador *quiero* métricas estándar y gráficos *para*
  interpretar la calidad del modelo.
  * CA-4.1.1: Se generan `metrics.csv`, matrices de confusión y curvas ROC.
* **HU-4.2** — *Como* responsable técnico *quiero* la importancia de variables
  *para* explicar las decisiones del modelo.
  * CA-4.2.1: Se produce `feature_importance.png/.csv` para el mejor modelo.

### Épica 5 — Predicción e interfaz

**Features:** carga del modelo; reconstrucción de estado; predicción de un partido;
visualización de probabilidades.

* **HU-5.1** — *Como* usuario final *quiero* elegir dos selecciones y obtener las
  probabilidades de resultado *para* anticipar el partido.
  * CA-5.1.1: La interfaz muestra las probabilidades de cada clase.
  * CA-5.1.2: Las probabilidades suman 1 (±0,01).

## 8. Arquitectura del sistema

Arquitectura por capas, con un punto de unión (*seam*) entre fuentes de datos y
el resto del pipeline: tanto los datos reales como los sintéticos producen el
**mismo esquema** (`matches`, `attrs`), por lo que el resto del sistema no cambia.

```
┌─────────────────────────────────────────────────────────────┐
│                        FUENTES DE DATOS                       │
│   scrapers (Transfermarkt/Sofascore/FIFA)  |  sintético       │
│        └───────────── fallback ────────────┘                  │
│                  matches  +  attrs (mismo esquema)            │
└───────────────────────────────┬─────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│             INGENIERÍA DE CARACTERÍSTICAS (sin leakage)       │
│   FeatureBuilder: Elo, forma, ranking, valor, rating, h2h...  │
│                      → features.csv                          │
└───────────────────────────────┬─────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 ENTRENAMIENTO Y EVALUACIÓN                    │
│  LogReg · Árbol · Random Forest · XGBoost · línea base       │
│  split temporal · TimeSeriesSplit · métricas · ROC · importancia│
│            → reports/ (métricas, figuras, best_model)        │
└───────────────────────────────┬─────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   INTERFAZ DE PREDICCIÓN                      │
│         Streamlit: elegir equipos → probabilidades           │
└─────────────────────────────────────────────────────────────┘
```

**Tecnologías:** Python 3.11, pandas, scikit-learn, xgboost (opcional),
matplotlib/seaborn, requests + BeautifulSoup, Streamlit, pytest.

## 9. Dataset y diccionario de variables

### 9.1 Tablas crudas

**`matches`** (una fila por partido):

| Columna     | Tipo  | Descripción                          |
| ----------- | ----- | ------------------------------------ |
| date        | fecha | Fecha del partido                    |
| home_team   | texto | Selección local                      |
| away_team   | texto | Selección visitante                  |
| home_score  | entero| Goles del local                      |
| away_score  | entero| Goles del visitante                  |
| neutral     | bool  | Cancha neutral (sí/no)               |

**`attrs`** (fotografía mensual por selección):

| Columna           | Tipo  | Descripción                                   |
| ----------------- | ----- | --------------------------------------------- |
| date              | fecha | Fecha del snapshot                            |
| team              | texto | Selección                                     |
| market_value      | float | Valor de mercado total (millones €)           |
| market_value_var  | float | Variación respecto al snapshot anterior        |
| fifa_rank         | entero| Posición en el ranking FIFA (1 = mejor)        |
| avg_rating        | float | Rating medio de jugadores (escala Sofascore)   |
| injured           | entero| Cantidad de lesionados                         |

### 9.2 Variables del modelo (features) y justificación

| Variable                       | Nivel | Justificación                                                   |
| ------------------------------ | ----- | --------------------------------------------------------------- |
| `elo_diff`, `elo_exp_home`     | 1     | Fuerza acumulada; predictor robusto del resultado.              |
| `form_ppg_*`, `form_gd_*`      | 1     | Estado competitivo actual (forma reciente).                     |
| `fifa_rank_diff`               | 2     | Referencia histórica de nivel relativo.                         |
| `mktval_diff`, `mktval_var_*`  | 3     | Aproxima la calidad individual y su tendencia.                  |
| `rating_*`, `rating_diff`      | 2     | Desempeño reciente de los jugadores.                            |
| `injured_*`                    | 4     | Disponibilidad y rendimiento esperado.                          |
| `h2h_home_winrate`             | 5     | Historial directo entre ambas selecciones.                      |
| `rest_days_*`                  | 5     | Fatiga/descanso entre partidos.                                 |
| `neutral`                      | —     | Contexto de localía (relevante en Mundial).                     |

> **Sobre "resultados dentro del Mundial".** El desempeño observado en el propio
> torneo es muy informativo del estado actual del equipo, pero **debe tratarse
> como variable de desempeño reciente calculada solo con partidos anteriores** al
> que se predice. No se afirma que "la historia no importa": su peso lo determinan
> los datos y el modelo (ver importancia de variables).

## 10. Ingeniería de características y prevención de fuga de información

La fuga de información (*data leakage*) es el error que más infla artificialmente
las métricas en este tipo de proyectos. Para predecir un partido de fecha *D*
solo se pueden usar datos **anteriores** a *D*.

**Mecanismo de garantía:** `FeatureBuilder` procesa los partidos **ordenados por
fecha** en una sola pasada. Para cada partido (1) registra el estado **previo**
como features y (2) **solo después** actualiza el estado con el resultado. Así es
estructuralmente imposible que una feature contenga información del propio partido.
La prueba **CP-04** verifica esta propiedad automáticamente.

## 11. Modelado y evaluación

* **Modelos:** Regresión Logística (con estandarización), Árbol de Decisión,
  Random Forest, XGBoost (opcional) y una línea base de clase mayoritaria.
* **Preprocesamiento:** imputación de faltantes por mediana; estandarización para
  modelos lineales (los modelos de árbol no la requieren).
* **Validación:** *split temporal* (80% antiguo / 20% reciente) + `TimeSeriesSplit`
  para validación cruzada respetando el orden temporal.
* **Métricas:** accuracy, precision/recall/F1 *macro*, ROC AUC (One-vs-Rest).
* **Selección del mejor modelo:** mayor F1 *macro* en el conjunto de prueba.
* **Explicabilidad:** importancia de variables (nativa en árboles; coeficientes en
  regresión; permutación como respaldo).

## 12. Flujo de Machine Learning (pipeline)

1. **Configuración** (`config.yaml`): fuente, modelos, parámetros de Elo y split.
2. **Datos** (`build_dataset.py`): obtiene `matches`/`attrs` y calcula `features`.
3. **Entrenamiento** (`train.py`): entrena, valida y compara; guarda artefactos.
4. **Evaluación** (`evaluate.py`): métricas y figuras.
5. **Predicción** (`streamlit_app.py`): interfaz para un partido concreto.

## 13. Riesgos y mitigaciones

| ID    | Riesgo                                              | Prob. | Impacto | Mitigación                                                        |
| ----- | --------------------------------------------------- | ----- | ------- | ----------------------------------------------------------------- |
| R-01  | Fuentes externas sin API estable / cambios de HTML  | Alta  | Alto    | Caché + *retries* + **fallback a datos sintéticos** sin abortar.  |
| R-02  | Fuga de información temporal                         | Media | Alto    | Procesamiento ordenado por fecha; prueba automática CP-04.        |
| R-03  | Pocos datos si se limita al Mundial                 | Alta  | Alto    | Entrenar con todos los partidos internacionales; validar en Mundial.|
| R-04  | Datos faltantes (lesiones, ratings)                 | Media | Medio   | Imputación por mediana; variables tolerantes a NaN.               |
| R-05  | Restricciones legales/términos de uso del scraping  | Media | Medio   | Bajo volumen, uso académico, posibilidad de dataset licenciado.   |
| R-06  | Sobreajuste                                         | Media | Medio   | Validación temporal, regularización, límites de profundidad.      |
| R-07  | Desbalance de clases (empates)                      | Media | Medio   | F1 macro como métrica; análisis de matriz de confusión.           |

## 14. Plan de pruebas (resumen)

El detalle está en `casos_de_prueba.md`. Cubre: generación de datos, esquema de
features, **no fuga de información**, validez del objetivo, manejo de faltantes,
entrenamiento, validación cruzada y predicción. Las pruebas automatizadas se
ejecutan con `pytest -q`.

## 15. Gestión de defectos (resumen)

El registro completo está en `gestion_defectos.md`, con ID, descripción,
severidad, prioridad, estado, responsable, evidencia y fecha.

## 16. Trazabilidad (resumen)

La matriz completa está en `matriz_trazabilidad.md` y vincula cada requisito con
su épica/feature, el componente de código que lo implementa y el caso de prueba
que lo verifica.

## 17. Glosario

* **Feature:** variable de entrada del modelo.
* **Data leakage:** uso indebido de información futura al construir variables.
* **Elo:** sistema de puntuación de fuerza relativa actualizado tras cada partido.
* **F1 macro:** promedio no ponderado del F1 por clase.
* **ROC AUC (OvR):** área bajo la curva ROC en esquema uno-contra-el-resto.
* **TimeSeriesSplit:** validación cruzada que respeta el orden temporal.
