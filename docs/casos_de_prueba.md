# Casos de prueba

Proyecto: Predicción de resultados de partidos del Mundial · Versión 1.0

Las pruebas automatizadas viven en `tests/test_pipeline.py` y se ejecutan con:

```powershell
pytest -q
```

Estado: ⬜ Pendiente · ✅ Aprobado · ❌ Fallido (completar tras la ejecución).

---

## CP-01 — Generación/carga de datos

* **Requisito:** RF-01, RF-04
* **Prueba automatizada:** `test_synthetic_shapes`
* **Precondición:** configuración válida en `config.yaml`.
* **Pasos:** generar las tablas `matches` y `attrs`.
* **Datos de entrada:** `n_teams=16`, 2015–2018, `matches_per_year=300`.
* **Resultado esperado:** `matches` tiene 1200 filas y el esquema definido;
  `attrs` contiene las columnas de atributos; `home_team != away_team` en todas.
* **Estado:** ⬜
* **Evidencia:** salida de `pytest` + `data/processed/*.csv`.

## CP-02 — Etiqueta objetivo correcta

* **Requisito:** RF-04
* **Prueba automatizada:** `test_result_label`, `test_target_distribution_valid`
* **Pasos:** calcular la etiqueta para marcadores conocidos y validar el dominio.
* **Resultado esperado:** `(2,0)→2`, `(1,1)→1`, `(0,3)→0`; objetivo ⊆ {0,1,2} y
  ninguna clase supera el 97% (no degenerado).
* **Estado:** ⬜

## CP-03 — Esquema de features

* **Requisito:** RF-04, RNF-03
* **Prueba automatizada:** `test_feature_columns_exclude_ids`
* **Resultado esperado:** las columnas de features excluyen `date`, `home_team`,
  `away_team` y `target`; incluyen `elo_diff` y `form_ppg_diff`.
* **Estado:** ⬜

## CP-04 — No fuga de información (crítico)

* **Requisito:** RF-03, RNF-05
* **Prueba automatizada:** `test_no_leakage_first_match`
* **Pasos:** construir features y tomar el primer partido del histórico.
* **Resultado esperado:** `form_ppg_home/away`, `rest_days_home` y
  `h2h_home_winrate` son `NaN`; `elo_home == elo_away == 1500`.
* **Justificación:** demuestra que ninguna feature usa información del propio
  partido ni de partidos futuros.
* **Estado:** ⬜

## CP-05 — El modelo supera la línea base

* **Requisito:** RF-05, RF-07
* **Prueba automatizada:** `test_model_beats_baseline`
* **Pasos:** split temporal; entrenar Random Forest; comparar contra la clase
  mayoritaria.
* **Resultado esperado:** accuracy del modelo ≥ accuracy de la línea base.
* **Estado:** ⬜

## CP-06 — Entrenamiento y artefactos (manual)

* **Requisito:** RF-05, RF-06, RF-08, RF-09, RF-10
* **Pasos:** `python run_all.py`.
* **Resultado esperado:** se crean `reports/metrics.csv`,
  `reports/confusion_*.png`, `reports/roc_*.png`,
  `reports/feature_importance.png`, `reports/best_model.joblib` y
  `reports/model_meta.json`; cada modelo aparece en `metrics.csv`.
* **Estado:** ⬜
* **Evidencia:** carpeta `reports/`.

## CP-07 — Validación cruzada temporal (manual)

* **Requisito:** RF-07
* **Pasos:** ejecutar el entrenamiento y revisar la columna `cv_acc`.
* **Resultado esperado:** `cv_acc` se calcula con `TimeSeriesSplit` (5 splits)
  para cada modelo y no es `NaN`.
* **Estado:** ⬜

## CP-08 — Manejo de valores faltantes

* **Requisito:** RF-12
* **Pasos:** introducir `NaN` en columnas de atributos y entrenar.
* **Resultado esperado:** la imputación por mediana evita errores; el pipeline
  finaliza correctamente.
* **Estado:** ⬜

## CP-09 — Fallback de fuente de datos

* **Requisito:** RF-02, RNF-02
* **Pasos:** configurar `data.source: scrape` sin integración disponible.
* **Resultado esperado:** el sistema registra el aviso y continúa con datos
  sintéticos; **no aborta**.
* **Estado:** ⬜

## CP-10 — Predicción por interfaz

* **Requisito:** RF-11
* **Pasos:** `streamlit run src/app/streamlit_app.py`; elegir dos selecciones y
  predecir.
* **Resultado esperado:** se muestran probabilidades por clase que suman 1 (±0,01)
  y un gráfico de barras.
* **Estado:** ⬜
* **Evidencia:** captura de pantalla de la interfaz.

## CP-11 — Equipos sin historial suficiente

* **Requisito:** RF-03, RF-12
* **Pasos:** predecir un partido entre dos selecciones con muy pocos partidos
  previos.
* **Resultado esperado:** las features de forma/h2h resultan `NaN` y son imputadas;
  el sistema devuelve una predicción válida sin error.
* **Estado:** ⬜

## CP-12 — Reproducibilidad

* **Requisito:** RNF-01
* **Pasos:** ejecutar dos veces el pipeline con la misma semilla.
* **Resultado esperado:** `metrics.csv` idéntico entre ejecuciones.
* **Estado:** ⬜
