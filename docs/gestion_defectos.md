# Gestión de defectos

Proyecto: Predicción de resultados de partidos del Mundial · Versión 1.0

**Severidad:** Crítica · Alta · Media · Baja
**Prioridad:** Alta · Media · Baja
**Estado:** Abierto · En progreso · Resuelto · Cerrado · Rechazado

> Tabla de ejemplo con defectos representativos detectados durante el desarrollo.
> Actualizar a medida que se ejecuten las pruebas y se registren nuevos hallazgos.

| ID     | Descripción                                                                 | Caso de prueba | Severidad | Prioridad | Estado    | Responsable | Fecha       | Evidencia / Solución                                                                 |
| ------ | --------------------------------------------------------------------------- | -------------- | --------- | --------- | --------- | ----------- | ----------- | ------------------------------------------------------------------------------------ |
| DEF-01 | Las features de forma incluían el partido actual (fuga de información).      | CP-04          | Crítica   | Alta      | Resuelto  | Equipo      | 2026-06-22  | Se reordenó el flujo: registrar features **antes** de actualizar el estado.          |
| DEF-02 | `roc_auc` fallaba en multiclase por formato de `predict_proba`.             | CP-06          | Alta      | Alta      | Resuelto  | Equipo      | 2026-06-23  | Se usa `multi_class="ovr"` con `labels=clases` y manejo de excepción.                |
| DEF-03 | El pipeline abortaba si `xgboost` no estaba instalado.                       | CP-06          | Media     | Media     | Resuelto  | Equipo      | 2026-06-23  | Import protegido; si falta, se omite el modelo con aviso.                            |
| DEF-04 | Errores de red en scraping detenían toda la ejecución.                      | CP-09          | Alta      | Alta      | Resuelto  | Equipo      | 2026-06-24  | Caché + reintentos + fallback automático a datos sintéticos.                        |
| DEF-05 | Valores faltantes en atributos rompían el entrenamiento.                    | CP-08          | Alta      | Alta      | Resuelto  | Equipo      | 2026-06-24  | `SimpleImputer(strategy="median")` en todos los pipelines.                           |
| DEF-06 | Split aleatorio mezclaba fechas y sobreestimaba el desempeño.               | CP-07          | Alta      | Alta      | Resuelto  | Equipo      | 2026-06-25  | Se reemplazó por split temporal + `TimeSeriesSplit`.                                 |
| DEF-07 | La interfaz fallaba al elegir el mismo equipo como local y visitante.       | CP-10          | Baja      | Baja      | Resuelto  | Equipo      | 2026-06-26  | Validación que exige dos selecciones distintas.                                     |
| DEF-08 | Predicción con selecciones sin historial generaba excepción.               | CP-11          | Media     | Media     | Resuelto  | Equipo      | 2026-06-26  | Features `NaN` imputadas; estado base de Elo asegurado.                              |
| DEF-09 | Probabilidades de la interfaz no se mostraban ordenadas.                    | CP-10          | Baja      | Baja      | Cerrado   | Equipo      | 2026-06-27  | Se ordena el `DataFrame` por probabilidad descendente.                              |
| DEF-10 | Acentos mal codificados al guardar CSV en Windows.                          | CP-01          | Baja      | Media     | Resuelto  | Equipo      | 2026-06-27  | Se fuerza `encoding="utf-8"` en todas las escrituras.                                |

## Procedimiento de gestión

1. **Detección:** al ejecutar un caso de prueba o durante el desarrollo.
2. **Registro:** alta en esta tabla con ID correlativo y caso de prueba asociado.
3. **Clasificación:** asignar severidad y prioridad.
4. **Resolución:** corrección en el código y verificación con el caso de prueba.
5. **Cierre:** estado `Cerrado` con evidencia (salida de prueba, captura o commit).
