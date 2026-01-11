# Mejoras al Benchmark de Prescriptive Process Monitoring

## 📊 Estado Actual vs. Benchmarks de LLMs

### ✅ Lo que ya está bien:

1. **Métricas relevantes**: Net Gain (OPE-IPW), Lift vs BAU, % Intervenciones, % Violaciones, AUC-Qini
2. **Evaluador común**: `BenchmarkEvaluator` unifica el cálculo de métricas
3. **Comparación con baseline**: Se compara con Business As Usual (BAU)
4. **Reproducibilidad**: Split temporal 80/20, random_state fijo

### ⚠️ Lo que falta (comparado con benchmarks de LLMs):

#### 1. **Múltiples Datasets**
- **Actual**: Solo BPI 2017 sample (40 casos)
- **Ideal**: BPI 2012, BPI 2017 (completo), BPI 2019, BPI 2020, Helpdesk, etc.
- **Razón**: Los benchmarks de LLMs usan múltiples datasets para evaluar generalización

#### 2. **Estadísticas Robustas**
- **Actual**: Un solo run por modelo
- **Ideal**: Múltiples runs (5-10) con diferentes seeds, reportar media ± std, CI 95%
- **Razón**: Los benchmarks de LLMs reportan intervalos de confianza para comparaciones justas

#### 3. **Múltiples Objetivos**
- **Actual**: Solo maximizar Net Gain
- **Ideal**: También evaluar reducción de tiempo, reducción de costos, evitar violaciones
- **Razón**: Diferentes aplicaciones tienen diferentes objetivos

#### 4. **Comparación Justa**
- **Actual**: Cada modelo puede usar diferentes features/preprocesamiento
- **Ideal**: Mismo train/test split, mismo preprocesamiento, mismos features base
- **Razón**: Los benchmarks de LLMs garantizan condiciones iguales para todos

#### 5. **Ablación Studies**
- **Actual**: No hay estudios de ablación
- **Ideal**: Variar hiperparámetros clave, features, políticas
- **Razón**: Entender qué componentes son críticos

## 🎯 Propuesta de Mejora

### Tabla de Leaderboard Mejorada

```
| Paper (Modelo) | Dataset | 💰 Net Gain ($) (OPE-IPW) | 📈 Lift vs BAU | 📉 % Intervenciones | 🛡️ % Violación | 🎯 AUC-Qini | 🐢 Latencia | 🧠 Complejidad |
|---------------|---------|---------------------------|----------------|---------------------|----------------|-------------|-------------|---------------|
| CausalForest   | BPI2017 | 0.72 ± 0.15 [0.57, 0.87]  | +424.55 ± 50.2 | 100.00              | 0.00           | -78.75      | 5.2         | Media (CPU)   |
| IPWEstimator   | BPI2017 | -0.14 ± 0.08 [-0.22, -0.06] | +37.73 ± 20.1 | 75.00               | 0.00           | -58.33      | 2.1         | Media (CPU)   |
```

**Mejoras en la tabla:**
- ✅ Intervalos de confianza (CI 95%) para Net Gain
- ✅ Desviación estándar (± std) para todas las métricas
- ✅ Múltiples datasets (una fila por modelo×dataset)
- ✅ Formato claro y comparable

### Implementación

Se ha creado `benchmark_leaderboard.py` que:

1. **Agrega resultados de múltiples runs**:
   ```python
   leaderboard = BenchmarkLeaderboard(datasets=['BPI2017'], n_runs=5)
   leaderboard.add_result(ModelResult(...))
   ```

2. **Calcula estadísticas robustas**:
   - Media y desviación estándar
   - Intervalos de confianza 95% (usando t-distribution)
   - Número de runs

3. **Genera tabla markdown**:
   - Formato compatible con GitHub
   - Fácil de incluir en papers/documentación

4. **Guarda múltiples formatos**:
   - CSV con estadísticas agregadas
   - JSON con resultados raw (todos los runs)
   - Markdown para documentación

## 📋 Checklist para un Benchmark Completo

### Fase 1: Estadísticas Robustas (Prioridad Alta)
- [x] Crear `BenchmarkLeaderboard` class
- [ ] Modificar scripts de evaluación para ejecutar múltiples runs
- [ ] Agregar cálculo de intervalos de confianza
- [ ] Reportar media ± std en tablas

### Fase 2: Múltiples Datasets (Prioridad Media)
- [ ] Agregar soporte para BPI 2012
- [ ] Agregar soporte para BPI 2019
- [ ] Agregar soporte para BPI 2020
- [ ] Crear pipeline unificado de carga de datos

### Fase 3: Comparación Justa (Prioridad Media)
- [ ] Estandarizar features base (mismo conjunto para todos)
- [ ] Estandarizar preprocesamiento
- [ ] Garantizar mismo train/test split para todos los modelos
- [ ] Documentar configuración exacta

### Fase 4: Múltiples Objetivos (Prioridad Baja)
- [ ] Agregar métrica de reducción de tiempo
- [ ] Agregar métrica de reducción de costos
- [ ] Agregar métrica de compliance (sin violaciones)

### Fase 5: Ablación Studies (Prioridad Baja)
- [ ] Variar hiperparámetros clave
- [ ] Evaluar impacto de diferentes features
- [ ] Evaluar impacto de diferentes políticas

## 🔧 Cómo Usar el Nuevo Leaderboard

### Ejemplo 1: Evaluar un modelo con múltiples runs

```python
from benchmark_leaderboard import BenchmarkLeaderboard, ModelResult

# Crear leaderboard
leaderboard = BenchmarkLeaderboard(
    datasets=['BPI2017'],
    n_runs=5,
    random_seeds=[42, 43, 44, 45, 46]
)

# Ejecutar evaluación múltiples veces
for run_id, seed in enumerate(leaderboard.random_seeds):
    # ... entrenar modelo con seed ...
    # ... evaluar modelo ...
    
    result = ModelResult(
        model_name='CausalForest',
        dataset='BPI2017',
        run_id=run_id,
        net_gain=0.72,
        lift_vs_bau=424.55,
        intervention_percentage=100.0,
        violation_percentage=0.0,
        auc_qini=-78.75,
        latency_ms=5.2,
        training_complexity='Media (CPU - Forest)'
    )
    leaderboard.add_result(result)

# Calcular estadísticas y generar tabla
df_stats = leaderboard.compute_statistics()
markdown_table = leaderboard.generate_markdown_table(df_stats)
leaderboard.save_results('results/benchmark/leaderboard', df_stats, markdown_table)
```

### Ejemplo 2: Integrar con script existente

Modificar `test_causal_effect_estimation.py` para:

1. Ejecutar cada modelo N veces con diferentes seeds
2. Agregar resultados al leaderboard
3. Generar tabla final con estadísticas

## 📊 Formato de Salida

### CSV (`leaderboard_stats.csv`)
```csv
Model,Dataset,N Runs,Net Gain ($),Net Gain CI 95%,Lift vs BAU (%),% Intervenciones,% Violaciones,AUC-Qini,Latencia (ms),Complejidad
CausalForest,BPI2017,5,0.72 ± 0.15,[0.57, 0.87],424.55 ± 50.2,100.00,0.00,-78.75,5.2,Media (CPU - Forest)
```

### Markdown (`leaderboard.md`)
```markdown
# Prescriptive Process Monitoring Benchmark Leaderboard

| Paper (Modelo) | Dataset | 💰 Net Gain ($) (OPE-IPW) | 📈 Lift vs BAU | ...
|---------------|---------|---------------------------|----------------|----
| CausalForest   | BPI2017 | 0.72 ± 0.15 [0.57, 0.87]  | +424.55 ± 50.2 | ...
```

## 🎓 Referencias: Benchmarks de LLMs

- **GLUE**: 9 tareas, múltiples datasets, reporta media y std
- **SuperGLUE**: 8 tareas, leaderboard público, métricas estandarizadas
- **MMLU**: 57 tareas, múltiples dominios, reporta accuracy por dominio
- **HELM**: Evaluación holística, múltiples métricas, múltiples datasets

**Principios clave que aplicamos:**
1. Múltiples runs para robustez estadística
2. Intervalos de confianza para comparaciones justas
3. Múltiples datasets para evaluar generalización
4. Métricas estandarizadas y reproducibles
5. Formato claro y comparable

## ✅ Conclusión

El benchmark actual es **bueno como punto de partida**, pero necesita mejoras para ser comparable con estándares de benchmarks de LLMs. Las mejoras principales son:

1. **Estadísticas robustas** (múltiples runs, CI)
2. **Múltiples datasets**
3. **Comparación justa** (mismo split, mismo preprocesamiento)
4. **Formato estándar** (tabla markdown/CSV)

El archivo `benchmark_leaderboard.py` implementa estas mejoras y puede integrarse con los scripts de evaluación existentes.

