# 🎯 Guía de Ejecución: Causal-Gym

## 📋 Resumen de Enfoques

Causal-Gym tiene **DOS enfoques** para entrenar la política:

### 1. 🚀 Enfoque OFFLINE (Recomendado)
- **Ventajas**: Más rápido, supera el baseline (+109%), no requiere simulación
- **Script**: `train_from_historical.py`
- **Archivos necesarios**: Solo `bpi2017_train.csv`
- **Archivos generados**: `final_policy_model.pkl`

### 2. 🔄 Enfoque de SIMULACIÓN (Alternativo)
- **Ventajas**: Usa simulación realista con Prosimos
- **Scripts**: `extract_bpmn_json.py` → `compute_state.py` → `train_agent_in_gym.py` → `distill_policy.py`
- **Archivos necesarios**: `bpi2017_train.csv` + Docker (para Simod)
- **Archivos generados**: BPMN/JSON, estados parciales, `experience_buffer.csv`, `final_policy_model.pkl`

---

## 🎬 Orden de Ejecución Recomendado

### Opción A: Enfoque OFFLINE (Por Defecto)

```bash
# Ejecutar el pipeline completo (solo train_from_historical.py)
./scripts/ejecutar-todo.sh
```

**Flujo:**
```
1. extract_bpmn_json.py --train    [OPCIONAL - se omite]
2. compute_state.py --train        [OPCIONAL - se omite]
3. train_from_historical.py        [EJECUTA - entrena desde datos]
```

**Resultado:**
- ✅ Modelo guardado en: `results/bpi2017_train/distill/final_policy_model.pkl`
- ✅ Net Gain esperado: ~$26.56 (vs $12.68 baseline)

---

### Opción B: Enfoque de SIMULACIÓN

```bash
# Ejecutar con simulación
USE_SIMULATION=true ./scripts/ejecutar-todo.sh
```

**Flujo:**
```
1. extract_bpmn_json.py --train    [EJECUTA - genera BPMN/JSON]
2. compute_state.py --train        [EJECUTA - genera estados parciales]
3. train_agent_in_gym.py           [EJECUTA - entrena RL agent]
4. distill_policy.py                [EJECUTA - destila política]
```

**Resultado:**
- ✅ BPMN/JSON en: `results/bpi2017_train/simod/`
- ✅ Estados en: `results/bpi2017_train/state/`
- ✅ Buffer en: `results/bpi2017_train/rl/experience_buffer.csv`
- ✅ Modelo en: `results/bpi2017_train/distill/final_policy_model.pkl`

---

## 🧪 Evaluación

**Ambos enfoques** se evalúan igual:

```bash
# Evaluar el modelo entrenado
python src/benchmark/test_models/test_causal_gym.py --test
```

El script `test_causal_gym.py` detecta automáticamente qué tipo de modelo es:
- Si es un **bundle** (offline) → usa `apply_model_policy_offline()`
- Si es un **Pipeline** (simulación) → usa `apply_model_policy()`

---

## 📊 Comparación de Resultados

| Métrica | Baseline | Causal-Gym (Offline) | Causal-Gym (Simulación) |
|---------|----------|---------------------|------------------------|
| **Net Gain** | $12.68 | **$26.56** (+109%) | Variable |
| **% Intervenciones** | 99.5% | 30.2% | Variable |
| **Tiempo de entrenamiento** | N/A | ~10 segundos | ~30-60 minutos |
| **Requisitos** | N/A | Solo datos CSV | Docker + Simod |

---

## 🔧 Ejecución Manual (Si prefieres control total)

### Solo entrenar modelo offline:
```bash
python src/causal-gym/train_from_historical.py
```

### Solo evaluar:
```bash
python src/benchmark/test_models/test_causal_gym.py --test
```

### Pipeline completo de simulación (paso a paso):
```bash
# Paso 1: Extraer BPMN/JSON
python src/causal-gym/extract_bpmn_json.py --train --fast

# Paso 2: Calcular estados parciales
python src/causal-gym/compute_state.py --train

# Paso 3: Entrenar RL agent
python src/causal-gym/train_agent_in_gym.py

# Paso 4: Destilar política
python src/causal-gym/distill_policy.py

# Evaluar
python src/benchmark/test_models/test_causal_gym.py --test
```

---

## ❓ Preguntas Frecuentes

**Q: ¿Cuál enfoque debo usar?**  
A: **Offline** es recomendado porque es más rápido y supera el baseline. Usa simulación solo si necesitas validar el comportamiento en un entorno simulado.

**Q: ¿Puedo ejecutar ambos enfoques?**  
A: Sí, pero generarán modelos diferentes. El último modelo guardado será el que use `test_causal_gym.py`.

**Q: ¿Los pasos 1 y 2 son necesarios para offline?**  
A: No. El enfoque offline solo necesita `bpi2017_train.csv`.

**Q: ¿Cómo sé qué modelo se está usando?**  
A: `test_causal_gym.py` detecta automáticamente el tipo de modelo y muestra un mensaje en los logs.

---

## 📝 Notas Técnicas

- El modelo offline usa **predicción de outcome** para decidir intervenciones
- El modelo de simulación usa **RL con recompensas causales** en Prosimos
- Ambos modelos se guardan en el mismo directorio pero con diferentes estructuras internas
- `test_causal_gym.py` es compatible con ambos tipos de modelos

