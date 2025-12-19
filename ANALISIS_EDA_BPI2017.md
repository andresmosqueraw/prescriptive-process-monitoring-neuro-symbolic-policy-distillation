# Análisis EDA BPI Challenge 2017 - Validación de Definiciones

## 📊 Resumen General

- **Total de casos**: 31,509
- **Total de eventos**: 1,202,267
- **Actividades únicas**: 26
- **Recursos únicos**: 149
- **Duración promedio**: 21.90 días
- **Eventos por caso (promedio)**: 38.16

## 🎯 Definición de Outcome (Éxito)

### Actividades relacionadas con "Accepted":

Según `activity_distribution.csv`:
- **`O_Accepted`**: 17,228 eventos (1.43%) - **Esta es la definición correcta**
- **`A_Accepted`**: 31,509 eventos (2.62%) - **Paso intermedio, NO es el outcome final**
- **`A_Pending`**: 17,228 eventos (1.43%) - **Coincide exactamente con O_Accepted**

### Conclusión:
✅ **Outcome = `O_Accepted`** es CORRECTO
- `A_Accepted` es un paso intermedio (todos los casos pasan por aquí)
- `A_Pending` ocurre cuando `O_Accepted` ocurre (mismo número de casos)
- `O_Accepted` representa el préstamo aceptado y listo para desembolso

## 💰 Definición de Treatment (Intervención Costosa)

### Actividades de llamada (W_Call):

Según `activity_distribution.csv`:
- **`W_Call after offers`**: 191,092 eventos (15.89%)
- **`W_Call incomplete files`**: 168,529 eventos (14.02%)
- **Total**: 359,621 eventos (29.91% de todos los eventos)

### Análisis:
✅ **Treatment = `['W_Call after offers', 'W_Call incomplete files']`** es CORRECTO

**Observación importante**: 
- 99.53% de los casos tienen al menos una de estas actividades
- Esto es **esperado** en BPI 2017, ya que casi todos los casos requieren alguna llamada manual
- No es un error, es una característica del proceso real

## 📈 Estadísticas Clave

### Distribución de Actividades (Top 10):
1. `W_Validate application`: 209,496 (17.43%)
2. `W_Call after offers`: 191,092 (15.89%) ← **Treatment**
3. `W_Call incomplete files`: 168,529 (14.02%) ← **Treatment**
4. `W_Complete application`: 148,900 (12.38%)
5. `W_Handle leads`: 47,264 (3.93%)
6. `O_Create Offer`: 42,995 (3.58%)
7. `O_Created`: 42,995 (3.58%)
8. `O_Sent (mail and online)`: 39,707 (3.30%)
9. `A_Validating`: 38,816 (3.23%)
10. `A_Accepted`: 31,509 (2.62%) ← Paso intermedio

### Actividades de Outcome:
- `O_Accepted`: 17,228 (1.43%) ← **Outcome final (éxito)**
- `A_Pending`: 17,228 (1.43%) ← Coincide con O_Accepted
- `O_Refused`: 4,695 (0.39%) ← Fracaso
- `A_Cancelled`: 10,431 (0.87%) ← Fracaso
- `A_Denied`: 3,753 (0.31%) ← Fracaso

## ✅ Validación de Definiciones Actuales

### Treatment (Intervención):
```python
treatment_activities = [
    'W_Call after offers', 
    'W_Call incomplete files'
]
```
✅ **CORRECTO** - Estas son las únicas actividades de llamada manual costosa

### Outcome (Éxito):
```python
success_activities = ['O_Accepted']
```
✅ **CORRECTO** - `O_Accepted` es el outcome final de éxito

### Propensity Score:
✅ **CORRECTO** - Usa solo `num_events` y `duration_days` (sin data leakage)

## 🔍 Observaciones Importantes

1. **Alto % de Treatment (99.53%)**:
   - Esto es **normal** en BPI 2017
   - Casi todos los casos requieren alguna intervención manual
   - No es un error en la definición

2. **Relación A_Accepted vs O_Accepted**:
   - `A_Accepted` (31,509 casos) es un paso intermedio
   - `O_Accepted` (17,228 casos) es el outcome final
   - Solo ~54.7% de los casos que pasan por `A_Accepted` llegan a `O_Accepted`

3. **Duración del proceso**:
   - Promedio: 21.90 días
   - Mediana: 19.09 días
   - Rango: 0.00 - 286.07 días

## 📝 Recomendaciones

1. ✅ **Mantener definiciones actuales** - Son correctas según el EDA
2. ✅ **El warning de 99.53% es esperado** - No es un error
3. ✅ **Propensity Score sin data leakage** - Correcto
4. ✅ **Outcome = O_Accepted** - Correcto (no usar A_Accepted)

## 🎯 Métricas del Baseline

Con las definiciones actuales:
- **Net Gain**: $8.02
- **Lift vs BAU**: -37.68%
- **% Intervenciones**: 99.53% (esperado)
- **% Violaciones**: 0.00% (con days_since_last_intervention=999)

Estas métricas son **válidas** y representan correctamente el proceso histórico de BPI 2017.
