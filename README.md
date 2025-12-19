# Prescriptive Process Monitoring: Neuro-Symbolic Policy Distillation

Este proyecto implementa un pipeline completo para monitoreo prescriptivo de procesos de negocio utilizando técnicas de aprendizaje por refuerzo (RL) y destilación de políticas neuro-simbólicas.

## 📋 Descripción

El pipeline consta de 4 fases principales:

1. **Extracción de modelos BPMN y JSON** (`extract_bpmn_json.py`): Descubre modelos de proceso desde logs de eventos usando Simod
2. **Cálculo de estado parcial** (`compute_state.py`): Calcula el estado parcial del proceso en puntos de corte temporales
3. **Entrenamiento de agente RL** (`train_agent_in_gym.py`): Entrena un agente de aprendizaje por refuerzo en un entorno "Causal-Gym" con guards simbólicos y recompensas causales
4. **Destilación de política** (`distill_policy.py`): Destila la política del agente RL en un modelo interpretable y rápido para producción

## 📁 Estructura del Proyecto

```
prescriptive-process-monitoring-neuro-symbolic-policy-distillation/
├── src/                          # Código fuente Python
│   ├── extract_bpmn_json.py     # Fase 1: Extracción BPMN/JSON
│   ├── compute_state.py          # Fase 2: Cálculo de estado parcial
│   ├── train_agent_in_gym.py     # Fase 3: Entrenamiento RL
│   └── distill_policy.py         # Fase 4: Destilación de política
├── scripts/                      # Scripts de automatización
│   ├── ejecutar-todo.sh          # Ejecuta todo el pipeline
│   └── install_dependencies.sh   # Instala dependencias
├── configs/                      # Archivos de configuración
│   └── config.yaml               # Configuración principal
├── data/                         # Datos generados
│   ├── generado-simod/           # Modelos BPMN y JSON
│   ├── generado-state/           # Estados parciales calculados
│   ├── generado-rl-train/       # Experience buffer del RL
│   └── final_policy_model.pkl   # Modelo final destilado
├── logs/                         # Logs de eventos de entrada
├── paper/                        # Documentos del paper
├── requirements.txt              # Dependencias Python
└── README.md                     # Este archivo
```

## 🚀 Inicio Rápido

### Prerrequisitos

- **Python 3.8+**
- **Docker** (para ejecutar Simod)
- **Git**

### Instalación y Ejecución

El script `ejecutar-todo.sh` automatiza todo el proceso:

```bash
# Desde la raíz del proyecto
./scripts/ejecutar-todo.sh
```

Este script:
1. ✅ Crea automáticamente el entorno virtual si no existe
2. ✅ Instala todas las dependencias
3. ✅ Ejecuta las 4 fases del pipeline en secuencia
4. ✅ Maneja errores y muestra progreso

### Configuración

Edita `configs/config.yaml` para ajustar:
- Ruta del log de eventos
- Mapeo de columnas del log
- Parámetros de Simod
- Configuración de entrenamiento RL
- Parámetros de destilación

## 🐍 Entorno Virtual

### Creación Automática

El script `ejecutar-todo.sh` crea automáticamente el entorno virtual si no existe. No necesitas crearlo manualmente.

### Activación Manual (Opcional)

Si prefieres trabajar manualmente:

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### Desactivar

```bash
deactivate
```

## 📖 Uso Detallado

### Ejecutar Pipeline Completo

```bash
./scripts/ejecutar-todo.sh
```

### Ejecutar Fases Individuales

Si prefieres ejecutar cada fase por separado:

```bash
# Activar entorno virtual
source venv/bin/activate

# Fase 1: Extraer BPMN y JSON
python src/extract_bpmn_json.py

# Fase 2: Calcular estado parcial
python src/compute_state.py

# Fase 3: Entrenar agente RL
python src/train_agent_in_gym.py

# Fase 4: Destilar política
python src/distill_policy.py
```

## 📝 Fases del Pipeline

### Fase 1: extract_bpmn_json.py

Extrae modelos BPMN y JSON desde un log de eventos usando Simod.

**Requisitos:**
- Docker instalado y funcionando
- Imagen de Simod: `nokal/simod` (se descarga automáticamente)

**Configuración:**
- Edita `configs/config.yaml` en la sección `log_config` para especificar:
  - `log_path`: Ruta al archivo CSV del log
  - `column_mapping`: Mapeo de columnas (case, activity, resource, start_time, end_time)

**Archivos generados:**
- `data/generado-simod/<log_name>.bpmn` - Modelo BPMN descubierto
- `data/generado-simod/<log_name>.json` - Parámetros estocásticos

**Ejemplo:**
```bash
python src/extract_bpmn_json.py logs/PurchasingExample.csv
```

### Fase 2: compute_state.py

Calcula el estado parcial del proceso en puntos de corte temporales usando `ongoing-bps-state-short-term`.

**Requisitos:**
- Archivos generados por Fase 1 (`.bpmn` y `.json`)
- Log de eventos original (`.csv`)

**Configuración:**
- Edita `configs/config.yaml` en la sección `state_config`:
  - `cut_points`: Lista de timestamps para calcular estados (o `null` para usar automático)
  - `column_mapping`: Mapeo de columnas (si difiere del log_config)

**Archivos generados:**
- `data/generado-state/<log_name>_process_state_<timestamp>.json` - Estados parciales

**Ejemplo:**
```bash
python src/compute_state.py
```

### Fase 3: train_agent_in_gym.py

Entrena un agente de aprendizaje por refuerzo en un entorno "Causal-Gym" con:
- **Symbolic Safety Guards**: Reglas de seguridad que restringen acciones
- **Causal Rewards**: Recompensas basadas en estimación causal (IPW)

**Requisitos:**
- Archivos generados por Fases 1 y 2
- Prosimos instalado (ver `requirements.txt`)

**Configuración:**
- Edita `configs/config.yaml` en la sección `rl_config`:
  - `episodes`: Número de episodios de entrenamiento
  - `learning_rate`: Tasa de aprendizaje
  - `epsilon`: Exploración inicial (ε-greedy)

**Archivos generados:**
- `data/generado-rl-train/experience_buffer.csv` - Buffer de experiencias

**Ejemplo:**
```bash
python src/train_agent_in_gym.py
```

### Fase 4: distill_policy.py

Destila la política del agente RL en un modelo interpretable (Decision Tree) para producción.

**Requisitos:**
- Experience buffer generado por Fase 3

**Configuración:**
- Edita `configs/config.yaml` en la sección `distill_config`:
  - `min_samples_split`: Mínimo de muestras para dividir nodo
  - `max_depth`: Profundidad máxima del árbol
  - `quality_threshold`: Umbral de calidad para filtrar experiencias

**Archivos generados:**
- `data/final_policy_model.pkl` - Modelo destilado
- Reglas SQL/IF-THEN exportadas (opcional)

**Ejemplo:**
```bash
python src/distill_policy.py
```

## ⚙️ Configuración

El archivo `configs/config.yaml` contiene toda la configuración del pipeline:

```yaml
# Rutas a repositorios externos
external_repos:
  ongoing_bps_state_path: null  # o "/ruta/a/ongoing-bps-state-short-term"
  prosimos_path: null  # o "/ruta/a/Prosimos"

# Configuración del log
log_config:
  log_path: logs/PurchasingExample.csv
  column_mapping:
    case: "caseid"
    activity: "task"
    resource: "user"
    start_time: "start_timestamp"
    end_time: "end_timestamp"

# Configuración de Simod
simod_config:
  version: 5
  control_flow:
    mining_algorithm: "sm2"
    # ... más opciones

# Configuración de estado parcial
state_config:
  cut_points: null  # null = automático

# Configuración de RL
rl_config:
  episodes: 10
  learning_rate: 0.01
  # ... más opciones

# Configuración de destilación
distill_config:
  min_samples_split: 10
  max_depth: 10
  # ... más opciones
```

## 📦 Dependencias

Las dependencias principales incluyen:

- `ongoing-process-state` - Cálculo de estado parcial
- `pix-framework` - Framework para análisis de procesos
- `pandas`, `numpy` - Manipulación de datos
- `scikit-learn` - Machine learning
- `pyyaml` - Manejo de configuración
- `Prosimos` - Simulador de procesos (instalado desde repositorio local)

Ver `requirements.txt` para la lista completa.

### Rutas a Repositorios Externos

El proyecto requiere acceso a repositorios externos (`ongoing-bps-state-short-term` y `Prosimos`). **DEBES configurarlos en `configs/config.yaml`**:

```yaml
external_repos:
  # Ruta al repositorio ongoing-bps-state-short-term
  # REQUERIDO: Debe estar configurada, no hay fallback automático
  ongoing_bps_state_path: /ruta/a/ongoing-bps-state-short-term
  
  # Ruta al repositorio Prosimos
  # REQUERIDO: Debe estar configurada, no hay fallback automático
  prosimos_path: /ruta/a/Prosimos
```

**IMPORTANTE:** Las rutas deben estar configuradas correctamente en `config.yaml`. Si no están configuradas o son incorrectas, los scripts mostrarán un error y terminarán.

**Instalación de Prosimos:**

Prosimos debe instalarse desde su repositorio local. Si está configurado en `config.yaml`, usa esa ruta:

```bash
# Edita configs/config.yaml con la ruta a Prosimos, luego:
pip install -e /ruta/a/Prosimos

# O edita requirements.txt con la ruta correcta
```

## 📊 Archivos Generados

Después de ejecutar el pipeline completo, encontrarás:

```
data/
├── generado-simod/
│   ├── <log_name>.bpmn          # Modelo BPMN
│   └── <log_name>.json           # Parámetros estocásticos
├── generado-state/
│   └── <log_name>_process_state_<timestamp>.json  # Estados parciales
├── generado-rl-train/
│   └── experience_buffer.csv     # Buffer de experiencias RL
└── final_policy_model.pkl        # Modelo final destilado
```

## 🔧 Solución de Problemas

### Error: "No se encontró el entorno virtual"

El script `ejecutar-todo.sh` crea automáticamente el entorno virtual. Si persiste el error, ejecuta manualmente:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "Docker no está corriendo"

Asegúrate de que Docker esté instalado y corriendo:

```bash
docker --version
docker ps  # Debe funcionar sin errores
```

### Error: "No se encontró ongoing-bps-state-short-term"

**Solución:**

Configura la ruta en `configs/config.yaml`:
```yaml
external_repos:
  ongoing_bps_state_path: /ruta/a/ongoing-bps-state-short-term
```

Verifica que:
- La ruta existe en el sistema de archivos
- La ruta es absoluta y correcta
- El directorio contiene el código de `ongoing-bps-state-short-term`

### Error: "Prosimos no encontrado"

**Solución:**

1. **Configurar en config.yaml:**
   ```yaml
   external_repos:
     prosimos_path: /ruta/a/Prosimos
   ```

2. **Verificar que la ruta existe y contiene el directorio 'prosimos'**

3. **Instalar Prosimos desde su repositorio:**
   ```bash
   pip install -e /ruta/a/Prosimos
   ```

## 📚 Referencias

- **Simod**: Herramienta para descubrimiento de modelos de proceso
- **Prosimos**: Simulador estocástico de procesos de negocio
- **Ongoing BPS State**: Cálculo de estado parcial de procesos

## 📄 Licencia

Ver archivo `LICENSE` para más detalles.

## 👥 Contribución

Este es un proyecto de investigación. Para contribuciones, contacta a los mantenedores.

---

**Última actualización:** 2024
