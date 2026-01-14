#!/bin/bash
# Script para ejecutar el pipeline de Causal-Gym
#
# ENFOQUE PRINCIPAL (OFFLINE - RECOMENDADO):
#   Solo ejecuta train_from_historical.py que entrena directamente desde datos históricos.
#   Este enfoque supera el baseline (+109%) y es más rápido.
#
# ENFOQUE ALTERNATIVO (SIMULACIÓN):
#   Si quieres usar simulación con Prosimos, ejecuta:
#   1) extract_bpmn_json.py --train
#   2) compute_state.py --train
#   3) train_agent_in_gym.py
#   4) distill_policy.py
#
# Para evaluar cualquier enfoque:
#   python src/benchmark/test_models/test_causal_gym.py --test

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_DIR="$PROJECT_ROOT/venv"
SRC_DIR="$PROJECT_ROOT/src/causal-gym"

# Verificar que existe el entorno virtual
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ No se encontró el entorno virtual en: $VENV_DIR"
    echo "   Creando entorno virtual..."
    python3 -m venv "$VENV_DIR" || {
        echo "   Error al crear el entorno virtual"
        exit 1
    }
    echo "✅ Entorno virtual creado"
    echo "   Instalando dependencias..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$PROJECT_ROOT/requirements.txt" || {
        echo "   Error al instalar dependencias"
    exit 1
    }
    deactivate
    echo "✅ Dependencias instaladas"
fi

# Activar entorno virtual
source "$VENV_DIR/bin/activate"
echo "✅ Entorno virtual activado: $VENV_DIR"
echo "   Python: $(which python)"
echo "   Versión: $(python --version)"
echo ""

# Cambiar al directorio raíz del proyecto
cd "$PROJECT_ROOT"

# Agregar src/ al PYTHONPATH para que los módulos encuentren utils
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Función para manejar errores
handle_error() {
    echo ""
    echo "❌ Error en: $1"
    echo "   El proceso se detuvo"
    deactivate
    exit 1
}

# Verificar si se quiere usar enfoque de simulación (requiere pasos 1 y 2)
USE_SIMULATION=${USE_SIMULATION:-false}

if [ "$USE_SIMULATION" = "true" ]; then
    # Paso 1: Ejecutar extract_bpmn_json.py (usando --train para usar bpi2017_train.csv)
    echo "=================================================================================="
    echo "📋 PASO 1: Extrayendo BPMN y JSON con Simod (usando train set)"
    echo "=================================================================================="
    echo ""

    if [ -f "$SRC_DIR/extract_bpmn_json.py" ]; then
        PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH" python "$SRC_DIR/extract_bpmn_json.py" --train --fast || handle_error "extract_bpmn_json.py"
    else
        echo "❌ No se encontró: $SRC_DIR/extract_bpmn_json.py"
        deactivate
        exit 1
    fi

    echo ""
    echo "=================================================================================="
    echo "✅ PASO 1 COMPLETADO"
    echo "=================================================================================="
    echo ""

    # Paso 2: Ejecutar compute_state.py (usando --train para usar bpi2017_train.csv)
    echo "=================================================================================="
    echo "📋 PASO 2: Calculando estado parcial del proceso (usando train set)"
    echo "=================================================================================="
    echo ""

    if [ -f "$SRC_DIR/compute_state.py" ]; then
        PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH" python "$SRC_DIR/compute_state.py" --train || handle_error "compute_state.py"
    else
        echo "❌ No se encontró: $SRC_DIR/compute_state.py"
        deactivate
        exit 1
    fi

    echo ""
    echo "=================================================================================="
    echo "✅ PASO 2 COMPLETADO"
    echo "=================================================================================="
    echo ""
else
    echo "=================================================================================="
    echo "⏭️  PASOS 1 y 2 OMITIDOS (no necesarios para enfoque offline)"
    echo "=================================================================================="
    echo "💡 Los pasos 1 y 2 (extract_bpmn_json y compute_state) solo son necesarios"
    echo "   para el enfoque de simulación. El enfoque offline los omite."
    echo ""
    echo "   Para usar simulación, ejecuta: USE_SIMULATION=true ./scripts/ejecutar-todo.sh"
    echo ""
fi

# Paso 3: Ejecutar train_from_historical.py (ENFOQUE OFFLINE - RECOMENDADO)
echo "=================================================================================="
echo "📋 PASO 3: Entrenando política desde datos históricos (OFFLINE RL)"
echo "=================================================================================="
echo ""
echo "💡 Este enfoque entrena directamente desde datos históricos sin simulación."
echo "   Ventajas: Más rápido, supera el baseline (+109%), no requiere BPMN/JSON."
echo ""

if [ -f "$SRC_DIR/train_from_historical.py" ]; then
    PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH" python "$SRC_DIR/train_from_historical.py" || handle_error "train_from_historical.py"
else
    echo "❌ No se encontró: $SRC_DIR/train_from_historical.py"
    deactivate
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ PASO 3 COMPLETADO"
echo "=================================================================================="
echo ""
echo "💡 NOTA: Los pasos 1 y 2 (extract_bpmn_json y compute_state) NO son necesarios"
echo "   para el enfoque offline. Solo se ejecutaron por compatibilidad."
echo ""
echo "💡 Si prefieres usar el enfoque de SIMULACIÓN (más lento, requiere BPMN/JSON),"
echo "   puedes ejecutar manualmente:"
echo "   - train_agent_in_gym.py (requiere pasos 1 y 2)"
echo "   - distill_policy.py"
echo ""

# Leer rutas desde config.yaml usando Python (antes de desactivar venv)
CONFIG_FILE="$PROJECT_ROOT/configs/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  No se encontró config.yaml, usando rutas por defecto"
    SIMOD_DIR="data/generado-simod"
    STATE_DIR="data/generado-state"
    RL_BUFFER="data/generado-rl-train/experience_buffer.csv"
    DISTILL_MODEL="data/final_policy_model.pkl"
else
    # Leer rutas desde config.yaml (venv aún activo)
    SIMOD_DIR=$(python -c "
import yaml
import os
import sys

config_path = '$CONFIG_FILE'
if not os.path.exists(config_path):
    sys.exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

log_config = config.get('log_config', {})
script_config = config.get('script_config', {})
output_dir_base = script_config.get('output_dir')

# Obtener nombre del log
# Cuando se usa --train, el log es bpi2017_train.csv, así que usar ese nombre
train_log_path = os.path.join('$PROJECT_ROOT', 'logs', 'BPI2017', 'processed', 'bpi2017_train.csv')
if os.path.exists(train_log_path):
    # Usar el nombre del log de train
    log_name = 'bpi2017_train'
else:
    # Fallback: leer desde config.yaml
    log_path = log_config.get('log_path')
    if log_path:
        if not os.path.isabs(log_path):
            base_dir = '$PROJECT_ROOT'
            log_path = os.path.join(base_dir, log_path)
        log_name = os.path.splitext(os.path.basename(log_path))[0]
        if log_name.endswith('.xes'):
            log_name = os.path.splitext(log_name)[0]
    else:
        log_name = 'default'

if output_dir_base:
    # Construir ruta con nombre del log: results/{log_name}/simod/
    # Retornar ruta RELATIVA al PROJECT_ROOT (misma lógica que build_output_path)
    output_dir_base = output_dir_base.rstrip('/')
    if os.path.isabs(output_dir_base):
        parts = output_dir_base.split(os.sep)
        if len(parts) > 1:
            parts.insert(-1, log_name)
            output_dir = os.sep.join(parts)
        else:
            output_dir = os.path.join(output_dir_base, log_name, 'simod')
        # Convertir a relativa
        try:
            output_dir = os.path.relpath(output_dir, '$PROJECT_ROOT')
        except ValueError:
            # Si no se puede convertir (diferentes drives en Windows), mantener absoluta
            pass
    else:
        # Ruta relativa: "results/simod" -> "results/{log_name}/simod"
        parts = output_dir_base.split('/')
        if len(parts) >= 2:
            base_dir_name = parts[0]  # "results"
            output_dir = os.path.join(base_dir_name, log_name, 'simod')
        elif len(parts) == 1:
            # Solo "simod" -> "{log_name}/simod"
            output_dir = os.path.join(log_name, 'simod')
        else:
            output_dir = os.path.join(log_name, 'simod')
    print(output_dir)
else:
    print('data/generado-simod')
" 2>/dev/null || echo "data/generado-simod")
    
    STATE_DIR=$(python -c "
import yaml
import os
import sys

config_path = '$CONFIG_FILE'
if not os.path.exists(config_path):
    sys.exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

log_config = config.get('log_config', {})
script_config = config.get('script_config', {})
state_output_dir_base = script_config.get('state_output_dir')

# Obtener nombre del log
# Cuando se usa --train, el log es bpi2017_train.csv, así que usar ese nombre
train_log_path = os.path.join('$PROJECT_ROOT', 'logs', 'BPI2017', 'processed', 'bpi2017_train.csv')
if os.path.exists(train_log_path):
    # Usar el nombre del log de train
    log_name = 'bpi2017_train'
else:
    # Fallback: leer desde config.yaml
    log_path = log_config.get('log_path')
    if log_path:
        if not os.path.isabs(log_path):
            base_dir = '$PROJECT_ROOT'
            log_path = os.path.join(base_dir, log_path)
        log_name = os.path.splitext(os.path.basename(log_path))[0]
        if log_name.endswith('.xes'):
            log_name = os.path.splitext(log_name)[0]
    else:
        log_name = 'default'

if state_output_dir_base:
    # Construir ruta con nombre del log: results/{log_name}/state/
    # Retornar ruta RELATIVA al PROJECT_ROOT (misma lógica que build_output_path)
    state_output_dir_base = state_output_dir_base.rstrip('/')
    if os.path.isabs(state_output_dir_base):
        parts = state_output_dir_base.split(os.sep)
        if len(parts) > 1:
            parts.insert(-1, log_name)
            state_output_dir = os.sep.join(parts)
        else:
            state_output_dir = os.path.join(state_output_dir_base, log_name, 'state')
        # Convertir a relativa
        try:
            state_output_dir = os.path.relpath(state_output_dir, '$PROJECT_ROOT')
        except ValueError:
            pass
    else:
        # Ruta relativa: "results/state" -> "results/{log_name}/state"
        parts = state_output_dir_base.split('/')
        if len(parts) >= 2:
            base_dir_name = parts[0]  # "results"
            state_output_dir = os.path.join(base_dir_name, log_name, 'state')
        elif len(parts) == 1:
            state_output_dir = os.path.join(log_name, 'state')
        else:
            state_output_dir = os.path.join(log_name, 'state')
    print(state_output_dir)
else:
    print('data/generado-state')
" 2>/dev/null || echo "data/generado-state")
    
    RL_BUFFER=$(python -c "
import yaml
import os
import sys

config_path = '$CONFIG_FILE'
if not os.path.exists(config_path):
    sys.exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

log_config = config.get('log_config', {})
script_config = config.get('script_config', {})
distill_config = config.get('distill_config', {})
rl_output_dir_base = script_config.get('rl_output_dir')
input_csv = distill_config.get('input_csv')

# Obtener nombre del log
# Cuando se usa --train, el log es bpi2017_train.csv, así que usar ese nombre
train_log_path = os.path.join('$PROJECT_ROOT', 'logs', 'BPI2017', 'processed', 'bpi2017_train.csv')
if os.path.exists(train_log_path):
    # Usar el nombre del log de train
    log_name = 'bpi2017_train'
else:
    # Fallback: leer desde config.yaml
    log_path = log_config.get('log_path')
    if log_path:
        if not os.path.isabs(log_path):
            base_dir = '$PROJECT_ROOT'
            log_path = os.path.join(base_dir, log_path)
        log_name = os.path.splitext(os.path.basename(log_path))[0]
        if log_name.endswith('.xes'):
            log_name = os.path.splitext(log_name)[0]
    else:
        log_name = 'default'

# Preferir input_csv de distill_config, luego construir desde rl_output_dir_base
if input_csv:
    # Si es absoluta, convertir a relativa al PROJECT_ROOT
    if os.path.isabs(input_csv):
        try:
            input_csv = os.path.relpath(input_csv, '$PROJECT_ROOT')
        except ValueError:
            pass
    print(input_csv)
elif rl_output_dir_base:
    # Construir ruta con nombre del log: results/{log_name}/rl/experience_buffer.csv
    # Retornar ruta RELATIVA al PROJECT_ROOT (misma lógica que build_output_path)
    rl_output_dir_base = rl_output_dir_base.rstrip('/')
    if os.path.isabs(rl_output_dir_base):
        parts = rl_output_dir_base.split(os.sep)
        if len(parts) > 1:
            parts.insert(-1, log_name)
            rl_output_dir = os.sep.join(parts)
        else:
            rl_output_dir = os.path.join(rl_output_dir_base, log_name, 'rl')
        # Convertir a relativa
        try:
            rl_output_dir = os.path.relpath(rl_output_dir, '$PROJECT_ROOT')
        except ValueError:
            pass
    else:
        # Ruta relativa: "results/rl" -> "results/{log_name}/rl"
        parts = rl_output_dir_base.split('/')
        if len(parts) >= 2:
            base_dir_name = parts[0]  # "results"
            rl_output_dir = os.path.join(base_dir_name, log_name, 'rl')
        elif len(parts) == 1:
            rl_output_dir = os.path.join(log_name, 'rl')
        else:
            rl_output_dir = os.path.join(log_name, 'rl')
    print(os.path.join(rl_output_dir, 'experience_buffer.csv'))
else:
    print('data/generado-rl-train/experience_buffer.csv')
" 2>/dev/null || echo "data/generado-rl-train/experience_buffer.csv")
    
    DISTILL_MODEL=$(python -c "
import yaml
import os
import sys

config_path = '$CONFIG_FILE'
if not os.path.exists(config_path):
    sys.exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

log_config = config.get('log_config', {})
script_config = config.get('script_config', {})
distill_config = config.get('distill_config', {})
output_model = distill_config.get('output_model')
distill_output_dir_base = script_config.get('distill_output_dir')

# Obtener nombre del log
# Cuando se usa --train, el log es bpi2017_train.csv, así que usar ese nombre
train_log_path = os.path.join('$PROJECT_ROOT', 'logs', 'BPI2017', 'processed', 'bpi2017_train.csv')
if os.path.exists(train_log_path):
    # Usar el nombre del log de train
    log_name = 'bpi2017_train'
else:
    # Fallback: leer desde config.yaml
    log_path = log_config.get('log_path')
    if log_path:
        if not os.path.isabs(log_path):
            base_dir = '$PROJECT_ROOT'
            log_path = os.path.join(base_dir, log_path)
        log_name = os.path.splitext(os.path.basename(log_path))[0]
        if log_name.endswith('.xes'):
            log_name = os.path.splitext(log_name)[0]
    else:
        log_name = 'default'

if output_model:
    # Si es absoluta, convertir a relativa al PROJECT_ROOT
    if os.path.isabs(output_model):
        try:
            output_model = os.path.relpath(output_model, '$PROJECT_ROOT')
        except ValueError:
            pass  # Si no se puede convertir, mantener absoluta
    print(output_model)
elif distill_output_dir_base:
    # Construir ruta con nombre del log: results/{log_name}/distill/final_policy_model.pkl
    # Retornar ruta RELATIVA al PROJECT_ROOT (el bash la convertirá a absoluta)
    distill_output_dir_base = distill_output_dir_base.rstrip('/')
    if os.path.isabs(distill_output_dir_base):
        parts = distill_output_dir_base.split(os.sep)
        if len(parts) > 1:
            parts.insert(-1, log_name)
            distill_output_dir = os.sep.join(parts)
        else:
            distill_output_dir = os.path.join(distill_output_dir_base, log_name, 'distill')
        # Convertir a relativa si es posible
        try:
            distill_output_dir = os.path.relpath(distill_output_dir, '$PROJECT_ROOT')
        except ValueError:
            pass
    else:
        parts = distill_output_dir_base.split('/')
        if len(parts) >= 2:
            base_dir_name = parts[0]
            distill_output_dir = os.path.join(base_dir_name, log_name, 'distill')
        else:
            distill_output_dir = os.path.join(log_name, 'distill')
    print(os.path.join(distill_output_dir, 'final_policy_model.pkl'))
else:
    print('data/final_policy_model.pkl')
" 2>/dev/null || echo "data/final_policy_model.pkl")
fi

# Desactivar entorno virtual
deactivate

echo ""
echo "🎉 ¡Pipeline completado exitosamente!"
echo ""
echo "📁 Archivos generados:"

# Resolver rutas relativas a absolutas si es necesario
if [ ! -z "$SIMOD_DIR" ] && [ "${SIMOD_DIR#/}" = "$SIMOD_DIR" ]; then
    SIMOD_DIR="$PROJECT_ROOT/$SIMOD_DIR"
fi
if [ ! -z "$STATE_DIR" ] && [ "${STATE_DIR#/}" = "$STATE_DIR" ]; then
    STATE_DIR="$PROJECT_ROOT/$STATE_DIR"
fi
if [ ! -z "$RL_BUFFER" ] && [ "${RL_BUFFER#/}" = "$RL_BUFFER" ]; then
    RL_BUFFER="$PROJECT_ROOT/$RL_BUFFER"
fi
if [ ! -z "$DISTILL_MODEL" ] && [ "${DISTILL_MODEL#/}" = "$DISTILL_MODEL" ]; then
    DISTILL_MODEL="$PROJECT_ROOT/$DISTILL_MODEL"
fi

# Mostrar rutas (relativas al proyecto si es posible)
# Usar python del sistema para convertir rutas (no requiere venv)
rel_simod_dir=$(python3 -c "import os; print(os.path.relpath('$SIMOD_DIR', '$PROJECT_ROOT'))" 2>/dev/null || echo "$SIMOD_DIR")
rel_state_dir=$(python3 -c "import os; print(os.path.relpath('$STATE_DIR', '$PROJECT_ROOT'))" 2>/dev/null || echo "$STATE_DIR")
rel_rl_buffer=$(python3 -c "import os; print(os.path.relpath('$RL_BUFFER', '$PROJECT_ROOT'))" 2>/dev/null || echo "$RL_BUFFER")
rel_distill_model=$(python3 -c "import os; print(os.path.relpath('$DISTILL_MODEL', '$PROJECT_ROOT'))" 2>/dev/null || echo "$DISTILL_MODEL")

# Mostrar archivos según el enfoque usado
if [ "$USE_SIMULATION" = "true" ]; then
    # Enfoque de simulación: mostrar BPMN, estados, buffer y modelo destilado
    echo "   • BPMN y JSON: $rel_simod_dir"
    if [ -d "$SIMOD_DIR" ]; then
        bpmn_count=$(find "$SIMOD_DIR" -name "*.bpmn" 2>/dev/null | wc -l)
        json_count=$(find "$SIMOD_DIR" -name "*.json" 2>/dev/null | wc -l)
        if [ "$bpmn_count" -gt 0 ] || [ "$json_count" -gt 0 ]; then
            echo "     (✓ $bpmn_count archivo(s) .bpmn, $json_count archivo(s) .json)"
        fi
    fi

    echo "   • Estado parcial: $rel_state_dir"
    if [ -d "$STATE_DIR" ]; then
        state_count=$(find "$STATE_DIR" -name "*.json" 2>/dev/null | wc -l)
        if [ "$state_count" -gt 0 ]; then
            echo "     (✓ $state_count archivo(s) de estado)"
        fi
    fi

    if [ -f "$RL_BUFFER" ]; then
        echo "   • Experience buffer (RL): $rel_rl_buffer"
        buffer_size=$(wc -l < "$RL_BUFFER" 2>/dev/null || echo "0")
        if [ "$buffer_size" -gt 0 ]; then
            echo "     (✓ $(($buffer_size - 1)) experiencias)"
        fi
    else
        echo "   • Experience buffer (RL): (no generado)"
    fi

    if [ -f "$DISTILL_MODEL" ]; then
        echo "   • Modelo de política destilada: $rel_distill_model"
        model_size=$(du -h "$DISTILL_MODEL" 2>/dev/null | cut -f1)
        if [ ! -z "$model_size" ]; then
            echo "     (✓ Tamaño: $model_size)"
        fi
    else
        echo "   • Modelo de política destilada: (no generado)"
    fi
else
    # Enfoque offline: solo mostrar modelo
    OFFLINE_MODEL="$PROJECT_ROOT/results/bpi2017_train/distill/final_policy_model.pkl"
    if [ -f "$OFFLINE_MODEL" ]; then
        rel_offline_model=$(python3 -c "import os; print(os.path.relpath('$OFFLINE_MODEL', '$PROJECT_ROOT'))" 2>/dev/null || echo "$OFFLINE_MODEL")
        model_size=$(du -h "$OFFLINE_MODEL" 2>/dev/null | cut -f1)
        echo "   • Modelo de política (OFFLINE): $rel_offline_model"
        if [ ! -z "$model_size" ]; then
            echo "     (✓ Tamaño: $model_size)"
        fi
    else
        echo "   • Modelo de política (OFFLINE): (no generado)"
    fi
fi
