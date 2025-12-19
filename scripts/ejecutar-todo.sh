#!/bin/bash
# Script para ejecutar todo el pipeline:
#   1) extract_bpmn_json.py
#   2) compute_state.py
#   3) train_agent_in_gym.py
#   4) distill_policy.py

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_DIR="$PROJECT_ROOT/venv"
SRC_DIR="$PROJECT_ROOT/src"

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

# Función para manejar errores
handle_error() {
    echo ""
    echo "❌ Error en: $1"
    echo "   El proceso se detuvo"
    deactivate
    exit 1
}

# Paso 1: Ejecutar extract_bpmn_json.py
echo "=================================================================================="
echo "📋 PASO 1: Extrayendo BPMN y JSON con Simod"
echo "=================================================================================="
echo ""

if [ -f "$SRC_DIR/extract_bpmn_json.py" ]; then
    python "$SRC_DIR/extract_bpmn_json.py" || handle_error "extract_bpmn_json.py"
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

# Paso 2: Ejecutar compute_state.py
echo "=================================================================================="
echo "📋 PASO 2: Calculando estado parcial del proceso"
echo "=================================================================================="
echo ""

if [ -f "$SRC_DIR/compute_state.py" ]; then
    python "$SRC_DIR/compute_state.py" || handle_error "compute_state.py"
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

# Paso 3: Ejecutar train_agent_in_gym.py
echo "=================================================================================="
echo "📋 PASO 3: Entrenando agente (Causal-Gym / Neuro-Simbólico)"
echo "=================================================================================="
echo ""

if [ -f "$SRC_DIR/train_agent_in_gym.py" ]; then
    python "$SRC_DIR/train_agent_in_gym.py" || handle_error "train_agent_in_gym.py"
else
    echo "❌ No se encontró: $SRC_DIR/train_agent_in_gym.py"
    deactivate
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ PASO 3 COMPLETADO"
echo "=================================================================================="
echo ""

# Paso 4: Ejecutar distill_policy.py
echo "=================================================================================="
echo "📋 PASO 4: Destilando política (Policy Distillation / Imitation Learning)"
echo "=================================================================================="
echo ""

if [ -f "$SRC_DIR/distill_policy.py" ]; then
    python "$SRC_DIR/distill_policy.py" || handle_error "distill_policy.py"
else
    echo "❌ No se encontró: $SRC_DIR/distill_policy.py"
    deactivate
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ PASO 4 COMPLETADO"
echo "=================================================================================="
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

script_config = config.get('script_config', {})
output_dir = script_config.get('output_dir')
if output_dir:
    if not os.path.isabs(output_dir):
        base_dir = '$PROJECT_ROOT'
        output_dir = os.path.join(base_dir, output_dir)
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

script_config = config.get('script_config', {})
state_output_dir = script_config.get('state_output_dir')
if state_output_dir:
    if not os.path.isabs(state_output_dir):
        base_dir = '$PROJECT_ROOT'
        state_output_dir = os.path.join(base_dir, state_output_dir)
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

script_config = config.get('script_config', {})
distill_config = config.get('distill_config', {})
rl_output_dir = script_config.get('rl_output_dir')
input_csv = distill_config.get('input_csv')

# Preferir input_csv de distill_config, luego rl_output_dir
if input_csv:
    if not os.path.isabs(input_csv):
        base_dir = '$PROJECT_ROOT'
        input_csv = os.path.join(base_dir, input_csv)
    print(input_csv)
elif rl_output_dir:
    if not os.path.isabs(rl_output_dir):
        base_dir = '$PROJECT_ROOT'
        rl_output_dir = os.path.join(base_dir, rl_output_dir)
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

distill_config = config.get('distill_config', {})
output_model = distill_config.get('output_model')
if output_model:
    if not os.path.isabs(output_model):
        base_dir = '$PROJECT_ROOT'
        output_model = os.path.join(base_dir, output_model)
    print(output_model)
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
