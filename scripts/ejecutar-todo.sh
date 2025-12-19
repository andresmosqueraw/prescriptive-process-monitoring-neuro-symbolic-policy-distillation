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

# Desactivar entorno virtual
deactivate

echo ""
echo "🎉 ¡Pipeline completado exitosamente!"
echo ""
echo "📁 Archivos generados:"
echo "   • BPMN y JSON: data/generado-simod/"
echo "   • Estado parcial: data/generado-state/"
if [ -f "data/generado-rl-train/experience_buffer.csv" ]; then
    echo "   • Experience buffer (RL): data/generado-rl-train/experience_buffer.csv"
else
    echo "   • Experience buffer (RL): (no generado)"
fi
if [ -f "data/final_policy_model.pkl" ]; then
    echo "   • Modelo de política destilada: data/final_policy_model.pkl"
else
    echo "   • Modelo de política destilada: (no generado)"
fi
