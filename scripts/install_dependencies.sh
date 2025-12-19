#!/bin/bash
# Script para instalar dependencias de DeclarativeProcessSimulation en el venv de nuevo/

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
DECLARATIVE_DIR="$SCRIPT_DIR/../.."

echo "🔧 Instalando dependencias de DeclarativeProcessSimulation..."
echo ""

# Verificar que existe el entorno virtual
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ No se encontró el entorno virtual en: $VENV_DIR"
    echo "   Ejecuta primero: python3 -m venv venv"
    exit 1
fi

# Activar entorno virtual
source "$VENV_DIR/bin/activate"
echo "✅ Entorno virtual activado: $VENV_DIR"
echo "   Python: $(which python)"
echo "   Versión: $(python --version)"
echo ""

# Verificar que existe requirements.txt de DeclarativeProcessSimulation
REQUIREMENTS_FILE="$DECLARATIVE_DIR/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ No se encontró requirements.txt en: $REQUIREMENTS_FILE"
    exit 1
fi

echo "📦 Instalando dependencias desde: $REQUIREMENTS_FILE"
echo "   (Filtrando referencias a ongoing_bps_state/Prosimos que no aplican aquí)"
echo ""

# Crear archivo temporal sin la línea problemática de ongoing_bps_state
TEMP_REQUIREMENTS="/tmp/requirements_declarative_filtered.txt"
grep -v "ongoing_bps_state" "$REQUIREMENTS_FILE" > "$TEMP_REQUIREMENTS"

# Instalar dependencias
pip install --upgrade pip

# Intentar instalar desde requirements.txt (puede fallar en algunas versiones específicas)
echo "   Instalando dependencias (esto puede tardar varios minutos)..."
pip install -r "$TEMP_REQUIREMENTS" || {
    echo ""
    echo "⚠️  Algunas dependencias con versiones específicas fallaron"
    echo "   Instalando dependencias críticas manualmente..."
    
    # Instalar dependencias críticas sin versiones específicas
    pip install tensorflow keras pandas pm4py networkx scikit-learn hyperopt jellyfish opyenxes lxml matplotlib nltk ipywidgets xmltodict beautifulsoup4
    
    # Intentar instalar support-modules desde git
    pip install git+http://github.com/Mcamargo85/support_modules.git || echo "⚠️  No se pudo instalar support-modules desde git"
}

# Limpiar archivo temporal
rm -f "$TEMP_REQUIREMENTS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dependencias instaladas exitosamente"
    echo ""
    echo "🔍 Verificando instalación..."
    python -c "import tensorflow; print('✅ tensorflow:', tensorflow.__version__)" 2>&1
    python -c "import keras; print('✅ keras:', keras.__version__)" 2>&1
    python -c "import pandas; print('✅ pandas:', pandas.__version__)" 2>&1
    python -c "import pm4py; print('✅ pm4py:', pm4py.__version__)" 2>&1
else
    echo ""
    echo "❌ Error instalando dependencias"
    exit 1
fi

deactivate
echo ""
echo "🎉 ¡Instalación completada!"

