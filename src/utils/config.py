"""
Utilidades para carga de configuración.
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Carga la configuración desde el archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración. Si es None, busca configs/config.yaml
                    relativo al directorio del proyecto.
    
    Returns:
        Diccionario con la configuración cargada, o None si hay error.
    """
    if config_path is None:
        # Determinar directorio base del proyecto
        # Asumimos que este módulo está en src/utils/
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file))
        base_dir = os.path.dirname(src_dir)
        config_path = os.path.join(base_dir, "configs", "config.yaml")
    
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception:
        return None
