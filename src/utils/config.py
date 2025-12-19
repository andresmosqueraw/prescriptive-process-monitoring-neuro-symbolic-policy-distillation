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


def get_log_name_from_path(log_path: str) -> str:
    """
    Extrae el nombre del log desde la ruta (sin extensión).
    
    Args:
        log_path: Ruta al archivo del log
    
    Returns:
        Nombre del log sin extensión
    """
    filename = os.path.basename(log_path)
    log_name = os.path.splitext(filename)[0]
    # Si termina en .xes (por ejemplo, si era .xes.gz)
    if log_name.endswith('.xes'):
        log_name = os.path.splitext(log_name)[0]
    return log_name


def build_output_path(
    base_path: Optional[str],
    log_name: str,
    subdirectory: str,
    default_base: str = "data"
) -> str:
    """
    Construye una ruta de salida incluyendo el nombre del log.
    
    Si base_path es "results/simod/", log_name es "PurchasingExample" y subdirectory es "simod",
    retorna "results/PurchasingExample/simod/".
    
    Args:
        base_path: Ruta base desde config.yaml (ej: "results/simod/")
        log_name: Nombre del log (sin extensión)
        subdirectory: Subdirectorio final (ej: "simod", "state", "rl", "distill")
        default_base: Base por defecto si base_path es None
    
    Returns:
        Ruta completa con el nombre del log incluido
    """
    # Determinar directorio base del proyecto
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file))
    project_root = os.path.dirname(src_dir)
    
    if base_path is None:
        # Usar default_base
        final_path = os.path.join(project_root, default_base, f"generado-{subdirectory}")
    else:
        # Limpiar la ruta (remover barras finales)
        base_path = base_path.rstrip('/')
        
        # Si es absoluta, trabajar con ella directamente
        if os.path.isabs(base_path):
            # Ej: "/path/to/results/simod" -> "/path/to/results/{log_name}/simod"
            parts = base_path.split(os.sep)
            # Insertar log_name antes del último componente (subdirectory)
            if len(parts) > 1:
                parts.insert(-1, log_name)
                final_path = os.sep.join(parts)
            else:
                final_path = os.path.join(base_path, log_name, subdirectory)
        else:
            # Si es relativa, construir desde project_root
            # Ej: "results/simod" -> "results/{log_name}/simod"
            parts = base_path.split('/')
            if len(parts) >= 2:
                # Caso: "results/simod" -> "results/{log_name}/simod"
                base_dir = parts[0]  # "results"
                # Reemplazar el último componente con log_name y luego subdirectory
                final_path = os.path.join(project_root, base_dir, log_name, subdirectory)
            elif len(parts) == 1:
                # Caso: "simod" -> "{log_name}/simod"
                final_path = os.path.join(project_root, log_name, subdirectory)
            else:
                # Fallback: solo log_name/subdirectory
                final_path = os.path.join(project_root, log_name, subdirectory)
    
    return os.path.abspath(final_path)
