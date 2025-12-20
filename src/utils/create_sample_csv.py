#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para crear una muestra del log BPI 2017 CSV para pruebas rápidas.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar src/ al PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = src/utils/
src_dir = os.path.dirname(script_dir)  # src/
project_root = os.path.dirname(src_dir)  # proyecto raíz
# Agregar src/ al path para importar utils
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.config import load_config


def create_sample_from_csv(
    input_path: str,
    output_path: str,
    max_cases: int = 1000,
    sample_percentage: float = 0.1
) -> bool:
    """
    Crea una muestra del log CSV.
    
    Args:
        input_path: Ruta al archivo CSV original
        output_path: Ruta donde guardar la muestra
        max_cases: Número máximo de casos a extraer
        sample_percentage: Porcentaje de casos a extraer (0.0 a 1.0)
    
    Returns:
        True si se creó exitosamente, False en caso contrario
    """
    logger.info("=" * 80)
    logger.info("CREANDO MUESTRA DEL LOG BPI 2017 (CSV)")
    logger.info("=" * 80)
    logger.info(f"Archivo original: {input_path}")
    logger.info(f"Archivo de salida: {output_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"No se encontró el archivo: {input_path}")
        return False
    
    try:
        logger.info("Leyendo log CSV...")
        # Leer CSV en chunks si es muy grande
        df = pd.read_csv(input_path, low_memory=False)
        
        # Obtener casos únicos
        case_col = 'case:concept:name'
        if case_col not in df.columns:
            logger.error(f"No se encontró la columna {case_col}")
            return False
        
        unique_cases = df[case_col].unique()
        total_cases = len(unique_cases)
        logger.info(f"Total de casos en el log: {total_cases}")
        
        # Calcular número de casos a extraer
        # Priorizar max_cases si está especificado
        cases_by_percentage = int(total_cases * sample_percentage)
        if max_cases > 0:
            # Usar max_cases si está especificado (limitado al total disponible)
            num_cases_to_extract = min(max_cases, total_cases)
        else:
            # Usar porcentaje si max_cases no está especificado
            num_cases_to_extract = cases_by_percentage
        
        logger.info(f"Extrayendo {num_cases_to_extract} casos "
                   f"(de {total_cases} totales, {num_cases_to_extract/total_cases*100:.1f}%)")
        
        # Seleccionar casos aleatoriamente
        np.random.seed(42)  # Para reproducibilidad
        selected_cases = np.random.choice(unique_cases, size=num_cases_to_extract, replace=False)
        
        # Filtrar DataFrame
        df_sample = df[df[case_col].isin(selected_cases)].copy()
        logger.info(f"Casos seleccionados: {len(df_sample)} eventos de {num_cases_to_extract} casos")
        
        # Guardar muestra
        logger.info(f"Guardando muestra en: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_sample.to_csv(output_path, index=False)
        
        # Verificar tamaño
        original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        sample_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        logger.info("=" * 80)
        logger.info("MUESTRA CREADA EXITOSAMENTE")
        logger.info("=" * 80)
        logger.info(f"Casos originales: {total_cases}")
        logger.info(f"Casos en muestra: {num_cases_to_extract}")
        logger.info(f"Tamaño original: {original_size:.2f} MB")
        logger.info(f"Tamaño muestra: {sample_size:.2f} MB")
        logger.info(f"Reducción: {(1 - sample_size/original_size)*100:.1f}%")
        logger.info(f"Archivo guardado en: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creando muestra: {e}", exc_info=True)
        return False


def main():
    """Función principal"""
    config = load_config()
    if config is None:
        logger.error("No se pudo cargar la configuración")
        sys.exit(1)
    
    # Obtener rutas desde config
    log_config = config.get("log_config", {})
    bpi2017_config = log_config.get("bpi2017", {})
    
    # Encontrar directorio raíz del proyecto
    # Este script está en src/utils/, así que subimos DOS niveles
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = src/utils/
    utils_dir = os.path.dirname(script_dir)  # src/
    project_root = os.path.dirname(utils_dir)  # proyecto raíz
    
    # Ruta del log original - usar csv_path de bpi2017
    original_csv = bpi2017_config.get("csv_path", "logs/BPI2017/bpi-challenge-2017.csv")
    if not os.path.isabs(original_csv):
        original_csv = os.path.join(project_root, original_csv)
    
    # Ruta de salida para la muestra
    sample_csv = os.path.join(project_root, "logs/BPI2017/bpi-challenge-2017-sample.csv")
    
    # Crear muestra con 20 casos
    success = create_sample_from_csv(
        input_path=original_csv,
        output_path=sample_csv,
        max_cases=20,  # 20 casos para pruebas muy rápidas
        sample_percentage=1.0  # Ignorar porcentaje, usar max_cases directamente
    )
    
    if success:
        logger.info("")
        logger.info("✅ Muestra CSV creada exitosamente")
        logger.info(f"   Actualiza config.yaml para usar: logs/BPI2017/bpi-challenge-2017-sample.csv")
        sys.exit(0)
    else:
        logger.error("❌ Error creando la muestra")
        sys.exit(1)


if __name__ == "__main__":
    main()
