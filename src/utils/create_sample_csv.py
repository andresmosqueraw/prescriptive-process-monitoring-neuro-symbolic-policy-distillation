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
    n_cases_with_intervention: int = 20,
    n_cases_without_intervention: int = 20
) -> bool:
    """
    Crea una muestra balanceada del log CSV con casos con y sin intervención.
    
    Args:
        input_path: Ruta al archivo CSV original
        output_path: Ruta donde guardar la muestra
        n_cases_with_intervention: Número de casos CON intervención (T=1) a extraer
        n_cases_without_intervention: Número de casos SIN intervención (T=0) a extraer
    
    Returns:
        True si se creó exitosamente, False en caso contrario
    """
    logger.info("=" * 80)
    logger.info("CREANDO MUESTRA BALANCEADA DEL LOG BPI 2017 (CSV)")
    logger.info("=" * 80)
    logger.info(f"Archivo original: {input_path}")
    logger.info(f"Archivo de salida: {output_path}")
    logger.info(f"Casos CON intervención (T=1): {n_cases_with_intervention}")
    logger.info(f"Casos SIN intervención (T=0): {n_cases_without_intervention}")
    logger.info(f"Total de casos en muestra: {n_cases_with_intervention + n_cases_without_intervention}")
    
    if not os.path.exists(input_path):
        logger.error(f"No se encontró el archivo: {input_path}")
        return False
    
    try:
        logger.info("Leyendo log CSV...")
        df = pd.read_csv(input_path, low_memory=False)
        
        # Obtener casos únicos
        case_col = 'case:concept:name'
        act_col = 'concept:name'
        
        if case_col not in df.columns:
            logger.error(f"No se encontró la columna {case_col}")
            return False
        
        if act_col not in df.columns:
            logger.error(f"No se encontró la columna {act_col}")
            return False
        
        unique_cases = df[case_col].unique()
        total_cases = len(unique_cases)
        logger.info(f"Total de casos en el log: {total_cases}")
        
        # Identificar casos con intervención (basado en eda_bpi2017.py)
        intervention_activities = [
            'W_Call after offers',
            'W_Call incomplete files'
        ]
        
        logger.info("Identificando casos con y sin intervención...")
        mask_intervention = df[act_col].isin(intervention_activities)
        
        # Si no hay coincidencias exactas, buscar por patrón
        if mask_intervention.sum() == 0:
            logger.info("No se encontraron coincidencias exactas, buscando por patrón 'W_Call'...")
            mask_intervention = df[act_col].astype(str).str.contains('W_Call', case=False, na=False)
        
        # Obtener IDs de casos con intervención
        cases_with_intervention = df.loc[mask_intervention, case_col].unique()
        cases_without_intervention = np.setdiff1d(unique_cases, cases_with_intervention)
        
        n_total_with = len(cases_with_intervention)
        n_total_without = len(cases_without_intervention)
        
        logger.info(f"Casos CON intervención disponibles: {n_total_with}")
        logger.info(f"Casos SIN intervención disponibles: {n_total_without}")
        
        # Verificar que hay suficientes casos de cada tipo
        if n_total_with < n_cases_with_intervention:
            logger.warning(f"⚠️  Solo hay {n_total_with} casos con intervención, pero se solicitan {n_cases_with_intervention}")
            logger.warning(f"   Usando todos los {n_total_with} casos disponibles")
            n_cases_with_intervention = n_total_with
        
        if n_total_without < n_cases_without_intervention:
            logger.warning(f"⚠️  Solo hay {n_total_without} casos sin intervención, pero se solicitan {n_cases_without_intervention}")
            logger.warning(f"   Usando todos los {n_total_without} casos disponibles")
            n_cases_without_intervention = n_total_without
        
        # Seleccionar casos aleatoriamente de cada grupo
        np.random.seed(42)  # Para reproducibilidad
        
        if n_cases_with_intervention > 0:
            selected_with = np.random.choice(
                cases_with_intervention, 
                size=min(n_cases_with_intervention, n_total_with), 
                replace=False
            )
        else:
            selected_with = np.array([])
        
        if n_cases_without_intervention > 0:
            selected_without = np.random.choice(
                cases_without_intervention, 
                size=min(n_cases_without_intervention, n_total_without), 
                replace=False
            )
        else:
            selected_without = np.array([])
        
        # Combinar casos seleccionados
        selected_cases = np.concatenate([selected_with, selected_without])
        total_selected = len(selected_cases)
        
        logger.info(f"Casos seleccionados: {len(selected_with)} con intervención, {len(selected_without)} sin intervención")
        logger.info(f"Total de casos en muestra: {total_selected}")
        
        # Filtrar DataFrame
        df_sample = df[df[case_col].isin(selected_cases)].copy()
        logger.info(f"Eventos en muestra: {len(df_sample)} eventos de {total_selected} casos")
        
        # Verificar la muestra
        sample_cases_with = df_sample.loc[df_sample[act_col].isin(intervention_activities), case_col].unique()
        if len(sample_cases_with) == 0:
            sample_cases_with = df_sample.loc[
                df_sample[act_col].astype(str).str.contains('W_Call', case=False, na=False), 
                case_col
            ].unique()
        sample_cases_without = np.setdiff1d(selected_cases, sample_cases_with)
        
        logger.info(f"Verificación de muestra:")
        logger.info(f"  Casos CON intervención en muestra: {len(sample_cases_with)}")
        logger.info(f"  Casos SIN intervención en muestra: {len(sample_cases_without)}")
        
        # Guardar muestra
        logger.info(f"Guardando muestra en: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_sample.to_csv(output_path, index=False)
        
        # Verificar tamaño
        original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        sample_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        logger.info("=" * 80)
        logger.info("MUESTRA BALANCEADA CREADA EXITOSAMENTE")
        logger.info("=" * 80)
        logger.info(f"Casos originales: {total_cases}")
        logger.info(f"Casos en muestra: {total_selected}")
        logger.info(f"  - Con intervención (T=1): {len(sample_cases_with)}")
        logger.info(f"  - Sin intervención (T=0): {len(sample_cases_without)}")
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
    
    # Crear muestra balanceada: 20 casos con intervención + 20 casos sin intervención = 40 casos totales
    success = create_sample_from_csv(
        input_path=original_csv,
        output_path=sample_csv,
        n_cases_with_intervention=20,  # 20 casos CON intervención (T=1)
        n_cases_without_intervention=20  # 20 casos SIN intervención (T=0)
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
